#!/usr/bin/env python3
"""Edit ONE task row of cl_v3/STATUS.md atomically (flock) and recompute the summary.

Several agents edit STATUS.md concurrently; never hand-edit the file — use this::

    python docs/cl_v3/update_status.py --task Stack-Cube --owner W1-G \
        --set "Init spec=cube yaw ±π, base yaw ±π, robot ±10°" --gate I=ok --gate B=ok \
        --set "Baseline SR (n, HEAD)=0.480 (128, abc1234)" --append-notes "..."

Columns: Task | Owner | Group | v2 SR | Init spec | I | B | A | S | T | V |
         Baseline SR (n, HEAD) | Current SR (n, HEAD) | Strategy | Notes
``--set "Column=value"`` sets any column by header name. ``--decision "D3|question|decided"``
appends a decision row. ``--infra "item-substring|state|notes"`` edits an infra row.
"""

from __future__ import annotations

import argparse
import fcntl
import re
from datetime import date
from pathlib import Path

STATUS = Path(__file__).resolve().parent / "STATUS.md"
GATES = ["I", "B", "A", "S", "T", "V"]
HEADER_PREFIX = "| Task | Owner | Group |"


def _split(row: str) -> list[str]:
  return [c.strip() for c in row.strip().strip("|").split("|")]


def _join(cells: list[str]) -> str:
  return "| " + " | ".join(cells) + " |"


def _sr(cell: str) -> float | None:
  m = re.match(r"\s*([01]\.\d+)", cell)
  return float(m.group(1)) if m else None


def _n(cell: str) -> int:
  m = re.search(r"\((?:n\s*=\s*)?(\d+)", cell)
  return int(m.group(1)) if m else 0


def _recompute(lines: list[str]) -> list[str]:
  header = None
  rows = []
  in_table = False
  for ln in lines:
    if ln.startswith(HEADER_PREFIX):
      header = _split(ln)
      in_table = True
      continue
    if in_table and ln.startswith("|---"):
      continue
    if in_table and ln.startswith("|"):
      rows.append(_split(ln))
    elif in_table:
      in_table = False
  if header is None:
    return lines
  col = {h: i for i, h in enumerate(header)}
  gi = {g: col[g] for g in GATES}
  cur = col["Current SR (n, HEAD)"]

  def ok(r, g):
    return len(r) > gi[g] and r[gi[g]] == "ok"

  def at(r, thr):
    if len(r) <= cur:
      return False
    v = _sr(r[cur])
    return v is not None and v >= thr and _n(r[cur]) >= 128

  repl = {
    "Tasks in scope": len(rows),
    "Init spec frozen (I)": sum(ok(r, "I") for r in rows),
    "Baselined on v3 spec (B)": sum(ok(r, "B") for r in rows),
    "Teacher >= 0.90 @ n=128 (T)": sum(ok(r, "T") or at(r, 0.90) for r in rows),
    "Teacher >= 0.97 @ n=128": sum(at(r, 0.97) for r in rows),
    "Video published (V)": sum(ok(r, "V") for r in rows),
    "All six gates green": sum(all(ok(r, g) for g in GATES) for r in rows),
    "Blocked / needs decision": sum(any(len(r) > gi[g] and r[gi[g]] == "x" for g in GATES) for r in rows),
  }
  out = []
  for ln in lines:
    m = re.match(r"^\| (.+?) \| (\d+) \|$", ln.strip())
    if m and m.group(1) in repl:
      out.append(f"| {m.group(1)} | {repl[m.group(1)]} |")
    elif ln.startswith("Last updated:"):
      out.append(f"Last updated: {date.today().isoformat()} (recomputed from rows by update_status.py)")
    else:
      out.append(ln)
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", help="row name, e.g. Stack-Cube")
  ap.add_argument("--owner")
  ap.add_argument("--measured-on", help="Measurement revision note shown above the scoreboard")
  ap.add_argument("--gate", action="append", default=[], help="I=ok | B=~ | T=x | V=-")
  ap.add_argument("--set", action="append", default=[], help='"Column=value" (header name)')
  ap.add_argument("--notes")
  ap.add_argument("--append-notes")
  ap.add_argument("--decision", help="'D3 | question | decided'")
  ap.add_argument("--infra", help="'item-substring | state | notes'")
  a = ap.parse_args()

  with open(STATUS, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.read().split("\n")
    header = next((_split(ln) for ln in lines if ln.startswith(HEADER_PREFIX)), None)
    if header is None:
      raise SystemExit("task table header not found")
    col = {h: i for i, h in enumerate(header)}
    if a.task:
      hit = False
      for i, ln in enumerate(lines):
        if ln.startswith(f"| {a.task} |"):
          c = _split(ln)
          if len(c) < len(header):
            c += [""] * (len(header) - len(c))
          if a.owner is not None:
            c[col["Owner"]] = a.owner
          for g in a.gate:
            k, v = g.split("=", 1)
            c[col[k]] = v
          for s in a.set:
            k, v = s.split("=", 1)
            if k not in col:
              raise SystemExit(f"unknown column '{k}'; columns: {list(col)}")
            c[col[k]] = v.replace("|", "/")
          if a.notes is not None:
            c[col["Notes"]] = a.notes.replace("|", "/")
          if a.append_notes:
            n = a.append_notes.replace("|", "/")
            c[col["Notes"]] = (c[col["Notes"]] + "; " if c[col["Notes"]] else "") + n
          lines[i] = _join(c)
          hit = True
          break
      if not hit:
        raise SystemExit(f"row '{a.task}' not found in {STATUS}")
    if a.decision:
      cells = [x.strip() for x in a.decision.split("|")]
      cells = [cells[0], cells[1], date.today().isoformat(), cells[2] if len(cells) > 2 else ""]
      idx = max(i for i, ln in enumerate(lines) if ln.startswith("| # | Question |")) + 1
      while idx + 1 < len(lines) and lines[idx + 1].startswith("|"):
        idx += 1
      lines.insert(idx + 1, _join(cells))
    if a.infra:
      sub, state, notes = [x.strip() for x in a.infra.split("|")]
      for i, ln in enumerate(lines):
        if ln.startswith("| ") and sub in ln and "| W0 |" in ln:
          c = _split(ln)
          c[2] = state
          c[3] = notes
          lines[i] = _join(c)
    if a.measured_on is not None:
      lines = [f"Measured on HEAD: {a.measured_on}" if ln.startswith("Measured on HEAD:") else ln for ln in lines]
    lines = _recompute(lines)
    f.seek(0)
    f.truncate()
    f.write("\n".join(lines))
  print("STATUS.md updated")


if __name__ == "__main__":
  main()
