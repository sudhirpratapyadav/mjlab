#!/usr/bin/env python3
"""Edit ONE task row of STATUS.md atomically (flock) and recompute the summary.

Five agents edit STATUS.md concurrently; never hand-edit the file — use this::

    python docs/cl_v2/update_status.py --task Lift-Cube --owner W1-a \
        --chosen "YCB 077 Rubik's cube (46 mm)" --gate G1=ok --gate G2=ok \
        --sr "1.000 -> 0.984 (128, abc1234)" --notes "grasp width 46 mm"

Columns: Task | Owner | Current asset | Chosen asset | G1..G7 | Teacher SR | Notes.
Unspecified fields are left as they are. ``--ledger`` appends a provenance row.
``--decision "D5|question|decided"`` appends a decision row.
"""

from __future__ import annotations

import argparse
import fcntl
import re
from datetime import date
from pathlib import Path

STATUS = Path(__file__).resolve().parent / "STATUS.md"
GATES = ["G1", "G2", "G3", "G4", "G5", "G6", "G7"]


def _split(row: str) -> list[str]:
  return [c.strip() for c in row.strip().strip("|").split("|")]


def _join(cells: list[str]) -> str:
  return "| " + " | ".join(cells) + " |"


def _recompute(lines: list[str]) -> list[str]:
  rows = []
  in_task_table = False
  for ln in lines:
    if ln.startswith("| Task | Owner |"):
      in_task_table = True
      continue
    if in_task_table and ln.startswith("|---"):
      continue
    if in_task_table and ln.startswith("|"):
      rows.append(_split(ln))
    elif in_task_table and not ln.startswith("|"):
      in_task_table = False
  n = len(rows)
  gi = {g: 4 + i for i, g in enumerate(GATES)}
  all_green = sum(all(r[gi[g]] == "ok" for g in GATES) for r in rows)
  g1 = sum(r[gi["G1"]] == "ok" for r in rows)
  g345 = sum(all(r[gi[g]] == "ok" for g in ("G3", "G4", "G5")) for r in rows)
  g6 = sum(r[gi["G6"]] == "ok" for r in rows)
  g7 = sum(r[gi["G7"]] == "ok" for r in rows)
  blocked = sum(any(r[gi[g]] == "x" for g in GATES) for r in rows)
  repl = {
    "Tasks in scope": n,
    "All seven gates green": all_green,
    "Asset chosen (G1)": g1,
    "Physics + init + success verified (G3–G5)": g345,
    "Visual accepted (G6)": g6,
    "Teacher regression recorded (G7)": g7,
    "Blocked / needs decision": blocked,
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
  ap.add_argument("--task", help="row name, e.g. Lift-Cube")
  ap.add_argument("--owner")
  ap.add_argument("--current")
  ap.add_argument("--chosen")
  ap.add_argument("--gate", action="append", default=[], help="G3=ok | G3=x | G3=~ | G3=-")
  ap.add_argument("--sr")
  ap.add_argument("--notes")
  ap.add_argument("--append-notes")
  ap.add_argument("--ledger", help="'key | asset | url | license | real scale checked | used by'")
  ap.add_argument("--decision", help="'D5 | question | decided'")
  ap.add_argument("--infra", help="'item-substring | state | notes'")
  a = ap.parse_args()

  with open(STATUS, "r+") as f:
    fcntl.flock(f, fcntl.LOCK_EX)
    lines = f.read().split("\n")
    if a.task:
      hit = False
      for i, ln in enumerate(lines):
        if ln.startswith(f"| {a.task} |"):
          c = _split(ln)
          if len(c) < 13:
            c += [""] * (13 - len(c))
          if a.owner is not None: c[1] = a.owner
          if a.current is not None: c[2] = a.current
          if a.chosen is not None: c[3] = a.chosen
          for g in a.gate:
            k, v = g.split("=", 1)
            c[4 + GATES.index(k)] = v
          if a.sr is not None: c[11] = a.sr
          if a.notes is not None: c[12] = a.notes
          if a.append_notes: c[12] = (c[12] + "; " if c[12] else "") + a.append_notes
          lines[i] = _join(c)
          hit = True
          break
      if not hit:
        raise SystemExit(f"row '{a.task}' not found in {STATUS}")
    if a.ledger:
      cells = [x.strip() for x in a.ledger.split("|")]
      idx = max(i for i, ln in enumerate(lines) if ln.startswith("| Key | Asset |")) + 1
      while idx + 1 < len(lines) and lines[idx + 1].startswith("|"):
        idx += 1
      # replace the empty placeholder row if present
      if lines[idx].strip().replace("|", "").strip() == "":
        lines[idx] = _join(cells)
      else:
        lines.insert(idx + 1, _join(cells))
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
        if ln.startswith("| `") and sub in ln:
          c = _split(ln)
          c[2] = state
          c[3] = notes
          lines[i] = _join(c)
    lines = _recompute(lines)
    f.seek(0)
    f.truncate()
    f.write("\n".join(lines))
  print("STATUS.md updated")


if __name__ == "__main__":
  main()
