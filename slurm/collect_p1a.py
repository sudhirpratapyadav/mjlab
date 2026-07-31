#!/usr/bin/env python3
"""Collect P0 run results from logs and refresh the status tables in
docs/P1_EXPERIMENTS.md.

Each run's final per-task success rates are the LAST block of
"  - <Task>: KL=... | StudentSucc=..." lines in its log; the final average SR is
their mean. A run counts as complete only once "Training complete" appears.

Usage:  python3 slurm/collect_p0.py [--write]
        (without --write, prints the table and changes nothing)
"""
import argparse
import re
import statistics
from pathlib import Path

REPO = Path("/ihub/homedirs/svs_ald/sudhir/mjlab")
LOGS = REPO / "logs"
DOC = REPO / "src/mjlab/continual_distill/docs/P1_EXPERIMENTS.md"

SUCC_RE = re.compile(r"^\s+- (\w+): KL=[\d.eE+-]+ \| .*StudentSucc=([\d.]+)", re.M)

RUNS = (
    [(f"ewc_{o}_s{s}", "P1-5") for o in ("best", "worst") for s in (0, 1, 2)]
    + [(f"l2_best_s{s}", "P1-5") for s in (0, 1, 2)]
    + [(f"lrw{w}_lr1e5_s{s}", "LRW") for w in (512, 1024, 2048, 4096) for s in (0, 1, 2)]
)


def parse(name):
    """Return (status, per_task dict, avg) for one run."""
    log = LOGS / f"cl_{name}.log"
    if not log.exists():
        return "queued", {}, None
    text = log.read_text(errors="ignore")
    done = "Training complete" in text

    # Take the final evaluation sweep: the last contiguous run of per-task lines.
    hits = SUCC_RE.findall(text)
    if not hits:
        return ("running" if not done else "complete"), {}, None

    # Keep the last occurrence of each task name (final eval overwrites earlier ones).
    per_task = {}
    for task, val in hits:
        per_task[task] = float(val)
    avg = statistics.fmean(per_task.values()) if per_task else None

    if done:
        return "complete", per_task, avg
    return "running", per_task, avg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="update the results section of the doc")
    args = ap.parse_args()

    icon = {"queued": "⏳", "running": "🔄", "complete": "✅", "failed": "❌"}
    rows, done_n = [], 0
    for name, group in RUNS:
        status, per_task, avg = parse(name)
        if status == "complete":
            done_n += 1
        rows.append((group, name, status, per_task, avg))
        avg_s = f"{avg:.3f}" if avg is not None else "—"
        detail = "  ".join(f"{k}={v:.2f}" for k, v in sorted(per_task.items()))
        print(f"{icon[status]} {group:5s} {name:20s} avg={avg_s:>6s}  {detail}")

    print(f"\ncomplete: {done_n}/{len(RUNS)}")

    # group means
    print("\n-- group means (complete runs only) --")
    for group in ("P1-5", "LRW"):
        vals = [a for g, _, s, _, a in rows if g == group and s == "complete" and a is not None]
        if vals:
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(f"{group}: {statistics.fmean(vals):.3f} ± {sd:.3f}  (n={len(vals)})")

    if args.write and DOC.exists():
        text = DOC.read_text()
        block = ["## Results", "", "_Auto-collected by `slurm/collect_p0.py`._", "",
                 "| group | run | status | avg SR | per-task |", "|---|---|---|---|---|"]
        for group, name, status, per_task, avg in rows:
            avg_s = f"{avg:.3f}" if avg is not None else "—"
            detail = ", ".join(f"{k} {v:.2f}" for k, v in sorted(per_task.items())) or "—"
            block.append(f"| {group} | `{name}` | {icon[status]} {status} | {avg_s} | {detail} |")
        block.append("")
        new = re.sub(r"## Results\n.*?(?=\n## )", "\n".join(block) + "\n", text, flags=re.S)
        DOC.write_text(new)
        print(f"\nwrote {DOC}")


if __name__ == "__main__":
    main()
