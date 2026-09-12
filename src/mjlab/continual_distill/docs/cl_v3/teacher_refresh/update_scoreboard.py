"""Keep corrected-rule measurements separate from historical teacher rates."""

import json
import subprocess
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
status = here.parent / "STATUS.md"
lines = status.read_text().splitlines()
header = next(line for line in lines if line.startswith("| Task |"))
columns = [cell.strip() for cell in header.strip("|").split("|")]
current_index = columns.index("Current SR (n, HEAD)")
rows = {}
for line in lines:
  cells = [cell.strip() for cell in line.strip("|").split("|")]
  if len(cells) == len(columns) and cells[0] not in ("Task", "---"):
    rows[cells[0]] = cells


def update(args):
  subprocess.run(
    [sys.executable, str(here.parent / "update_status.py"), *args],
    check=True,
    capture_output=True,
  )


for name, row in rows.items():
  if name in ("Reorient-Object", "Cage-Drag", "Peg-Insertion"):
    continue
  prior = row[current_index]
  if not prior.startswith("Historical:"):
    update(
      [
        "--task",
        name,
        "--gate",
        "B=~",
        "--gate",
        "T=~",
        "--gate",
        "V=~",
        "--set",
        "Current SR (n, HEAD)=Historical: " + prior,
        "--append-notes",
        "September 11 audit: previous rate/video retained as historical evidence; corrected-rule confirmation pending.",
      ]
    )

for name, directory in (
  ("Cage-Drag", "baseline32"),
  ("Peg-Insertion", "gpu_baseline32"),
):
  result = json.loads(
    (here / directory / f"Mjlab-{name}-Franka/result.json").read_text()
  )
  measured = f"{result['sr']:.4f} ({result['n']}, {result['source_sha256'][:12]})"
  update(
    [
      "--task",
      name,
      "--owner",
      "teacher-refresh",
      "--gate",
      "B=~",
      "--gate",
      "T=~",
      "--gate",
      "V=~",
      "--set",
      "Baseline SR (n, HEAD)=" + measured,
      "--set",
      "Current SR (n, HEAD)=" + measured,
      "--notes",
      f"2026-09-11 corrected rules, {result['device']}, seed {result['stats_seed']}; n=32 iteration baseline only. Original controller retained: tested alternatives did not improve it. Historical website clips await replacement; see teacher_refresh/WORK_LOG.md.",
    ]
  )

result = json.loads(
  (here / "videos/Mjlab-Reorient-Object-Franka/result.json").read_text()
)
baseline = result["baseline"]
update(
  [
    "--task",
    "Reorient-Object",
    "--owner",
    "teacher-refresh",
    "--gate",
    "B=ok",
    "--gate",
    "A=ok",
    "--gate",
    "S=ok",
    "--gate",
    "T=~",
    "--gate",
    "V=ok",
    "--set",
    f"Baseline SR (n, HEAD)={baseline['sr']:.4f} (128, {baseline['source_sha256'][:12]}; archived original teacher)",
    "--set",
    f"Current SR (n, HEAD)={result['sr']:.4f} (128, {result['source_sha256'][:12]})",
    "--set",
    "Strategy=Preserve the acquired grasp across CLOSE to LIFT; remove the accidental one-step OPEN pulse.",
    "--notes",
    "2026-09-11 corrected rules: 97/128 to 100/128; identical initial qpos, seed 20260911, GPU 3. Modest measured increase, below 90%, no significance claim. Exact measured success/failure clips published; 37 regression tests passed.",
    "--measured-on",
    "September 11 audited task rules; source fingerprints accompany each fresh measurement. Historical rows await confirmation.",
  ]
)
print(
  "Updated 25 rows through update_status.py; historical rates excluded from current competence counts."
)
