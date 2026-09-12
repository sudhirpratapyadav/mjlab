"""Validate matched measurements and attach the comparison to reviewed clips."""

import argparse
import json
from pathlib import Path

import imageio_ffmpeg


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--video", type=Path, required=True)
  parser.add_argument("--strategy", required=True)
  parser.add_argument("--notes", required=True)
  args = parser.parse_args()
  baseline = json.loads(args.baseline.read_text())
  path = args.video / "result.json"
  result = json.loads(path.read_text())
  for key in (
    "task_id",
    "n",
    "stats_seed",
    "stats_batch_size",
    "device",
    "evaluation_protocol",
    "task_revision",
  ):
    assert result[key] == baseline[key], key
  assert result["n"] == 128
  for row in (baseline, result):
    assert len(row["outcomes"]) == row["n"]
    assert sum(x["success"] for x in row["outcomes"]) == row["num_success"]
  assert result["num_success"] > baseline["num_success"], "No measured improvement"
  for label, filename in (("success", "teacher.mp4"), ("failure", "failure.mp4")):
    if label not in result["clips"]:
      continue
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(args.video / filename))
    clip = result["clips"][label]
    assert frames == clip["encoded_frames"]
    assert abs(seconds - clip["simulation_seconds"]) < 0.03
  paired = dict(gained=0, lost=0, both_success=0, both_failure=0)
  for before, after in zip(baseline["outcomes"], result["outcomes"], strict=True):
    assert before["env"] == after["env"]
    if before["success"] == after["success"]:
      paired["both_success" if after["success"] else "both_failure"] += 1
    else:
      paired["gained" if after["success"] else "lost"] += 1
  result["baseline"] = {
    key: baseline[key]
    for key in (
      "n",
      "num_success",
      "sr",
      "stats_seed",
      "stats_batch_size",
      "device",
      "source_sha256",
      "task_revision",
    )
  }
  result["baseline"]["teacher_override_sha256"] = baseline.get(
    "teacher_override_sha256"
  )
  result["paired_changes"] = paired
  path.write_text(json.dumps(result, indent=2) + "\n")
  task = dict(
    init_spec="Unchanged randomized training distribution, with observation noise and original episode budget.",
    strategy=args.strategy,
    notes=args.notes,
    gates={
      "I": "ok",
      "B": "ok",
      "A": "ok",
      "S": "ok",
      "T": "ok" if result["pass_bar"] else "~",
      "V": "ok",
    },
  )
  (args.video / "task.json").write_text(json.dumps(task, indent=2) + "\n")
  print(
    json.dumps(
      dict(
        task=result["task_id"],
        before=baseline["num_success"],
        after=result["num_success"],
        paired=paired,
      ),
      indent=2,
    )
  )


if __name__ == "__main__":
  main()
