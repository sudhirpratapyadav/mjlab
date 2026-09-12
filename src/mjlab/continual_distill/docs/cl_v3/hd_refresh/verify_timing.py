"""Check every encoded HD clip against its recorded simulation duration."""

import argparse
import json
from pathlib import Path
import imageio_ffmpeg

p = argparse.ArgumentParser()
p.add_argument("--root", type=Path, required=True)
p.add_argument("--expected", type=int, default=25)
a = p.parse_args()
folders = sorted((a.root / "videos").glob("*/result.json"))
assert len(folders) == a.expected, (len(folders), a.expected)
checks = []
for path in folders:
  r = json.loads(path.read_text())
  assert r["physics_dt"] == 0.005 and r["control_dt"] == 0.02 and r["decimation"] == 4
  assert r["width"] == 1920 and r["height"] == 1080 and r["fps"] == 50
  if 0 < r["num_success"] < r["n"]:
    assert r["has_teacher_clip"] and r["has_failure_clip"], path
  for label, filename in [("success", "teacher.mp4"), ("failure", "failure.mp4")]:
    clip = path.parent / filename
    if not clip.exists():
      continue
    timing = r["clips"][label]
    reader = imageio_ffmpeg.read_frames(str(clip))
    meta = next(reader)
    reader.close()
    frames, duration = imageio_ffmpeg.count_frames_and_secs(str(clip))
    assert meta["size"] == (1920, 1080) and meta["fps"] == 50, (clip, meta)
    assert frames == timing["encoded_frames"] == timing["control_steps"], (
      clip,
      frames,
      timing,
    )
    assert abs(duration - timing["simulation_seconds"]) < 0.011, (
      clip,
      duration,
      timing,
    )
    checks.append(
      dict(
        task=path.parent.name,
        outcome=label,
        width=1920,
        height=1080,
        fps=50,
        frames=frames,
        duration=duration,
        simulation_seconds=timing["simulation_seconds"],
      )
    )
report = dict(
  tasks=len(folders),
  clips=len(checks),
  physics_hz=200,
  control_hz=50,
  all_intermediate_rates_have_both_clips=True,
  checks=checks,
)
(a.root / "timing_verification.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
