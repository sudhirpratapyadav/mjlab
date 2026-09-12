"""Check recorded state/video integrity and assemble the goal-state contact sheet."""

import hashlib
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image


def main():
  root = Path("docs/task_geometry_review/verified_tasks")
  rows = json.loads((root / "results.json").read_text())
  assert len(rows) == 25
  source = hashlib.sha256()
  for folder in (
    "src/mjlab/tasks/manipulation",
    "src/mjlab/asset_zoo",
    "src/mjlab/continual_distill/classical",
  ):
    for path in sorted(Path(folder).rglob("*")):
      if path.suffix in (".py", ".xml"):
        source.update(str(path).encode())
        source.update(path.read_bytes())
  sheet = Image.new("RGB", (1920, 1440))
  for index, row in enumerate(rows):
    folder = root / row["task"]
    assert row["source_sha256"] == source.hexdigest(), row["task"]
    assert row["oracle"]["accepted"], row["task"]
    assert not row["oracle"]["deep_contacts"], row["task"]
    assert all(
      reset["finite"] and not reset["success"] and not reset["deep_contacts"]
      for reset in row["resets"]
    ), row["task"]
    with np.load(folder / "trajectory.npz") as trajectory:
      assert len(trajectory["qpos"]) == row["rollout"]["steps"] + 1
      assert all(np.isfinite(trajectory[key]).all() for key in trajectory.files)
    with np.load(folder / "goal_state.npz") as state:
      assert all(np.isfinite(state[key]).all() for key in state.files)
    video = str(folder / "rollout.mp4")
    subprocess.run(
      [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", video, "-f", "null", "-"],
      check=True,
    )
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(video)
    assert frames == row["video"]["frames"], row["task"]
    assert abs(seconds - row["rollout"]["simulation_seconds"]) <= 0.11
    with Image.open(folder / "goal_detail.png") as detail:
      sheet.paste(detail.resize((384, 288)), ((index % 5) * 384, (index // 5) * 288))
  sheet.save(root / "all_goals.jpg")
  result = dict(
    tasks=len(rows),
    decoded_videos=len(rows),
    goal_states_without_deep_contacts=len(rows),
    finite_trajectories=len(rows),
    source_sha256=source.hexdigest(),
  )
  (root / "artifact_verification.json").write_text(json.dumps(result, indent=2) + "\n")
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
