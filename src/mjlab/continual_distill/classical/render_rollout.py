#!/usr/bin/env python3
"""Render a classical-teacher rollout to mp4, for any registered task.

Replaces the old hardcoded-4-task / PNG-frame version of this script. This one:

  * Reads the task list from ``CLASSICAL_POLICIES``
    (``mjlab.continual_distill.classical.CLASSICAL_POLICIES``) instead of a
    hardcoded dict, so it works for any task with a registered teacher, including
    ones added after this file was written.
  * Writes H.264 mp4, not PNG frames. Two clips per task, each optional depending
    on what the rollout actually produced:
      - ``teacher.mp4``  a representative SUCCESSFUL episode (omitted if the
        teacher never succeeded in the sampled episodes).
      - ``failure.mp4``  a representative FAILED episode (omitted if the teacher
        never failed).
    Plus ``thumb.jpg`` (first frame of whichever of the two exists) and
    ``result.json`` (SR/n/HEAD/date sidecar for the site to render on the task's
    card).
  * Reuses the offscreen-render path from ``mjlab.scripts.record_task_videos``:
    same viewer-framing overrides (WORLD-origin camera solved from the scene's own
    bounding box) and the same moviepy H.264 encode call. Do not reinvent this —
    import and call it.

Two-phase design, so ``--num-episodes`` can be the real measurement protocol
number (32 while iterating, 128 to claim the 0.90 bar) without rendering (and
disk-writing) 128 episodes worth of frames:

  1. **Stats phase** — batched, no rendering. ``--num-episodes`` envs (chunked by
     ``--batch-size``) each run exactly one episode; success is read the same way
     ``test_classical.py`` reads it (``cmd.episode_success``, maxed over the
     episode because the env auto-resets on timeout and would otherwise clear the
     flag before a post-loop read saw it). This is the SR/n written to
     ``result.json`` and is the same number a caller would get from
     ``test_classical.py`` at the same n.
  2. **Render phase** — single env (``num_envs=1``), rendered. Runs up to
     ``--max-render-episodes`` sequential single-env episodes, stopping as soon as
     both a success and a failure clip have been captured (or the render budget
     is exhausted). This bounds render cost independent of ``--num-episodes``.

Usage:
    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python \
        -m mjlab.continual_distill.classical.render_rollout \
        --task Mjlab-Lift-Cube-Franka \
        --num-episodes 128 \
        --out /path/to/local/Mjlab-Lift-Cube-Franka \
        --device cuda:0

Must run on a GPU node with ``MUJOCO_GL=egl`` (the login node has no EGL device).

``--debug`` keeps the old script's cheap single-env eyeballing behaviour alive: it
additionally dumps PNG frames (every ``--debug-every`` steps) and a per-step phase
print for the render-phase episodes, into ``<out>/debug_frames/``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.record_task_videos import RecordConfig, _autoframe_camera
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

#: Phase-1 competence bar (phase_1_plan.md section 1 / 4).
COMPETENCE_BAR = 0.90

_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git_head() -> str:
  try:
    out = subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"],
      cwd=_REPO_ROOT,
      capture_output=True,
      text=True,
      check=True,
    )
    return out.stdout.strip()
  except Exception:  # noqa: BLE001 — HEAD lookup must never crash the render.
    return "unknown"


def _build_env(
  task_id: str, num_envs: int, device: str, render: bool, width: int, height: int
) -> ManagerBasedRlEnv:
  env_cfg = load_env_cfg(task_id, play=False)
  env_cfg.scene.num_envs = num_envs
  if render:
    env_cfg.viewer.width = width
    env_cfg.viewer.height = height
    # Same framing override as record_task_videos.record_task: the task cfgs
    # track link0 (TRACKING mode ignores `lookat`), so a WORLD free camera is
    # needed for lookat to take effect; lookat/distance are then solved from the
    # scene's own bounding box in _autoframe_camera, since a fixed guess frames
    # some tasks (mechanisms at z=0.5, floor-level lift scenes) badly.
    from mjlab.viewer import ViewerConfig

    env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    # Side three-quarter view from the robot's right (-y), slightly in front of the
    # plate-mounted mechanisms so their handles face the camera (azimuth 150 looked at
    # every mechanism from BEHIND its plate and hid the handle; lead, cl_v2).
    env_cfg.viewer.elevation = -24.0
    env_cfg.viewer.azimuth = 60.0
    env_cfg.viewer.lookat = (0.3, 0.0, 0.4)
    env_cfg.viewer.distance = 1.6
    env_cfg.viewer.env_idx = 0
    return ManagerBasedRlEnv(
      cfg=env_cfg, device=device, render_mode="rgb_array"
    )
  return ManagerBasedRlEnv(cfg=env_cfg, device=device)


def _success_term(env: ManagerBasedRlEnv):
  return env.command_manager.get_term(next(iter(env.command_manager.active_terms)))


@dataclass
class Stats:
  n: int
  num_success: int

  @property
  def sr(self) -> float:
    return self.num_success / self.n if self.n else 0.0


def run_stats_phase(
  task_id: str, num_episodes: int, batch_size: int, device: str
) -> Stats:
  """Batched, unrendered: ``num_episodes`` envs, one episode each, chunked."""
  total = 0
  succ = 0
  remaining = num_episodes
  while remaining > 0:
    b = min(batch_size, remaining)
    env = _build_env(task_id, b, device, render=False, width=0, height=0)
    try:
      policy = CLASSICAL_POLICIES[task_id](num_envs=b)
      episode_length = int(env.max_episode_length)
      obs, _ = env.reset()
      policy.reset()
      cmd = _success_term(env)
      ep_success = np.zeros(b)
      for _t in range(episode_length):
        actions = policy(obs["policy"].detach().cpu().numpy())
        obs, *_ = env.step(torch.from_numpy(actions).to(device))
        ep_success = np.maximum(
          ep_success, cmd.episode_success.detach().cpu().numpy()
        )
      succ += int(ep_success.sum())
      total += b
    finally:
      env.close()
    remaining -= b
  return Stats(n=total, num_success=succ)


def _resample_fps(frames: list[np.ndarray], control_dt: float, fps: int):
  src_fps = 1.0 / control_dt
  if fps < src_fps:
    idx = np.linspace(0, len(frames) - 1, max(1, int(round(len(frames) / src_fps * fps))))
    return [frames[int(round(i))] for i in idx]
  return frames


def _write_mp4(frames: list[np.ndarray], fps: int, out_path: Path) -> None:
  from moviepy import ImageSequenceClip

  out_path.parent.mkdir(parents=True, exist_ok=True)
  clip = ImageSequenceClip(frames, fps=fps)
  clip.write_videofile(str(out_path), codec="libx264", logger=None, audio=False)
  clip.close()


#: If the stats phase already measured SR outside this band (on n >= 32, so it is
#: not a fluke), do not keep burning render episodes hunting for the other outcome
#: after the first opposite-of-guaranteed clip is captured — e.g. Reach-Target
#: measures 1.000 over 128 episodes; a failure clip is not going to show up in the
#: next 16 single-env attempts, and spending the full render budget looking for one
#: just makes every near-perfect or characterised-zero task slow to render.
_NEAR_CERTAIN_LOW = 0.02
_NEAR_CERTAIN_HIGH = 0.98
_NEAR_CERTAIN_MIN_N = 32


def run_render_phase(
  task_id: str,
  device: str,
  out_dir: Path,
  max_render_episodes: int,
  fps: int,
  width: int,
  height: int,
  debug: bool,
  debug_every: int,
  stats: "Stats | None" = None,
) -> dict:
  """Single-env, rendered. Returns which clips were written and episode counts."""
  skip_success_search = (
    stats is not None
    and stats.n >= _NEAR_CERTAIN_MIN_N
    and stats.sr <= _NEAR_CERTAIN_LOW
  )
  skip_failure_search = (
    stats is not None
    and stats.n >= _NEAR_CERTAIN_MIN_N
    and stats.sr >= _NEAR_CERTAIN_HIGH
  )
  env = _build_env(task_id, 1, device, render=True, width=width, height=height)
  result = {
    "has_teacher_clip": False,
    "has_failure_clip": False,
    "render_episodes_used": 0,
  }
  debug_dir = out_dir / "debug_frames"
  try:
    # Reset FIRST: a mocap asset only takes its sampled per-env pose when its command
    # term resamples, i.e. on the first reset. Framing before that measures the MJCF
    # pose -- Throw-To-Bin's bin is written to x 0.78-0.90 but sits at 0.55 in the XML.
    env.reset()
    _autoframe_camera(env, RecordConfig())
    policy = CLASSICAL_POLICIES[task_id](num_envs=1)
    episode_length = int(env.max_episode_length)
    control_dt = float(env.step_dt)
    cmd = _success_term(env)

    success_frames: list[np.ndarray] | None = None
    failure_frames: list[np.ndarray] | None = None

    if debug:
      import imageio.v2 as imageio

      debug_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(max_render_episodes):
      have_success = success_frames is not None or skip_success_search
      have_failure = failure_frames is not None or skip_failure_search
      if have_success and have_failure:
        break
      obs, _ = env.reset()
      policy.reset()
      ep_success = np.zeros(1)
      frames: list[np.ndarray] = []
      for t in range(episode_length):
        obs_np = obs["policy"].detach().cpu().numpy()
        actions = policy(obs_np)
        obs, *_ = env.step(torch.from_numpy(actions).to(device))
        ep_success = np.maximum(
          ep_success, cmd.episode_success.detach().cpu().numpy()
        )
        frame = env.render()
        if frame is not None:
          frame = np.asarray(frame)
          if frame.ndim == 4:
            frame = frame[0]
          if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
          frames.append(frame)
          if debug and t % debug_every == 0:
            ph = int(policy._phase[0]) if hasattr(policy, "_phase") else -1
            imageio.imwrite(debug_dir / f"ep{ep:02d}_t{t:03d}_ph{ph}.png", frame)
      result["render_episodes_used"] += 1
      succeeded = bool(ep_success[0] > 0)
      if debug:
        print(
          f"[render] ep={ep} steps={episode_length} succeeded={succeeded} "
          f"frames={len(frames)}",
          flush=True,
        )
      if succeeded and success_frames is None:
        success_frames = frames
      elif not succeeded and failure_frames is None:
        failure_frames = frames

    if success_frames:
      out_frames = _resample_fps(success_frames, control_dt, fps)
      _write_mp4(out_frames, fps, out_dir / "teacher.mp4")
      result["has_teacher_clip"] = True
    if failure_frames:
      out_frames = _resample_fps(failure_frames, control_dt, fps)
      _write_mp4(out_frames, fps, out_dir / "failure.mp4")
      result["has_failure_clip"] = True

    thumb_src = success_frames or failure_frames
    if thumb_src:
      import imageio.v2 as imageio

      mid = thumb_src[len(thumb_src) // 2]
      out_dir.mkdir(parents=True, exist_ok=True)
      imageio.imwrite(out_dir / "thumb.jpg", mid)
  finally:
    env.close()
  return result


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=sorted(CLASSICAL_POLICIES))
  parser.add_argument(
    "--num-episodes",
    type=int,
    default=32,
    help="Episodes for the SR/n reported in result.json (stats phase, unrendered). "
    "Use 128 to claim the 0.90 bar; 32 while iterating.",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    help="Max parallel envs for the stats phase.",
  )
  parser.add_argument(
    "--max-render-episodes",
    type=int,
    default=12,
    help="Cap on single-env rendered episodes while looking for one success clip "
    "and one failure clip.",
  )
  parser.add_argument("--out", required=True, type=Path, help="Output directory.")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--fps", type=int, default=30)
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument(
    "--skip-stats",
    action="store_true",
    help="Skip the batched stats phase (render only). result.json gets sr=null.",
  )
  parser.add_argument(
    "--debug",
    action="store_true",
    help="Also dump PNG frames + a per-episode print for the render phase "
    "(cheap: reuses frames already rendered). Old render_rollout.py behaviour.",
  )
  parser.add_argument("--debug-every", type=int, default=5)
  args = parser.parse_args()

  if args.task not in CLASSICAL_POLICIES:
    raise SystemExit(
      f"{args.task!r} has no registered classical teacher. "
      f"Known tasks: {sorted(CLASSICAL_POLICIES)}"
    )

  configure_torch_backends()
  device = args.device
  if device.startswith("cuda") and not torch.cuda.is_available():
    print(f"WARNING: cuda not available, falling back to cpu (requested {device})")
    device = "cpu"

  args.out.mkdir(parents=True, exist_ok=True)

  stats: Stats | None = None
  if not args.skip_stats:
    print(f"[stats] {args.task}: {args.num_episodes} episodes, batch<= {args.batch_size}")
    stats = run_stats_phase(args.task, args.num_episodes, args.batch_size, device)
    print(f"[stats] SR = {stats.sr:.3f} ({stats.num_success}/{stats.n})")

  print(f"[render] {args.task}: up to {args.max_render_episodes} single-env episodes")
  render_result = run_render_phase(
    args.task,
    device,
    args.out,
    args.max_render_episodes,
    args.fps,
    args.width,
    args.height,
    args.debug,
    args.debug_every,
    stats,
  )
  print(
    f"[render] teacher.mp4={render_result['has_teacher_clip']} "
    f"failure.mp4={render_result['has_failure_clip']} "
    f"(used {render_result['render_episodes_used']} render episodes)"
  )

  head = _git_head()
  sidecar = {
    "task_id": args.task,
    "sr": round(stats.sr, 4) if stats else None,
    "n": stats.n if stats else None,
    "num_success": stats.num_success if stats else None,
    "bar": COMPETENCE_BAR,
    "pass_bar": (stats.sr >= COMPETENCE_BAR) if stats else None,
    "head": head,
    "date": date.today().isoformat(),
    "has_teacher_clip": render_result["has_teacher_clip"],
    "has_failure_clip": render_result["has_failure_clip"],
    "render_episodes_used": render_result["render_episodes_used"],
    "fps": args.fps,
  }
  (args.out / "result.json").write_text(json.dumps(sidecar, indent=2))
  print(f"[done] wrote {args.out}/result.json")
  if stats is None:
    print(
      "NOTE: --skip-stats was set; result.json has sr=null. The site card needs a "
      "real SR — re-run without --skip-stats, or fill it in by hand before publishing."
    )
  if not render_result["has_teacher_clip"] and not render_result["has_failure_clip"]:
    print(
      f"WARNING: no clip captured at all in {render_result['render_episodes_used']} "
      "render episodes — increase --max-render-episodes."
    )


if __name__ == "__main__":
  main()
