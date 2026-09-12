#!/usr/bin/env python3
"""Evaluate teachers and record genuine outcome clips at real-time speed.

Default output is 1920x1080, 50 fps. Search uses unrendered simulation states;
selected episodes are rendered from those exact states, with no action replay.
Success rates come from independent first episodes, never automatic reset retries.
Use --stats-from only when the change affects visualization/encoding rather than
physics, teacher actions, reset distributions or success criteria.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.base import ClassicalPolicyBase
from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture
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


def source_fingerprint() -> str:
  """Include uncommitted teacher and physics sources in video provenance."""
  root = _REPO_ROOT / "src/mjlab"
  paths = []
  for folder in ("asset_zoo", "tasks/manipulation", "continual_distill/classical", "entity", "viewer"):
    paths.extend(p for p in (root / folder).rglob("*")
                 if p.is_file() and p.suffix in {".py", ".xml", ".obj", ".stl"})
  digest = hashlib.sha256()
  for path in sorted(paths):
    digest.update(str(path.relative_to(root)).encode())
    digest.update(path.read_bytes())
  return digest.hexdigest()


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
  task_id: str, num_episodes: int, batch_size: int, device: str, seed: int = 20260910
) -> Stats:
  """Batched, unrendered: ``num_episodes`` envs, one episode each, chunked."""
  total = 0
  succ = 0
  remaining = num_episodes
  while remaining > 0:
    b = min(batch_size, remaining)
    ManagerBasedRlEnv.seed(seed + total)
    env = _build_env(task_id, b, device, render=False, width=0, height=0)
    try:
      policy = CLASSICAL_POLICIES[task_id](num_envs=b)
      episode_length = int(env.max_episode_length)
      obs, _ = env.reset()
      policy.reset()
      cmd = _success_term(env)
      ep_success = np.zeros(b, dtype=bool)
      alive = np.ones(b, dtype=bool)
      with TerminalSuccessCapture(env, cmd) as capture:
        for _t in range(episode_length):
          policy_obs = obs["policy"].detach().cpu().numpy()
          kwargs = {"active_env_ids": np.flatnonzero(alive)} if isinstance(policy, ClassicalPolicyBase) else {}
          actions = policy(policy_obs, **kwargs)
          obs, _, terminated, truncated, _ = capture.step(torch.from_numpy(actions).to(device))
          success = torch.where(terminated | truncated, capture.terminal_success, cmd.episode_success > 0)
          ep_success |= alive & success.detach().cpu().numpy()
          # Success is latched by the task; further actions cannot change this trial.
          alive &= ~((terminated | truncated).detach().cpu().numpy() | ep_success)
          if not alive.any():
            break
      succ += int(ep_success.sum())
      total += b
    finally:
      env.close()
    remaining -= b
  return Stats(n=total, num_success=succ)


def _resample_fps(frames: list[np.ndarray], control_dt: float, fps: int):
  if not frames:
    return []
  count = max(1, round(len(frames) * control_dt * fps))
  indices = np.minimum((np.arange(count) / (fps * control_dt)).astype(int), len(frames) - 1)
  return [frames[i] for i in indices]


def _write_mp4(frames: list[np.ndarray], fps: int, out_path: Path) -> None:
  from moviepy import ImageSequenceClip

  out_path.parent.mkdir(parents=True, exist_ok=True)
  clip = ImageSequenceClip(frames, fps=fps)
  clip.write_videofile(str(out_path), codec="libx264", logger=None, audio=False)
  clip.close()


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
  seed: int = 20270910,
) -> dict:
  """Find both real outcomes, then render recorded states without re-simulation."""
  from mjlab.continual_distill.classical.record_outcomes import record_outcomes
  return record_outcomes(task_id, device, out_dir, max_render_episodes, fps,
                         width, height, stats, seed, debug=debug, debug_every=debug_every)


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
    default=256,
    help="Cap on single-env rendered episodes while looking for one success clip "
    "and one failure clip.",
  )
  parser.add_argument("--stats-from", type=Path, help="Reuse a verified measurement when only visualization/encoding changes.")
  parser.add_argument("--out", required=True, type=Path, help="Output directory.")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=20260910)
  parser.add_argument("--fps", type=int, default=50)
  parser.add_argument("--width", type=int, default=1920)
  parser.add_argument("--height", type=int, default=1080)
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

  fingerprint = source_fingerprint()
  stats: Stats | None = None
  if args.stats_from:
    previous = json.loads(args.stats_from.read_text())
    if previous['task_id'] != args.task or previous['evaluation_protocol'] != 'single_episode_no_reset_retries':
      raise ValueError('Incompatible previous measurement')
    stats = Stats(n=previous['n'], num_success=previous['num_success'])
  elif not args.skip_stats:
    print(f"[stats] {args.task}: {args.num_episodes} episodes, batch<= {args.batch_size}")
    stats = run_stats_phase(args.task, args.num_episodes, args.batch_size, device, args.seed)
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
    seed=args.seed + 10000,
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
    "source_sha256": fingerprint,
    "stats_seed": args.seed,
    "stats_batch_size": args.batch_size,
    "render_seed": args.seed + 10000,
    "evaluation_protocol": "single_episode_no_reset_retries",
    "geometry_revision": "2026-09-10-full-collision",
    "date": date.today().isoformat(),
    "has_teacher_clip": render_result["has_teacher_clip"],
    "has_failure_clip": render_result["has_failure_clip"],
    "render_episodes_used": render_result["render_episodes_used"],
    "fps": args.fps,
    "clip_end": "first termination/timeout or 0.5 seconds after latched success",
  }
  sidecar.update(render_result)
  sidecar['visualization_revision'] = 'axis-goals-hd-20260910'
  if args.stats_from:
    sidecar['stats_source_sha256'] = previous.get('stats_source_sha256', previous['source_sha256'])
    sidecar['stats_reused_visual_only'] = True
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
