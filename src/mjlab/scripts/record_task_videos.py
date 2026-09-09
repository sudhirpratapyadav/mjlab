"""Record a short video of every benchmark task, driven by a random agent.

Why a separate script instead of ``play.py --video``: play.py refuses to record with
dummy agents (``render_mode`` is only set when ``TRAINED_MODE``), and it plays exactly
one task per invocation from a checkpoint. Here we want the opposite — no checkpoints,
every registered task, a fixed-length clip each, for eyeballing scene layout, asset
placement and camera framing.

This is a VISUAL INSPECTION tool, not an evaluation: a random agent shows you what the
scene looks like and that it steps, never whether the task is solvable.

Rendering is offscreen (``render_mode="rgb_array"`` -> ``OffscreenRenderer``), so it
needs a working GL backend. On this cluster that means running on a GPU node with
``MUJOCO_GL=egl``; the login node has no EGL device and will fail.

Usage:
    MUJOCO_GL=egl python -m mjlab.scripts.record_task_videos --out-dir videos
    MUJOCO_GL=egl python -m mjlab.scripts.record_task_videos --keyword Franka --seconds 3
"""

from __future__ import annotations

import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro

import mjlab  # noqa: F401  (import side-effect: registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation import benchmark
from mjlab.tasks.registry import load_env_cfg, load_taxonomy
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class RecordConfig:
  out_dir: Path = Path("videos")
  """Directory for the .mp4 files and the manifest."""
  keyword: str | None = None
  """Only record tasks whose ID contains this substring."""
  seconds: float = 3.0
  """Clip length in seconds of SIMULATED time."""
  fps: int = 30
  """Output frame rate. Frames are resampled from the env's control rate."""
  width: int = 640
  height: int = 480
  num_envs: int = 1
  device: str | None = None
  seed: int = 0
  action_scale: float = 1.0
  """Random actions are drawn from U(-1, 1) * action_scale."""
  camera_distance: float | None = None
  """Override the viewer distance. Defaults to a close manipulation-workspace framing.

  The per-task viewer defaults are tuned for interactive play (distance 1.5-2.0 from
  ``link0``, elevation -5 to -10). Rendered small they put the robot base centre-frame
  and leave the object — the thing you actually want to inspect — tiny and near the
  bottom edge. The defaults below frame the workspace instead."""
  camera_elevation: float | None = None
  camera_azimuth: float | None = None
  camera_lookat_z: float | None = None
  """Raise the camera target off the floor to centre the workspace, not the base."""
  raw_camera: bool = False
  """Use each task's own viewer config verbatim, without the framing overrides."""
  tasks: tuple[str, ...] = field(default_factory=tuple)
  """Explicit task IDs. Overrides --keyword when non-empty."""


def _select_tasks(cfg: RecordConfig) -> list[str]:
  if cfg.tasks:
    return list(cfg.tasks)
  task_ids = sorted(benchmark.all_benchmark_tasks().keys())
  if cfg.keyword:
    task_ids = [t for t in task_ids if cfg.keyword.lower() in t.lower()]
  return task_ids


def _autoframe_camera(env: ManagerBasedRlEnv, cfg: RecordConfig) -> tuple:
  """Point the offscreen camera at the scene's own bounding box.

  Scene extents vary widely across the suite (a floor-level lift scene, a mechanism
  mounted at z=0.5, the taller LEAP setups), so any single hardcoded lookat/distance
  frames some tasks badly — mechanisms end up cropped or the object becomes a few
  pixels near the bottom edge. Instead measure the compiled geometry of env 0 and fit
  the camera to it.

  Returns the (lookat, distance) actually applied, for the manifest.
  """
  import mujoco

  renderer = env._offline_renderer
  if renderer is None:  # pragma: no cover — render_mode guarantees this exists
    return ((0.0, 0.0, 0.0), 0.0)

  model = renderer._model
  data = mujoco.MjData(model)
  data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  # Mocap bodies carry NO qpos. Thirteen asset_zoo objects hang off mocap roots that the
  # command terms write per-env (every articulated mechanism, plus container / ledge /
  # wall), so without syncing them the box below is measured with all of them still at
  # their MJCF pose. Same fix, same reason, as viewer/offscreen_renderer.py:68-71.
  if model.nmocap:
    data.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(model, data)

  # Frame the MANIPULATED SCENE, not the robot. Fitting the box to every geom lets the
  # ~1 m tall Franka set the frame, and the object of interest (a 5 cm cube on the
  # floor) ends up half-cropped at the bottom edge (W1-a, cl_v2). So: drop the ground
  # plane (effectively infinite), drop every robot geom, and centre on what is left.
  # The robot is still in shot because it works inside that box; the distance floor
  # below keeps the wrist and forearm visible above a small object.
  keep = [
    i for i in range(model.ngeom)
    if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
    and not (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("robot/")
  ]
  if not keep:  # asset-free task (Reach-Target): fall back to the whole scene
    keep = [
      i for i in range(model.ngeom) if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_PLANE
    ]
  if not keep:
    return (tuple(renderer._cam.lookat), float(renderer._cam.distance))

  pos = data.geom_xpos[keep]
  rbound = model.geom_rbound[keep][:, None]
  lo = (pos - rbound).min(axis=0)
  hi = (pos + rbound).max(axis=0)
  center = (lo + hi) / 2.0
  extent = float(np.linalg.norm(hi - lo))

  # Object-centred framing: a small free object gets a ~1.1 m shot that still shows
  # the arm from the elbow down; a 0.8 m cabinet door gets ~1.7 m. Capped so a large
  # mechanism never pushes the camera out of the room.
  distance = float(min(2.0, max(0.75, 0.9 * extent + 0.30)))
  # Look slightly above the object box so the gripper's approach is in frame.
  center = center + np.array([0.0, 0.0, 0.06])

  cam = renderer._cam
  cam.lookat[:] = center
  cam.distance = distance
  if cfg.camera_distance is not None:
    cam.distance = cfg.camera_distance
  if cfg.camera_elevation is not None:
    cam.elevation = cfg.camera_elevation
  if cfg.camera_azimuth is not None:
    cam.azimuth = cfg.camera_azimuth
  if cfg.camera_lookat_z is not None:
    cam.lookat[2] = cfg.camera_lookat_z
  return (tuple(round(float(x), 3) for x in cam.lookat), round(float(cam.distance), 3))


def record_task(task_id: str, cfg: RecordConfig, device: str) -> dict:
  """Record one task. Returns a manifest entry; never raises."""
  entry: dict = {"task_id": task_id, "ok": False}
  env = None
  try:
    env_cfg = load_env_cfg(task_id, play=True)
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.viewer.width = cfg.width
    env_cfg.viewer.height = cfg.height
    if not cfg.raw_camera:
      # Frame the manipulation workspace, not the robot base.
      #
      # The task cfgs use origin_type=ASSET_BODY (tracking link0). In TRACKING mode
      # MuJoCo overwrites camera.lookat with the tracked body's position every frame,
      # so `lookat` is silently ignored and the camera always centres on the base at
      # floor level — the arm fills the frame and the object is a few pixels near the
      # bottom edge. Switching to a WORLD free camera makes lookat authoritative.
      #
      # lookat/distance are then SOLVED from the scene's own bounding box rather than
      # hardcoded: scene extents differ a lot across tasks (a floor-level lift scene
      # vs a mechanism mounted at z=0.5 vs the taller LEAP setups), and any fixed
      # guess frames some subset of them badly.
      from mjlab.viewer import ViewerConfig

      env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
      env_cfg.viewer.elevation = -28.0
      env_cfg.viewer.azimuth = 150.0
      # Filled in after the env is built (needs compiled geometry).
      env_cfg.viewer.lookat = (0.3, 0.0, 0.4)
      env_cfg.viewer.distance = 1.6
    if cfg.camera_distance is not None:
      env_cfg.viewer.distance = cfg.camera_distance
    if cfg.camera_elevation is not None:
      env_cfg.viewer.elevation = cfg.camera_elevation
    if cfg.camera_azimuth is not None:
      env_cfg.viewer.azimuth = cfg.camera_azimuth
    if cfg.camera_lookat_z is not None:
      lookat = list(env_cfg.viewer.lookat)
      lookat[2] = cfg.camera_lookat_z
      env_cfg.viewer.lookat = tuple(lookat)
    # Single env: the renderer draws up to 32 worlds as extra geoms, which would
    # overlay duplicate arms on top of each other in one frame.
    env_cfg.viewer.env_idx = 0

    # play=True sets a ~forever episode length, which is what we want: the clip should
    # show continuous motion rather than resetting mid-way.
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
    env.reset()

    camera = _autoframe_camera(env, cfg) if not cfg.raw_camera else None

    control_dt = float(env.step_dt)
    n_steps = max(1, int(round(cfg.seconds / control_dt)))
    # Render every step, then resample to the requested fps at write time.
    action_dim = int(env.action_manager.total_action_dim)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    frames: list[np.ndarray] = []
    t0 = time.time()
    for _ in range(n_steps):
      action = (
        torch.rand(
          (cfg.num_envs, action_dim), device=device, generator=generator
        )
        * 2.0
        - 1.0
      ) * cfg.action_scale
      env.step(action)
      frame = env.render()
      if frame is None:
        raise RuntimeError("render() returned None despite render_mode='rgb_array'")
      frame = np.asarray(frame)
      if frame.ndim == 4:  # (num_envs, H, W, 3)
        frame = frame[0]
      if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
      frames.append(frame)
    sim_time = time.time() - t0

    # Resample control-rate frames to the requested output fps.
    src_fps = 1.0 / control_dt
    if cfg.fps < src_fps:
      idx = np.linspace(0, len(frames) - 1, int(round(cfg.seconds * cfg.fps)))
      out_frames = [frames[int(round(i))] for i in idx]
    else:
      out_frames = frames

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{task_id}.mp4"

    from moviepy import ImageSequenceClip

    clip = ImageSequenceClip(out_frames, fps=cfg.fps)
    clip.write_videofile(str(out_path), logger=None, audio=False)
    clip.close()

    taxonomy = load_taxonomy(task_id)
    entry.update(
      ok=True,
      file=out_path.name,
      steps=n_steps,
      frames=len(out_frames),
      seconds=round(len(out_frames) / cfg.fps, 2),
      control_dt=round(control_dt, 4),
      obs_dim=int(env.observation_manager.group_obs_dim["policy"][0]),
      action_dim=action_dim,
      episode_length_s=float(env_cfg.episode_length_s),
      render_seconds=round(sim_time, 1),
      camera={"lookat": camera[0], "distance": camera[1]} if camera else None,
      taxonomy=taxonomy.as_dict() if taxonomy is not None else None,
    )
    print(f"[OK]   {task_id}  ({entry['frames']} frames, {sim_time:.0f}s render)")
  except Exception as exc:  # noqa: BLE001 — record and continue to the next task
    entry["error"] = f"{type(exc).__name__}: {exc}"
    print(f"[FAIL] {task_id}: {entry['error']}")
    traceback.print_exc()
  finally:
    if env is not None:
      try:
        env.close()
      except Exception:  # noqa: BLE001 — teardown noise must not mask results
        pass
  return entry


def main(cfg: RecordConfig) -> None:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  task_ids = _select_tasks(cfg)

  print(f"Recording {len(task_ids)} task(s) to {cfg.out_dir} on {device}")
  entries = [record_task(t, cfg, device) for t in task_ids]

  manifest = {
    "num_tasks": len(entries),
    "num_ok": sum(e["ok"] for e in entries),
    "settings": {
      "seconds": cfg.seconds,
      "fps": cfg.fps,
      "width": cfg.width,
      "height": cfg.height,
      "agent": "random",
      "action_scale": cfg.action_scale,
      "seed": cfg.seed,
    },
    "videos": entries,
  }
  cfg.out_dir.mkdir(parents=True, exist_ok=True)
  (cfg.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

  ok = manifest["num_ok"]
  print(f"\n{ok}/{len(entries)} recorded -> {cfg.out_dir}/manifest.json")
  for e in entries:
    if not e["ok"]:
      print(f"  FAILED {e['task_id']}: {e.get('error')}")


if __name__ == "__main__":
  main(tyro.cli(RecordConfig))
