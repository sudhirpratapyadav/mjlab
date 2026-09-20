#!/usr/bin/env python3
"""Render a video of the CL-V5 student policy on one task from a checkpoint.

Headless (EGL) render, deterministic (mean) student action. Rolls out
`--num-envs` parallel episodes and picks a successful one if any succeeded
(falls back to the highest-return failure), matching CL-V4's success-preferred
convention.

Usage:
    python render_student.py --checkpoint path/to/checkpoint.pkl \
        --task-name DragPull --output-dir out/
"""

import argparse
import os
import pickle
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio_ffmpeg
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import torch
from PIL import Image

from mjlab.continual_distill.utils import SharedStudentPolicy, StudentPolicy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def load_checkpoint(checkpoint_path: Path) -> dict:
  with checkpoint_path.open("rb") as f:
    checkpoint = pickle.load(f)

  def numpy_to_jax(tree):
    import jax.tree_util as tree_util
    return tree_util.tree_map(lambda x: jnp.array(x) if isinstance(x, np.ndarray) else x, tree)

  checkpoint["params"] = numpy_to_jax(checkpoint["params"])
  if isinstance(checkpoint["normalizer_params"], dict):
    checkpoint["normalizer_params"] = {
      "mean": jnp.array(checkpoint["normalizer_params"]["mean"]),
      "std": jnp.array(checkpoint["normalizer_params"]["std"]),
    }
  else:
    checkpoint["normalizer_params"] = numpy_to_jax(checkpoint["normalizer_params"])
  return checkpoint


def build_action_fn(student_policy, params, normalizer_params, task_idx, action_dim, architecture):
  task_idx_jax = jnp.asarray(task_idx, dtype=jnp.int32)

  @jax.jit
  def get_action(obs_jax):
    if isinstance(normalizer_params, dict):
      normalized_obs = (obs_jax - normalizer_params["mean"]) / (normalizer_params["std"] + 1e-8)
    else:
      normalized_obs = normalizer_params.normalize(obs_jax)
    if architecture == "shared_resnet":
      head_logits = student_policy.network.apply(params, normalized_obs, task_idx_jax)
    else:
      logits = student_policy.network.apply(params, normalized_obs)
      head_dim = 2 * action_dim
      start = task_idx_jax * head_dim
      head_logits = jax.lax.dynamic_slice_in_dim(logits, start_index=start, slice_size=head_dim, axis=-1)
    loc, _scale = jnp.split(head_logits, 2, axis=-1)
    return loc

  return get_action


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task-name", type=str, required=True, help="Task name as it appears in checkpoint['task_sequence']")
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=8)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--fps", type=int, default=25)
  parser.add_argument("--frame-stride", type=int, default=2)
  args = parser.parse_args()

  args.output_dir.mkdir(parents=True, exist_ok=True)

  ckpt = load_checkpoint(args.checkpoint)
  task_sequence = list(ckpt["task_sequence"])
  env_ids = list(ckpt["env_ids"])
  if args.task_name not in task_sequence:
    raise SystemExit(f"Task {args.task_name!r} not in checkpoint task_sequence {task_sequence}")
  task_idx = task_sequence.index(args.task_name)
  env_id = env_ids[task_idx]
  print(f"Task {args.task_name} -> index {task_idx}, env_id {env_id}")

  cfg = load_env_cfg(env_id)
  cfg.scene.num_envs = args.num_envs
  cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg, device="cuda:0")

  architecture = ckpt.get("architecture", "heads")
  if architecture == "shared_resnet":
    arch_kwargs = ckpt.get("architecture_kwargs", {})
    student_policy = SharedStudentPolicy(
      obs_size=int(ckpt["obs_dim"]),
      action_size=int(ckpt["action_dim"]),
      num_tasks=int(ckpt["num_tasks"]),
      width=int(arch_kwargs.get("width", 2048)),
      num_blocks=int(arch_kwargs.get("num_blocks", 6)),
      embed_dim=int(arch_kwargs.get("embed_dim", 32)),
      min_std=float(ckpt["student_min_std"]),
    )
  else:
    student_policy = StudentPolicy(
      obs_size=int(ckpt["obs_dim"]),
      action_size=int(ckpt["action_dim"]),
      num_tasks=int(ckpt["num_tasks"]),
      hidden_dims=tuple(int(h) for h in ckpt["hidden_dims"]),
      min_std=float(ckpt["student_min_std"]),
    )
  print(f"Architecture: {architecture}")
  get_action = build_action_fn(
    student_policy, ckpt["params"], ckpt["normalizer_params"], task_idx, int(ckpt["action_dim"]), architecture
  )

  episode_length = env.max_episode_length
  print(f"Episode length: {episode_length} steps, num_envs={args.num_envs}")

  obs, _ = env.reset(seed=args.seed)
  command = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
  latched_success = np.zeros(args.num_envs, dtype=bool)
  episode_returns = np.zeros(args.num_envs)

  model = env.sim.mj_model
  model.vis.global_.offwidth = 960
  model.vis.global_.offheight = 540
  data = mujoco.MjData(model)
  renderer = mujoco.Renderer(model, 540, 960)
  camera = mujoco.MjvCamera()
  camera.lookat[:] = [0.43, 0, 0.21]
  camera.distance = 1.30
  camera.azimuth = 60
  camera.elevation = -28
  options = mujoco.MjvOption()
  options.geomgroup[3] = 0
  options.sitegroup[:] = 0

  frames_per_env = {i: [] for i in range(args.num_envs)}

  for step in range(episode_length):
    obs_array = obs["policy"].cpu().numpy()
    obs_jax = jnp.array(obs_array)
    action_jax = get_action(obs_jax)
    action_torch = torch.from_numpy(np.array(action_jax))
    obs, reward, terminated, truncated, info = env.step(action_torch)
    episode_returns += reward.cpu().numpy()
    latched_success = np.maximum(latched_success, command.episode_success.detach().cpu().numpy() > 0)

    if step % args.frame_stride == 0:
      qpos = env.sim.data.qpos.cpu().numpy()
      for i in range(args.num_envs):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera, scene_option=options)
        frames_per_env[i].append(renderer.render().copy())

  success_envs = [i for i in range(args.num_envs) if latched_success[i]]
  if success_envs:
    chosen = max(success_envs, key=lambda i: episode_returns[i])
    label = "success"
  else:
    chosen = int(np.argmax(episode_returns))
    label = "failure"

  frames = frames_per_env[chosen]
  path = args.output_dir / f"{args.task_name}-{label}.mp4"
  writer = imageio_ffmpeg.write_frames(
    str(path), (960, 540), fps=args.fps, codec="libx264", pix_fmt_out="yuv420p",
    macro_block_size=1, output_params=["-crf", "20", "-preset", "fast", "-threads", "2"],
    ffmpeg_log_level="error",
  )
  writer.send(None)
  for frame_index, frame in enumerate(frames):
    writer.send(np.ascontiguousarray(frame))
    if frame_index in (0, len(frames) // 2, len(frames) - 1):
      Image.fromarray(frame).save(args.output_dir / f"{args.task_name}-{label}-{frame_index:04d}.png")
  writer.close()

  print(f"Saved {path} ({label}, success_rate={latched_success.mean():.3f} over {args.num_envs} envs)")


if __name__ == "__main__":
  main()
