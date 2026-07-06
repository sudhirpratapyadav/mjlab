#!/usr/bin/env python3
"""Parameter sweep / failure-mode stats for the OpenDoor classical policy.

For each config: rollout N envs x E episodes, report success rate, mean/max
door angle reached (from |o2g| chord), and how many envs ever reached the
pull phase.
"""

import argparse
import itertools

import numpy as np
import torch

import mjlab.continual_distill.classical.open_door as od
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def run_config(env, num_envs, episodes, overrides):
  for k, v in overrides.items():
    setattr(od, k, v)
  policy = od.OpenDoorClassicalPolicy(num_envs=num_envs)
  if "max_dq" in overrides:
    policy.max_dq = overrides["max_dq"]

  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
  episode_length = int(env.max_episode_length)
  r2 = 2 * float(np.linalg.norm(od._R0))

  succ_all, theta_max_all, pulled_all = [], [], []
  for _ep in range(episodes):
    obs, _ = env.reset()
    policy.reset()
    ep_succ = np.zeros(num_envs)
    theta_max = np.full(num_envs, -1.0)
    pulled = np.zeros(num_envs, dtype=bool)
    for _t in range(episode_length):
      obs_np = obs["policy"].detach().cpu().numpy()
      actions = policy(obs_np)
      obs, *_ = env.step(torch.from_numpy(actions).to(env.device))
      o2g = obs_np[:, 43:46]
      chord = np.clip(np.linalg.norm(o2g, axis=1) / r2, 0, 1)
      theta = np.degrees(np.pi / 2 - 2 * np.arcsin(chord))
      theta_max = np.maximum(theta_max, theta)
      pulled |= policy._phase == 3
      ep_succ = np.maximum(ep_succ, cmd.episode_success.detach().cpu().numpy())
    succ_all.append(ep_succ)
    theta_max_all.append(theta_max)
    pulled_all.append(pulled)

  succ = np.concatenate(succ_all)
  th = np.concatenate(theta_max_all)
  pl = np.concatenate(pulled_all)
  return {
    "success": succ.mean(),
    "reached_pull": pl.mean(),
    "theta_mean": th.mean(),
    "theta_p50": np.median(th),
    "theta_p90": np.percentile(th, 90),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--episodes", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  configure_torch_backends()
  env_cfg = load_env_cfg("Mjlab-Open-Door-Franka", play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

  import numpy as _np
  grid = [
    # (name, overrides)
    ("arc_rot", {"PULL_MODE": "arc", "PULL_ROT": "rotate"}),
    ("arc_free", {"PULL_MODE": "arc", "PULL_ROT": "free"}),
    ("arc_fixed", {"PULL_MODE": "arc", "PULL_ROT": "fixed"}),
    ("tan_rot", {"PULL_MODE": "tangent", "PULL_ROT": "rotate"}),
    ("tan_free", {"PULL_MODE": "tangent", "PULL_ROT": "free"}),
    ("arc_rot_dth20", {"PULL_MODE": "arc", "PULL_ROT": "rotate", "PULL_DTHETA": _np.radians(20.0)}),
    ("arc_rot_dth8", {"PULL_MODE": "arc", "PULL_ROT": "rotate", "PULL_DTHETA": _np.radians(8.0)}),
  ]
  for name, ov in grid:
    stats = run_config(env, args.num_envs, args.episodes, ov)
    print(
      f"{name:12s} succ={stats['success']:.3f} pull={stats['reached_pull']:.2f} "
      f"thetaMean={stats['theta_mean']:5.1f} p50={stats['theta_p50']:5.1f} "
      f"p90={stats['theta_p90']:5.1f}",
      flush=True,
    )
  env.close()


if __name__ == "__main__":
  main()
