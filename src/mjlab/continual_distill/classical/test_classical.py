#!/usr/bin/env python3
"""Test a classical (scripted) teacher policy in the mjlab environment.

Usage:
    python -m mjlab.continual_distill.classical.test_classical \
        --task Mjlab-Push-Button-Franka --num-envs 16 --num-episodes 2
"""

import argparse

import numpy as np
import torch

from mjlab.continual_distill.classical import (
  OpenDoorClassicalPolicy,
  OpenDrawerClassicalPolicy,
  PushButtonClassicalPolicy,
  PushCuboidClassicalPolicy,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

POLICIES = {
  "Mjlab-Push-Button-Franka": PushButtonClassicalPolicy,
  "Mjlab-Push-Cuboid-Franka": PushCuboidClassicalPolicy,
  "Mjlab-Open-Door-Franka": OpenDoorClassicalPolicy,
  "Mjlab-Open-Drawer-Franka": OpenDrawerClassicalPolicy,
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Mjlab-Push-Button-Franka")
  parser.add_argument("--num-envs", type=int, default=16)
  parser.add_argument("--num-episodes", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

  policy = POLICIES[args.task](num_envs=args.num_envs)

  episode_length = int(env.max_episode_length)
  print(f"episode_length={episode_length}, num_envs={args.num_envs}")

  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
  successes = []
  for ep in range(args.num_episodes):
    obs, _ = env.reset()
    policy.reset()
    # Track success DURING the episode; the env auto-resets on timeout, which
    # clears episode_success before a post-loop read would see it.
    ep_success = np.zeros(args.num_envs)
    for _t in range(episode_length):
      obs_np = obs["policy"].detach().cpu().numpy()
      actions = policy(obs_np)
      obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(actions).to(args.device)
      )
      ep_success = np.maximum(
        ep_success, cmd.episode_success.detach().cpu().numpy()
      )
    successes.append(ep_success)
    print(
      f"episode {ep}: success={ep_success.mean():.3f} "
      f"({int(ep_success.sum())}/{args.num_envs})"
    )

  all_succ = np.concatenate(successes)
  print(f"\nOVERALL success rate: {all_succ.mean():.3f} over {len(all_succ)} episodes")
  env.close()


if __name__ == "__main__":
  main()
