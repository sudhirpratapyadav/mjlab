#!/usr/bin/env python3
"""Verbose single-policy rollout debugger: prints phase, errors, terminations.

Usage:
    python -m mjlab.continual_distill.classical.debug_rollout \
        --task Mjlab-Open-Door-Franka --num-envs 4 --print-envs 2
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
  parser.add_argument("--task", default="Mjlab-Open-Door-Franka")
  parser.add_argument("--num-envs", type=int, default=4)
  parser.add_argument("--print-envs", type=int, default=2)
  parser.add_argument("--print-every", type=int, default=10)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  policy = POLICIES[args.task](num_envs=args.num_envs)

  episode_length = int(env.max_episode_length)
  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))

  term_mgr = env.termination_manager
  print(f"termination terms: {term_mgr.active_terms}")

  obs, _ = env.reset()
  policy.reset()
  for t in range(episode_length):
    obs_np = obs["policy"].detach().cpu().numpy()
    actions = policy(obs_np)
    obs, reward, terminated, truncated, info = env.step(
      torch.from_numpy(actions).to(args.device)
    )
    term = terminated.detach().cpu().numpy()
    trunc = truncated.detach().cpu().numpy()
    succ = cmd.episode_success.detach().cpu().numpy()
    if t % args.print_every == 0 or term.any() or trunc.any():
      for i in range(min(args.print_envs, args.num_envs)):
        o = obs_np[i]
        gto = o[40:43]
        o2g = o[43:46]
        chord = np.clip(np.linalg.norm(o2g) / 1.102, 0.0, 1.0)
        theta = np.pi / 2 - 2 * np.arcsin(chord)
        ph = int(policy._phase[i]) if hasattr(policy, "_phase") else -1
        print(
          f"t={t:3d} env{i} phase={ph} |gto|={np.linalg.norm(gto):.3f} "
          f"gto=[{gto[0]:+.3f} {gto[1]:+.3f} {gto[2]:+.3f}] "
          f"|o2g|={np.linalg.norm(o2g):.3f} theta={np.degrees(theta):+6.1f} "
          f"succ={succ[i]:.0f} term={term[i]} trunc={trunc[i]}"
        )
      if term.any():
        # which termination fired?
        for name in term_mgr.active_terms:
          val = term_mgr.get_term(name).detach().cpu().numpy()
          if val.any():
            print(f"   -> termination '{name}' fired for envs {np.nonzero(val)[0]}")
  env.close()


if __name__ == "__main__":
  main()
