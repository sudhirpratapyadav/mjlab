#!/usr/bin/env python3
"""Stage 1: collect BC training data from a classical expert with noise.

The env executes the expert action PLUS exploration noise, so trajectories
visit states off the expert's deterministic corridor; the stored label is
always the expert's CLEAN action at the visited state (DART-style), teaching
the BC policy how to recover.

Usage:
    python -m mjlab.continual_distill.bc.collect_bc_data \
        --task Mjlab-Push-Button-Franka --num-samples 200000 \
        --num-envs 512 --noise-std 1.0 --clean-fraction 0.25
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

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
  parser.add_argument("--num-samples", type=int, default=200000)
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--noise-std", type=float, default=1.0,
                      help="Std of Gaussian noise added to EXECUTED actions "
                           "(action units; env scale is 0.04 rad/unit).")
  parser.add_argument("--clean-fraction", type=float, default=0.25,
                      help="Fraction of rollout rounds executed without noise.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  rng = np.random.default_rng(args.seed)

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  episode_length = int(env.max_episode_length)

  policy = POLICIES[args.task](num_envs=args.num_envs)
  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))

  samples_per_round = args.num_envs * episode_length
  num_rounds = int(np.ceil(args.num_samples / samples_per_round))
  num_clean = max(1, int(round(num_rounds * args.clean_fraction)))
  print(f"{num_rounds} rounds x {samples_per_round} samples "
        f"({num_clean} clean, {num_rounds - num_clean} noisy std={args.noise_std})")

  all_obs, all_labels = [], []
  succ_stats = []
  for rnd in range(num_rounds):
    noisy = rnd >= num_clean
    obs, _ = env.reset()
    policy.reset()
    succ = np.zeros(args.num_envs)
    for _t in tqdm(range(episode_length), desc=f"round {rnd} ({'noisy' if noisy else 'clean'})"):
      obs_np = obs["policy"].detach().cpu().numpy()
      expert_actions = policy(obs_np)
      all_obs.append(obs_np.astype(np.float32))
      all_labels.append(expert_actions.astype(np.float32))
      exec_actions = expert_actions
      if noisy:
        exec_actions = expert_actions + rng.normal(
          0.0, args.noise_std, size=expert_actions.shape
        ).astype(np.float32)
      obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(exec_actions).to(args.device)
      )
      done = (terminated | truncated).detach().cpu().numpy()
      if done.any():
        policy.reset(np.where(done)[0])
      succ = np.maximum(succ, cmd.episode_success.detach().cpu().numpy())
    succ_stats.append((noisy, succ.mean()))
    print(f"round {rnd} ({'noisy' if noisy else 'clean'}): success={succ.mean():.3f}")

  observations = np.concatenate(all_obs, axis=0)
  labels = np.concatenate(all_labels, axis=0)
  print(f"Total samples: {len(observations)}")

  out_root = args.output_dir or (Path(__file__).resolve().parents[1] / "bc_datasets")
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  folder = out_root / f"{args.task.replace('-', '_')}_bcdata_{stamp}"
  folder.mkdir(parents=True, exist_ok=True)
  with (folder / "bc_data.pkl").open("wb") as f:
    pickle.dump({
      "observations": observations,
      "expert_actions": labels,
      "metadata": {
        "task": args.task,
        "expert": policy.__class__.__name__,
        "noise_std": args.noise_std,
        "clean_fraction": args.clean_fraction,
        "num_envs": args.num_envs,
        "episode_length": episode_length,
        "rounds": succ_stats,
        "seed": args.seed,
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
      },
    }, f)
  print(f"Saved to {folder}/bc_data.pkl")
  env.close()


if __name__ == "__main__":
  main()
