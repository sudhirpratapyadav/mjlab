#!/usr/bin/env python3
"""Collect a teacher dataset from a classical (scripted) policy.

Mirrors extract_teacher_dataset.py output format exactly:
  <output>/<Task>_classical_<timestamp>/
    data.pkl     {'observations': (N,obs), 'action_targets': (N,2*act) [mean|logstd], 'metadata': {...}}
    teacher.pkl  {'teacher_type': 'classical', 'classical_class': ..., 'action_std': (act,), ...}

Usage:
    python -m mjlab.continual_distill.classical.collect_classical_dataset \
        --task Mjlab-Push-Button-Franka --num-samples 153600 --num-envs 512
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
  parser.add_argument("--num-samples", type=int, default=153600)
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--action-std", type=float, default=0.2,
                      help="Fixed per-dim std stored as the teacher's label std.")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args()

  configure_torch_backends()
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  episode_length = int(env.max_episode_length)

  policy_cls = POLICIES[args.task]
  policy = policy_cls(num_envs=args.num_envs)

  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))

  samples_per_round = args.num_envs * episode_length
  num_rounds = int(np.ceil(args.num_samples / samples_per_round))
  print(f"Collecting {num_rounds} rounds x {samples_per_round} = "
        f"{num_rounds * samples_per_round} samples (requested {args.num_samples})")

  all_obs, all_means = [], []
  success_all = []
  for rnd in range(num_rounds):
    obs, _ = env.reset()
    policy.reset()
    succ = np.zeros(args.num_envs)
    for _t in tqdm(range(episode_length), desc=f"round {rnd}"):
      obs_np = obs["policy"].detach().cpu().numpy()
      actions = policy(obs_np)
      all_obs.append(obs_np.astype(np.float32))
      all_means.append(actions.astype(np.float32))
      obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(actions).to(args.device)
      )
      succ = np.maximum(succ, cmd.episode_success.detach().cpu().numpy())
    success_all.append(succ)
    print(f"round {rnd}: success={succ.mean():.3f}")

  observations = np.concatenate(all_obs, axis=0)
  action_means = np.concatenate(all_means, axis=0)
  action_dim = action_means.shape[1]
  action_std = np.full((action_dim,), args.action_std, dtype=np.float32)
  action_logstds = np.tile(np.log(action_std + 1e-8), (len(action_means), 1))
  action_targets = np.concatenate([action_means, action_logstds], axis=1)

  mean_success = float(np.concatenate(success_all).mean())
  print(f"\nTotal samples: {len(observations)}, teacher success: {mean_success:.3f}")

  out_root = args.output_dir or (Path(__file__).resolve().parents[1] / "teacher_datasets")
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  folder = out_root / f"{args.task.replace('-', '_')}_classical_{stamp}"
  folder.mkdir(parents=True, exist_ok=True)

  with (folder / "data.pkl").open("wb") as f:
    pickle.dump({
      "observations": observations,
      "action_targets": action_targets,
      "metadata": {
        "task": args.task,
        "obs_dim": observations.shape[1],
        "action_dim": action_dim,
        "num_samples_requested": args.num_samples,
        "num_envs": args.num_envs,
        "episode_length": episode_length,
        "total_samples": len(observations),
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "device": args.device,
        "teacher_type": "classical",
        "teacher_success": mean_success,
      },
    }, f)

  with (folder / "teacher.pkl").open("wb") as f:
    pickle.dump({
      "teacher_type": "classical",
      "classical_class": policy_cls.__name__,
      "action_std": action_std,
      "task": args.task,
      "obs_dim": observations.shape[1],
      "action_dim": action_dim,
      "checkpoint_path": "classical",
      "checkpoint_name": policy_cls.__name__,
      "hidden_dims": (),
      "obs_normalizer_mean": np.zeros(observations.shape[1], dtype=np.float32),
      "obs_normalizer_std": np.ones(observations.shape[1], dtype=np.float32),
      "jax_params": None,
    }, f)

  print(f"Saved to {folder}")
  env.close()


if __name__ == "__main__":
  main()
