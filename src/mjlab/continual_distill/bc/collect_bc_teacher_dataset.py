#!/usr/bin/env python3
"""Stage 3: collect the distillation dataset from a trained BC teacher.

Deterministic (mean-action) rollouts of the BC teacher, saving data.pkl +
teacher.pkl in the same folder layout extract_teacher_dataset.py produces —
ready to be referenced from tasks.yaml.

Usage:
    python -m mjlab.continual_distill.bc.collect_bc_teacher_dataset \
        --teacher .../bc_teachers/<run>/teacher.pkl --num-samples 153600
"""

import argparse
import pickle
import shutil
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch
from tqdm import tqdm

from mjlab.continual_distill.utils import TeacherPolicy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--teacher", type=Path, required=True)
  parser.add_argument("--num-samples", type=int, default=153600)
  parser.add_argument("--num-envs", type=int, default=512)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args()

  with args.teacher.open("rb") as f:
    ti = pickle.load(f)
  task = ti["task"]
  obs_dim, action_dim = ti["obs_dim"], ti["action_dim"]

  teacher = TeacherPolicy(obs_size=obs_dim, action_size=action_dim,
                          hidden_dims=tuple(ti["hidden_dims"]))
  params = ti["jax_params"]
  nm = jnp.asarray(ti["obs_normalizer_mean"])
  ns = jnp.asarray(ti["obs_normalizer_std"])
  action_std = np.asarray(ti["action_std"], dtype=np.float32)

  @jax.jit
  def act_fn(ob):
    return teacher.actor.apply(params, (ob - nm) / (ns + 1e-8))

  configure_torch_backends()
  torch.manual_seed(args.seed)
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
  episode_length = int(env.max_episode_length)

  samples_per_round = args.num_envs * episode_length
  num_rounds = int(np.ceil(args.num_samples / samples_per_round))
  print(f"Collecting {num_rounds} rounds x {samples_per_round} samples")

  all_obs, all_means = [], []
  succ_all = []
  for rnd in range(num_rounds):
    obs, _ = env.reset()
    succ = np.zeros(args.num_envs)
    for _t in tqdm(range(episode_length), desc=f"round {rnd}"):
      obs_np = obs["policy"].detach().cpu().numpy()
      means = np.array(act_fn(jnp.asarray(obs_np)))
      all_obs.append(obs_np.astype(np.float32))
      all_means.append(means.astype(np.float32))
      obs, reward, terminated, truncated, info = env.step(
        torch.from_numpy(means).to(args.device)
      )
      succ = np.maximum(succ, cmd.episode_success.detach().cpu().numpy())
    succ_all.append(succ)
    print(f"round {rnd}: success={succ.mean():.3f}")

  observations = np.concatenate(all_obs, axis=0)
  action_means = np.concatenate(all_means, axis=0)
  logstds = np.tile(np.log(action_std + 1e-8), (len(action_means), 1))
  action_targets = np.concatenate([action_means, logstds], axis=1)
  mean_success = float(np.concatenate(succ_all).mean())
  print(f"Total: {len(observations)} samples, BC teacher success {mean_success:.3f}")

  out_root = args.output_dir or (Path(__file__).resolve().parents[1] / "teacher_datasets")
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  folder = out_root / f"{task.replace('-', '_')}_bc_{stamp}"
  folder.mkdir(parents=True, exist_ok=True)
  with (folder / "data.pkl").open("wb") as f:
    pickle.dump({
      "observations": observations,
      "action_targets": action_targets,
      "metadata": {
        "task": task,
        "obs_dim": obs_dim,
        "action_dim": action_dim,
        "num_samples_requested": args.num_samples,
        "num_envs": args.num_envs,
        "episode_length": episode_length,
        "total_samples": len(observations),
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "device": args.device,
        "teacher_type": "bc",
        "teacher_success": mean_success,
      },
    }, f)
  shutil.copy(args.teacher, folder / "teacher.pkl")
  print(f"Saved to {folder}")
  env.close()


if __name__ == "__main__":
  main()
