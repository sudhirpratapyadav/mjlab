#!/usr/bin/env python3
"""Stage 2: train a BC policy on expert data and save it as a teacher.pkl.

The network is the exact TeacherActorMLP used by the RL teachers (ELU,
512-256-128 by default), with the same observation-normalizer convention, so
the saved teacher.pkl is a drop-in replacement anywhere an RL teacher.pkl is
accepted (extract/eval/distill).

Usage:
    python -m mjlab.continual_distill.bc.train_bc \
        --bc-data .../bc_data.pkl --epochs 50 --eval-envs 64
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import torch

from mjlab.continual_distill.utils import ObservationNormalizer, TeacherPolicy
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--bc-data", type=Path, required=True)
  parser.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256, 128])
  parser.add_argument("--epochs", type=int, default=50)
  parser.add_argument("--batch-size", type=int, default=1024)
  parser.add_argument("--learning-rate", type=float, default=3e-4)
  parser.add_argument("--train-fraction", type=float, default=0.9)
  parser.add_argument("--action-std", type=float, default=0.2,
                      help="Fixed std stored with the teacher (matches the "
                           "classical-teacher label convention).")
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--eval-envs", type=int, default=64)
  parser.add_argument("--eval-episodes", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--output-dir", type=Path, default=None)
  args = parser.parse_args()

  with args.bc_data.open("rb") as f:
    data = pickle.load(f)
  obs = np.asarray(data["observations"], dtype=np.float32)
  act = np.asarray(data["expert_actions"], dtype=np.float32)
  task = data["metadata"]["task"]
  obs_dim, action_dim = obs.shape[1], act.shape[1]
  print(f"BC data: {obs.shape[0]} samples, task={task}")

  # Normalizer from the data.
  norm_mean = obs.mean(axis=0)
  norm_std = obs.std(axis=0) + 1e-6

  rng = np.random.default_rng(args.seed)
  perm = rng.permutation(len(obs))
  n_train = int(len(obs) * args.train_fraction)
  tr_idx, va_idx = perm[:n_train], perm[n_train:]

  teacher = TeacherPolicy(obs_size=obs_dim, action_size=action_dim,
                          hidden_dims=tuple(args.hidden_dims))
  params = teacher.init(jax.random.PRNGKey(args.seed))
  tx = optax.adam(args.learning_rate)
  opt_state = tx.init(params)

  norm_mean_j = jnp.asarray(norm_mean)
  norm_std_j = jnp.asarray(norm_std)

  @jax.jit
  def train_step(params, opt_state, ob, ac):
    def loss_fn(p):
      pred = teacher.actor.apply(p, (ob - norm_mean_j) / (norm_std_j + 1e-8))
      return jnp.mean(jnp.square(pred - ac))
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = tx.update(grads, opt_state)
    return optax.apply_updates(params, updates), opt_state, loss

  @jax.jit
  def val_loss_fn(params, ob, ac):
    pred = teacher.actor.apply(params, (ob - norm_mean_j) / (norm_std_j + 1e-8))
    return jnp.mean(jnp.square(pred - ac))

  obs_j = jnp.asarray(obs)
  act_j = jnp.asarray(act)
  steps_per_epoch = max(1, n_train // args.batch_size)
  for epoch in range(args.epochs):
    ep_perm = rng.permutation(n_train)
    losses = []
    for s in range(steps_per_epoch):
      idx = tr_idx[ep_perm[s * args.batch_size:(s + 1) * args.batch_size]]
      params, opt_state, loss = train_step(params, opt_state, obs_j[idx], act_j[idx])
      losses.append(float(loss))
    if epoch % 5 == 0 or epoch == args.epochs - 1:
      vl = float(val_loss_fn(params, obs_j[va_idx], act_j[va_idx]))
      print(f"epoch {epoch:3d}: train_mse={np.mean(losses):.4f} val_mse={vl:.4f}")

  # ---- env validation ----
  print("\nValidating BC policy in env...")
  configure_torch_backends()
  env_cfg = load_env_cfg(task, play=False)
  env_cfg.scene.num_envs = args.eval_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
  episode_length = int(env.max_episode_length)

  @jax.jit
  def act_fn(ob):
    return teacher.actor.apply(params, (ob - norm_mean_j) / (norm_std_j + 1e-8))

  succ_all = []
  for ep in range(args.eval_episodes):
    o, _ = env.reset()
    succ = np.zeros(args.eval_envs)
    for _t in range(episode_length):
      a = np.array(act_fn(jnp.asarray(o["policy"].detach().cpu().numpy())))
      o, r, term, trunc, info = env.step(torch.from_numpy(a).to(args.device))
      succ = np.maximum(succ, cmd.episode_success.detach().cpu().numpy())
    succ_all.append(succ)
    print(f"  eval episode {ep}: success={succ.mean():.3f}")
  success_rate = float(np.concatenate(succ_all).mean())
  print(f"BC policy env success: {success_rate:.3f}")
  env.close()

  # ---- save teacher.pkl (RL-teacher-compatible format) ----
  out_root = args.output_dir or (Path(__file__).resolve().parents[1] / "bc_teachers")
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  folder = out_root / f"{task.replace('-', '_')}_bc_{stamp}"
  folder.mkdir(parents=True, exist_ok=True)
  with (folder / "teacher.pkl").open("wb") as f:
    pickle.dump({
      "jax_params": jax.device_get(params),
      "action_std": np.full((action_dim,), args.action_std, dtype=np.float32),
      "obs_normalizer_mean": norm_mean.astype(np.float32),
      "obs_normalizer_std": norm_std.astype(np.float32),
      "hidden_dims": tuple(args.hidden_dims),
      "checkpoint_path": str(args.bc_data),
      "checkpoint_name": "bc_teacher",
      "task": task,
      "obs_dim": obs_dim,
      "action_dim": action_dim,
      "teacher_type": "bc",
      "env_success_rate": success_rate,
    }, f)
  print(f"Saved BC teacher to {folder}/teacher.pkl")


if __name__ == "__main__":
  main()
