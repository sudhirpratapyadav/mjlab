"""Quick solvability check for a benchmark task: train briefly, read success.

Unlike ``benchmark_smoke`` (which only checks a task BUILDS), this trains a task for a
short budget and reports the env success metric — the honest check that a new task's
reward shaping and success predicate actually work. Fast tasks (reach) solve in ~1 min;
contact-rich tasks need a larger ``--iters``.

Not a substitute for a full training run — it is a smoke-test for *learnability*.

Usage:
    python -m mjlab.scripts.benchmark_validate --task Mjlab-Reach-Target-Franka --iters 150
"""

from __future__ import annotations

from dataclasses import asdict

import torch
import tyro

import mjlab  # noqa: F401  (registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def _read_success(env: ManagerBasedRlEnv) -> tuple[float, float]:
  """Return (mean episode_success, mean goal_error) from whichever command exposes them."""
  cm = env.unwrapped.command_manager
  for name in cm.active_terms:
    term = cm.get_term(name)
    m = getattr(term, "metrics", {})
    if "episode_success" in m:
      succ = m["episode_success"].float().mean().item()
      err = m["goal_error"].float().mean().item() if "goal_error" in m else float("nan")
      return succ, err
  return float("nan"), float("nan")


def validate(
  task: str,
  iters: int = 150,
  num_envs: int = 1024,
  device: str | None = None,
  log_dir: str = "/tmp/mjlab_benchmark_validate",
) -> float:
  """Train ``task`` for ``iters`` iterations and print/return mean episode success."""
  dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device=dev)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

  from rsl_rl.runners import OnPolicyRunner

  agent_cfg = asdict(load_rl_cfg(task))
  runner = OnPolicyRunner(wrapped, agent_cfg, log_dir, dev)
  print(f"=== validating {task}: {iters} iters x {num_envs} envs ===", flush=True)
  runner.learn(num_learning_iterations=iters, init_at_random_ep_len=True)

  succ, err = _read_success(env)
  verdict = "SOLVABLE" if succ > 0.5 else ("PARTIAL" if succ > 0.15 else "NOT-LEARNING")
  print(f"\nRESULT {task}: episode_success={succ:.3f} goal_error={err:.3f} -> {verdict}", flush=True)
  wrapped.close()
  return succ


def main():
  tyro.cli(validate)


if __name__ == "__main__":
  main()
