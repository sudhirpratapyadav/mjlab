"""Smoke-test benchmark manipulation tasks: build, reset, and step each env.

Structural validation for the manipulation-diversity benchmark. Does NOT check task
solvability (that needs training) — it checks that a task is wired correctly: the env
builds, resets, and steps with finite rewards and the expected obs/action shapes. Run
this after authoring any new task.

Usage:
    python -m mjlab.scripts.benchmark_smoke                 # all tagged benchmark tasks
    python -m mjlab.scripts.benchmark_smoke --keyword Lift  # subset by keyword
    python -m mjlab.scripts.benchmark_smoke --num-envs 8 --steps 20
"""

from __future__ import annotations

import os
import traceback
from dataclasses import dataclass

import torch
import tyro
from prettytable import PrettyTable

import mjlab  # noqa: F401  (import side-effect: registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation import benchmark
from mjlab.tasks.registry import load_env_cfg


@dataclass
class SmokeResult:
  task_id: str
  ok: bool
  obs_dim: int | None = None
  action_dim: int | None = None
  detail: str = ""


def smoke_test_task(
  task_id: str,
  num_envs: int = 4,
  steps: int = 10,
  device: str | None = None,
) -> SmokeResult:
  """Build, reset, and step one task; return a structured pass/fail result."""
  dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
  try:
    cfg = load_env_cfg(task_id, test=True)
    cfg.scene.num_envs = num_envs
    env = ManagerBasedRlEnv(cfg, device=dev)
    obs, _ = env.reset()
    obs_dim = int(obs["policy"].shape[-1])
    action_dim = int(env.action_manager.total_action_dim)
    for _ in range(steps):
      action = torch.zeros((num_envs, action_dim), device=dev)
      obs, rew, term, trunc, info = env.step(action)
      if not torch.isfinite(rew).all():
        return SmokeResult(task_id, False, obs_dim, action_dim, "non-finite reward")
      if not torch.isfinite(obs["policy"]).all():
        return SmokeResult(task_id, False, obs_dim, action_dim, "non-finite obs")
    # Free GPU state between tasks.
    del env
    if dev == "cuda":
      torch.cuda.empty_cache()
    return SmokeResult(task_id, True, obs_dim, action_dim, "ok")
  except Exception as exc:  # noqa: BLE001 — report, don't abort the sweep
    return SmokeResult(task_id, False, detail=f"{type(exc).__name__}: {exc}")


def run(
  keyword: str | None = None,
  num_envs: int = 4,
  steps: int = 10,
  device: str | None = None,
  verbose: bool = False,
) -> int:
  """Smoke-test benchmark tasks and print a summary table.

  Args:
    keyword: Only test tasks whose id contains this (case-insensitive).
    num_envs: Parallel envs per task.
    steps: Zero-action steps to take per task.
    device: "cuda"/"cpu"; auto if None.
    verbose: Print full traceback for failures.

  Returns the number of failing tasks (0 == all green; usable as an exit code).
  """
  task_ids = sorted(benchmark.all_benchmark_tasks())
  if keyword:
    task_ids = [t for t in task_ids if keyword.lower() in t.lower()]

  results: list[SmokeResult] = []
  for task_id in task_ids:
    res = smoke_test_task(task_id, num_envs=num_envs, steps=steps, device=device)
    results.append(res)
    if verbose and not res.ok:
      print(f"\n--- {task_id} FAILED ---\n{res.detail}\n")

  table = PrettyTable(["Task ID", "Status", "obs", "action", "detail"])
  table.align["Task ID"] = "l"
  table.align["detail"] = "l"
  for r in results:
    table.add_row(
      [
        r.task_id,
        "PASS" if r.ok else "FAIL",
        r.obs_dim if r.obs_dim is not None else "-",
        r.action_dim if r.action_dim is not None else "-",
        r.detail,
      ]
    )
  print(table)

  n_fail = sum(not r.ok for r in results)
  print(f"\n{len(results) - n_fail}/{len(results)} passed.")
  return n_fail


def main():
  n_fail = tyro.cli(run)
  # Flush then hard-exit: warp/mujoco + torch CUDA contexts can segfault (exit 139)
  # during interpreter teardown even after a clean run. os._exit skips teardown.
  import sys
  sys.stdout.flush()
  sys.stderr.flush()
  os._exit(1 if n_fail else 0)


if __name__ == "__main__":
  main()
