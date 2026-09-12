"""Read-only current configuration inventory for the active RL-teacher suite."""

import dataclasses
import hashlib
import importlib.metadata
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from mjlab.tasks.manipulation.benchmark import DEFERRED_CL_TASKS, active_cl_tasks
from mjlab.tasks.manipulation.franka_interface import OBS_DIM, VERSION
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

here = Path(__file__).resolve().parent


def pack(x):
  if dataclasses.is_dataclass(x):
    return {f.name: pack(getattr(x, f.name)) for f in dataclasses.fields(x)}
  if isinstance(x, dict):
    return {str(k): pack(v) for k, v in x.items()}
  if isinstance(x, (tuple, list)):
    return [pack(v) for v in x]
  if callable(x):
    return x.__module__ + "." + x.__qualname__
  if x is None or isinstance(x, (str, int, float, bool)):
    return x
  return str(x)


rows = []
for task in active_cl_tasks():
  cfg = load_env_cfg(task)
  rl = load_rl_cfg(task)
  assert OBS_DIM == 60 and list(cfg.observations["policy"].terms) == [VERSION]
  assert cfg.observations["policy"] == cfg.observations["critic"]
  assert rl.clip_actions == 1 and rl.policy.init_noise_std == 0.2
  rows.append(
    dict(
      task=task,
      obs_dim=OBS_DIM,
      interface=VERSION,
      actions=pack(cfg.actions),
      episode_s=cfg.episode_length_s,
      dt=cfg.sim.mujoco.timestep * cfg.decimation,
      num_envs=cfg.scene.num_envs,
      gravity=list(cfg.sim.mujoco.gravity),
      rewards=pack(cfg.rewards),
      curriculum=pack(cfg.curriculum),
      terminations=pack(cfg.terminations),
      rl=pack(rl),
    )
  )
assert len(rows) == 24
packages = {}
for name in [
  "torch",
  "mujoco",
  "mujoco-warp",
  "warp-lang",
  "rsl-rl-lib",
  "tensordict",
  "tensorboard",
]:
  try:
    packages[name] = importlib.metadata.version(name)
  except importlib.metadata.PackageNotFoundError:
    packages[name] = "not found under this distribution name"
paths = [
  *Path("src/mjlab/tasks/manipulation").rglob("*.py"),
  *Path("src/mjlab/rl").rglob("*.py"),
  Path("src/mjlab/scripts/train.py"),
  Path("src/mjlab/utils/gpu.py"),
]
data = dict(
  checked_utc=datetime.now(timezone.utc).isoformat(),
  active_count=24,
  deferred=DEFERRED_CL_TASKS,
  git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
  git_status=subprocess.check_output(
    ["git", "status", "--short"], text=True
  ).splitlines(),
  package_versions=packages,
  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  tasks=rows,
)
(here / "evidence/preflight.json").write_text(json.dumps(data, indent=2) + "\n")
print(
  json.dumps(
    dict(
      active_count=24,
      obs_dim=OBS_DIM,
      gravity_off=sum(not any(r["gravity"]) for r in rows),
      packages=packages,
    ),
    indent=2,
  )
)
