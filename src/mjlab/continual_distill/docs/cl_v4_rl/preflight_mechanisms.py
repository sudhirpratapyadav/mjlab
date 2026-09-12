"""Finite-state preflight for remaining mechanism recipes; no learning claim."""
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from rl_recipes import apply_recipe

HERE = Path(__file__).resolve().parent
CASES = ["Flip-Switch", "Open-Door", "Open-Lid", "Push-Flap", "Turn-Lever", "Rotate-Valve", "Slide-Window", "Axial-Extract"]
rows = []
output = HERE / "evidence/mechanism_preflight2.json"
if output.exists():
  raise FileExistsError(output)
for short in CASES:
  task = f"Mjlab-{short}-Franka"
  cfg = TrainConfig.from_task(task)
  invariant_fields = ("observations", "actions", "commands", "terminations", "events", "scene", "sim", "episode_length_s")
  before = {field: repr(getattr(cfg.env, field)) for field in invariant_fields}
  apply_recipe(cfg, "mechanism_v1")
  assert before == {field: repr(getattr(cfg.env, field)) for field in invariant_fields}
  cfg.env.scene.num_envs = 16
  cfg.env.seed = 20260912
  env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
  try:
    obs, _ = env.reset()
    for _ in range(32):
      obs, reward, *_ = env.step(torch.randn(16, 8, device=env.device) * 0.1)
      assert obs["policy"].shape == (16, 60)
      assert all(torch.isfinite(tensor).all() for tensor in (obs["policy"], reward, env.sim.data.qpos, env.sim.data.qvel))
    rows.append({"task": task, "recipe": "mechanism_v1", "finite": True, "interface_unchanged": True, "num_envs": 16, "steps": 32, "seed": 20260912, "gravity": list(cfg.env.sim.mujoco.gravity)})
    print("PASS", task, flush=True)
  finally:
    env.close()
output.write_text(json.dumps(rows, indent=2) + "\n")
