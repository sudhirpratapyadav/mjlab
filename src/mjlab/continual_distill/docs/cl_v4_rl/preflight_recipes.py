"""Numerical preflight for the next recipe batch before expensive learning."""

import json
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from rl_recipes import apply_recipe

HERE=Path(__file__).resolve().parent
CASES=[("Lift-Cube","lift_v1"),("Reorient-Object","reorient_v1"),
       ("Push-Button","mechanism_v1"),("Open-Drawer","mechanism_v1"),
       ("Flip-Switch","mechanism_v1"),("Push-Cuboid","stable_v1")]
rows=[]
for short,recipe in CASES:
  task=f"Mjlab-{short}-Franka"
  cfg=TrainConfig.from_task(task)
  apply_recipe(cfg,recipe)
  cfg.env.scene.num_envs=16
  cfg.env.seed=20260912
  env=ManagerBasedRlEnv(cfg.env,device="cuda:0")
  try:
    obs,_=env.reset()
    initial=env.reward_manager.compute(env.step_dt).clone()
    for _ in range(32):
      action=torch.randn(16,8,device=env.device)*0.1
      obs,reward,*_=env.step(action)
      assert obs['policy'].shape==(16,60)
      assert torch.isfinite(obs['policy']).all() and torch.isfinite(reward).all()
      assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
    row={'task':task,'recipe':recipe,'finite':True,'obs_dim':60,'action_dim':8,'num_envs':16,'control_steps':32,'initial_reward_mean':float(initial.mean()),'gravity':list(cfg.env.sim.mujoco.gravity)}
    rows.append(row)
    print('PASS',row,flush=True)
  finally:
    env.close()
(HERE/'evidence/recipe_preflight.json').write_text(json.dumps(rows,indent=2)+'\n')
