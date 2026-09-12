"""Check real bounded-policy exploration and episode-long open-cage validity."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from bounded_policy import BoundedActorCritic
from rl_recipes import apply_recipe
from wandb_config import configure, ENTITY, PROJECT

HERE = Path(__file__).resolve().parent
output = HERE/'evidence/bounded_preflight.json'
if output.exists():
  raise FileExistsError(output)
rows = []
for task,recipe in [('Mjlab-Cage-Drag-Franka','cage_v1'),('Mjlab-Reorient-Object-Franka','reorient_v2'),('Mjlab-Lift-Cube-Franka','lift_v2')]:
  cfg = TrainConfig.from_task(task)
  apply_recipe(cfg,recipe)
  cfg.env.scene.num_envs = 64
  cfg.env.seed = 20260912
  env = ManagerBasedRlEnv(cfg.env,device='cuda:0')
  wrapped = RslRlVecEnvWrapper(env,clip_actions=cfg.agent.clip_actions)
  try:
    values = asdict(cfg.agent.policy)
    values.pop('class_name')
    obs = wrapped.get_observations()
    policy = BoundedActorCritic(obs,cfg.agent.obs_groups,8,**values).to(env.device)
    aperture_min = 1.0
    peak_mean = 0.0
    command = env.command_manager.get_term(env.command_manager.active_terms[0])
    for step in range(200):
      with torch.no_grad():
        policy.update_normalization(obs)
        actions = policy.act(obs)
        peak_mean = max(peak_mean,float(policy.action_mean.abs().max()))
        obs,reward,done,extras = wrapped.step(actions)
      assert obs['policy'].shape == (64,60)
      assert all(torch.isfinite(value).all() for value in [obs['policy'],reward,env.sim.data.qpos,env.sim.data.qvel])
      if recipe == 'cage_v1':
        aperture_min = min(aperture_min,float(command._aperture().min()))
    if recipe == 'cage_v1':
      assert aperture_min > command.cfg.aperture_min
    rows.append({'task':task,'recipe':recipe,'num_envs':64,'steps':200,'finite':True,'peak_absolute_mean':peak_mean,'minimum_aperture':aperture_min if recipe=='cage_v1' else None,'initial_mean':cfg.agent.policy.initial_mean,'interface':'franka_shared_60_v2'})
    print('PASS',rows[-1],flush=True)
  finally:
    wrapped.close()
output.write_text(json.dumps(rows,indent=2)+'\n')
configure()
import wandb
with wandb.init(entity=ENTITY,project=PROJECT,name='bounded-policy-preflight',job_type='readiness') as run:
  run.log({'finite_cases':len(rows),'cage_minimum_aperture':rows[0]['minimum_aperture']})
  artifact = wandb.Artifact('bounded-policy-preflight',type='readiness')
  artifact.add_file(str(output));run.log_artifact(artifact)
