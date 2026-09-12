"""Exercise completion recipes with real physics before expensive pilots."""
from dataclasses import asdict
import json
from pathlib import Path
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from bounded_policy import BoundedActorCritic
from completion_reward import safe_grasp_target
from rl_recipes import apply_recipe
from wandb_config import configure,ENTITY,PROJECT

HERE=Path(__file__).resolve().parent
output=HERE/'evidence/completion_preflight.json'
if output.exists():
  raise FileExistsError(output)
rows=[]
for short in ['Stack-Cube','Place-In-Container','Peg-Insertion']:
  task=f'Mjlab-{short}-Franka'
  cfg=TrainConfig.from_task(task)
  apply_recipe(cfg,'completion_v1')
  cfg.env.scene.num_envs=32
  cfg.env.seed=20260912
  env=ManagerBasedRlEnv(cfg.env,device='cuda:0')
  wrapped=RslRlVecEnvWrapper(env,clip_actions=cfg.agent.clip_actions)
  try:
    obs=wrapped.get_observations()
    values=asdict(cfg.agent.policy);values.pop('class_name')
    policy=BoundedActorCritic(obs,cfg.agent.obs_groups,8,**values).to(env.device)
    command=env.command_manager.get_term(env.command_manager.active_terms[0])
    grasp_height=(safe_grasp_target(command)[:,2]-env.scene.env_origins[:,2]).mean().item()
    total_reward=0.
    for _ in range(100):
      with torch.no_grad():
        policy.update_normalization(obs)
        obs,reward,*_=wrapped.step(policy.act(obs))
      assert obs['policy'].shape==(32,60)
      assert all(torch.isfinite(v).all() for v in [obs['policy'],reward,env.sim.data.qpos,env.sim.data.qvel])
      total_reward+=float(reward.mean())
    rows.append({'task':task,'recipe':'completion_v1','finite':True,'num_envs':32,'steps':100,'initial_grasp_height':grasp_height,'mean_return':total_reward,'interface':'franka_shared_60_v2'})
    print('PASS',rows[-1],flush=True)
  finally:
    wrapped.close()
output.write_text(json.dumps(rows,indent=2)+'\n')
configure()
import wandb
with wandb.init(entity=ENTITY,project=PROJECT,name='completion-reward-preflight',job_type='readiness') as run:
  run.log({'finite_cases':len(rows)})
  artifact=wandb.Artifact('completion-reward-preflight',type='readiness');artifact.add_file(str(output));run.log_artifact(artifact)
