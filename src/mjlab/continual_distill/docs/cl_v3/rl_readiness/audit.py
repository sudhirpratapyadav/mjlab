"""Read-only RL configuration and short numerical smoke audit of the CL25 tasks."""
import json, dataclasses, traceback
from pathlib import Path
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.continual_distill.classical import CLASSICAL_POLICIES

here=Path(__file__).resolve().parent
names=sorted(p.parent.name for p in (here.parent/'teacher_refresh/all_remaining/baseline128').glob('*/result.json'))+['Mjlab-Reorient-Object-Franka']
def pack(x):
 if dataclasses.is_dataclass(x): return {f.name:pack(getattr(x,f.name)) for f in dataclasses.fields(x)}
 if isinstance(x,dict):return {str(k):pack(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [pack(v) for v in x]
 if callable(x):return getattr(x,'__name__',str(x))
 if isinstance(x,(str,int,float,bool)) or x is None:return x
 return str(x)
rows=[]
for name in names:
 row={'task':name}
 env=None
 try:
  cfg=load_env_cfg(name);rl=load_rl_cfg(name)
  row.update(actions=pack(cfg.actions),observations=pack(cfg.observations),rewards=pack(cfg.rewards),curriculum=pack(cfg.curriculum),terminations=pack(cfg.terminations),rl=pack(rl),episode_s=cfg.episode_length_s,dt=cfg.sim.mujoco.timestep*cfg.decimation,commands=pack(cfg.commands),configured_num_envs=cfg.scene.num_envs)
  cfg.scene.num_envs=4;cfg.seed=20260913
  env=ManagerBasedRlEnv(cfg,device='cuda:0');obs,_=env.reset()
  row['obs_dims']={k:list(v.shape) for k,v in obs.items()}
  row['obs_terms']=pack(env.observation_manager._group_obs_term_dim)
  row['origins']=env.scene.env_origins.cpu().tolist()
  term=env.action_manager.get_term('robot_joint_pos')
  row['action_dim']=env.action_manager.total_action_dim
  row['offset']=term.offset[0].cpu().tolist() if torch.is_tensor(term.offset) else term.offset
  robot=env.scene['robot'];limits=robot.data.soft_joint_pos_limits[0,:7]
  row['arm_soft_limits']=limits.cpu().tolist()
  row['raw_action_at_soft_limits']=((limits-torch.as_tensor(row['offset'][:7],device=limits.device)[:,None])/.04).cpu().tolist()
  teacher=CLASSICAL_POLICIES[name](num_envs=4);a=teacher(obs['policy'].cpu().numpy())
  row['initial_teacher_max_abs_arm_action']=float(abs(a[:,:7]).max())
  row['finite']=all(bool(torch.isfinite(v).all()) for v in obs.values())
  means=[];terms=[]
  for step in range(12):
   action=torch.randn(4,row['action_dim'],device='cuda:0')
   obs,reward,terminated,truncated,info=env.step(action)
   row['finite'] &= bool(torch.isfinite(reward).all()) and all(bool(torch.isfinite(v).all()) for v in obs.values())
   means.append(float(reward.mean()));terms.append(env.reward_manager._step_reward.mean(0).cpu().tolist())
  row['random_12step_reward_range']=[min(means),max(means)]
  row['reward_terms']=env.reward_manager._term_names
  row['mean_weighted_reward_rates']=torch.tensor(terms).mean(0).tolist()
  row['status']='pass'
 except Exception:
  row['status']='error';row['error']=traceback.format_exc();print(row['error'],flush=True)
 finally:
  if env is not None:env.close()
 rows.append(row);(here/'audit.json').write_text(json.dumps(rows,indent=2)+'\n');print('AUDIT',name,row['status'],row.get('obs_dims'),flush=True)
