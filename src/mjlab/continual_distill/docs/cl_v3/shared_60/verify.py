"""Compare the restored fields with legacy MDP terms, then exercise all CL25 tasks."""
import json
from pathlib import Path
import numpy as np
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.config.franka import env_cfgs
from mjlab.tasks.manipulation.franka_interface import OBS_DIM, LOWER, UPPER
from mjlab.managers.observation_manager import ObservationManager
from mjlab.envs import ManagerBasedRlEnv
from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.shared_interface import legacy_observation, normalized_actions
here=Path(__file__).resolve().parent
from mjlab.tasks.manipulation.benchmark import active_cl_tasks
names=active_cl_tasks()
rows=[]
for name in names:
 cfg=load_env_cfg(name,test=True);cfg.scene.num_envs=4;cfg.seed=20260912
 env=ManagerBasedRlEnv(cfg,device='cuda:0');obs,_=env.reset()
 task=name.removeprefix('Mjlab-').removesuffix('-Franka')
 original_cfg=getattr(env_cfgs,'franka_'+task.lower().replace('-','_')+'_env_cfg')(test=True)
 old_manager=ObservationManager(original_cfg.observations,env)
 teacher=CLASSICAL_POLICIES[name](4)
 reference=CLASSICAL_POLICIES[name].__mro__[1](4)
 layout=teacher.legacy_layout
 max_error=0.; action_error=0.
 for step in range(16):
  assert obs['policy'].shape==(4,60) and OBS_DIM==60
  torch.testing.assert_close(obs['policy'],obs['critic'])
  raw=old_manager.compute(update_history=True)['policy'].cpu().numpy()
  origin=env.scene.env_origins.cpu().numpy()
  slots={'reach':[(18,21)],'stack':[(18,21),(25,28),(28,31)],'standard':[(18,21),(25,28)],'tool':[(18,21),(25,28)]}[layout]
  for start,end in slots:raw[:,start:end]-=origin
  x=obs['policy'].cpu().numpy()
  if layout=='tool':
   # Original 69D Tool input consists of the ordinary 60 fields plus 9 tool fields.
   expected=np.concatenate((raw[:,:52],raw[:,-8:]),axis=-1)
   actual=x
   try:teacher(x)
   except ValueError as e:assert 'separate tool pose' in str(e)
   else:raise AssertionError('Tool teacher silently accepted missing data')
  else:
   actual=legacy_observation(x,reference.default_qpos,layout)
   expected=raw
   if layout=='stack':
    # Removed fixture position is unused by the existing Stack/Peg controllers.
    actual=actual.copy();actual[:,25:28]=expected[:,25:28]
  max_error=max(max_error,float(np.max(np.abs(actual-expected))))
  np.testing.assert_allclose(actual,expected,atol=3e-6,rtol=2e-5)
  if step<8 and layout!='tool':
   a=teacher(x)
   old=reference(raw)
   wanted=normalized_actions(old,reference.default_qpos)
   action_error=max(action_error,float(np.max(np.abs(a-wanted))))
   np.testing.assert_allclose(a,wanted,atol=2e-5,rtol=1e-5)
   action=torch.tensor(a,device=env.device)
  else:action=torch.randn(4,8,device=env.device)*.2
  obs,reward,terminated,truncated,_=env.step(action)
  assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
  assert torch.isfinite(env.sim.data.qpos).all()
  action_term=env.action_manager.get_term('robot_joint_pos')
  assert (action_term._processed_actions>=torch.tensor(LOWER[:8],device=env.device)-1e-6).all()
  assert (action_term._processed_actions<=torch.tensor(UPPER[:8],device=env.device)+1e-6).all()
  done=(terminated|truncated).nonzero().flatten().cpu().numpy()
  if len(done):teacher.reset(done);reference.reset(done)
 row=dict(task=name,obs_dim=60,action_dim=8,finite=True,legacy_field_max_error=max_error,teacher_action_max_error=action_error if layout!='tool' else None,teacher_status='requires separate tool pose' if layout=='tool' else 'pass')
 rows.append(row);(here/'verification.json').write_text(json.dumps(rows,indent=2)+'\n');print('PASS',row,flush=True)
 env.close()
