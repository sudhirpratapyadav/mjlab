"""Compare teacher conversion against the original independently computed MDP terms."""
import json
from pathlib import Path
import numpy as np
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.config.franka import env_cfgs
from mjlab.tasks.manipulation.franka_interface import SLICES, shared_observation
from mjlab.managers.observation_manager import ObservationManager
from mjlab.envs import ManagerBasedRlEnv
from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.shared_interface import legacy_observation
here=Path(__file__).resolve().parent
rows=[]
for task in ['Reach-Target','Lift-Cube','Peg-Insertion','Tool-Pull']:
 name=f'Mjlab-{task}-Franka';cfg=load_env_cfg(name);cfg.scene.num_envs=4
 env=ManagerBasedRlEnv(cfg,device='cuda:0');obs,_=env.reset()
 old_cfg=getattr(env_cfgs,'franka_'+task.lower().replace('-','_')+'_env_cfg')(test=True)
 old_manager=ObservationManager(old_cfg.observations,env)
 teacher=CLASSICAL_POLICIES[name](4)
 params=next(iter(cfg.observations['policy'].terms.values())).params
 max_error=0.
 for step in range(4):
  old=old_manager.compute(update_history=True)['policy'].cpu().numpy()
  # Independent old terms contain world positions. Shift just their position fields.
  origin=env.scene.env_origins.cpu().numpy()
  slots={'reach':[(18,21)],'stack':[(18,21),(25,28),(28,31)],'standard':[(18,21),(25,28)],'tool':[(18,21),(25,28)]}[teacher.legacy_layout]
  for start,end in slots:old[:,start:end]-=origin
  converted=legacy_observation(obs['policy'].cpu().numpy(),teacher.default_qpos,teacher.legacy_layout)
  max_error=max(max_error,float(np.max(np.abs(old-converted))))
  np.testing.assert_allclose(converted,old,atol=3e-6,rtol=2e-5)
  # Missing role fields must be exactly zero; Reach still observes its real goal.
  if task=='Reach-Target':
   for key in ['object_position','object_velocity','object_quaternion','fixture_position','tool_position','gripper_to_object','object_to_goal','presence']:
    assert torch.count_nonzero(obs['policy'][:,SLICES[key]])==0,key
  action=teacher(obs['policy'].cpu().numpy())
  obs,*_=env.step(torch.tensor(action,device=env.device))
 rows.append(dict(task=name,legacy_max_error=max_error));print(rows[-1],flush=True)
 env.close()
(here/'legacy_comparison.json').write_text(json.dumps(rows,indent=2)+'\n')
