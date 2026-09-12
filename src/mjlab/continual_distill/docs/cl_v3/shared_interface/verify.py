"""GPU integration checks of the common contract, independent of PPO convergence."""
import json
from pathlib import Path
import numpy as np
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.franka_interface import OBS_DIM, SLICES, CENTER, HALF_RANGE, LOWER, UPPER
from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.shared_interface import legacy_observation, normalized_actions
here=Path(__file__).resolve().parent
names=[r['task'] for r in json.loads((here.parent/'rl_readiness/audit.json').read_text())]
results=[]
for name in names:
 cfg=load_env_cfg(name);cfg.scene.num_envs=4;cfg.seed=20260912
 env=ManagerBasedRlEnv(cfg,device='cuda:0')
 obs,_=env.reset();term=env.action_manager.get_term('robot_joint_pos')
 assert obs['policy'].shape==(4,OBS_DIM)
 torch.testing.assert_close(obs['policy'],obs['critic'])
 assert env.action_manager.total_action_dim==8
 for x,expected in [(-2,LOWER[:8]),(0,CENTER[:8]),(2,UPPER[:8])]:
  term.process_actions(torch.full((4,8),float(x),device=env.device))
  torch.testing.assert_close(term._processed_actions,torch.tensor(expected,device=env.device).expand(4,-1))
 term.reset()
 teacher=CLASSICAL_POLICIES[name](num_envs=4)
 reference=CLASSICAL_POLICIES[name](num_envs=4)
 max_error=0.
 for i in range(16):
  x=obs['policy'].cpu().numpy()
  a=teacher(x)
  old=reference(legacy_observation(x,reference.default_qpos,reference.legacy_layout))
  expected=normalized_actions(old,reference.default_qpos)
  max_error=max(max_error,float(np.max(np.abs(a-expected))))
  np.testing.assert_allclose(a,expected,atol=2e-6)
  assert np.isfinite(a).all() and np.abs(a).max()<=1
  obs,reward,term_done,trunc,info=env.step(torch.tensor(a,device=env.device))
  assert torch.isfinite(reward).all() and torch.isfinite(obs['policy']).all()
  torch.testing.assert_close(obs['policy'],obs['critic'])
  # Physical units round-trip and local position convention.
  robot=env.scene['robot'];ids=[robot.joint_names.index(f'joint{j}') for j in range(1,8)]+[robot.joint_names.index('finger_joint1'),robot.joint_names.index('finger_joint2')]
  q=obs['policy'][:,SLICES['joint_position']]*torch.tensor(HALF_RANGE,device=env.device)+torch.tensor(CENTER,device=env.device)
  torch.testing.assert_close(q,robot.data.joint_pos[:,ids],atol=2e-6,rtol=1e-5)
  torch.testing.assert_close(obs['policy'][:,SLICES['gripper_position']],robot.data.site_pos_w[:,robot.site_names.index('gripper')]-env.scene.env_origins)
  done=(term_done|trunc).nonzero().flatten().cpu().numpy()
  if len(done):teacher.reset(done);reference.reset(done)
 for i in range(8):
  a = torch.randn(4, 8, device=env.device) * load_rl_cfg(name).policy.init_noise_std
  obs, reward, *_ = env.step(a)
  assert torch.isfinite(reward).all() and torch.isfinite(env.sim.data.qpos).all()
  assert torch.isfinite(obs['policy']).all()
  assert (term._processed_actions >= torch.tensor(LOWER[:8], device=env.device)-1e-6).all()
  assert (term._processed_actions <= torch.tensor(UPPER[:8], device=env.device)+1e-6).all()
 results.append(dict(task=name,obs_dim=OBS_DIM,action_dim=8,finite=True,teacher_steps=16,random_steps=8,teacher_adapter_max_error=max_error))
 (here/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
 print('PASS',name,flush=True)
 env.close()
