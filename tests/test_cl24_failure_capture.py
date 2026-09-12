"""Retain the lead-up to failures even when native reset replaced bad state."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_failure_history_survives_native_state_reset(monkeypatch,tmp_path):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  module = importlib.import_module('teacher_runner')
  data = SimpleNamespace(**{name:torch.zeros(2,width) for name,width in
    [('qpos',9),('qvel',9),('ctrl',8),('qacc_warmstart',9),('mocap_pos',3),('mocap_quat',4)]})
  steps = 0
  def step(action):
    nonlocal steps
    steps += 1
    data.qpos.add_(1)
    obs,reward = torch.zeros(2,60),torch.ones(2)
    if steps == 9:
      # The native loop may already have reset the failing world, but its
      # computed reward/observation remains invalid and must still stop PPO.
      data.qpos[1].zero_()
      obs[1,0] = float('nan')
      reward[1] = float('nan')
    return {'policy':obs},reward,torch.zeros(2,dtype=torch.bool),{}
  env = SimpleNamespace(step=step,unwrapped=SimpleNamespace(sim=SimpleNamespace(data=data)))
  policy = torch.nn.Linear(60,8)
  def init(self,*args,**kwargs):
    self.env = env
    self.alg = SimpleNamespace(policy=policy,optimizer=torch.optim.Adam(policy.parameters()),update=lambda:{})
    self.logger = SimpleNamespace(log=lambda **kw:None,log_dir=str(tmp_path))
    self.current_learning_iteration = 123
  monkeypatch.setattr(module.OnPolicyRunner,'__init__',init)
  runner = module.TeacherRunner(capture_pre_step=True)
  for _ in range(8):
    runner.env.step(torch.zeros(2,8))
  with pytest.raises(RuntimeError,match='rollout_state_or_reward'):
    runner.env.step(torch.zeros(2,8))
  info = json.loads((tmp_path/'numerical_failure.json').read_text())
  assert info['nonfinite_simulator_env_ids'] == []
  assert info['nonfinite_reward_env_ids'] == [1]
  assert info['nonfinite_observation_env_ids'] == [1]
  assert info['captured_history_steps'] == 8
  saved = torch.load(tmp_path/'numerical_failure.pt',weights_only=False)
  assert [float(state['qpos'][0,0]) for state in saved['state_history']] == list(range(1,9))
  assert float(saved['previous_state']['qpos'][1,0]) == 8
  assert torch.isfinite(saved['qpos']).all()
  assert torch.isnan(saved['rewards'][1])
