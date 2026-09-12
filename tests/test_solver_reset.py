"""Episode resets clear stale acceleration without perturbing ongoing worlds."""
from types import SimpleNamespace

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.sim.sim import Simulation


@pytest.mark.parametrize('ids', [None, slice(1,3), torch.tensor([0,2]), torch.tensor([],dtype=torch.long)])
def test_solver_reset_is_selective_and_preserves_physical_state(ids):
  sim = object.__new__(Simulation)
  warm = torch.arange(20,dtype=torch.float32).reshape(4,5)+1
  original = warm.clone()
  qpos, qvel, ctrl = torch.randn(4,6), torch.randn(4,5), torch.randn(4,3)
  before = [x.clone() for x in (qpos,qvel,ctrl)]
  sim._data_bridge = SimpleNamespace(qacc_warmstart=warm,qpos=qpos,qvel=qvel,ctrl=ctrl)
  sim.reset_solver_state(ids)
  expected = original.clone()
  expected[slice(None) if ids is None else ids] = 0
  torch.testing.assert_close(warm,expected,rtol=0,atol=0)
  for actual,saved in zip((qpos,qvel,ctrl),before):
    torch.testing.assert_close(actual,saved,rtol=0,atol=0)


def test_real_episode_reset_clears_cache_before_reset_pose_forward():
  sim = object.__new__(Simulation)
  warm = torch.ones(3,9)
  sim._data_bridge = SimpleNamespace(qacc_warmstart=warm)
  ids = torch.tensor([1])
  calls = []
  def check_cache():
    assert torch.equal(warm[1],torch.zeros(9))
    assert torch.equal(warm[[0,2]],torch.ones(2,9))
    calls.append('forward')
  sim.forward = check_cache
  manager = SimpleNamespace(reset=lambda ids:{},compute=lambda **kw:None)
  env = SimpleNamespace(sim=sim,curriculum_manager=manager,
    scene=SimpleNamespace(reset=lambda ids:None,write_data_to_sim=lambda:None),
    event_manager=SimpleNamespace(available_modes=['reset'],apply=lambda **kw:None,reset=lambda ids:{}),
    cfg=SimpleNamespace(decimation=4),_sim_step_counter=8,extras={},
    observation_manager=manager,action_manager=manager,reward_manager=manager,
    command_manager=manager,termination_manager=manager,episode_length_buf=torch.tensor([5,7,9]))
  ManagerBasedRlEnv._reset_idx(env,ids)
  assert calls == ['forward']
  assert env.episode_length_buf.tolist() == [5,0,9]
