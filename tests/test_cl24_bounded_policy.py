"""Bounded means must survive optimization and checkpoint reload unchanged."""
import importlib
from dataclasses import asdict
from pathlib import Path
import torch
import copy
import pytest
from tensordict import TensorDict
from mjlab.scripts.train import TrainConfig


def test_bounded_policy_initialization_and_roundtrip(monkeypatch):
  stage = Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes = importlib.import_module('rl_recipes')
  module = importlib.import_module('bounded_policy')
  cfg = TrainConfig.from_task('Mjlab-Cage-Drag-Franka')
  recipes.apply_recipe(cfg,'cage_v1')
  obs = TensorDict({'policy':torch.randn(64,60),'critic':torch.randn(64,60)},batch_size=[64])
  values = asdict(cfg.agent.policy)
  values.pop('class_name')
  policy = module.BoundedActorCritic(obs, cfg.agent.obs_groups, 8, **values)
  mean = policy.act_inference(obs)
  assert mean.shape == (64,8) and mean.abs().max() <= 1
  assert mean[:,-1].min() > .98
  action = cfg.env.actions['robot_joint_pos']
  init = cfg.env.scene.entities['robot'].init_state.joint_pos
  for i in range(7):
    name = f'joint{i+1}'
    mapped = cfg.agent.policy.initial_mean[i]*action.scale[name]+action.offset[name]
    assert abs(mapped-init[name]) < 1e-6
  sampled = policy.act(obs)
  loss = -policy.get_actions_log_prob(sampled.detach()).mean() + policy.evaluate(obs).square().mean()
  optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
  optimizer.zero_grad();loss.backward();optimizer.step()
  policy.eval()
  clone = module.BoundedActorCritic(obs,cfg.agent.obs_groups,8,**values)
  clone.load_state_dict(policy.state_dict());clone.eval()
  torch.testing.assert_close(policy.act_inference(obs),clone.act_inference(obs))
  assert torch.isfinite(policy.log_std.exp()).all() and (policy.log_std.exp()>0).all()
  # Even extreme normalized observations cannot create an unbounded mean.
  huge = TensorDict({'policy':torch.randn(64,60)*1e6,'critic':torch.randn(64,60)},batch_size=[64])
  assert policy.act_inference(huge).abs().max() <= 1

  # A resume intervention must preserve deterministic behavior, normalization,
  # and all learned parameters and optimizer state outside gripper exploration.
  runner_module = importlib.import_module('teacher_runner')
  before = copy.deepcopy(policy.state_dict())
  moments = {p:copy.deepcopy(v) for p,v in optimizer.state.items()}
  output_before = policy.act_inference(obs).detach().clone()
  runner_module.reset_gripper_exploration(policy, optimizer, .2)
  torch.testing.assert_close(policy.act_inference(obs), output_before, rtol=0, atol=0)
  for name, value in policy.state_dict().items():
    if name == 'log_std':
      torch.testing.assert_close(value[:-1], before[name][:-1], rtol=0, atol=0)
      assert value[-1].exp().item() == pytest.approx(.2)
    else:
      torch.testing.assert_close(value, before[name], rtol=0, atol=0)
  for parameter, old_state in moments.items():
    for key, old in old_state.items():
      new = optimizer.state[parameter][key]
      if parameter is policy.log_std and key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        assert new[-1] == 0
        torch.testing.assert_close(new[:-1], old[:-1], rtol=0, atol=0)
      else:
        torch.testing.assert_close(new, old, rtol=0, atol=0)
  after = copy.deepcopy(policy.state_dict())
  for invalid in (0., -1., float('nan'), float('inf')):
    with pytest.raises(ValueError):
      runner_module.reset_gripper_exploration(policy, optimizer, invalid)
  for name, value in policy.state_dict().items():
    torch.testing.assert_close(value, after[name], rtol=0, atol=0)
