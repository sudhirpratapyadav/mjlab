"""Bounded means must survive optimization and checkpoint reload unchanged."""
import importlib
from dataclasses import asdict
from pathlib import Path
import torch
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
