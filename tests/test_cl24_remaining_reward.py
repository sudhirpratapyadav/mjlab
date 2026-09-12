"""Shaping predictions must distinguish correct flights from plausible misses."""
import importlib
from pathlib import Path
import torch


def module(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  return importlib.import_module('remaining_reward')


def test_slide_prediction_distinguishes_undershoot_overshoot_and_sideways(monkeypatch):
  reward = module(monkeypatch)
  start = torch.tensor([[.4,0.]]).repeat(4,1)
  mu = .04
  speed = (2*mu*9.81*.5)**.5
  velocity = torch.tensor([[speed,0.],[speed*.5,0.],[speed*1.5,0.],[0.,speed]])
  landing = reward.sliding_endpoint(start,velocity,mu)
  goal = torch.tensor([.9,0.])
  error = torch.linalg.vector_norm(landing-goal,dim=-1)
  assert error[0]<1e-6 and (error[1:]>.3).all()
  torch.testing.assert_close(reward.sliding_endpoint(start,torch.zeros_like(velocity),mu),start)


def test_strike_target_crosses_wedge_contact_without_mutating_state(monkeypatch):
  reward = module(monkeypatch)
  position = torch.tensor([[.4,.1,.0127],[.4,-.1,1.0127]])
  original = position.clone()
  direction = torch.tensor([[1.,0.],[-1.,0.]])
  floor = torch.tensor([0.,1.])
  old = reward.strike_approach_target(position,direction,floor)
  driven = reward.strike_approach_target(position,direction,floor,.012)
  # Independently derive the first pad/cylinder contact in the horizontal
  # plane from registered geometry:38.1mm puck radius,30mm half-gap,
  # 8.75mm pad half-length. The old seating waypoint leaves a real gap.
  contact_standoff = (.0381**2-.030**2)**.5+.00875
  old_standoff = ((position[:,:2]-old[:,:2])*direction).sum(-1)
  new_standoff = ((position[:,:2]-driven[:,:2])*direction).sum(-1)
  assert (old_standoff>contact_standoff+.002).all()
  assert (new_standoff<contact_standoff-.008).all()
  torch.testing.assert_close(position,original)
  torch.testing.assert_close(driven[:,2],floor+.028)
  torch.testing.assert_close(old[:,1],driven[:,1])


def test_strike_drive_recipe_preserves_benchmark_and_policy(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes = importlib.import_module('rl_recipes')
  old,new = (TrainConfig.from_task('Mjlab-Strike-Slide-Franka') for _ in range(2))
  recipes.apply_recipe(old,'strike_v1')
  recipes.apply_recipe(new,'strike_v2')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('contact_drive')==.012
  assert new.env.rewards['reach_object'].params.pop('precision_weight')==2.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_ballistic_prediction_checks_rim_crossing_and_downward_branch(monkeypatch):
  reward = module(monkeypatch)
  position = torch.tensor([[.5,0.,.4],[.5,0.,.05],[.5,0.,.4]])
  velocity = torch.tensor([[1.,0.,0.],[1.,0.,0.],[1.,0.,-1.]])
  height = torch.full((3,),.14)
  landing,valid = reward.ballistic_endpoint(position,velocity,height)
  assert valid.tolist()==[True,False,True]
  expected = .5+(2*(.4-.14)/9.81)**.5
  torch.testing.assert_close(landing[0,0],torch.tensor(expected))
  assert landing[2,0]<landing[0,0]
  assert torch.isfinite(landing).all()
