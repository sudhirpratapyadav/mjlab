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
