"""Opposed pad normals reject top pressing without whole-box containment."""
import importlib
import math
from pathlib import Path
import torch


def module(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  return importlib.import_module('contact_grasp')


def test_contact_direction_object_identity_world_and_buffer_validity(monkeypatch):
  m=module(monkeypatch)
  n=8
  pads=torch.tensor([[[0.,-.03,0.],[0.,.03,0.]]]).expand(n,-1,-1)
  geoms=torch.tensor([[10,20],[11,20]]).repeat(n,1)
  normals=torch.tensor([[0.,1.,0.],[0.,-1.,0.]]).repeat(n,1)
  distances=torch.zeros(2*n)
  worlds=torch.arange(n).repeat_interleave(2)
  normals[2:4]=torch.tensor([0.,0.,-1.]) # Same top face, two pads.
  normals[4:6]=torch.tensor([0.,1.,0.])  # Both normals in same direction.
  geoms[7,1]=30                        # Other object's contact cannot help.
  geoms[8:10]=geoms[8:10].flip(1)      # Geom ordering reversal must be handled.
  normals[8:10]*=-1
  distances[11]=.01                    # Separated pads are not grasping.
  # Worlds6/7 contain plausible-looking stale contacts past nacon.
  mask=m.opposed_contact_mask(geoms,normals,distances,worlds,12,torch.tensor([10,11]),torch.tensor([20]),pads)
  assert mask.tolist()==[True,False,False,False,True,False,False,False]
  worlds[0:2]=-1                      # Invalid world IDs must not leak to0.
  assert not m.opposed_contact_mask(geoms,normals,distances,worlds,12,torch.tensor([10,11]),torch.tensor([20]),pads)[0]


def test_rotated_cube_centerline_is_narrower_than_full_projection(monkeypatch):
  m=module(monkeypatch)
  angle=math.pi/6
  direction=torch.tensor([[math.cos(angle),math.sin(angle),0.]])
  width,valid=m.box_ray_width(torch.zeros(1,3),direction,torch.full((3,),-.023),torch.full((3,),.023))
  assert valid.item()
  torch.testing.assert_close(width,torch.tensor([.046/math.cos(angle)]))
  projected=.046*(math.cos(angle)+math.sin(angle))
  assert width.item()<projected-.008


def test_ray_miss_parallel_axes_and_zero_direction_remain_finite(monkeypatch):
  m=module(monkeypatch)
  origin=torch.tensor([[0.,0.,0.],[.1,0.,0.],[0.,0.,0.]])
  direction=torch.tensor([[0.,1.,0.],[0.,1.,0.],[0.,0.,0.]])
  width,valid=m.box_ray_width(origin,direction,torch.full((3,),-.023),torch.full((3,),.023))
  assert valid.tolist()==[True,False,False]
  torch.testing.assert_close(width,torch.tensor([.046,0.,0.]))


def test_slender_peg_fallback_uses_cross_section_without_changing_true_ray_hits(monkeypatch):
  m=module(monkeypatch)
  angle=math.radians(18)
  direction=torch.tensor([[math.cos(angle),0.,math.sin(angle)]]).expand(2,-1)
  lower=torch.tensor([-.0125,-.0125,0.]);upper=torch.tensor([.0125,.0125,.1])
  # First center ray misses sideways by7.5mm, within the8.8mm finite pad width.
  origin=torch.tensor([[0.,.02,.07],[0.,0.,.05]])
  projected=((upper-lower)*direction.abs()).sum(-1)
  old=m.box_capture_width(origin,direction,lower,upper,projected)
  new=m.box_capture_width(origin,direction,lower,upper,projected,True)
  assert old[0]>.05 and .025<new[0]<.028
  torch.testing.assert_close(new,torch.full((2,),.025/math.cos(angle)))
  torch.testing.assert_close(new[1],old[1])


def test_peg_fallback_recipe_changes_only_training_width_estimate(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Peg-Insertion-Franka') for _ in range(2))
  recipes.apply_recipe(old,'completion_v6');recipes.apply_recipe(new,'peg_v1')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['stack'].params.pop('centered_fallback')
  assert repr(old.env.rewards)==repr(new.env.rewards)
