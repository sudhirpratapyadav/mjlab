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
