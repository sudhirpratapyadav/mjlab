"""Project a live orientation onto an axis-alignment goal, preserving free rotation."""

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_mul


def nearest_axis_goal(quat, body_axis, target_axis, symmetric=False):
  current = quat_apply(quat, body_axis.expand(len(quat), -1))
  target = target_axis.clone()
  dot = (current * target).sum(-1, keepdim=True)
  if symmetric:
    target = torch.where(dot < 0, -target, target)
    dot = dot.abs()
  correction = torch.cat([1 + dot, torch.linalg.cross(current, target)], dim=-1)
  opposite = dot[:, 0] < -0.999999
  if opposite.any():
    # At 180 degrees, use a body-attached perpendicular axis to preserve continuity
    # under free rotations, instead of choosing an arbitrary fixed world heading.
    basis = torch.eye(3, device=quat.device, dtype=quat.dtype)[body_axis.abs().argmin()]
    perpendicular = torch.linalg.cross(body_axis, basis)
    perpendicular = perpendicular / torch.linalg.norm(perpendicular)
    axis = quat_apply(quat, perpendicular.expand(len(quat), -1))
    correction[opposite] = torch.cat([torch.zeros_like(dot), axis], dim=-1)[opposite]
  correction = correction / torch.linalg.norm(correction, dim=-1, keepdim=True)
  result = quat_mul(correction, quat)
  return result / torch.linalg.norm(result, dim=-1, keepdim=True)
