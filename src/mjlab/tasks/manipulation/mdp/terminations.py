from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return torch.any(sensor.data.found, dim=-1)


def object_out_of_bounds(
  env: ManagerBasedRlEnv,
  object_name: str,
  x_bounds: tuple[float, float] = (0.0, 1.0),
  y_bounds: tuple[float, float] = (-0.5, 0.5),
) -> torch.Tensor:
  """Terminate if object position exceeds bounds in X or Y direction.

  Args:
    env: The environment.
    object_name: Name of the object entity to check.
    x_bounds: (min, max) bounds for X position relative to env origin (meters).
    y_bounds: (min, max) bounds for Y position relative to env origin (meters).

  Returns:
    Boolean tensor indicating which environments should terminate.
  """
  obj: Entity = env.scene[object_name]
  object_pos_w = obj.data.root_link_pos_w
  env_origins = env.scene.env_origins

  # Get object position relative to environment origin
  relative_pos = object_pos_w - env_origins

  # Unpack bounds
  x_min, x_max = x_bounds
  y_min, y_max = y_bounds

  # Check if X or Y position exceeds bounds
  x_out = (relative_pos[:, 0] > x_max) | (relative_pos[:, 0] < x_min)
  y_out = (relative_pos[:, 1] > y_max) | (relative_pos[:, 1] < y_min)

  return x_out | y_out
