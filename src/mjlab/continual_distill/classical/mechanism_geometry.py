"""Known mechanism geometry used by observation-only teachers.

These are static asset parameters, never simulator state. Deriving them from the
same MJCF as the task avoids silently interpreting a visualization offset as FK.
"""
from functools import lru_cache
from importlib import import_module

import mujoco
import numpy as np


@lru_cache(maxsize=None)
def joint_goal_displacement(asset: str, joint: str, target: float) -> np.ndarray:
  """Target-site minus rest-site, expressed in the fixed mount's frame."""
  module = import_module(f"mjlab.asset_zoo.objects.articulated.{asset}.{asset}_constants")
  model = getattr(module, f"get_{asset}_spec")().compile()
  data = mujoco.MjData(model)
  joint_id = model.joint(joint).id
  site = model.site("object_site").id
  root = model.site_bodyid[site]
  while model.body_parentid[root] != 0:
    root = model.body_parentid[root]
  data.qpos[model.jnt_qposadr[joint_id]] = 0
  mujoco.mj_forward(model, data)
  rest = data.site_xpos[site].copy()
  mount_rotation = data.xmat[root].reshape(3, 3).copy()
  data.qpos[model.jnt_qposadr[joint_id]] = target
  mujoco.mj_forward(model, data)
  displacement = mount_rotation.T @ (data.site_xpos[site] - rest)
  displacement.setflags(write=False)
  return displacement
