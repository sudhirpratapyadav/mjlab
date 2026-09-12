"""Scripted Drag-Pull teacher (CL-V3, W1-P).

Same carton and the same ``PushingCommand`` predicate as Push-Cuboid (3-D goal error
< 0.03, latched); the object spawns in the FAR half of the grasp box and the goal in the
NEAR half, so the goal direction points back toward the base and the contact face is
the object's far face. The planar paddle push in ``push_cuboid.PlanarPushPolicy`` is
direction-agnostic (contact face = the face whose outward normal is most opposite to
the goal direction, read from ``object_orientation``), so this teacher is that policy
with the task's tolerance. History and measurements: ``docs/cl_v3/logs/W1-P.md``;
the phase-1 contact-point-servo teacher (0.477 / 0.391) is described in
``docs/cl25/phase_1/LOGS.md`` (W1-b, 2026-09-08).

The far face sits up to ~0.59 m from the base at ride height (the spawn band is inset
by the carton's x half-extent for exactly this reason, `_ENGAGE_INSET` in
``franka_drag_pull_env_cfg``); the full-frame top-down IK is kept because the paddle
must face the object, and the approach hovers low so the wrist never has to fold at
the reach limit.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.push_cuboid import PlanarPushPolicy


class DragPullClassicalPolicy(PlanarPushPolicy):
  """Engage the carton's far face with the closed paddle and pull it to the goal."""

  OBJ_HALF = np.array([0.0365, 0.0446, 0.0150])
  SUCCESS_TOL = 0.03
  CAGE = False
  GRIPPER_ACTION = -1.0
  # Analytical support replaces the learned height bias for the inward pull.
  gravity_compensation = True
  Z_KI = 0.0
  Z_BIAS_MAX = 0.0
  Z_FF_PER_LEAD = 0.0
