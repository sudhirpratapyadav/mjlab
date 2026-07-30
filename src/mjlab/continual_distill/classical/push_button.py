"""Scripted PushButton teacher policy.

Strategy (per env, observation-only):
  Phase 0 (HOVER):  move the gripper to a point above the button handle with
                    the hand pointing straight down, fingers closed.
  Phase 1 (PRESS):  descend onto the handle and keep pushing below its current
                    position until the slide joint bottoms out (success at
                    >=2cm displacement; we just keep pressing).

Observation layout (60 dims, PushButton task):
  0:9    robot_joint_pos (relative to NEUTRAL default)
  9:18   robot_joint_vel
  18:21  object_pos      (button handle site, world frame)
  21:25  object_quat
  25:28  gripper_pos     (gripper site, world frame)
  28:34  gripper_orientation (rot6d)
  34:40  object_orientation  (rot6d)
  40:43  gripper_to_object vector
  43:46  object_to_goal vector
  46:52  goal_orientation_diff (rot6d)
  52:60  control_qpos_diff
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Hand pointing straight down: 180 deg rotation about x (z-axis -> -z).
_ROT_DOWN = np.array(
  [
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
  ]
)

# Approach-axis target: EE z-axis tilted slightly forward from straight down,
# yaw left free. The button now sits at radial 0.44-0.48 (was 0.65), well
# inside the comfortable envelope, so much less forward tilt is needed.
_AXIS_DOWN = np.array([0.20, 0.0, -0.980])
_AXIS_DOWN = _AXIS_DOWN / np.linalg.norm(_AXIS_DOWN)

HOVER_HEIGHT = 0.07  # m above the handle in phase 0
HOVER_XY_TOL = 0.03  # xy alignment tolerance to start pressing
HOVER_Z_TOL = 0.04
PRESS_DEPTH = 0.03  # target this far below the handle's current position
GRIPPER_CLOSED = -1.0  # action -> finger target 0.0 (closed)


class PushButtonClassicalPolicy(ClassicalPolicyBase):
  """Press the button by hovering above the handle and pushing down."""

  # The button mount moved DOWN (cap now at z~0.16, was ~0.6): from the
  # NEUTRAL start the gripper site begins ~0.85m ABOVE the cap instead of
  # roughly level with it. The old default rate (max_dq 0.05, max_pos_err
  # 0.08) turned that descent into a 130-step crawl and the press never
  # happened inside the 150-step budget. Descend briskly, then slow down for
  # the press so the cap is not batted sideways.
  max_dq = 0.20
  max_pos_err = 0.25

  def _target_error(self, i: int, obs_i: np.ndarray):
    # Brisk while travelling, gentle once pressing.
    if self._phase[i] == 0:
      self.max_dq = 0.20
      self.max_pos_err = 0.25
    else:
      self.max_dq = 0.06
      self.max_pos_err = 0.06
    # gripper_to_object = handle_pos - gripper_pos (env-origin offsets cancel).
    gto = obs_i[40:43]

    if self._phase[i] == 0:
      pos_err = gto + np.array([0.0, 0.0, HOVER_HEIGHT])
      xy_err = np.linalg.norm(gto[:2])
      z_err = abs(pos_err[2])
      if xy_err < HOVER_XY_TOL and z_err < HOVER_Z_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    else:
      # Aim below the handle's *current* position so we keep pressing as it
      # slides down. Stay xy-locked onto the handle.
      pos_err = gto + np.array([0.0, 0.0, -PRESS_DEPTH])

    return pos_err, _AXIS_DOWN, GRIPPER_CLOSED
