"""Scripted PushCuboid teacher policy.

Strategy (mimics what the RL teacher was observed doing): descend until the
gripper is just above the cuboid, then "shepherd" it — keep the closed
fingers riding at cuboid height slightly behind the object relative to the
goal and creep toward the goal so the fingers nudge the object along.

No orientation objective: the ee_ground_collision sensor matches the whole
link7 SUBTREE (hand + fingers), and any orientation constraint prevents the
IK from reaching the required ~4-6cm site height. Position-only IK lets the
arm tilt however it needs (the posture term keeps it sane).

Observation layout matches push_button (60 dims): object=cuboid.
  40:43 gripper_to_object, 43:46 object_to_goal.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

HOVER_HEIGHT = 0.15  # above the object while aligning
RIDE_HEIGHT = 0.035  # site height above object center while pushing
BEHIND = 0.045  # stay this far behind the object (opposite the goal)
ALIGN_TOL = 0.04
GOAL_TOL = 0.015
DESCENT_RATE = 0.05
GRIPPER_CLOSED = -1.0


class PushCuboidClassicalPolicy(ClassicalPolicyBase):
  step_clip_mode = "per_joint"  # shepherding was tuned under legacy clipping
  """Shepherd the cuboid to the goal with closed fingers at cuboid height."""

  DEFAULT_QPOS = HOME_QPOS  # push_cuboid env uses get_franka_robot_cfg (home)
  max_dq = 0.08  # correct kinematics allow brisker motion; episode is 150 steps

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto = obs_i[40:43]  # object - gripper
    o2g = obs_i[43:46]  # goal - object

    o2g_xy = o2g[:2]
    dist_goal = np.linalg.norm(o2g_xy)
    d = o2g_xy / (dist_goal + 1e-8)
    behind = np.array([-d[0] * BEHIND, -d[1] * BEHIND, 0.0])

    if dist_goal < GOAL_TOL or self._phase[i] == 3:
      # Done: back off behind the object and hold altitude.
      self._phase[i] = 3
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
    elif self._phase[i] == 0:
      # Align above a point slightly behind the object.
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.08:
        self._phase[i] = 1
    elif self._phase[i] == 1:
      # Descend to ride height just behind the object.
      pos_err = gto + behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 2
    else:
      # Aim well through the object toward the goal: push speed scales with
      # how far the commanded point sits beyond the contact face. Back off
      # near the goal to avoid overshooting.
      adv = min(0.07, dist_goal + 0.02)
      pos_err = gto + np.array([d[0] * adv, d[1] * adv, RIDE_HEIGHT])

    pos_err[2] = max(pos_err[2], -DESCENT_RATE)  # rate-limit descents
    return pos_err, None, GRIPPER_CLOSED
