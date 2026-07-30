"""Scripted PushCuboid teacher policy.

Strategy (mimics what the RL teacher was observed doing): descend until the
gripper is just above the cuboid, then "shepherd" it — keep the closed
fingers riding at cuboid height slightly behind the object relative to the
goal and creep toward the goal so the fingers nudge the object along.

Orientation: the EE approach axis is held roughly DOWN (axis-only target, yaw
free). This was previously position-only IK, on the reasoning that any
orientation constraint stopped the arm reaching the low ride height when the
cuboid sat at x 0.60-0.80 — the edge of the reach envelope. The cuboid now
spawns at x 0.30-0.41, well inside the comfortable cone, so the constraint is
affordable, and it is what keeps the hand off the floor: ee_ground_collision
matches the whole link7 SUBTREE, and with free orientation the wrist tilts and
drives the hand capsule into the ground plane (there is no table any more).
In a top-down pose the lowest collision geom (the finger pads) sits only
~1.2cm below the gripper site, so a modest ride height clears the floor.

Observation layout matches push_button (60 dims): object=cuboid.
  40:43 gripper_to_object, 43:46 object_to_goal.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# EE approach axis: straight down, yaw free. Keeps the hand capsule and finger
# pads horizontal so the only thing near the floor is the fingertip pads.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

HOVER_HEIGHT = 0.15  # above the object while aligning
# Site height above the cuboid CENTRE (centre is at z=0.015 on the ground; the
# box is only 3cm tall, so its TOP is at z=0.03).
#
# This is a tight window and both sides of it bite. The finger pads reach only
# ~1.2cm below the gripper site in a top-down pose, so:
#   - site too HIGH: the pads clear the top of the 3cm box entirely and the
#     "push" becomes a hover — the pusher grazes the cuboid without ever making
#     a face-on contact and the object drifts a few cm instead of tracking.
#     (site at centre+0.05 => pads at z~0.053, well above the 0.03 top.)
#   - site too LOW: the pads, and then the hand capsule, touch the ground plane
#     and ee_ground_collision terminates the episode.
# Put the pads at the box's mid-height (z~0.015) => site at centre + ~0.027,
# which contacts the vertical face squarely and still leaves floor clearance.
RIDE_HEIGHT = 0.028
BEHIND = 0.055  # stay this far behind the object (opposite the goal)
ALIGN_TOL = 0.04
GOAL_TOL = 0.015
DESCENT_RATE = 0.05
GRIPPER_CLOSED = -1.0
# How far ahead of the current contact point the shepherd waypoint is placed
# each step. This is the push "lead": too small and the pusher never loads the
# cuboid, too large and the IK runs through the object to the far side.
#
# It must EXCEED the BEHIND standoff, or the two cancel and the commanded point
# lands back on the object centre: the pusher then holds station touching the
# cuboid without loading it and |o2g| stays flat. ADVANCE - BEHIND is the
# effective penetration into the contact face, which is what generates the push.
ADVANCE = 0.085


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
      # Shepherd: hold the pusher on the BEHIND side of the cuboid and creep the
      # whole contact point toward the goal.
      #
      # This used to aim `adv` metres THROUGH the object toward the goal, on the
      # theory that commanding past the contact face sets the push speed. With
      # the object no longer pinned against a table that reasoning fails: the IK
      # has no contact model, so it simply drives the site through the cuboid and
      # parks ~8cm on the FAR side, between object and goal — measured `along`
      # settles at -0.08 and the cuboid gets shoved AWAY from the goal, which is
      # why |o2g| stayed flat for the whole episode.
      #
      # Instead: target = object + behind-offset, then step that point toward the
      # goal by a bounded advance. The pusher stays on the correct side and the
      # advance is what generates contact force.
      # Taper the lead as the goal nears (less overshoot), but never below the
      # BEHIND standoff or the net penetration goes to zero and the push stalls
      # short of the 2cm success threshold.
      adv = max(BEHIND + 0.015, min(ADVANCE, dist_goal + BEHIND))
      pos_err = (
        gto
        + behind
        + np.array([d[0] * adv, d[1] * adv, RIDE_HEIGHT])
      )

    pos_err[2] = max(pos_err[2], -DESCENT_RATE)  # rate-limit descents
    return pos_err, _DOWN_AXIS, GRIPPER_CLOSED
