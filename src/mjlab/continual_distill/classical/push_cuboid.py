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
# Minimum "behind-ness" (component of object-minus-gripper along the push
# direction) that still counts as a usable contact. Below this the pusher has
# drawn level with the 8x8cm box's centre and is about to overtake it, so the
# shepherd phase bails out and re-approaches from behind. Measured successes hold
# +0.03..+0.05 here; measured failures decay through 0 to -0.05 and never recover.
MIN_BEHIND = 0.012
# Steps spent backing off behind the contact face once a re-seat is triggered.
# Acts as hysteresis: without it the shepherd re-triggers every other step.
RESEAT_STEPS = 12
# Absolute lower bound on the commanded gripper-site height, in the robot base
# frame. See the guard at the end of ``_target_error``.
FLOOR_MIN_Z = 0.030


class PushCuboidClassicalPolicy(ClassicalPolicyBase):
  step_clip_mode = "per_joint"  # shepherding was tuned under legacy clipping
  """Shepherd the cuboid to the goal with closed fingers at cuboid height."""

  DEFAULT_QPOS = HOME_QPOS  # push_cuboid env uses get_franka_robot_cfg (home)
  max_dq = 0.08  # correct kinematics allow brisker motion; episode is 150 steps

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._reseat = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._reseat[env_ids] = 0

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
      #
      # RE-APPROACH WHEN THE PUSHER OVERTAKES THE OBJECT. Measured over 32
      # episodes, this is the dominant failure and it is a slow structural drift,
      # not a tuning miss. Define `along` = component of (object - gripper) along
      # the push direction; it is POSITIVE while the pusher is correctly behind
      # the object. In the runs that succeed it stays at +0.03..+0.05 throughout.
      # In the runs that fail it starts at +0.04 and decays through zero to
      # -0.03..-0.05, after which the pusher sits BETWEEN the object and the goal
      # and every further advance shoves the cuboid backwards -- |o2g| plateaus
      # and never reaches the 2cm threshold.
      #
      # The drift is built into the tapering lead: as `dist_goal` shrinks, `adv`
      # collapses toward the BEHIND standoff, net penetration goes to zero, and
      # the DLS steady-state bias is then free to walk the site through the
      # 8x8cm box to the far side. Creeping onward from there cannot recover.
      #
      # So treat "behind the object" as a hard geometric precondition: if the
      # pusher has lost it, stop pushing and go back around to re-acquire the
      # contact face. A re-approach costs ~15 steps and restores a working
      # contact; continuing to creep wastes the rest of the episode.
      # The recovery is a CHEAP LATERAL RE-SEAT, not a return to the hover phase.
      # Sending this back to phase 0 was tried and is much worse: phase 0 climbs
      # to HOVER_HEIGHT and has to descend again, ~25 steps out of 150, so the
      # policy thrashed p2->p0->p1->p2 every ten steps and the object barely moved
      # at all (measured |o2g| 0.366 -> 0.337 over 140 steps). Instead, stay at
      # ride height and slide back around behind the contact face, with
      # hysteresis so a single noisy sample cannot trigger a re-seat.
      along = float(gto[:2] @ d)
      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        # Track a point well behind the object at ride height; no advance term,
        # so the pusher backs off and re-acquires the face without loading it.
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      else:
        adv = max(BEHIND + 0.015, min(ADVANCE, dist_goal + BEHIND))
        pos_err = (
          gto
          + behind
          + np.array([d[0] * adv, d[1] * adv, RIDE_HEIGHT])
        )

    pos_err[2] = max(pos_err[2], -DESCENT_RATE)  # rate-limit descents

    # ABSOLUTE FLOOR GUARD. Every height above is expressed relative to the
    # OBSERVED object centre, and that observation carries +-1cm of noise per
    # axis; combined with the DLS solve's vertical sag, nothing stopped the site
    # being commanded below the floor. ``ee_ground_collision`` then terminates
    # the episode and the env AUTO-RESETS IN PLACE without notifying the harness,
    # so the state machine keeps shepherding an object that is no longer there.
    #
    # The clearance figure the teachers in this package used to assume (1.24cm
    # below the site) counted the whole link7 subtree, most of which is
    # contype=conaffinity=0 and cannot collide at all. Over the geoms that
    # genuinely collide it is 1.4cm to the fingertip pads, and the hand capsule's
    # bounding volume reaches ~3.1cm below the site. Hence the same 0.030 floor
    # the shared grasp spine now uses.
    #
    # Site height comes from FK on the arm's own joint angles, which is expressed
    # in the ROBOT BASE frame and so is free of the per-env scene-origin offset --
    # legal in a way that reading the absolute gripper_pos observation is not.
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err, _DOWN_AXIS, GRIPPER_CLOSED
