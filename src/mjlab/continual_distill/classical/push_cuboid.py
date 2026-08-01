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
# STOP BAND. Success is |object - goal| < 0.02 (success_threshold in
# push_cuboid_env_cfg). This was 0.015 — TIGHTER than the success window — so the
# teacher kept pushing after the box was already successful, and the measured
# distance trace shows exactly that: env0 sits at 0.008/0.013/0.018/0.016 between
# t=80 and t=140, i.e. inside the window, and is repeatedly knocked back out,
# ending at 0.023. The box was not failing to arrive; it was being shoved through.
#
# Stop band. Success is `norm(target_pos - object_pos) < 0.02` in 3D. MEASURED on
# the running env: the target z is 0.0150 and the cuboid rests at 0.0149, so the
# permanent z-error is 0.0001 and effectively the full 2 cm is available in-plane.
# (The 0.03 `goal_z_height` default in PushingCommandCfg is overridden for Franka —
# do not reason from the dataclass default, as an earlier pass here did.)
#
# MEASURED: 0.009 (with the brake below) gave 0.031/0.000, well below the 0.015
# baseline — parking early wastes the remaining budget instead of continuing to
# close. Kept at the original 0.015.
GOAL_TOL = 0.015
DESCENT_RATE = 0.05
GRIPPER_CLOSED = -1.0
# WIDE TWO-POINT CONTACT. Fully closed (-1.0) puts the two finger pads against
# each other, so the pusher contacts the 8 cm rear face at essentially ONE point.
# Any lateral offset e from the face centre then applies a torque F*e about the
# box centre, and the box rotates or squirts sideways instead of sliding along
# the goal line — which is what the measured backward excursions were
# (-0.051, -0.148) once sustained force was added.
#
# Holding the fingers PARTLY OPEN spans the pads across the face instead. The
# pair is self-centring and applies no net torque, and unlike straddling the box
# it needs no fitting tolerance: the pads sit at ~+-0.02 on an 8 cm face, well
# inside the +-0.0476 finger travel, so the +-1 cm observation noise cannot cause
# a miss. Still a push — the box stays on the ground throughout.
#
# MEASURED AND REVERTED: -0.15 gave 0.000/0.094, no better than fully closed.
# The torque argument above is sound in isolation but is evidently not the
# binding constraint either. Kept fully closed.
GRIPPER_PUSH_SPAN = GRIPPER_CLOSED
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

# -- contact-point servo (see the strategy note in _target_error) -------------
# Geometry, from cuboid.xml: box is 0.08 x 0.08 x 0.03, so half-width 0.04 along
# whichever axis is being pushed. The fingertip pad adds ~0.01 of radius.
HALF_WIDTH = 0.04
PAD_RADIUS = 0.010
# Clearance between the pad surface and the box face at zero advance. Small and
# positive: the reference must sit OUTSIDE the box (which is the whole point of
# this rewrite) but close enough that a modest `step` makes contact immediately.
STANDOFF = 0.004
# Advance = GAIN * remaining distance, clipped. GAIN < 1 makes it a proportional
# controller on the push; STEP_MAX bounds the penetration so the commanded point
# can never cross to the far face even at maximum.
GAIN = 0.60
STEP_MIN = 0.012
# MEASURED transport budget (16 envs, instrumented): the box starts 0.127 m from
# the goal and the previous cap moved it only 0.057 m in 150 steps -- 0.38 mm/step
# against the 0.71 mm/step needed to reach the 2 cm window, a 1.89x shortfall.
# The failure is transport RATE, not the overtake (measured: 1/16 overtook, 15/16
# were still shepherding at timeout).
#
# The earlier 0.011 came from requiring the COMMANDED point to stay outside the
# box. That is stricter than necessary: contact stops the pusher well short of its
# setpoint, so what must stay outside the box is where the pusher ACTUALLY is, not
# where it is aimed. A larger advance is simply a larger contact force, which is
# what moves the box. The overtake guard (MIN_BEHIND + re-seat) remains as the
# safety net for the case where the pusher does slip past.
STEP_MAX = 0.030
# Endgame: success is a 2cm window, so inside FINE_ZONE the advance is throttled
# to a fraction of the remaining distance to avoid overshooting through it.
FINE_ZONE = 0.06
FINE_GAIN = 0.35
# Cross-track correction. Keeps the pusher on the object->goal line so contact
# stays square and the box tracks straight instead of rotating off-axis.
CROSS_GAIN = 0.8
# Command lead (rad) during the shepherd phase. The joint command integrates from
# the previous COMMAND rather than the lagging actual position, up to this far
# ahead — which is what converts a blocked position servo into a sustained push.
# open_door uses 0.35 for its drag; the cuboid needs less force (0.49 N of sliding
# friction vs a hinge), and too much lead near the ground risks the floor guard.
PUSH_CMD_LEAD = 0.20
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
  # 0.08 was roughly half what the two teachers that actually work on this budget
  # use (push_button 0.20 at SR 1.000, open_drawer 0.15 at 0.81), and the measured
  # bottleneck here is transport rate: 0.38 mm/step achieved against 0.71 needed.
  # The floor guard below is what protects against a brisk rate near the ground,
  # not the rate cap itself.
  max_dq = 0.15

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._reseat = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._reseat[env_ids] = 0

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto = obs_i[40:43]  # object - gripper
    o2g = obs_i[43:46]  # goal - object

    # SUSTAINED PUSH FORCE. Measured per-25-step box displacement: +0.050 in the
    # first 25 steps, then ~0.000 for the remaining 125, while `along` stayed a
    # healthy +0.03..+0.04 — i.e. the pusher is correctly seated and in contact,
    # and the box simply stops moving. That is not a control-law error, it is
    # plain position control: the arm servos TO its commanded point and stops, so
    # once the box resists there is no force left to push through the 0.49 N of
    # sliding friction. It explains why tripling the advance, the gain and max_dq
    # all failed to move the 0.06 m plateau — those set WHERE and HOW FAST the arm
    # goes, not how hard it presses.
    #
    # cmd_lead_max is the mechanism the base class provides for exactly this
    # ("sustained lead = sustained servo force"), and open_door is the only other
    # teacher that uses it.
    #
    # MEASURED AND REVERTED: enabling it at 0.20 during the shepherd phase made
    # things WORSE — 0.125/0.062 against a 0.094-0.156 baseline — and the per-25-step
    # trace picked up large NEGATIVE excursions (-0.051, -0.148, -0.056) that the
    # zero-lead runs did not have. Sustained force through a single fingertip on an
    # 8 cm face applies a torque about the box centre, so the extra push rotates the
    # cuboid or squirts it sideways instead of sliding it along the goal line. The
    # missing ingredient is a SQUARE contact, not more force. Left disabled.
    self.cmd_lead_max = 0.0

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
      # STRATEGY CHANGE (contact-point servo). The law above commanded
      #   target = object + (-d*BEHIND) + d*adv
      # whose net penetration (adv - BEHIND) is +0.030 for essentially the whole
      # push, i.e. the commanded point sits INSIDE the 8cm box the entire time.
      # The IK has no contact model, so it is permanently solving to drive the
      # fingertip through the workpiece; contact blocks the site, the DLS solve
      # keeps integrating, and the accumulated bias eventually squirts the site
      # around the box. That IS the overtake, and MIN_BEHIND/RESEAT_STEPS are
      # patches on a reference that is unreachable by construction.
      #
      # Instead servo the CONTACT POINT itself, never commanding anything inside
      # the box:
      #   contact face  = object centre - d*(half_width + pad_radius)
      #   commanded pt  = contact face - d*STANDOFF + d*step
      # `step` is a bounded closed-loop advance proportional to the remaining
      # distance, so the reference is always OUTSIDE the box and the penetration
      # that generates push force comes from `step` alone, which is explicitly
      # capped. Cross-track error is corrected separately so the pusher stays on
      # the goal line rather than drifting off the face.
      along = float(gto[:2] @ d)

      # Lateral (cross-track) offset of the pusher from the object->goal line.
      # Positive `cross` means the pusher sits off to one side; zeroing it keeps
      # the contact square, which is what stops the box from being pushed at an
      # angle and rotating away.
      n = np.array([-d[1], d[0]])
      cross = float(gto[:2] @ n)

      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      else:
        # Where the pusher should sit: just off the rear face, on the goal line.
        seat_back = HALF_WIDTH + PAD_RADIUS + STANDOFF
        # Closed-loop advance: push hard while far, ease off near the goal so the
        # 2cm success window is not overshot. Bounded so penetration can never
        # reach the far side of the box.
        step = float(np.clip(GAIN * dist_goal, STEP_MIN, STEP_MAX))
        if dist_goal < FINE_ZONE:
          step = min(step, dist_goal * FINE_GAIN)
        # NOTE: a hard terminal brake was tried here (throttle `step` to 4 mm, then
        # 8 mm, inside the last 3-5 cm) on the theory that the box — which moves up
        # to 23 mm/step — can jump across the 20 mm success window between two
        # control steps. Measured WORSE both times (0.000 at 4 mm, 0.031/0.000 at
        # 8 mm vs a 0.094-0.156 baseline): braking spends the remaining step budget
        # without closing the distance. Deliberately absent.

        # Target expressed as an error from the CURRENT gripper position:
        #   (object - gripper) + (-d * seat_back) + (d * step) - (n * cross)
        # The -n*cross term drives the pusher back onto the goal line.
        pos_err = gto + np.array([
          -d[0] * seat_back + d[0] * step - n[0] * cross * CROSS_GAIN,
          -d[1] * seat_back + d[1] * step - n[1] * cross * CROSS_GAIN,
          RIDE_HEIGHT,
        ])

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
    return pos_err, _DOWN_AXIS, GRIPPER_PUSH_SPAN
