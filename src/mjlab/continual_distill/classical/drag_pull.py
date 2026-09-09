"""Scripted DragPull teacher policy.

Motion profile: PLANAR DRAG-PULL — the engagement-inverted twin of push_cuboid.
The cuboid spawns in the FAR half of the corner-safe grasp box and the goal sits
in the NEAR half (see ``franka_drag_pull_env_cfg``), so every episode requires
the gripper to reach BEYOND the object (engage its far face, away from the
robot base) and then retract, dragging the object back toward the base. Same
object (``get_cuboid_cfg`` — 0.08 x 0.08 x 0.03 box, half-extent 0.04, resting
at z=0.015) and the same ``PushingCommand`` success predicate as push_cuboid,
so the physics is identical; only the spawn/goal geometry is inverted.

Because the underlying push mechanics do not care about compass direction —
only about "which side of the object is the gripper standing on relative to
the goal" — this teacher reuses push_cuboid's proven CONTACT-POINT SERVO
strategy verbatim, expressed in terms of ``d`` = unit(goal - object). ``d``
happens to point back toward the robot base here instead of away from it, and
"behind" (``-d``) is therefore the FAR side of the object — exactly the face
the task wants engaged. No sign flips or direction-specific logic are needed;
the vector algebra is direction-agnostic by construction.

Strategy: descend until the gripper (fingers held closed, acting as a single
wide pusher) rides at cuboid height on the far side of the object, then creep
the contact point toward the goal, staying square-on so the box slides rather
than rotates. See push_cuboid.py for the full derivation of the standoff /
advance / re-seat geometry — the constants below START from its final (v3,
kept) configuration, with ``GOAL_TOL`` retuned for this task's looser 0.03 m
success threshold (push_cuboid's is 0.02 m).

MEASURED MECHANISM BUG (the actual instrument-before-tune finding here, not a
constant retune): push_cuboid's ``RIDE_HEIGHT = 0.028`` places the fingertip
pads at object_centre + 0.028 - ~0.012 (pad-below-site offset) = object_centre
+ 0.016, i.e. only 1 mm above the cuboid's top face (object_centre + 0.015,
half-height 0.015). Copied verbatim, this teacher stalled hard on a subset of
envs — instrumented via a per-step trace of ``gto``/``o2g``/``along``, several
envs showed the pusher correctly "behind" the object (``along`` +0.03..+0.05,
not an overtake) while the object simply did not move for 70+ steps. The
1 mm clearance is consistent with the pad intermittently grazing over the top
edge of the box instead of catching its face squarely. Lowering
``RIDE_HEIGHT`` to 0.020 (pads at object_centre + 0.008, well inside the
0.015 half-height) alone took SR from 0.156 (n=32) to 0.469 (n=32, same seed)
— a MECHANISM fix, not a tuning one. ``STEP_MAX``/``GAIN`` were then raised
modestly (0.030->0.040, 0.60->0.80) on top of that, which measured SR (n=128,
HEAD 1127d12) of **0.492** — see LOGS.md for the full run history including
what did NOT help (MIN_BEHIND/RESEAT_STEPS retuning made it worse).

RESIDUAL FAILURE MECHANISM (below 0.90). Even after the ride-height fix, a
minority of envs plateau at a roughly-constant distance for most of the
episode with a healthy ``along`` (i.e. NOT an overtake) — the same
"position-servo generates a steady-state offset but not enough sustained
force to keep sliding against friction" mechanism push_cuboid's own history
describes, and which push_cuboid's own extensive retuning (7 measured
strategies) never fully solved either. A second, smaller group of envs get
very close (within ~1cm of the 0.03 m window) without quite latching —
endgame precision, also as in push_cuboid. Retract-direction pulling does not
introduce a new failure mode beyond what push mechanism already has; it
inherits push_cuboid's two known unresolved mechanisms rather than adding one.

Observation layout (60 dims, identical structure/order to push_cuboid,
object=cuboid, command=drag_pull):
  [ 0: 9] robot_joint_pos          [ 9:18] robot_joint_vel
  [18:21] object_pos               [21:25] object_quat
  [25:28] gripper_pos              [28:34] gripper_orientation
  [34:40] object_orientation
  [40:43] gripper_to_object  <- USE  [43:46] object_to_goal      <- USE
  [46:52] goal_orientation_diff     [52:60] control_qpos_diff

Orientation: EE approach axis held DOWN (axis-only, yaw free) — as in
push_cuboid, a full orientation target is not needed because the pusher acts
as a fully-closed, direction-agnostic contact point.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

HOVER_HEIGHT = 0.15
# Site height above the cuboid CENTRE. push_cuboid's value (0.028) puts the
# pads only 1mm above the box's top face (see module docstring) — MEASURED to
# cause stalls (pad grazes the top edge instead of the face). 0.020 puts the
# pads mid-face (object_centre + ~0.008, well inside the 0.015 half-height);
# too low and the hand capsule touches the floor and ee_ground_collision fires.
RIDE_HEIGHT = 0.020
BEHIND = 0.055  # stand-off on the far side of the object, opposite the goal
ALIGN_TOL = 0.04
# Success is |object - goal| < 0.03 (success_threshold in make_drag_pull_env_cfg).
# push_cuboid used 0.015 against a 0.02 predicate (a 0.005 margin so the teacher
# does not keep shoving once inside the window); keep the same absolute margin
# here: 0.03 - 0.005 = 0.025.
GOAL_TOL = 0.025
DESCENT_RATE = 0.05
GRIPPER_CLOSED = -1.0
GRIPPER_PUSH_SPAN = GRIPPER_CLOSED

# -- contact-point servo (see push_cuboid.py for the full derivation) --------
# cuboid.xml: box 0.08 x 0.08 x 0.03, half-width 0.04 along the pushed axis.
# Mean horizontal half-extent of the new cuboid (YCB 009_gelatin_box, collision box
# 0.073 x 0.089 x 0.030 => 0.0365 / 0.0446). See push_cuboid.py for why this stays a
# single scalar rather than the direction-dependent support width. RIDE_HEIGHT is
# unchanged: the box's half-HEIGHT is still 0.0150.
HALF_WIDTH = 0.0405
PAD_RADIUS = 0.010  # fingertip pad offset from the site with fingers closed
STANDOFF = 0.004  # clearance between the pad and the box face at zero advance
GAIN = 0.80  # push_cuboid used 0.60; MEASURED slightly better here at 0.80
STEP_MIN = 0.012
STEP_MAX = 0.040  # push_cuboid used 0.030; MEASURED slightly better at 0.040
FINE_ZONE = 0.06
FINE_GAIN = 0.35
CROSS_GAIN = 0.8
# Minimum "behind-ness" (component of object-minus-gripper along the push
# direction) that still counts as a usable contact; below this the pusher has
# drawn level with the box centre and is about to overtake it.
MIN_BEHIND = 0.012
RESEAT_STEPS = 12
FLOOR_MIN_Z = 0.030


class DragPullClassicalPolicy(ClassicalPolicyBase):
  step_clip_mode = "per_joint"  # inherited from push_cuboid's tuning
  """Engage the cuboid's far face with closed fingers and retract to the goal."""

  DEFAULT_QPOS = HOME_QPOS  # drag_pull env uses get_franka_robot_cfg (home)
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

    self.cmd_lead_max = 0.0  # see push_cuboid.py: sustained lead measured worse

    o2g_xy = o2g[:2]
    dist_goal = np.linalg.norm(o2g_xy)
    d = o2g_xy / (dist_goal + 1e-8)  # points from object toward goal (-> base)
    behind = np.array([-d[0] * BEHIND, -d[1] * BEHIND, 0.0])  # -> far side

    if dist_goal < GOAL_TOL or self._phase[i] == 3:
      self._phase[i] = 3
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
    elif self._phase[i] == 0:
      # Align above a point on the FAR side of the object (beyond it, away
      # from the base) before descending — this is the "engage far face" step.
      pos_err = gto + behind + np.array([0.0, 0.0, HOVER_HEIGHT])
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.08:
        self._phase[i] = 1
    elif self._phase[i] == 1:
      pos_err = gto + behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 2
    else:
      # Retract: hold the pusher on the far side of the cuboid and creep the
      # contact point toward the goal (toward the base). Never command a point
      # INSIDE the box — servo the contact face itself, exactly as
      # push_cuboid's v3 (contact-point servo, kept) does.
      along = float(gto[:2] @ d)  # + while correctly on the far side
      n = np.array([-d[1], d[0]])
      cross = float(gto[:2] @ n)

      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        pos_err = gto + 1.6 * behind + np.array([0.0, 0.0, RIDE_HEIGHT])
      else:
        seat_back = HALF_WIDTH + PAD_RADIUS + STANDOFF
        step = float(np.clip(GAIN * dist_goal, STEP_MIN, STEP_MAX))
        if dist_goal < FINE_ZONE:
          step = min(step, dist_goal * FINE_GAIN)
        pos_err = gto + np.array([
          -d[0] * seat_back + d[0] * step - n[0] * cross * CROSS_GAIN,
          -d[1] * seat_back + d[1] * step - n[1] * cross * CROSS_GAIN,
          RIDE_HEIGHT,
        ])

    pos_err[2] = max(pos_err[2], -DESCENT_RATE)

    # Absolute floor guard (see push_cuboid.py: geom_rbound overstates downward
    # reach; the true clearance to the fingertip pads is ~1.4cm, to the hand
    # capsule ~3.1cm). Computed from FK on the arm's own joints (robot-base
    # frame), so it is free of the per-env scene-origin offset.
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err, _DOWN_AXIS, GRIPPER_PUSH_SPAN
