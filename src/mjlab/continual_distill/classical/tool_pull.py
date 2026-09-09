"""Scripted ToolPull teacher policy.

WHAT THIS FILE IS, AND WHY IT IS NOT TOOL USE
=============================================
The task is named for tool use and the environment now genuinely supports it: the
policy observation is 69-D and carries ``gripper_to_tool`` (obs[52:55]) and
``tool_orientation`` (obs[55:61]), so the stick is fully observable. An earlier version
of this file argued tool use was impossible because the stick was invisible; that
argument is obsolete and was removed.

A full tool-use teacher was therefore implemented and measured, and it does not work.
The reason is mechanical, not a matter of tuning, and it is documented here so the next
author does not spend the effort again.

WHY THE STICK CANNOT BE PICKED UP (measured, 8 configurations)
--------------------------------------------------------------
The approach machinery works. With the floor guard fixed (see below) and integral action
on the descent, the gripper converges onto the stick's ``object_site`` to within
0.023-0.028 m in 6-8 of 8 envs, with the finger-closing axis measured at (0.00, 1.00,
0.00) -- exactly perpendicular to the shaft, which lies along world +x. The fingers then
close and STALL at qpos ~= 0.010 per side, which is precisely half the shaft's 22mm
width: they are genuinely in contact with the shaft, not closing on air.

The stick still never leaves the ground. Across the whole test its ``object_site`` z
stays pinned at 0.011, its resting height, while the gripper site rises to 0.35+. What
happens instead is visible in ``gripper_to_tool`` during the squeeze: its x component
grows monotonically from -0.004 to -0.042. **The squeeze ejects the stick axially.** The
shaft is a long thin dowel (0.26 m long, 22 mm across; a box of the same extents before
CL-V2) with its mass 9cm off the grasp point,
so any residual misalignment converts the pinch's normal force into a force along the
shaft, and it squirts out from between the pads before the grip can develop. Fingertip
slide friction is domain-randomised as low as 0.3, which is far too little to arrest it.

Configurations tried, all with the same outcome (stick_z pinned at 0.011):
  * grasp height 0.018 / 0.020 / 0.022 / 0.024 / 0.026 / 0.028 / 0.030 m
  * grasp point at ``object_site``, at the shaft's centre of mass (+0.09 along x), and
    at the hook bar (+0.21, +0.035)
  * finger-closing axis along world y (perpendicular to the shaft) and axis-only
    orientation targets, with close windows of 40 and 130 control steps
  * a slow guarded descent at 6mm/step with lateral integral action, which brought the
    approach residual down to 0.001-0.007 m and the collision count to 3

Even granting a successful grasp, the plan needs the stick moved ~0.12 m in +x AND
~0.23 m in +y to bring its hook bar (body offset (0.12, 0.035), i.e. (+0.21, +0.035)
from the grasp site) onto the far side of the puck. That is a full carry, not a nudge,
so a marginal grip would not survive it either.

Concluding: with this gripper and this stick geometry, tool use here needs a learned
teacher (or a stick whose grasp region is a graspable feature rather than a smooth
shaft). Reported as such rather than dressed up.

WHAT THIS FILE DOES INSTEAD, AND THE BUG THAT WAS ACTUALLY CAPPING IT
---------------------------------------------------------------------
A DIRECT closed-finger drag: reach past the puck with the fingers closed and shepherd it
inward along the line to the goal -- the same closed-finger push the push_cuboid teacher
uses, run outward-to-inward. ``tool_grasped`` is tracked by the command but does NOT gate
success, so a direct drag scores.

The previous direct-drag teacher measured 0.000 and its failure was blamed on the
strategy. It was not the strategy: it was ``floor_min_z = 0.026``. Measured directly on
the Franka model over the genuinely collidable link7-subtree geoms (``hand_capsule``,
``left_finger_pad``, ``right_finger_pad``; every other subtree geom has
contype=conaffinity=0 and is visual only), the lowest pad sits 1.4cm below the
``gripper`` site. A guard at 0.026 leaves pad material at 0.012 and the drag's transient
dip closes that immediately, so ``ee_ground_collision`` fired constantly and auto-reset
the env under a state machine the harness never tells about it. Raising the guard to
0.030 took the same strategy from 0.000 to a measured 0.125.

That number is honest but weak, and it is unstable between batches (0.250 and 0.000 on
two consecutive 16-env episodes) -- see the measurement note in the benchmark doc. The
puck spawns at radial 0.62-0.70, right at the edge of the arm's top-down envelope, and
shoving a 7.6cm, 160 g disc from there gives very little steering authority; ``object_out_of_
bounds`` (x outside (0,1), y outside (-0.5,0.5)) also fires readily on a puck being
pushed sideways. This teacher is a floor, not a solution.

SUCCESS (ToolPullCommand)
-------------------------
    || puck_site - (env_origin + (0.42, 0, 0.0127)) || < 0.07

OBSERVATION LAYOUT (69-D, measured against the live env, not assumed)
--------------------------------------------------------------------
    [ 0: 9] robot_joint_pos      [ 9:18] robot_joint_vel
    [18:21] object_pos (puck, ABSOLUTE -- carries the per-env origin, never used)
    [21:25] object_quat          [25:28] gripper_pos (ABSOLUTE, never used)
    [28:34] gripper_orientation  [34:40] object_orientation (puck)
    [40:43] gripper_to_object    <- puck - gripper, RELATIVE
    [43:46] object_to_goal       <- goal - puck, RELATIVE
    [46:52] goal_orientation_diff
    [52:55] gripper_to_tool      <- stick object_site - gripper, RELATIVE
    [55:61] tool_orientation     <- stick rot6d
    [61:69] control_qpos_diff

The env-local frame is recovered from the relative terms alone, because the goal sits at
a FIXED env-local offset:

    puck_local    = GOAL_LOCAL - o2g
    gripper_local = GOAL_LOCAL - o2g - gto

Both are offset-free (the scene origin cancels), which is what makes an env-local plan
legal here. Absolute terms are never used.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# Env-local goal, from ToolPullCommandCfg.goal_offset (z = the puck's half-thickness).
GOAL_LOCAL = np.array([0.42, 0.0, 0.0127])

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0

# The puck is a regulation 7.62 cm-diameter, 2.54 cm-thick ice-hockey puck (CL-V2, W1-b;
# it was a 7 cm x 2.4 cm primitive of 30 g, and is now 160 g). Ride the closed fingertips
# at this height: high enough that the pads clear the floor, low enough to catch the
# puck's rim rather than skate over it. RE-DERIVED: pads sit ~1.3 cm below the site, so
# 0.034 puts the pad bottom at 0.021 -- 83% of the way up the puck's 0-0.0254 face (it
# was 87% of the old 0-0.024 face), still clear of the 0.030 site floor guard. Unchanged.
RIDE_Z = 0.034
HOVER_Z = 0.16  # transit altitude while swinging out over the puck

# Bounded descent rate, in metres of commanded displacement per control step. Dropping in
# one commanded jump is what generated the ground contacts: at the converged IK pose the
# arm clears the floor comfortably, but the DLS path to that pose dips transiently, and
# out at radial 0.65 the arm is near-singular so the dip is large. Feeding the descent in
# slowly keeps the solver near its converged branch the whole way down.
DESCEND_RATE = 0.012

# Stand this far BEHIND the puck (on the far side from the goal) so the fingers make
# contact on its outer rim and push inward. Puck radius 0.0381 + a pad half-width 0.017.
BEHIND = 0.055
# Multiplier on the standoff during the shepherd phase, to compensate the DLS
# steady-state bias that otherwise pulls the gripper on top of the puck.
BEHIND_BIAS = 1.8

APPROACH_TOL = 0.030  # xy tolerance before dropping to ride height
GOAL_TOL = 0.045  # inside the 0.07 success radius, with margin
PUSH_STEP = 0.09  # per-step lead along the goal direction while shepherding


class ToolPullClassicalPolicy(ClassicalPolicyBase):
  """Direct closed-finger drag of the puck into the near zone.

  Not a tool-use policy -- see the module docstring for the measured reason the stick
  cannot be picked up. The state machine is deliberately flat (no grasp).
  """

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.10
  step_clip_mode = "per_joint"  # shepherding, like push_cuboid

  # THE VALUE THAT WAS CAPPING THIS TASK. The lowest COLLIDABLE link7-subtree geom is
  # 1.4cm below the ``gripper`` site (measured, not assumed), so the previous 0.026 left
  # pad material at 0.012 and ``ee_ground_collision`` fired on every transient dip.
  # Raising it to 0.030 took the identical strategy from 0.000 to 0.125.
  floor_min_z = 0.030

  # Phases: 0 swing out behind the puck at altitude, 1 descend to ride height,
  # 2 shepherd toward the goal, 3 done (back off).
  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ok = np.zeros(self.num_envs, dtype=bool)
      self._started = np.zeros(self.num_envs, dtype=bool)
    else:
      self._ok[env_ids] = False
      self._started[env_ids] = False

  # -- auto-reset detection --------------------------------------------------
  #
  # ``object_out_of_bounds`` fires readily here -- the puck starts at x~0.65 and is being
  # shoved around -- and ``ee_ground_collision`` fires on the drag. The harness only
  # calls policy.reset() BETWEEN episodes, never per-env on termination, so without this
  # the state machine stays in its shepherd phase against a freshly respawned puck it has
  # not approached and the arm walks off.
  #
  # Detected from the ROBOT state, not the object: ``reset_robot_joints`` puts the arm
  # back at exactly DEFAULT_QPOS, so ``joint_pos_rel`` collapses to noise-sized zero on
  # the first post-reset observation and never does so again once the arm is moving. An
  # earlier version watched for a jump in ``gripper_to_object`` instead; that misfires,
  # because at max_dq = 0.1 rad/step the EE legitimately travels ~7cm in one control step
  # and the observation carries up to 1.7cm of noise on top. ``_started`` latches so the
  # genuine at-home observation at the start of an episode is not read as a reset.
  reset_qpos_tol = 0.05

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    at_home = np.max(np.abs(obs_i[0:7])) < self.reset_qpos_tol
    if at_home and self._started[i]:
      self._phase[i] = 0
      self._phase_steps[i] = 0
      self._ok[i] = False
      self._started[i] = False
    elif not at_home:
      self._started[i] = True

  def _smooth(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not self._ok[i]:
      self._ema[i] = raw
      self._ok[i] = True
    else:
      self._ema[i] = 0.45 * raw + 0.55 * self._ema[i]
    return self._ema[i]

  def _target_error(self, i: int, obs_i: np.ndarray):
    err, rot, grip = self._plan(i, obs_i)
    # The site height is recovered by running FK on the arm's own joint angles
    # (``DEFAULT_QPOS + obs[0:9]``), which is expressed in the ROBOT BASE frame and so
    # carries no per-env scene-origin offset -- legal in a way that reading obs[25:28]
    # (absolute world gripper_pos) is not.
    z = float(self._fk(self.default_qpos + obs_i[0:9])[0][2])
    if z + err[2] < self.floor_min_z:
      err = np.asarray(err, dtype=np.float64).copy()
      err[2] = self.floor_min_z - z
    return err, rot, grip

  def _plan(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i)
    gto = self._smooth(i, obs_i[40:43])  # puck - gripper
    o2g = obs_i[43:46]  # goal - puck

    dist = float(np.linalg.norm(o2g[:2]))
    # Unit vector from the puck toward the goal (i.e. inward, toward the base).
    d = o2g[:2] / (dist + 1e-8)
    # Contact point: the puck's far rim, so the fingers push it inward.
    behind = np.array([-d[0] * BEHIND, -d[1] * BEHIND, 0.0])

    # Gripper height above the ground, recovered without any absolute term:
    # gripper_local = GOAL_LOCAL - o2g - gto, so its z is GOAL_LOCAL.z - o2g.z - gto.z.
    grip_z = GOAL_LOCAL[2] - o2g[2] - gto[2]

    if dist < GOAL_TOL or self._phase[i] == 3:
      # Done: lift straight up and hold, so the arm stops disturbing the puck while the
      # success latch is read.
      self._phase[i] = 3
      return np.array([0.0, 0.0, HOVER_Z - grip_z]), _DOWN_AXIS, GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Swing out to a point behind the puck, at transit altitude. Going high first
      # matters: a low straight-line approach ploughs the fingers through the puck from
      # the near side and shoves it further away.
      err = gto + behind
      err[2] = HOVER_Z - grip_z
      if np.linalg.norm(err[:2]) < APPROACH_TOL and self._phase_steps[i] > 6:
        self._phase[i] = 1
        self._phase_steps[i] = 0
      return err, _DOWN_AXIS, GRIPPER_CLOSED

    if self._phase[i] == 1:
      # Drop to ride height, still behind the puck, at a BOUNDED rate.
      err = gto + behind
      err[2] = np.clip(RIDE_Z - grip_z, -DESCEND_RATE, DESCEND_RATE)
      if abs(RIDE_Z - grip_z) < 0.012 or self._phase_steps[i] > 90:
        self._phase[i] = 2
        self._phase_steps[i] = 0
      return err, _DOWN_AXIS, GRIPPER_CLOSED

    # Shepherd: hold the contact point on the puck's far rim and creep inward.
    #
    # The lead is applied to the CONTACT POINT, and the standoff is deliberately not
    # cancelled: commanding ``gto + behind`` alone converges the gripper ONTO the puck
    # rather than behind it, because the DLS solve carries a 2-3cm steady-state bias that
    # eats the 5.2cm standoff. Measured symptom: the fingers straddle and PIN the puck
    # against the floor instead of pushing its rim, and ``goal_error`` stays flat for the
    # whole episode. Biasing the commanded contact point further out compensates.
    lead = min(dist, PUSH_STEP)
    err = gto + behind * BEHIND_BIAS + np.array([d[0] * lead, d[1] * lead, 0.0])
    err[2] = np.clip(RIDE_Z - grip_z, -DESCEND_RATE, DESCEND_RATE)
    return err, _DOWN_AXIS, GRIPPER_CLOSED
