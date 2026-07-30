"""Scripted ToolPull teacher policy.

TWO ENVIRONMENT FACTS THAT DETERMINE WHAT THIS TEACHER CAN BE
=============================================================

1. **The stick is not in the observation.** ``tool_pull_env_cfg.py``'s policy group
   binds every object term -- ``object_pos``, ``object_quat``, ``object_orientation``,
   ``gripper_to_object``, ``object_to_goal``, ``goal_orientation_diff`` -- to
   ``object_asset_name="puck"``. There is no stick term of any kind. The stick's pose
   is sampled uniformly over a 20cm x 19cm box every episode, so from the policy
   observation its grasp site is a random point inside a box roughly the size of the
   arm's whole grasp envelope. A tool-use teacher is therefore not merely hard here,
   it is **unobservable**: no function of the 60-D policy observation can locate the
   stick. (The command term tracks ``tool_grasped`` internally, but the policy never
   sees it.)

2. **The puck is not actually out of reach.** The stated premise is that the puck
   spawns beyond direct reach so the tool is mandatory. Measured against this repo's
   own IK on the Franka: driving the ``gripper`` site to the far corners of the puck's
   spawn box -- (0.62, 0.05, 0.05), (0.65, 0.11, 0.05), (0.70, 0.17, 0.05),
   (0.71, 0.05, 0.04) env-local -- converges to within 0.008-0.011m in a clean
   top-down pose every time. The box was pulled in from radial 0.78-0.88 (genuinely
   unreachable) to 0.62-0.71 to satisfy the workspace audit, and 0.71 is inside the
   arm's comfortable top-down envelope.

Given (1) and (2), the only teacher this observation space admits is a DIRECT drag:
reach past the puck with the fingers closed and sweep it into the near zone. That is
what is implemented. It is deliberately not dressed up as tool use -- see the
strategy note below.

SUCCESS (ToolPullCommand)
-------------------------
    || puck_site - (env_origin + (0.42, 0, 0.012)) || < 0.07
``tool_grasped`` is tracked but does NOT gate success, so a direct drag scores.

STRATEGY
--------
Reconstruct the env-local frame from two relative observations. The goal is at a
FIXED env-local offset, so with ``o2g`` = goal - puck and ``gto`` = puck - gripper:

    puck_local    = GOAL_LOCAL - o2g
    gripper_local = GOAL_LOCAL - o2g - gto

Both are offset-free (the scene origin cancels), which is what makes an absolute-ish
env-local plan legal here. Then: hover behind the puck (on the far side, away from
the goal), drop to puck height with the fingers closed, and shepherd it along the
straight line to the goal -- the same closed-finger push used by the push_cuboid
teacher, run outward-to-inward.

OBSERVATIONS: 60-D layout. [40:43] gripper_to_object (puck - gripper),
[43:46] object_to_goal (goal - puck).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# Env-local goal, from ToolPullCommandCfg.goal_offset.
GOAL_LOCAL = np.array([0.42, 0.0, 0.012])

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0

# The puck is a 7cm-diameter, 2.4cm-thick disc. Ride the closed fingertips at this
# height above the ground: high enough that the pads clear the floor (they sit ~4mm
# below the site), low enough to catch the puck's rim rather than skate over it.
RIDE_Z = 0.032
HOVER_Z = 0.16  # transit altitude while swinging out over the puck

# Rate at which the descent from HOVER_Z to RIDE_Z is allowed to proceed, in metres of
# commanded displacement per control step. Dropping in one commanded jump is what
# generated the ground contacts: at the converged IK pose the link7 subtree clears the
# floor comfortably (lowest geom at z=+0.022 for a site at 0.028, measured across the
# puck's spawn box), but the DLS path to that pose dips transiently, and out at radial
# 0.65 the arm is near-singular so the dip is large. Feeding the descent in slowly
# keeps the solver near its converged branch the whole way down.
DESCEND_RATE = 0.012

# Stand this far BEHIND the puck (on the far side from the goal) so the fingers make
# contact on its outer rim and push inward. Puck radius 0.035 + a pad half-width.
BEHIND = 0.052
# Multiplier on the standoff during the shepherd phase, to compensate the DLS
# steady-state bias that otherwise pulls the gripper on top of the puck.
BEHIND_BIAS = 1.8

APPROACH_TOL = 0.030  # xy tolerance before dropping to ride height
GOAL_TOL = 0.045  # inside the 0.07 success radius, with margin
PUSH_STEP = 0.09  # per-step lead along the goal direction while shepherding


class ToolPullClassicalPolicy(ClassicalPolicyBase):
  """Direct closed-finger drag of the puck into the near zone.

  Not a tool-use policy -- see the module docstring for why the observation space
  does not permit one. The state machine is deliberately flat (no grasp), because
  there is nothing to grasp that the policy can see.
  """

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.10
  step_clip_mode = "per_joint"  # shepherding, like push_cuboid

  # Phases: 0 swing out behind the puck at altitude, 1 descend to ride height,
  # 2 shepherd toward the goal, 3 done (back off).
  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ok = np.zeros(self.num_envs, dtype=bool)
      self._prev = np.zeros((self.num_envs, 3))
    else:
      self._ok[env_ids] = False
      self._prev[env_ids] = 0.0

  def _detect_reset(self, i: int, raw: np.ndarray) -> None:
    """Self-reset on the env's auto-reset.

    ``object_out_of_bounds`` (x outside (0,1), y outside (-0.5,0.5)) fires readily
    here -- the puck starts at x~0.65 and is being shoved around -- and the harness
    never tells the policy. Without this the state machine stays in its "shepherd"
    phase against a freshly respawned puck it has not approached, and the arm walks
    off. Detected as a one-step jump in the puck bearing larger than the arm can
    physically produce.
    """
    if self._ok[i] and np.linalg.norm(raw - self._prev[i]) > 0.12:
      self._phase[i] = 0
      self._phase_steps[i] = 0
      self._ok[i] = False
    self._prev[i] = raw

  def _smooth(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not self._ok[i]:
      self._ema[i] = raw
      self._ok[i] = True
    else:
      self._ema[i] = 0.45 * raw + 0.55 * self._ema[i]
    return self._ema[i]

  # Floor guard, for the same reason as the grasp tasks: ``ee_ground_collision``
  # matches the whole link7 subtree and the lowest fingertip geom is 1.24cm below the
  # ``gripper`` site. The site height is recovered by running FK on the arm's own
  # joint angles (``DEFAULT_QPOS + obs[0:9]``), which is expressed in the ROBOT BASE
  # frame and so carries no per-env scene-origin offset.
  floor_min_z = 0.026

  def _target_error(self, i: int, obs_i: np.ndarray):
    err, rot, grip = self._plan(i, obs_i)
    z = float(self._fk(self.default_qpos + obs_i[0:9])[0][2])
    if z + err[2] < self.floor_min_z:
      err = np.asarray(err, dtype=np.float64).copy()
      err[2] = self.floor_min_z - z
    return err, rot, grip

  def _plan(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i[40:43])
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
      # Done: lift straight up and hold, so the arm stops disturbing the puck while
      # the success latch is read.
      self._phase[i] = 3
      return np.array([0.0, 0.0, HOVER_Z - grip_z]), _DOWN_AXIS, GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Swing out to a point behind the puck, at transit altitude. Going high first
      # matters: a low straight-line approach ploughs the fingers through the puck
      # from the near side and shoves it further away.
      err = gto + behind
      err[2] = HOVER_Z - grip_z
      if np.linalg.norm(err[:2]) < APPROACH_TOL and self._phase_steps[i] > 6:
        self._phase[i] = 1
        self._phase_steps[i] = 0
      return err, _DOWN_AXIS, GRIPPER_CLOSED

    if self._phase[i] == 1:
      # Drop to ride height, still behind the puck, at a BOUNDED rate (see
      # DESCEND_RATE) so the solver does not swing the wrist through the floor.
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
    # rather than behind it, because the DLS solve carries a 2-3cm steady-state bias
    # that eats the 5.2cm standoff. Measured symptom: in the shepherd phase
    # ``gripper_to_object`` sits at ~0 (not at -behind) with the pads at puck height,
    # so the fingers straddle and PIN the puck against the floor instead of pushing
    # its rim, and ``goal_error`` stays flat at 0.25-0.28 for the whole episode.
    # Biasing the commanded contact point further out compensates for that bias.
    lead = min(dist, PUSH_STEP)
    err = gto + behind * BEHIND_BIAS + np.array([d[0] * lead, d[1] * lead, 0.0])
    err[2] = np.clip(RIDE_Z - grip_z, -DESCEND_RATE, DESCEND_RATE)
    return err, _DOWN_AXIS, GRIPPER_CLOSED
