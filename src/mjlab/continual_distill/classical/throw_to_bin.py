"""Scripted ThrowToBin teacher policy -- ballistic release (Lane D, D-3).

TASK
----
Reuses place_in_container's machinery unmodified (cube, container/bin,
PlaceInContainerCommand): grasp the 4cm cube (same object as Lift-Cube),
carry it toward the bin, release it in. CL-V2 (W1-b): the bin is now a real moulded
plastic basket, 191 x 183 x 129 mm, with a 16.3 cm inner clear span -- a 3x larger
landing area than the primitive's, which is the one thing that makes the ballistic
version of this task less hopeless. The one thing that changes is the
bin's placement: ``franka_throw_to_bin_env_cfg`` samples the bin at radial
0.78-0.90 (x 0.78-0.90, y -0.15..0.15) -- DELIBERATELY beyond the arm's
absolute ~0.85 m stretch (CONTEXT.md sec 6, ``strike_slide_env_cfg.py``'s own
docstring gives the same figure). The cube itself spawns in the ordinary
corner-safe grasp box (x ~0.30-0.46ish), well inside reach.

SUCCESS (PlaceInContainerCommand, reused unmodified)
-------------------------------------------------------
    lateral(cube - bin_site) < 0.0585  inside the footprint
    (cube - bin_site).z      < 0.093   below the rim
    (cube - bin_site).z      > -0.020  not tunnelled through the floor
    |cube linear velocity|   < 0.12    released and settled, not carried
Latched over the 250-step / 5.0 s episode.

WHY "HOLD ABOVE AND DROP" -- THE STRATEGY THAT SOLVES PLACE-IN-CONTAINER --
DOES NOT TRANSFER HERE
---------------------------------------------------------------------------
``place_in_container.py`` (the in-reach sibling of this task) solves its
task by centring the gripper directly ABOVE the bin's interior site at a
safe altitude, damping residual swing, and opening the fingers to let the
cube fall the last ~14cm -- deliberately NOT lowering into the bin, because
the fingertips would have to enter it past an 11.3cm rim. That strategy requires the gripper to REACH a point directly above the
bin. Here the bin sits partly or wholly beyond the arm's own absolute
stretch, so for a large fraction of sampled bin positions there is no
reachable point directly above the interior site at all -- the "hold above
and drop" precondition itself fails before release timing is ever the
question. See MEASURED section below for how much of the band this actually
costs.

STRATEGY -- grasp (lift_object's proven spine), reach toward the bin as far
as the arm goes, damp, release
-----------------------------------------------------------------------------
  0 HOVER    top-down, fingers open, align above the cube (identical
             constants to lift_object.py's LiftCubeClassicalPolicy -- same
             object, no reason to re-derive).
  1 DESCEND  straight down to the measured collision-safe grasp height.
  2 CLOSE    squeeze.
  3 CLIMB    straight up, holding, before moving laterally (dragging pops the
             grasp -- lift_object.py's own finding).
  4 REACH    drive ``object_to_goal`` toward "bin interior site + a small
             up offset" (mirrors place_in_container's DROP_HEIGHT convention).
             For an in-reach goal this converges normally; for an out-of-
             reach one the DLS solve simply saturates at its own steady-state
             extension short of the target and HOLDS there (same "punch a
             fixed, partly-unreachable target and let the solve saturate"
             behaviour topple_block/strike_slide/pivot_lift all rely on
             elsewhere in this package) -- there is no separate "give up"
             branch; whatever position is actually reached is where release
             happens.
  5 DAMP     hold station for a fixed step budget so residual swing settles
             before release (place_in_container's own finding: releasing into
             swing throws the cube past the wall).
  6 RELEASE  open fingers, hold position for the remainder of the episode.

Observations: 60-D layout, matches place_in_container/lift_object.
  40:43 gripper_to_object (cube - gripper), 43:46 object_to_goal
  (bin interior site - cube).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

# -- grasp geometry: identical to lift_object.py's LiftCubeClassicalPolicy --
# (same cube, same gripper -- no reason to re-derive; see that file's module
# docstring for the measured collision-floor derivation this reuses).
#
# CL-V2 (W1-b, on the lead's note): the shared cube is now W1-a's real 46 mm mini
# 3x3 (collision half-extents 0.0230 x 0.0226 x 0.0226), so its centre rests at
# **0.0226**, not 0.020. This file keeps its OWN copy of these constants rather
# than importing LiftCubeClassicalPolicy's, so W1-a's edit there did not reach it.
# ``GRASP_SITE_Z`` deliberately stays at 0.045: per lift_object.py it is the MEASURED
# ground-collision floor for the gripper site, not a function of the object, and the
# taller cube only improves the pad placement (pads at ~0.031-0.035 now sit at 69-78%
# of the cube's height instead of 78-88%). ``CLOSE_STEPS`` 15 matches W1-a's
# LiftCubeClassicalPolicy for the same cube.
GRASP_SITE_Z = 0.045
OBJ_CENTER_Z = 0.0226
CLOSE_STEPS = 15
HOVER_SITE_Z = 0.22
ALIGN_TOL = 0.020
ALIGN_TIMEOUT = 70
HELD_TOL = 0.06
SEAT_TOL = 0.012
DESCENT_RATE = 0.02
EMA_ALPHA = 0.4
CLIMB_STEPS = 30
CLIMB_ERR = 0.10
RESET_JUMP = 0.12

# -- reach / damp / release ----------------------------------------------------
# Height above the bin's interior site the gripper aims for -- mirrors
# place_in_container's DROP_HEIGHT, NOT reduced here: dropping from higher only
# matters once the site is actually reachable, which is the question this file
# measures, not tunes around. RE-DERIVED with the real basket (W1-b): its rim is
# 0.093 above the interior site (was 0.05), so the same clearance convention gives
# 0.15, matching place_in_container.
DROP_HEIGHT = 0.15
REACH_TIMEOUT = 120
REACH_SETTLE_TOL = 0.01
REACH_SETTLE_STEPS = 5
DAMP_STEPS = 25
RELEASE_HOLD = 15

FLOOR_MIN_Z = 0.030
MAX_WAYPOINT = 0.55

P_HOVER = 0
P_DESCEND = 1
P_CLOSE = 2
P_CLIMB = 3
P_REACH = 4
P_DAMP = 5
P_RELEASE = 6


class ThrowToBinClassicalPolicy(ClassicalPolicyBase):
  """Grasp the cube, reach as far toward the bin as the arm goes, release."""

  DEFAULT_QPOS = HOME_QPOS  # place_in_container-derived env uses HOME_QPOS
  max_dq = 0.05
  orientation_weight = 0.3

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._reach_settle = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._ema_init[env_ids] = False
      self._settle[env_ids] = 0
      self._reach_settle[env_ids] = 0

  def _rewind(self, i: int) -> None:
    self._phase[i] = P_HOVER
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._ema_init[i] = False
    self._reach_settle[i] = 0

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._gto_ema[i] = raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    return self._gto_ema[i]

  def _site_err_to(self, gto: np.ndarray, site_z: float) -> np.ndarray:
    return np.array([gto[0], gto[1], gto[2] + (site_z - OBJ_CENTER_Z)])

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]
    if self._ema_init[i]:
      if np.linalg.norm(raw_gto - self._prev_gto[i]) > RESET_JUMP:
        self._rewind(i)
    self._prev_gto[i] = raw_gto

    gto = self._gto(i, obs_i)
    o2g = obs_i[43:46]  # bin interior site - cube
    phase = self._phase[i]

    if phase == P_HOVER:
      pos_err = self._site_err_to(gto, HOVER_SITE_Z)
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.05:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = P_DESCEND
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      if self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = P_DESCEND
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return self._floor_guard(obs_i, pos_err), _DOWN_AXIS, GRIPPER_OPEN

    if phase == P_DESCEND:
      pos_err = self._site_err_to(gto, GRASP_SITE_Z)
      z_err = pos_err[2]
      pos_err[2] = max(z_err, -DESCENT_RATE)
      if abs(z_err) < SEAT_TOL and np.linalg.norm(pos_err[:2]) < 0.025:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = P_CLOSE
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > 120:
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
      return self._floor_guard(obs_i, pos_err), _DOWN_AXIS, GRIPPER_OPEN

    if phase == P_CLOSE:
      pos_err = self._site_err_to(gto, GRASP_SITE_Z)
      pos_err[:2] *= 0.5
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = P_CLIMB
        self._phase_steps[i] = 0
      return self._floor_guard(obs_i, pos_err), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_CLIMB:
      if self._phase_steps[i] < CLIMB_STEPS:
        return np.array([0.0, 0.0, CLIMB_ERR]), _DOWN_AXIS, GRIPPER_CLOSED
      if np.linalg.norm(gto) > HELD_TOL:
        # Grasp missed: retry rather than reaching toward the bin holding
        # nothing.
        self._rewind(i)
        return self._site_err_to(gto, HOVER_SITE_Z), _DOWN_AXIS, GRIPPER_OPEN
      self._phase[i] = P_REACH
      self._phase_steps[i] = 0
      return np.array([0.0, 0.0, 0.0]), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_REACH:
      # Aim above the bin's interior site. For an in-reach bin this converges
      # normally; for an out-of-reach one the DLS solve saturates at its own
      # steady-state extension and this simply stops improving -- there is no
      # separate branch for that, which is the point being measured.
      target = o2g + np.array([0.0, 0.0, DROP_HEIGHT])
      pos_err = target
      if np.linalg.norm(pos_err) < REACH_SETTLE_TOL:
        self._reach_settle[i] += 1
      else:
        self._reach_settle[i] = 0
      if (
        self._reach_settle[i] >= REACH_SETTLE_STEPS
        or self._phase_steps[i] > REACH_TIMEOUT
      ):
        self._phase[i] = P_DAMP
        self._phase_steps[i] = 0
      return self._floor_guard(obs_i, pos_err), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_DAMP:
      if self._phase_steps[i] >= DAMP_STEPS:
        self._phase[i] = P_RELEASE
        self._phase_steps[i] = 0
      return np.zeros(3), _DOWN_AXIS, GRIPPER_CLOSED

    # P_RELEASE: open and hold.
    return np.zeros(3), _DOWN_AXIS, GRIPPER_OPEN

  def _floor_guard(self, obs_i: np.ndarray, pos_err: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err
