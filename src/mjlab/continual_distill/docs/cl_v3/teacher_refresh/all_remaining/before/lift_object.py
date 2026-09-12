"""Scripted grasp-and-lift teacher, shared across all four free objects.

ONE strategy, four instantiations (cube / cylinder / sphere / ellipsoid). The
shapes differ only in grasp geometry, which is captured by a handful of class
attributes rather than by forking the file.

Strategy — a four-phase top-down pick:
  0 HOVER    fingers OPEN, align the gripper site above the object centre at
             HOVER_Z, EE z-axis pointing straight down (yaw free).
  1 DESCEND  drop VERTICALLY to the grasp height, still open, rate-limited.
  2 CLOSE    hold station and command the fingers shut for CLOSE_STEPS.
  3 LIFT     climb straight up to clear the ground, then drive
             ``object_to_goal`` to zero with the fingers still shut.

MEASURED GEOMETRY (all of this was verified in-sim, not assumed):

  * The ``gripper`` site is NOT 10cm behind the fingertips. With the EE pointing
    down, the lowest collision geometry of the whole hand (the finger pads) sits
    only 0.0135 m BELOW the site; the hand capsule is 0.028 m ABOVE it.
    ``workspace.SITE_TO_FINGERTIP = 0.10`` refers to a different reference and
    must NOT be applied here — doing so commands the site 10cm too high and the
    fingers never touch the object.

  * The binding constraint is the ``ee_ground_collision`` TERMINATION, which
    watches the entire link7 subtree against the terrain. Objects rest on the
    GROUND (no table), so the grasp happens a few centimetres above a live
    kill-switch. Sweeping in laterally at low altitude terminates the episode
    within ~50 steps every time — that is what makes this task hard, not the
    grasp itself.

    Measured, holding station over a cube for 250 steps (8 envs):
        direct approach to site z=0.08   ->  9 ground collisions
        direct approach to site z=0.09   ->  3 collisions
        hover-then-vertical-descend to site z=0.045 -> 0 collisions
        hover-then-vertical-descend to site z=0.035 -> 5 collisions
    The collisions in the direct case are TRANSIENTS on the swing-down path,
    not the hold pose. Hence: get high above the object first, align in xy,
    and only then come straight down. GRASP_SITE_Z = 0.045 is the floor.

  * At site z = 0.045 the pad bottoms are at z ~ 0.031 and the pad centres at
    z ~ 0.035, so the pads straddle the upper half of a 4cm-tall object. That
    is a valid grasp for every object here.

  * The finger CLOSING axis is the site y-axis; at full open the pads sit at
    +-0.0476 m (a 9.5cm gap), wider than every object here (max width 0.044),
    so yaw can be left free.

  * Franka finger action: POSITIVE opens, NEGATIVE closes.

Only RELATIVE observations are used (per-env scene-origin offsets cancel):
  obs[40:43] gripper_to_object (object - gripper site)
  obs[43:46] object_to_goal    (goal - object)
Success: object centre within 5cm of the goal, latched over the episode. Goals
hang in free space at z 0.10-0.35, so the object genuinely has to leave the
ground — shoving it along the floor cannot satisfy this.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0


class LiftObjectClassicalPolicy(ClassicalPolicyBase):
  """Top-down pick-and-place: hover -> vertical descend -> close -> carry."""

  DEFAULT_QPOS = HOME_QPOS  # lift envs use get_franka_robot_cfg (home)
  # Deliberately unhurried. Brisk motion near the ground is exactly what trips
  # the ee_ground_collision termination.
  max_dq = 0.05
  orientation_weight = 0.3  # keep the approach vertical; yaw stays free

  # -- per-object tuning ----------------------------------------------------
  # Absolute height (metres above the ground) the gripper SITE descends to for
  # the grasp. 0.045 is the measured collision floor; see the module docstring.
  GRASP_SITE_Z = 0.045
  # Height of the object's CENTRE above the ground when at rest. Used only to
  # convert the relative gripper_to_object vector into an absolute site height.
  OBJ_CENTER_Z = 0.020
  # Steps spent squeezing before lifting.
  CLOSE_STEPS = 15
  # Site height while hovering. High enough that the swing-in never dips the
  # wrist toward the terrain.
  HOVER_SITE_Z = 0.22
  # xy alignment tolerance before committing to the descent, and a hard cap on
  # how long to keep trying: the DLS solve has a steady-state bias of ~2cm, so
  # an over-tight tolerance simply never trips.
  ALIGN_TOL = 0.020
  ALIGN_TIMEOUT = 70
  # After the post-grasp climb, the object must still be within this distance of
  # the gripper site or the grasp is judged to have missed.
  HELD_TOL = 0.06
  # Descent is complete when the site is this close to the grasp height.
  SEAT_TOL = 0.012
  # Vertical rate cap on the commanded descent error.
  DESCENT_RATE = 0.02

  EMA_ALPHA = 0.4
  # Steps of straight-up climb immediately after the grasp, before chasing the
  # goal laterally. Dragging the object across the terrain pops it out of the
  # fingers, and the goal can be as low as z=0.10.
  CLIMB_STEPS = 30
  CLIMB_ERR = 0.10

  # If the observed object jumps by more than this in one control step, the env
  # auto-reset underneath us (a ground collision or an out-of-bounds ends the
  # episode mid-flight and respawns the object). The per-env state machine MUST
  # be rewound then, or the policy spends the rest of the episode in phase 3
  # "carrying" an object it never picked up. This single check is worth more
  # than any amount of grasp tuning.
  RESET_JUMP = 0.12

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._prev_gto = np.zeros((self.num_envs, 3))
    else:
      self._ema_init[env_ids] = False
      self._settle[env_ids] = 0

  def _rewind(self, i: int) -> None:
    self._phase[i] = 0
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._ema_init[i] = False

  # -- helpers --------------------------------------------------------------

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Smoothed object-minus-site vector (relative => offset-free)."""
    raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._gto_ema[i] = raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = self.EMA_ALPHA * raw + (1 - self.EMA_ALPHA) * self._gto_ema[i]
    return self._gto_ema[i]

  def _site_err_to(self, gto: np.ndarray, site_z: float) -> np.ndarray:
    """Position error that puts the site over the object at absolute height
    ``site_z``.

    The object centre rests at OBJ_CENTER_Z, so ``gto[2] = obj_z - site_z_now``
    gives the current site height implicitly and the desired vertical error is
    ``gto[2] + (site_z - OBJ_CENTER_Z)``. Everything stays relative, so the
    per-env scene-origin offsets cancel.
    """
    return np.array([gto[0], gto[1], gto[2] + (site_z - self.OBJ_CENTER_Z)])

  # -- policy ---------------------------------------------------------------

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]
    # --- detect a mid-episode auto-reset and rewind the state machine --------
    if self._ema_init[i]:
      if np.linalg.norm(raw_gto - self._prev_gto[i]) > self.RESET_JUMP:
        self._rewind(i)
    self._prev_gto[i] = raw_gto

    gto = self._gto(i, obs_i)
    o2g = obs_i[43:46]  # goal - object
    phase = self._phase[i]

    if phase == 0:
      # Get high and centred before going anywhere near the floor. The tolerance
      # is checked over consecutive steps: the DLS solve carries a steady-state
      # bias of a couple of cm, so a single-step tight test never trips and the
      # arm hovers for the whole episode.
      pos_err = self._site_err_to(gto, self.HOVER_SITE_Z)
      if np.linalg.norm(pos_err[:2]) < self.ALIGN_TOL and abs(pos_err[2]) < 0.05:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = 1
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      # Do not hover forever: after ALIGN_TIMEOUT steps take the best xy we have
      # and descend anyway. Descending slightly off-centre still often grasps;
      # never descending never does.
      if self._phase_steps[i] > self.ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return pos_err, _DOWN_AXIS, GRIPPER_OPEN

    if phase == 1:
      # Straight down. xy is already solved; the rate cap keeps the wrist from
      # swinging on the way in.
      pos_err = self._site_err_to(gto, self.GRASP_SITE_Z)
      z_err = pos_err[2]
      pos_err[2] = max(z_err, -self.DESCENT_RATE)
      if abs(z_err) < self.SEAT_TOL and np.linalg.norm(pos_err[:2]) < 0.025:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > 120:  # stalled: squeeze anyway rather than idle
        self._phase[i] = 2
        self._phase_steps[i] = 0
      return pos_err, _DOWN_AXIS, GRIPPER_OPEN

    if phase == 2:
      # Squeeze. Actively HOLD the grasp height — commanding zero vertical error
      # does not hold altitude, it just stops correcting, and the arm sags into
      # the terrain (measured: a ground-collision termination mid-close).
      pos_err = self._site_err_to(gto, self.GRASP_SITE_Z)
      pos_err[:2] *= 0.5  # gentle xy so the pads do not shove the object away
      if self._phase_steps[i] >= self.CLOSE_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
      return pos_err, _DOWN_AXIS, GRIPPER_CLOSED

    # phase 3: climb clear, then carry to the goal.
    if self._phase_steps[i] < self.CLIMB_STEPS:
      return np.array([0.0, 0.0, self.CLIMB_ERR]), _DOWN_AXIS, GRIPPER_CLOSED

    # Grasp check. If the object is still on the floor after the climb, the
    # grasp missed; go back and try again rather than flying to the goal
    # holding nothing (which drags the site to the edge of the workspace and
    # wastes the rest of the episode).
    if np.linalg.norm(gto) > self.HELD_TOL:
      self._rewind(i)
      return self._site_err_to(gto, self.HOVER_SITE_Z), _DOWN_AXIS, GRIPPER_OPEN

    # The object is rigidly held, so moving the site by d moves the object by d:
    # commanding o2g on the site drives the success metric directly.
    return o2g.copy(), _DOWN_AXIS, GRIPPER_CLOSED


class LiftCubeClassicalPolicy(LiftObjectClassicalPolicy):
  """Cube: a real 46 mm mini 3x3 (centre rests at 0.0226). Flat faces, easiest grasp.

  GRASP_SITE_Z stays at the inherited 0.045 — it is the MEASURED ground-collision
  floor for the site, not a function of the object, and the taller cube only helps:
  the pads (bottoms ~0.031, centres ~0.035) now sit at 69-78% of the cube's height
  instead of 78-88%, i.e. further from the top edge they used to risk sliding off.
  """

  OBJ_CENTER_Z = 0.0226
  CLOSE_STEPS = 15
  # CL-V3 (W1-N, 2026-09-09): the +-10 deg robot start noise took this teacher from 1.000 to
  # 0.945 (128). diagnose: 26/31 failures are `ee_ground_collision` with the fingers still
  # OPEN, most steps in HOVER -- the site is commanded to 0.22 but SINKS to 0.06-0.09 while
  # the arm folds in from the stretched (noisy) HOME pose, the 70-step hover times out
  # unaligned and DESCEND starts beside the cube at floor height. This is W1-G's gravity-sag
  # ratchet (logs/W1-G.md): with cmd_lead_max 0 the joint command is re-anchored to the
  # sagged actual joints every step. Bounded command lead = gravity compensation; set on
  # the cube class only (the collapsed variants keep the legacy behaviour).
  cmd_lead_max = 0.12


class LiftCylinderClassicalPolicy(LiftObjectClassicalPolicy):
  """Amber packer bottle r=0.0150 h=0.0532 upright (centre 0.0266). Curved side, flat ends."""

  OBJ_CENTER_Z = 0.0266
  CLOSE_STEPS = 18


class LiftSphereClassicalPolicy(LiftObjectClassicalPolicy):
  """Sphere r=0.022 (centre 0.022). Rolls; pads must meet near the equator."""

  OBJ_CENTER_Z = 0.022
  # A ball punishes lateral error: a pad that lands off-centre rolls it away.
  ALIGN_TOL = 0.008
  SEAT_TOL = 0.010
  CLOSE_STEPS = 24


class LiftEllipsoidClassicalPolicy(LiftObjectClassicalPolicy):
  """Ellipsoid semi-axes 0.035 x 0.018 x 0.018, lying down (centre 0.018).

  Grasping across the SHORT axis is what works. Yaw is left free: at full open
  the pads are 9.5cm apart, comfortably wider than the 0.070 long axis, so the
  fingers clear the object from any yaw and close onto whichever cross-section
  they meet.
  """

  OBJ_CENTER_Z = 0.018
  SEAT_TOL = 0.010
  CLOSE_STEPS = 22
