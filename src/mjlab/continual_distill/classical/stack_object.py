"""Scripted StackCube teacher policy + the shared grasp-and-transport spine.

All five multi-object / novel-success-shape tasks (stack, peg insertion,
place-in-container, reorient, tool-pull) share the same skeleton:

    hover over the object -> descend -> close -> lift -> translate -> ending

so that skeleton lives here as ``GraspTransportPolicy`` and each task subclasses it
and overrides only the pieces it needs (where "over the object" is, what the carry
target is, and what happens at the end).

GEOMETRY (measured from panda.xml, not guessed)
-----------------------------------------------
With the wrist in a top-down pose and the fingers CLOSED, the two fingertip pads sit
3.65mm BELOW the ``gripper`` site and straddle it symmetrically along the EE x-axis.
So the ``gripper`` site IS the grasp point: to grasp an object whose centre is at P,
drive the site to P (no 10cm tool offset). The EE **x-axis is the finger-closing
axis**; the EE z-axis is the approach axis.

OBSERVATIONS
------------
This file's task (Stack-Cube) uses the 51-D two-object layout:
    [37:40] gripper_to_object  (cube - gripper)
    [40:43] object_to_goal     (stack target - cube); the target tracks the base
Both are relative, so the per-env scene-origin offsets cancel. Absolute terms
(obs[18:21], obs[25:31]) are NEVER used — they differ per env by the scene origin.

SUCCESS (StackingCommand)
-------------------------
xy error < 0.03 AND |z error| < 0.02 between the cube's body origin and
base_pos + 0.035. So the cube has to be *released* onto the base: while it is still
held 5cm up the height test fails. Hence the explicit open-and-retreat ending.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  HOME_QPOS,
  ClassicalPolicyBase,
  lowest_hand_z,
)
from mjlab.continual_distill.classical.staged_stack import StagedStackPolicy

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

# Phases of the shared spine.
P_HOVER = 0  # get xy-aligned above the object, fingers open
P_DESCEND = 1  # come straight down onto the grasp point
P_CLOSE = 2  # squeeze, hold still
P_LIFT = 3  # raise clear of the ground / of neighbouring objects
P_CARRY = 4  # translate to above the drop point
P_PLACE = 5  # lower onto the drop point
P_RELEASE = 6  # open the fingers
P_RETREAT = 7  # rise away so the arm is not resting on the placed object
P_DONE = 8

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])


def down_frame(yaw: float) -> np.ndarray:
  """Full EE rotation matrix: z-axis straight down, finger-closing axis (EE x) at
  ``yaw`` in the world xy-plane.

  Constraining yaw matters whenever the object is not rotationally symmetric about
  the approach axis (the peg, the lying cylinder, the stick shaft). For symmetric
  objects prefer the axis-only target ``_DOWN_AXIS`` — it leaves a DOF free and so
  keeps poses feasible further out in the workspace.
  """
  c, s = np.cos(yaw), np.sin(yaw)
  ex = np.array([c, s, 0.0])  # finger-closing axis
  ez = np.array([0.0, 0.0, -1.0])  # approach axis
  ey = np.cross(ez, ex)
  return np.column_stack([ex, ey, ez])


def closing_frame(theta: float) -> np.ndarray:
  """Full EE rotation: z-axis straight down, FINGER-CLOSING axis (the site's y-axis,
  measured: the pads separate along site y, +-0.0476 at full open) at heading
  ``theta`` in the world xy-plane. ``down_frame`` puts the site x-axis at the yaw;
  this puts the closing axis there, which is what face alignment needs."""
  c, s = np.cos(theta), np.sin(theta)
  ey = np.array([c, s, 0.0])  # closing axis
  ez = np.array([0.0, 0.0, -1.0])  # approach axis
  ex = np.cross(ey, ez)
  return np.column_stack([ex, ey, ez])


def quat_yaw_wxyz(q: np.ndarray) -> float:
  w, x, y, z = q
  return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class GraspTransportPolicy(ClassicalPolicyBase):
  """Shared hover -> descend -> close -> lift -> carry -> ending state machine.

  Subclasses supply the task geometry through the hooks below. Everything is
  expressed as vectors FROM the gripper site, built only from relative observations.
  """

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.10

  # -- tunables (overridable per task) --------------------------------------
  hover_height = 0.12  # hover this far above the grasp point
  align_tol = 0.025  # xy tolerance before starting the descent
  descend_tol = 0.012  # 3D tolerance that counts as "on the grasp point"
  close_steps = 12  # steps to hold the squeeze before lifting
  lift_height = 0.16  # carry altitude above the grasp point
  lift_steps = 22
  # Fraction of ``lift_steps`` over which the lift command ramps from 0 to
  # ``lift_height`` (see P_LIFT). A fraction, not a step count, so that
  # subclasses which shorten ``lift_steps`` still reach full height in time.
  # DEFAULT 0.0 (= no ramp, the original behaviour). Ramping was measured over 96
  # episode-instances and made both cube tasks WORSE, not better -- see P_LIFT.
  lift_ramp_frac = 0.0
  carry_tol = 0.02  # xy tolerance above the drop point before lowering
  place_tol = 0.012
  release_steps = 12
  retreat_height = 0.15
  ema_alpha = 0.45  # observation smoothing (obs noise is +-1cm per axis)
  descend_integ_gain = 0.25  # nulls the DLS steady-state bias on the descent
  descend_integ_clip = 0.05
  carry_integ_gain = 0.18  # same, for the lateral carry/place (see _carry)
  carry_integ_clip = 0.06
  grasp_yaw = None  # None -> axis-only (yaw free); else a float in radians
  # Gripper-closed action value used for P_CLOSE/P_LIFT/_carry/_place. DEFAULT is the
  # original GRIPPER_CLOSED (-1.0, "fully closed") so every existing subclass
  # (peg_insertion, reorient_object) is byte-for-byte unaffected unless it opts in by
  # overriding this. See the P_CLOSE docstring for why a subclass might override it.
  grip_close_action = GRIPPER_CLOSED
  # Vertical offset added to the grasp point. In a top-down pose the lowest robot
  # COLLISION geom (a fingertip pad) sits 1.4cm below the ``gripper`` site -- and the
  # hand capsule's bounding volume reaches ~3.1cm below it -- and
  # ``ee_ground_collision`` terminates the episode on any robot/ground contact. So
  # grasping a ground-resting object exactly at its centre leaves under a centimetre
  # of clearance, and the descent's overshoot eats it. Grasp slightly high instead.
  grasp_z_offset = 0.0

  # -- CL-V3 opt-ins (W1-G, 2026-09-09). ALL default-off: subclasses that do not set
  # them (reorient_object) behave byte-for-byte as before. -----------------------
  #
  # grasp_style = "lift": HOVER/DESCEND/CLOSE/LIFT are lift_object.py's phases
  # (measured 1.000 on the same 46 mm cube): settle xy for ``lift_align_settle``
  # consecutive steps at an ABSOLUTE site height before descending, descend
  # vertically at a bounded rate to an ABSOLUTE grasp site height with a seat settle,
  # keep servoing xy (half gain) and z while closing, climb straight up, then a
  # held check that retries from HOVER. The diagnose trace of the old spine shows why
  # this matters: every failing Stack env had the cube shoved 30-55 mm at 0.6-0.85
  # m/s DURING DESCEND (the hand landed on it: fast descent, no settle, integrator
  # wind-up, grasp 1 cm deeper than the collision floor) before the pads ever closed.
  grasp_style = "spine"
  lift_hover_site_z = 0.22
  lift_grasp_site_z = 0.045  # lift_object.py's measured collision floor for the site
  lift_align_tol = 0.020
  lift_align_settle = 3
  lift_align_timeout = 70
  lift_seat_tol = 0.012
  lift_seat_xy_tol = 0.025
  lift_descent_rate = 0.02
  lift_descend_timeout = 120
  lift_close_steps = 15
  # CL-V3 (W1-G2): ramp the finger command from open to closed over this many steps
  # instead of slamming it (0 = the legacy instant close, the default for every existing
  # subclass). The finger actuator is a kp-350 position servo on a 15 g finger, so a
  # step command drives the pads in at ~0.4 m/s and the pad EDGE meets the object before
  # the face does. A squat cube shrugs that off; a 10 cm standing peg is toppled by it.
  lift_close_ramp = 0
  # CL-V3 (W1-G2), opt-in (0 = off, the default for every existing subclass): do not enter
  # CLOSE until the finger-closing axis is within this many radians of the latched target
  # heading. MEASURED on Peg-Insertion (diagnose, n = 32): the aperture at LIFT entry --
  # which is the width the pads actually stalled at -- runs 0.0249 to 0.0409 on a 25 mm
  # square peg whose DIAGONAL is 0.0354, median 0.0289. Anything above ~0.026 means the
  # pads met the peg on two EDGES rather than two faces, and two line contacts are a
  # HINGE: the peg then rotates freely about them, which is why 13/28 pegs first tilt
  # past 32 deg during LIFT and another 9 during CARRY, with the grip still closed.
  close_orient_tol = 0.0
  # Lateral gain while closing. The lift-style CLOSE keeps creeping the xy error at half
  # gain so the arm does not sag; for a tall, tippy object that same creep walks the pads
  # across the object and pushes it over, so it can be turned off per subclass.
  lift_close_xy_gain = 0.5
  lift_climb_steps = 30
  lift_climb_err = 0.10
  lift_held_tol = 0.06
  lift_max_dq = 0.05  # unhurried near the floor, like lift_object.py
  # Climb until the SITE reaches this absolute height (None = the legacy fixed step
  # count, which sends the site to ~0.37 m and makes the carry a fast 0.18 m descent).
  lift_climb_to = None
  # max_dq while carrying/placing a held object (None = the class max_dq).
  carry_max_dq = None
  # Small lateral integrator on the lift-style hover/descent (the yaw-constrained
  # frame below carries a 2-3 cm DLS bias that a proportional hover never closes).
  lift_integ_gain = 0.0
  lift_integ_clip = 0.04
  lift_integ_near = 0.05  # integrate only near the target (wind-up on the approach made the hover orbit)
  # yaw_align: point the finger-closing axis at the object's faces (nearest of the
  # ``yaw_symmetry``-spaced candidates to the current closing heading, latched per
  # attempt). Needs the object quaternion (w,x,y,z) at ``quat_slice``.
  yaw_align = False
  quat_slice = slice(21, 25)
  yaw_symmetry = np.pi / 2  # cube / square peg: 4-fold
  yaw_align_hysteresis = 0.6  # fraction of yaw_symmetry before re-latching
  # reset_detect = "jump": detect a mid-episode auto-reset from a teleport of the
  # arm (> reset_q_jump rad on any joint in one step) OR of the object (> reset_gto_jump
  # m). The default "home" test (|joint_pos_rel| < 0.05) cannot fire under the CL-V3
  # +-10 deg reset joint noise, so it silently stopped working on every task.
  reset_detect = "home"
  reset_q_jump = 0.25
  reset_gto_jump = 0.15
  # carry_timeout > 0: unconditional CARRY escape back to HOVER (phase-1 M3; Stack /
  # Place had none -- 9/17 baseline failures sat in CARRY for ~900 steps holding air).
  carry_timeout = 0
  # retry_from_done: after RETREAT, if ``_placed_ok`` says the object is not at the
  # goal, go back to HOVER and try again while the budget lasts (success is latched).
  retry_from_done = False
  max_attempts = 6
  # cmd_lead_max (base.py): integrate the joint command from the previous COMMAND, up
  # to this many rad ahead of the actual joints. MEASURED (CPU, Peg hover): with the
  # default 0 the command is re-anchored to the sagging actual joints every step and a
  # stretched arm SINKS ~1 cm/step while commanded up (site 0.28 -> 0.06 in 50 steps,
  # old spine and new alike); 0.15 holds the hover at 0.20-0.22. Opt-in per subclass.
  lead = 0.0

  def _act_single(self, i, obs_i):
    self.cmd_lead_max = self._lead_for(i)
    return super()._act_single(i, obs_i)

  def _lead_for(self, i: int) -> float:
    """Command lead for THIS env this step (default: the class value).

    A hook, not a constant, because the lead is a two-edged tool: it is what stops the
    gravity-sag ratchet on a hover, and it is also what lets the command sit ~0.12 rad
    (= ~6 cm at the site) BELOW the actual joints while the servo catches up. MEASURED
    (Peg-Insertion diagnose, 16/31 failures): during the insert the peg jams on the
    board rim, the teacher keeps commanding down, and the wrist sinks to site z 0.004 --
    1.8 cm past the 0.022 floor guard, which clamps the TARGET but cannot clamp a
    command that is already leading downward -- and ``ee_ground_collision`` fires.
    Subclasses that press an object into something override this to drop the lead
    during the pressing phase."""
    return self.lead

  # -- observation hooks (subclass overrides) --------------------------------
  #: index slice of gripper_to_object in this task's observation layout
  gto_slice = slice(37, 40)
  #: index slice of object_to_goal
  o2g_slice = slice(40, 43)

  def _gripper_to_grasp(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Vector from the gripper site to the point the fingers should close on."""
    return self._gto(i, obs_i) + np.array([0.0, 0.0, self.grasp_z_offset])

  def _grasp_to_drop(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Vector from the CURRENT grasp point to where the object should be released."""
    return obs_i[self.o2g_slice]

  def _approach_rot(self, i: int, obs_i: np.ndarray):
    if self.yaw_align:
      return closing_frame(self._closing_heading(i, obs_i))
    if self.grasp_yaw is None:
      return _DOWN_AXIS
    return down_frame(self.grasp_yaw)

  def _object_yaw(self, i: int, obs_i: np.ndarray) -> float:
    return quat_yaw_wxyz(obs_i[self.quat_slice])

  def _closing_heading(self, i: int, obs_i: np.ndarray) -> float:
    """Heading for the finger-closing axis: the object-face normal (object yaw plus a
    multiple of ``yaw_symmetry``) nearest the CURRENT closing heading, latched per
    attempt with hysteresis so obs noise cannot flip it between two candidates."""
    psi = self._object_yaw(i, obs_i)
    sym = self.yaw_symmetry
    if np.isnan(self._yaw_target[i]):
      _, r = self._fk(self.default_qpos + obs_i[0:9])
      ref = float(np.arctan2(r[1, 1], r[0, 1]))  # current closing heading
    else:
      ref = float(self._yaw_target[i])
    k = np.round((ref - psi) / sym)
    cand = psi + k * sym
    if np.isnan(self._yaw_target[i]):
      self._yaw_target[i] = cand
    else:
      d = (cand - self._yaw_target[i] + np.pi) % (2 * np.pi) - np.pi
      if abs(d) > self.yaw_align_hysteresis * sym:
        self._yaw_target[i] = cand
      else:
        # Track the object's (noisy) yaw smoothly around the latched candidate.
        self._yaw_target[i] += 0.3 * d
    return float(self._yaw_target[i])

  # -- state -----------------------------------------------------------------

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_ok = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._anchor = np.zeros((self.num_envs, 3))
      self._started = np.zeros(self.num_envs, dtype=bool)
      # CL-V3 opt-in state (harmless when the opt-ins are off).
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._attempts = np.zeros(self.num_envs, dtype=np.int64)
      self._prev_q = np.zeros((self.num_envs, 7))
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._have_prev = np.zeros(self.num_envs, dtype=bool)
      self._yaw_target = np.full(self.num_envs, np.nan)
      self._lift_integ = np.zeros((self.num_envs, 3))
      self._mean_sum = np.zeros((self.num_envs, 3))
      self._mean_n = np.zeros(self.num_envs, dtype=np.int64)
      self._mean_held = np.zeros(self.num_envs, dtype=bool)
      if not hasattr(self, "_base_max_dq"):
        self._base_max_dq = self.max_dq
    else:
      self._ema_ok[env_ids] = False
      self._integ[env_ids] = 0.0
      self._anchor[env_ids] = 0.0
      self._started[env_ids] = False
      self._settle[env_ids] = 0
      self._attempts[env_ids] = 0
      self._have_prev[env_ids] = False
      self._yaw_target[env_ids] = np.nan
      self._lift_integ[env_ids] = 0.0
      self._mean_sum[env_ids] = 0.0
      self._mean_n[env_ids] = 0
      self._mean_held[env_ids] = False

  # -- auto-reset detection --------------------------------------------------
  #
  # These envs terminate EARLY and often: ``ee_ground_collision`` fires whenever any
  # robot geom touches the ground plane (and the lowest fingertip geom is only 1.4cm
  # below the ``gripper`` site, so a top-down grasp of a 4cm cube has under a
  # centimetre of clearance), and ``object_out_of_bounds`` fires if the object is
  # knocked outside x in (0,1) / y in (-0.5,0.5). On termination the env auto-resets
  # in place, but the evaluation harness (``test_classical.py``) only ever calls
  # ``policy.reset()`` between EPISODES, never per-env on termination. So without the
  # check below the state machine keeps marching through phases that no longer match
  # the world -- it "lifts" and "carries" nothing while the freshly respawned cube
  # sits untouched, and the arm wanders to the edge of its workspace. This was worth
  # the entire task's success rate: it is the single largest effect in this file.
  #
  # Detection is purely observational (the harness gives us no reset signal). We use
  # the ROBOT STATE, not the object: ``reset_robot_joints`` puts the arm back at
  # exactly DEFAULT_QPOS with zero velocity, so ``joint_pos_rel`` and ``joint_vel``
  # both collapse to (noise-sized) zero on the first post-reset observation, and that
  # combination never occurs mid-episode once the arm is moving.
  # An earlier version watched for a jump in ``gripper_to_object`` instead; that
  # misfires, because at max_dq=0.1 rad/step the EE legitimately travels ~7cm in one
  # control step and the observation adds up to 1.7cm of noise on top.
  # ``_started`` latches once the arm has actually left home, so the genuine
  # at-home observation at the very start of an episode is not read as a reset.
  reset_qpos_tol = 0.05  # rad; obs noise on joint_pos_rel is only +-0.01

  def _rewind(self, i: int) -> None:
    """Back to HOVER for a fresh attempt (missed grasp, lost object, auto-reset)."""
    self._phase[i] = P_HOVER
    self._phase_steps[i] = 0
    self._ema_ok[i] = False
    self._integ[i] = 0.0
    self._settle[i] = 0
    self._yaw_target[i] = np.nan
    self._lift_integ[i] = 0.0
    self._mean_sum[i] = 0.0
    self._mean_n[i] = 0
    self._mean_held[i] = False
    self._attempts[i] += 1

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    if self.reset_detect == "jump":
      q = obs_i[0:7]
      g = obs_i[self.gto_slice]
      if self._have_prev[i]:
        q_jump = float(np.max(np.abs(q - self._prev_q[i])))
        g_jump = float(np.linalg.norm(g - self._prev_gto[i]))
        if q_jump > self.reset_q_jump or g_jump > self.reset_gto_jump:
          self._rewind(i)
          self._attempts[i] = 0
          self._on_reset(i)
      self._prev_q[i] = q
      self._prev_gto[i] = g
      self._have_prev[i] = True
      return
    at_home = np.max(np.abs(obs_i[0:7])) < self.reset_qpos_tol
    if at_home and self._started[i]:
      self._phase[i] = P_HOVER
      self._phase_steps[i] = 0
      self._ema_ok[i] = False
      self._integ[i] = 0.0
      self._started[i] = False
      self._on_reset(i)
    elif not at_home:
      self._started[i] = True

  def _on_reset(self, i: int) -> None:
    """Hook for subclasses with extra per-env latched state."""

  # -- floor guard -----------------------------------------------------------
  #
  # ``ee_ground_collision`` matches the whole link7 SUBTREE against the terrain and
  # terminates the episode outright. Without this guard the descent's overshoot
  # tripped it constantly: 68 terminations in 400 steps across 8 envs, and no
  # environment ever advanced past the descend phase.
  #
  # THE OLD 0.022 WAS BASED ON A WRONG CONSTANT and was itself a major bug. It came
  # from "the lowest geom sits 1.24cm below the site", which counted the whole link7
  # subtree -- but most of that subtree is contype=conaffinity=0, i.e. visual only.
  # Over the geoms that can ACTUALLY collide (``hand_capsule`` and the two finger
  # pads) the deepest is a finger pad at 1.38cm below the site.
  #
  # (An earlier note here claimed the hand capsule reaches ~3.1cm below the site.
  # That figure is ``geom_rbound`` -- a bounding SPHERE, which for a capsule mounted
  # above the site vastly overstates its downward reach. Projecting the capsule's
  # true half-extent onto world z puts its lowest point 1.69cm ABOVE the site, so it
  # cannot touch the floor before the pads do. Use true extents, not rbound: the same
  # trap produced phantom "buried in the floor" readings in the workspace audit.)
  #
  # So 0.022 left almost no margin and
  # ``ee_ground_collision`` fired on transient dips -- which silently auto-resets the
  # env WITHOUT telling the harness, so the state machine kept marching through
  # phases that no longer matched the world (instrumented elsewhere in this package
  # at 45 collisions in 600 steps x 8 envs, 34 of them during descent).
  #
  # This mattered more than any strategy change measured in this file: on the sibling
  # reorient task, 0.022 -> 0.030 alone moved success 0.125 -> 0.594.
  #
  # The site height is obtained by running FK on the arm's own joint angles:
  # ``q_abs = DEFAULT_QPOS + obs[0:9]`` (obs[0:9] is joint_pos_rel), which yields the
  # site position IN THE ROBOT BASE FRAME. That is env-local and therefore free of the
  # per-env scene-origin offset -- it is legal in a way that reading obs[25:28]
  # (absolute world gripper_pos) is not.
  floor_min_z = 0.030
  # CL-V3 (W1-G2), opt-in and default-off: guard the TRUE lowest collidable point of the
  # hand (pad corners + hand capsule, ``base.lowest_hand_z``) instead of the gripper site.
  # The two agree only when the hand is exactly vertical; the pad centres sit 0.0476 off
  # the site axis with the jaws open, so ~10 deg of wrist tilt drops the outer pad corner
  # from 11.9 mm below the site to 21 mm below it -- more than the whole clearance budget
  # of a floor-level grasp. Subclasses opt in with ``guard_mode = "hand"``.
  guard_mode = "site"
  hand_clearance = 0.006

  def _site_z(self, obs_i: np.ndarray) -> float:
    q_abs = self.default_qpos + obs_i[0:9]
    return float(self._fk(q_abs)[0][2])

  def _guard(self, err: np.ndarray, obs_i: np.ndarray) -> np.ndarray:
    """Clamp a commanded displacement so the hand never targets below the floor."""
    q_abs = self.default_qpos + obs_i[0:9]
    ee_pos, ee_rot = self._fk(q_abs)
    z = float(ee_pos[2])
    if self.guard_mode == "hand":
      low = lowest_hand_z(ee_pos, ee_rot, float(q_abs[7]))
      floor = self.hand_clearance + (z - low)  # equivalent site height for that clearance
    else:
      floor = self.floor_min_z
    if z + err[2] < floor:
      err = err.copy()
      err[2] = floor - z
    return err

  # CL-V3 (W1-G2, additive, default "ema" = unchanged): a RUNNING MEAN beats an EMA on a
  # static object by a large factor. ``gripper_to_object`` carries 10-13 mm of noise per
  # axis (W1-D2's measurement); an EMA at alpha 0.45 averages ~3 samples and still leaves
  # 6-7 mm of standard deviation, which is more than a 30 mm bore's whole clearance. But
  # the target is static in a KNOWN frame in every phase: before contact it does not move
  # in the WORLD, so averaging ``FK(q_obs) + gripper_to_object`` (the object's position in
  # the robot base frame) over every approach step collapses the error as 1/sqrt(n); after
  # the grasp it does not move in the HAND, so averaging ``gripper_to_object`` itself does
  # the same. The accumulators are restarted at the CLOSE boundary and on every rewind.
  gto_estimator = "ema"

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Smoothed gripper_to_object. Raw obs carries +-1cm uniform noise per axis;
    unsmoothed it makes the descent chatter and the grasp miss."""
    raw = obs_i[self.gto_slice]
    if self.gto_estimator == "mean":
      held = self._phase[i] >= P_CLOSE
      if held != bool(self._mean_held[i]):
        self._mean_sum[i] = 0.0
        self._mean_n[i] = 0
        self._mean_held[i] = held
      site = self._fk(self.default_qpos + obs_i[0:9])[0]
      sample = raw if held else site + raw
      self._mean_sum[i] += sample
      self._mean_n[i] += 1
      est = self._mean_sum[i] / self._mean_n[i]
      return est if held else est - site
    if not self._ema_ok[i]:
      self._ema[i] = raw
      self._ema_ok[i] = True
    else:
      self._ema[i] = self.ema_alpha * raw + (1 - self.ema_alpha) * self._ema[i]
    return self._ema[i]

  # -- the spine -------------------------------------------------------------

  def _target_error(self, i: int, obs_i: np.ndarray):
    """Public entry point: run the state machine, then apply the floor guard.

    Every subclass overrides ``_plan`` (or the ``_carry``/``_place`` hooks) rather
    than this, so the guard cannot be bypassed by accident.
    """
    err, rot, grip = self._plan(i, obs_i)
    return self._guard(np.asarray(err, dtype=np.float64), obs_i), rot, grip

  def _plan(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i)
    rot = self._approach_rot(i, obs_i)
    ph = self._phase[i]
    up = np.array([0.0, 0.0, 1.0])

    if self.grasp_style == "lift":
      # Per-env, per-call speed switch (base.py solves env i right after this
      # returns, so it never leaks into another env).
      if ph <= P_LIFT:
        self.max_dq = self.lift_max_dq
      elif ph in (P_CARRY, P_PLACE) and self.carry_max_dq is not None:
        self.max_dq = self.carry_max_dq
      else:
        self.max_dq = self._base_max_dq
      if ph <= P_LIFT:
        return self._plan_lift_grasp(i, obs_i, rot)

    if ph == P_CARRY and self.carry_timeout and self._phase_steps[i] > self.carry_timeout:
      # Unconditional escape (phase-1 M3): with nothing in hand ``object_to_goal``
      # never changes and the carry exit can never fire.
      self._rewind(i)
      ph = self._phase[i]

    if ph == P_DONE and self.retry_from_done:
      if (
        self._phase_steps[i] > 20
        and self._attempts[i] < self.max_attempts
        and not self._placed_ok(i, obs_i)
      ):
        self._rewind(i)
        ph = self._phase[i]

    if ph == P_HOVER:
      g = self._gripper_to_grasp(i, obs_i)
      err = g + up * self.hover_height
      if np.linalg.norm(err[:2]) < self.align_tol and self._phase_steps[i] > 4:
        self._phase[i] = P_DESCEND
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
      return err, rot, GRIPPER_OPEN

    if ph == P_DESCEND:
      g = self._gripper_to_grasp(i, obs_i)
      # Integrate the LATERAL error only. Integrating z as well winds up against the
      # floor guard (which clamps the commanded height but cannot clamp the
      # integrator), leaving a stored downward bias that then drives the wrist into
      # the ground the moment the guard releases. The vertical steady-state offset is
      # small anyway, and ``descend_tol`` plus the phase timeout absorb it.
      lateral = np.array([g[0], g[1], 0.0])
      self._integ[i] = np.clip(
        self._integ[i] + self.descend_integ_gain * lateral,
        -self.descend_integ_clip,
        self.descend_integ_clip,
      )
      if np.linalg.norm(g) < self.descend_tol or self._phase_steps[i] > 60:
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
      return g + self._integ[i], rot, GRIPPER_OPEN

    if ph == P_CLOSE:
      # Hold the pose exactly (zero position error) while the fingers squeeze.
      # Any residual command here drags the object out from between the pads.
      #
      # ``grip_close_action`` (default GRIPPER_CLOSED, i.e. "fully closed"): see the
      # W2-c retention finding in P_LIFT below -- commanding fully-closed keeps the
      # actuator driving hard against the object indefinitely, past first contact.
      if self._phase_steps[i] >= self.close_steps:
        self._phase[i] = P_LIFT
        self._phase_steps[i] = 0
      return np.zeros(3), rot, self.grip_close_action

    if ph == P_LIFT:
      # OPTIONAL lift ramp, DISABLED BY DEFAULT (``lift_ramp_frac = 0.0``), kept
      # only as a tunable because the diagnosis behind it is solid even though the
      # remedy did not pay.
      #
      # The original diagnosis (right symptom, wrong cause -- see below): losing the
      # object HERE is this spine's dominant failure. Instrumented over 32
      # episode-instances on both cube dependents, Stack lost the cube in 22/32 runs
      # (13 of them inside this phase) and place-in-container in 19/32 (12 here); of
      # the 10 Stack runs that never dropped it, 9 succeeded. So retention, not
      # placement precision, is what these tasks' success rates are mostly made of.
      #
      # THE REMEDY DID NOT FOLLOW: ramping the command (with a longer squeeze and a
      # deeper grasp) was measured over 96 episode-instances against a matched
      # 96-instance baseline and made BOTH tasks worse: Stack 0.354 -> 0.292,
      # place-in-container 0.302 -> 0.188. It also silently broke peg-insertion
      # when the ramp was an absolute step count rather than a fraction (peg
      # overrides ``lift_steps = 20``, so a fixed 14-step ramp left it at full
      # height for six steps and it never cleared the board: 0.03-0.06 -> 0.000).
      # Hence the ramp is off and the fraction form is retained so that no
      # subclass can be starved if anyone re-enables it.
      #
      # W2-c (2026-09-09), instrumented gripper/aperture/force traces, not just phase
      # counts: the ORIGINAL DIAGNOSIS NAMED THE RIGHT SYMPTOM AND THE WRONG PHASE.
      # The lift jerk is not the trigger. Per-step traces (gripper-object xy offset,
      # finger aperture, actuator8 force) across P_CLOSE through P_LIFT show that on
      # a large fraction of runs the object is EJECTED SIDEWAYS DURING P_CLOSE
      # ITSELF, before any lift command is ever issued -- aperture collapses
      # monotonically from the object's true contact width (~0.04) all the way to
      # ~0.00 (fully closed on nothing) while the gripper-object xy offset grows in
      # lockstep over the SAME several steps, both while ``P_CLOSE`` is still
      # commanding zero position error. Concretely (env 1 of one traced Stack-Cube
      # rollout): xy offset grew 0.007m -> 0.084m while aperture collapsed
      # 0.069 -> 0.005, entirely within P_CLOSE's fixed close_steps window; by the
      # time the phase-timer transitions to P_LIFT the object is often already gone,
      # and P_LIFT/P_CARRY only make the pre-existing loss OBSERVABLE (gripper-object
      # distance finally crosses a detection threshold) -- they are not what causes
      # it. This also explains, after the fact, why "longer squeeze" made things
      # worse above: more steps inside the very phase whose action ejects the object
      # is more exposure, not less; and why "ramped lift" made things worse: a slower
      # ramp cannot fix a loss that already happened one phase earlier, and it
      # prolongs the window in which a still-slipping grasp finishes escaping. The
      # applied force during this ejection is NOT small (actuator8 typically shows
      # 5-12N, decaying roughly with the aperture itself since it is a tendon-length
      # P-servo, not an independently regulated grip force) -- both successful holds
      # and ejections show comparable peak force, so the discriminator is whether the
      # aperture STABILIZES against the object (success) or is driven straight
      # through it to fully-closed (ejection), not how hard the actuator pushes.
      #
      # FIX ATTEMPTED: ``grip_close_action`` (see P_CLOSE) commands a PARTIAL close
      # instead of fully-closed, so the actuator's steady-state target sits near the
      # object's true contact width instead of continuing to drive past it once
      # contact is made. See StackObjectClassicalPolicy / PlaceInContainerClassicalPolicy
      # for the measured result and whether it was kept.
      ramp = max(1.0, self.lift_ramp_frac * self.lift_steps)
      frac = min(1.0, (self._phase_steps[i] + 1) / ramp)
      if self._phase_steps[i] >= self.lift_steps:
        self._phase[i] = P_CARRY
        self._phase_steps[i] = 0
      return up * (self.lift_height * frac), rot, self.grip_close_action

    if ph == P_CARRY:
      return self._carry(i, obs_i, rot)

    if ph == P_PLACE:
      return self._place(i, obs_i, rot)

    if ph == P_RELEASE:
      if self._phase_steps[i] >= self.release_steps:
        self._phase[i] = P_RETREAT
        self._phase_steps[i] = 0
      return np.zeros(3), rot, GRIPPER_OPEN

    if ph == P_RETREAT:
      if self._phase_steps[i] >= 30:
        self._phase[i] = P_DONE
        self._phase_steps[i] = 0
      return up * self.retreat_height, rot, GRIPPER_OPEN

    return up * self.retreat_height, rot, GRIPPER_OPEN

  # -- CL-V3 lift-style grasp (opt-in, see grasp_style) -----------------------

  def _placed_ok(self, i: int, obs_i: np.ndarray) -> bool:
    """Is the object at the goal? Default: 3-D object_to_goal within 3.5 cm."""
    return bool(np.linalg.norm(obs_i[self.o2g_slice]) < 0.035)

  def _plan_lift_grasp(self, i: int, obs_i: np.ndarray, rot):
    """lift_object.py's HOVER/DESCEND/CLOSE/LIFT, on this spine's phase numbers.

    Heights are ABSOLUTE site heights from FK on the arm's own joints (robot-base
    frame, so offset-free); xy comes from the (smoothed) gripper_to_grasp vector.
    """
    ph = self._phase[i]
    g = self._gripper_to_grasp(i, obs_i)
    z = self._site_z(obs_i)

    if (
      ph in (P_HOVER, P_DESCEND)
      and self.lift_integ_gain > 0.0
      and np.linalg.norm(g[:2]) < self.lift_integ_near
    ):
      self._lift_integ[i] = np.clip(
        self._lift_integ[i] + self.lift_integ_gain * np.array([g[0], g[1], 0.0]),
        -self.lift_integ_clip,
        self.lift_integ_clip,
      )

    if ph == P_HOVER:
      err = np.array([g[0], g[1], self.lift_hover_site_z - z])
      if np.linalg.norm(err[:2]) < self.lift_align_tol and abs(err[2]) < 0.05:
        self._settle[i] += 1
        if self._settle[i] >= self.lift_align_settle:
          self._phase[i] = P_DESCEND
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      if self._phase_steps[i] > self.lift_align_timeout:
        self._phase[i] = P_DESCEND
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return err + self._lift_integ[i], rot, GRIPPER_OPEN

    if ph == P_DESCEND:
      err = np.array([g[0], g[1], self.lift_grasp_site_z - z])
      z_err = err[2]
      err[2] = max(z_err, -self.lift_descent_rate)
      seated = abs(z_err) < self.lift_seat_tol and np.linalg.norm(err[:2]) < self.lift_seat_xy_tol
      if seated and self.close_orient_tol > 0.0 and self.yaw_align:
        _, r_cur = self._fk(self.default_qpos + obs_i[0:9])
        cur = float(np.arctan2(r_cur[1, 1], r_cur[0, 1]))
        tgt = float(self._yaw_target[i]) if not np.isnan(self._yaw_target[i]) else cur
        # The closing axis is a LINE, so the alignment error is mod pi.
        d_ang = (cur - tgt + np.pi / 2) % np.pi - np.pi / 2
        seated = abs(d_ang) < self.close_orient_tol
      if seated:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = P_CLOSE
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > self.lift_descend_timeout:
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return err + self._lift_integ[i], rot, GRIPPER_OPEN

    if ph == P_CLOSE:
      # Actively HOLD the grasp height and creep xy at half gain while squeezing
      # (commanding zero error lets the arm sag under gravity).
      k = self.lift_close_xy_gain
      err = np.array([k * g[0], k * g[1], self.lift_grasp_site_z - z])
      grip = self.grip_close_action
      if self.lift_close_ramp > 0:
        frac = min(1.0, (self._phase_steps[i] + 1) / float(self.lift_close_ramp))
        grip = GRIPPER_OPEN + (self.grip_close_action - GRIPPER_OPEN) * frac
      if self._phase_steps[i] >= self.lift_close_steps:
        self._phase[i] = P_LIFT
        self._phase_steps[i] = 0
      return err + self._lift_integ[i], rot, grip

    # P_LIFT: climb straight up, then check the object came along.
    climbing = self._phase_steps[i] < self.lift_climb_steps and (
      self.lift_climb_to is None or z < self.lift_climb_to
    )
    if climbing:
      return np.array([0.0, 0.0, self.lift_climb_err]), rot, self.grip_close_action
    if np.linalg.norm(g) > self.lift_held_tol:
      self._rewind(i)
      return np.array([g[0], g[1], self.lift_hover_site_z - z]), rot, GRIPPER_OPEN
    self._phase[i] = P_CARRY
    self._phase_steps[i] = 0
    self._integ[i] = 0.0
    return self._carry(i, obs_i, rot)

  # -- ending (subclasses specialise these two) ------------------------------

  def _carry(self, i, obs_i, rot):
    """Translate so the held object ends up above the drop point.

    INTEGRAL ACTION, not plain proportional. The DLS solve has a large lateral
    steady-state bias -- measured across the grasp envelope by iterating this repo's
    own ``_ik_step`` to convergence, it is 2.3cm mean / 3.1cm max even with yaw free
    (axis-only orientation target), and 2-3.6cm with yaw constrained. Every placement
    tolerance in this file's tasks is at or below that: Stack is 3cm, peg insertion
    1.5cm. So a proportional-only carry stalls at an offset bigger than the target and
    the task can never succeed -- the arm parks next to the base and waits out the
    episode. The integrator is what actually closes that gap.
    """
    d = self._grasp_to_drop(i, obs_i)
    lateral = np.array([d[0], d[1], 0.0])
    self._integ[i] = np.clip(
      self._integ[i] + self.carry_integ_gain * lateral,
      -self.carry_integ_clip,
      self.carry_integ_clip,
    )
    err = d + self._integ[i] + np.array([0.0, 0.0, self.hover_height])
    if np.linalg.norm(d[:2]) < self.carry_tol and self._phase_steps[i] > 10:
      self._phase[i] = P_PLACE
      self._phase_steps[i] = 0
      # The integrator is deliberately CARRIED OVER into the place phase: it is
      # holding out the same bias, and zeroing it lets the object drift straight back
      # off-target during the descent.
    return err, rot, self.grip_close_action

  def _place(self, i, obs_i, rot):
    """Lower until the object sits on the drop point, then release."""
    d = self._grasp_to_drop(i, obs_i)
    lateral = np.array([d[0], d[1], 0.0])
    self._integ[i] = np.clip(
      self._integ[i] + self.carry_integ_gain * lateral,
      -self.carry_integ_clip,
      self.carry_integ_clip,
    )
    if np.linalg.norm(d) < self.place_tol or self._phase_steps[i] > 70:
      self._phase[i] = P_RELEASE
      self._phase_steps[i] = 0
    return d + self._integ[i], rot, self.grip_close_action


class StackObjectClassicalPolicy(StagedStackPolicy):
  """Grasp safely, transport above the base, then lower and release in place."""

  legacy_layout = "stack"
