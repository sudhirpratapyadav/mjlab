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

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

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
  # Vertical offset added to the grasp point. In a top-down pose the lowest robot
  # COLLISION geom (a fingertip pad) sits 1.24cm below the ``gripper`` site, and
  # ``ee_ground_collision`` terminates the episode on any robot/ground contact. So
  # grasping a ground-resting object exactly at its centre leaves under a centimetre
  # of clearance, and the descent's overshoot eats it. Grasp slightly high instead.
  grasp_z_offset = 0.0

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
    if self.grasp_yaw is None:
      return _DOWN_AXIS
    return down_frame(self.grasp_yaw)

  # -- state -----------------------------------------------------------------

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_ok = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._anchor = np.zeros((self.num_envs, 3))
      self._started = np.zeros(self.num_envs, dtype=bool)
    else:
      self._ema_ok[env_ids] = False
      self._integ[env_ids] = 0.0
      self._anchor[env_ids] = 0.0
      self._started[env_ids] = False

  # -- auto-reset detection --------------------------------------------------
  #
  # These envs terminate EARLY and often: ``ee_ground_collision`` fires whenever any
  # robot geom touches the ground plane (and the lowest fingertip geom is only 1.24cm
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

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
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
  # terminates the episode outright. Measured in a top-down pose, the lowest geom in
  # that subtree (a fingertip pad, open or closed) sits 1.24cm below the ``gripper``
  # site, so the site must never be driven below ~0.02 to have any margin. Without
  # this guard the descent's overshoot tripped it constantly: 68 terminations in
  # 400 steps across 8 envs, and no environment ever advanced past the descend phase.
  #
  # The site height is obtained by running FK on the arm's own joint angles:
  # ``q_abs = DEFAULT_QPOS + obs[0:9]`` (obs[0:9] is joint_pos_rel), which yields the
  # site position IN THE ROBOT BASE FRAME. That is env-local and therefore free of the
  # per-env scene-origin offset -- it is legal in a way that reading obs[25:28]
  # (absolute world gripper_pos) is not.
  floor_min_z = 0.022

  def _site_z(self, obs_i: np.ndarray) -> float:
    q_abs = self.default_qpos + obs_i[0:9]
    return float(self._fk(q_abs)[0][2])

  def _guard(self, err: np.ndarray, obs_i: np.ndarray) -> np.ndarray:
    """Clamp a commanded displacement so the site never targets below the floor."""
    z = self._site_z(obs_i)
    lowest = z + err[2]
    if lowest < self.floor_min_z:
      err = err.copy()
      err[2] = self.floor_min_z - z
    return err

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Smoothed gripper_to_object. Raw obs carries +-1cm uniform noise per axis;
    unsmoothed it makes the descent chatter and the grasp miss."""
    raw = obs_i[self.gto_slice]
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
      if self._phase_steps[i] >= self.close_steps:
        self._phase[i] = P_LIFT
        self._phase_steps[i] = 0
      return np.zeros(3), rot, GRIPPER_CLOSED

    if ph == P_LIFT:
      if self._phase_steps[i] >= self.lift_steps:
        self._phase[i] = P_CARRY
        self._phase_steps[i] = 0
      return up * self.lift_height, rot, GRIPPER_CLOSED

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
    return err, rot, GRIPPER_CLOSED

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
    return d + self._integ[i], rot, GRIPPER_CLOSED


class StackObjectClassicalPolicy(GraspTransportPolicy):
  """Grasp the cube top-down, carry it over the cuboid base, lower and release.

  The 4cm cube is grasped at its centre (site == grasp point, see module docstring)
  and the drop target is ``object_to_goal``, which already points from the cube's
  centre to base_pos + 0.035 — i.e. the pose the success test wants. The only
  subtlety is the RELEASE: success needs |z| < 0.02 of that height while the object
  is free, so the fingers must open and the arm must retreat, otherwise the cube is
  held a finger-width high forever and the height test never fires.
  """

  gto_slice = slice(37, 40)
  o2g_slice = slice(40, 43)

  # The cube is 4cm; the base cuboid top is only 3cm above the ground. Keep the
  # carry altitude modest so lowering onto a 3cm-tall target is a short move, but
  # high enough to clear the base's 8x8cm footprint on the way in.
  hover_height = 0.11
  lift_height = 0.14
  align_tol = 0.02
  descend_tol = 0.010
  carry_tol = 0.018
  # Release slightly HIGH rather than pressing down: the cube's own weight seats it
  # and pressing pushes the base out from under it (both objects are free bodies).
  place_tol = 0.014
  # Grasp 1cm above the cube's centre. The 4cm cube spans z 0.00-0.04 and the pads
  # are 1.65cm tall, so pads at 0.018-0.034 still bite the upper half of the cube
  # while clearing the ground by 1.8cm -- enough that the descent's overshoot does
  # not trip ``ee_ground_collision``.
  grasp_z_offset = 0.010
