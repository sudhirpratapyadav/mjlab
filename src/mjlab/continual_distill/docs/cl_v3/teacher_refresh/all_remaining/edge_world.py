"""Scripted EdgeGrasp teacher policy (CL-V3 rewrite, W1-G, 2026-09-09).

THE TRAP THIS TASK IS BUILT AROUND
-----------------------------------
The plate is UNGRASPABLE where it spawns: a 150 mm round ceramic plate (collision:
cylinder r 0.075, half-height 0.0096, 0.18 kg, friction 0.5) lying flat on a 200 x 260 x
100 mm oak riser. Both in-plane widths are nearly twice the 0.08 m aperture and there is
no finger clearance under the rim. The only graspable configuration is thickness-wise
at an OVERHANG: push the plate over the riser's robot-facing edge, then pinch the exposed
rim across its 19 mm thickness and lift.

GEOMETRY (ledge.xml / plate.xml, measured)
-------------------------------------------
The riser is a static MOCAP body written per-env by ``EdgeGraspCommand``: box half
extents 0.10 (x) x 0.13 (y) x 0.05 (z), top at ``ledge_top_height`` = 0.10. CL-V3: the
riser has a yaw band of +-0.26 rad and the plate spawns on its top at RISER-frame
offsets rel_x in +-0.02, rel_y in +-0.05 (rotated with the riser); the plate's own yaw
is +-pi (a round plate: irrelevant). The robot faces the riser's -x face; pushing the
plate along the riser's -x axis creates the overhang.

RECOVERING THE RISER FRAME FROM RELATIVE OBSERVATIONS ONLY (60-D layout)
-------------------------------------------------------------------------
    [34:40] object_orientation   = plate rotation matrix rows 1,2  (obs[34:37] = row 1)
    [40:43] gripper_to_object    = plate - gripper
    [43:46] object_to_goal       = target - plate,  target = ledge + R(psi) @ GOAL_OFFSET
    [46:52] goal_orientation_diff = goal rows 0,1 - plate rows 0,1  (obs[49:52] = row 1 diff)
The goal marker is written with the RISER's quaternion, so
    riser row 1 = obs[49:52] + obs[34:37] = (sin psi, cos psi, 0)  ->  psi = atan2(r[0], r[1])
and then
    ledge_rel = o2g + gto - R(psi) @ GOAL_OFFSET = ledge - gripper       (offset-free)
    s_plate   = ((gto - ledge_rel) . x_r)                                  riser-frame x of the plate centre
    overhang  = -LEDGE_HALF_X - (s_plate - PLATE_R) = -0.025 - s_plate   how far the plate's near
                                                                         extreme sticks out past the edge
Absolute terms (object_pos, gripper_pos) are never used. The site height comes from FK
on the arm's own joints (robot-base frame = env-local).

STRATEGY
--------
1. PUSH (top-down, fingers CLOSED as one pad): contact the plate's far rim on the +x_r
   side, at the plate's mid-height (site 5 mm above the plate centre so the pad clears the
   riser top), creep along -x_r until ``overhang`` >= OVERHANG_TARGET (48 mm, well under
   the 75 mm half-length so the centre of mass stays on the riser).
2. RETREAT +x_r and up, still top-down (reorienting near the plate sweeps the fingertip
   through it -- phase-1 finding, kept).
3. SIDE pinch. The closing axis (site y) is VERTICAL; the approach axis is horizontal.
   Phase-1 fixed the hand heading at world +x (hand pointing at the riser): that puts the
   flange inside the robot's base column and the DLS solve never converges (4-15 cm,
   15-43 deg, the "limit cycle"). Measured with this repo's own IK (probe over the spawn
   band): a heading rotated ALPHA = 70 deg from the riser's +x axis toward +-y_r, with the
   closing axis pointing DOWN (site y = -z; which pad is on top is irrelevant), converges
   to 1.3-3 cm / 1-3 deg at every pinch point, flange at radial 0.22-0.30. The residual is
   nulled by integral action on the insert. The heading is angled rather than pure +-y
   because the hand capsule (r 0.04, centred 0.07 behind the site) would otherwise sit on
   the plate's rim: angling it moves the capsule radially outside the plate.
   The pinch point is INSET = 10 mm inside the plate's near extreme along the outward
   normal n = -x_r, at the plate's own height; approach from STANDOFF along the heading
   with the jaws open (+-0.04 m vertically -- the 19 mm rim slides between them).
4. CLOSE on the rim (aperture stalls at ~0.019), verify the plate is still at the pinch,
   retry the insert once if not.
5. LIFT straight up (+0.10) and slightly outward (+0.03 n): the plate hangs from the
   pinch; success needs its centre > riser top + 0.04, latched.

SUCCESS (EdgeGraspCommand): plate site z - (riser z + 0.10) > 0.04 AND xy drift from the
riser centre < 0.40, latched. Budget 300 steps (6 s) -- every phase has an unconditional
timeout and the phase budget sums to ~230.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GOAL_OFFSET = np.array([-0.13, 0.0, 0.18])  # EdgeGraspCommandCfg.goal_offset (riser frame)
PLATE_R = 0.075
LEDGE_HALF_X = 0.10

# Phases.
P_PUSH_HOVER = 0
P_PUSH_DESCEND = 1
P_PUSH_ADVANCE = 2
P_RETREAT = 3
P_SIDE_HOVER = 4
P_SIDE_INSERT = 5
P_CLOSE = 6
P_LIFT = 7
# CL-V3 (W1-G, 2026-09-09): appended, not inserted, so the ``phase >= P_SIDE_HOVER``
# solver switch and every existing comparison keep working.
P_SIDE_DROP = 8
PHASE_NAMES = {
  0: "PUSH_HOVER", 1: "PUSH_DESCEND", 2: "PUSH_ADVANCE", 3: "RETREAT",
  4: "SIDE_HOVER", 5: "SIDE_INSERT", 6: "CLOSE", 7: "LIFT", 8: "SIDE_DROP",
}

# -- push --------------------------------------------------------------------
PAD_RADIUS = 0.010
STANDOFF = 0.006
BEHIND = PLATE_R + PAD_RADIUS + STANDOFF  # 0.091 along +x_r from the plate centre
PUSH_HOVER_HEIGHT = 0.12
PUSH_ALIGN_TOL = 0.035
PUSH_HOVER_TIMEOUT = 25
# CL-V3 (W1-G2) -- THE PUSHER WAS FLIPPING THE PLATE OFF THE RISER, and that single event
# is both failure classes AND (embarrassingly) the entire score. MEASURED (diagnose, the
# `ik_joint_limit_factor` tree, n = 32): every one of the 12 successes latches during
# PUSH_ADVANCE (7), PUSH_HOVER (2), RETREAT (2) or SIDE_HOVER (1) -- i.e. the plate is
# levered up past riser_top + 0.04 by the closed gripper, never by the pinch; and 15 of
# the 20 failures are `ee_ground_collision` in CLOSE with the plate logged at z
# 0.004-0.047, i.e. the same lever that did not throw it high enough simply dropped it on
# the floor, where the side approach chases it down.
# Geometry: at 0.006 the closed pad spans plate_centre - 0.006 .. + 0.010, so its BOTTOM
# EDGE sits 6 mm below the plate's centre and only 4 mm above the riser's top face -- one
# servo dip and it is under the plate's rim, prying it up. 0.012 puts the pad bottom
# exactly at the plate's mid-height (9.6 mm of engagement below the plate's top face,
# 9.6 mm of clearance above the riser) so it can only ever push.
PUSH_SITE_ABOVE_PLATE = 0.006
FAR_EDGE_MARGIN = 0.014  # keep the pushing pad ON the riser top: beyond the far edge it sinks
# below the rim and flicks the plate up from underneath (diagnose #6, envs 1/31)
# CL-V3 (W1-G2) -- THE DESCENT IS WHAT FLINGS THE PLATE. Trace (diagnose, the fixed-push
# tree): in 17 of the 22 envs that lose the plate, it is already at z 0.112-0.152 (its
# resting height is 0.110) by the time RETREAT starts, and at the first step of
# PUSH_ADVANCE the gripper SITE is logged 13-20 mm BELOW the plate's centre -- i.e. the
# closed pad has descended past the plate's bottom face (half-thickness 9.6 mm) and the
# advance then wedges it under the rim and pries the plate up and off the riser. That
# single event is both the ground-collision failures (the side approach then chases a
# plate lying on the floor) AND, when it throws the plate high enough, every "success"
# this teacher had. At 0.03 m/step the descent covers the last 3 cm in one step with a
# 0.008 seat tolerance -- it cannot stop on a 19 mm-thick target. 0.010/step with a
# two-step seat settle can.
PUSH_DESCENT_RATE = 0.030
PUSH_SEAT_SETTLE = 2
PUSH_SEAT_TOL = 0.008
PUSH_DESCEND_TIMEOUT = 30
# CL-V3 (W1-G2) -- OVER-PUSHING IS THE DOMINANT FAILURE NOW. diagnose #9: 15 of 20
# first-terminations are ``ee_ground_collision`` in CLOSE, and the trace shows the PLATE
# at z 0.008-0.073 (its height on the riser is 0.110) in every one of them -- i.e. the
# plate had been shoved clean off the riser, the side phases then tracked it down to the
# floor (their target is the plate's own height) and the wrist followed it into the
# ground. The plate tips when its centre passes the riser's edge, i.e. at overhang
# 0.075 (centre 0.10 from the riser centre); the pad stops but the plate keeps sliding,
# so the exit has to leave that much margin. 0.028 exit / 0.042 target / 7 mm max step,
# plus a hard abort at 0.065 that goes straight to RETREAT.
# NEGATIVE RESULT (W1-G2, n=64): pushing LESS (target 0.042 / exit 0.028 / 7 mm step)
# measured 0.719 -> 0.453. With a 28 mm overhang the pinch point (INSET inside the near
# extreme) sits ~10 mm outboard of the riser's front face and the lower jaw has nowhere
# to go. The over-push is real but the cure is not a shorter push -- it is not CHASING
# the plate once it has left the riser (see PLATE_ON_RISER_MIN below).
OVERHANG_TARGET = 0.048
OVERHANG_EXIT = 0.036  # stop early: the plate keeps sliding after the pad stops
OVERHANG_ABORT = 0.070  # beyond this the plate is about to tip off the riser
PLATE_ON_RISER_MIN = 0.085  # plate centre height below which it has left the riser
PUSH_GAIN = 0.5
PUSH_STEP_MIN = 0.004
PUSH_STEP_MAX = 0.010  # MEASURED (CPU trace): at 0.02/step with the 0.12 lead the plate left the riser at ~1 m/s
PUSH_LEAD = 0.03  # command lead while pushing: a bounded push force (the wound-up lead is
# dropped at the push -> retreat transition; diagnose #6: the surge shoved the plate 4-7 cm)
MIN_BEHIND = 0.014
RESEAT_STEPS = 8
PUSH_TIMEOUT = 40

# -- retreat -------------------------------------------------------------------
RETREAT_STEPS = 18
RETREAT_UP_STEPS = 8
RETREAT_UP = 0.12
RETREAT_BACK = 0.04  # first: up and a little +x_r so the pad leaves the plate
# then TOWARD the robot past the near edge at altitude: the side hover must descend over
# free floor. MEASURED (diagnose #4, 25/32): hovering down to z 0.13 while still above the
# riser parks the horizontal hand's capsule (10 cm below the site) on the riser top.
RETREAT_OUT = 0.26  # along -x_r from the retreat start

# -- side pinch ------------------------------------------------------------------
ALPHA = np.radians(70.0)  # heading angle from +x_r toward +-y_r
INSET = 0.018  # pinch point this far inside the plate's near extreme (0.010 left the pads
# on the very edge once the insert's 1 cm radial tolerance was spent: 9/22 failures in
# diagnose #8 closed on NOTHING, final aperture 0.000)
SIDE_STANDOFF = 0.07  # start of the insert, behind the pinch along the heading
# Site height above the plate centre in the side phases. MEASURED (diagnose, 32 envs):
# at the plate's own height the hand capsule's bottom (site - 0.10 with the closing axis
# vertical) is 1.2 cm off the floor and 24/30 failures were ee_ground_collision in
# SIDE_HOVER/INSERT. 0.018 puts it at 3 cm; the open jaws (+-0.04) still straddle the
# 19 mm rim, and CLOSE ramps the site down to SIDE_Z_CLOSE so the pads meet the rim
# nearly symmetrically.
SIDE_Z_ABOVE = 0.020  # with the 20 deg pitch the capsule bottom is site - 0.072
SIDE_Z_CLOSE = 0.005  # symmetric closure on the rim
SIDE_FLOOR_MIN_Z = 0.105  # capsule bottom >= 3.3 cm
SIDE_HOVER_TOL = 0.05
SIDE_HOVER_COS = np.cos(np.radians(20.0))  # approach axis within 20 deg of the heading
SIDE_HOVER_TIMEOUT = 30
SIDE_INSERT_RADIAL_TOL = 0.008  # along the outward normal: what the pinch bite needs
SIDE_INSERT_Z_TOL = 0.020
SIDE_INSERT_TIMEOUT = 32

# CL-V3 (W1-G, 2026-09-09) -- APPROACH AT ALTITUDE, THEN DROP STRAIGHT DOWN.
# MEASURED (diagnose #7, n = 32): 11 of 21 failures are ``ee_ground_collision`` with the
# gripper SITE logged at z 0.075-0.11 during SIDE_HOVER -- BELOW ``SIDE_FLOOR_MIN_Z``
# (0.105) even though the guard was commanding it UP by 5 cm at the time. The old
# SIDE_HOVER asked for everything at once: reorient the wrist ~90 deg (top-down ->
# horizontal), translate ~0.26 m out past the near edge, AND descend ~0.12 m, all in one
# waypoint. The DLS trades position against orientation on that combined error and the
# site sags through the floor on the way, exactly the Stack-Cube lesson (settle the xy at
# an ABSOLUTE altitude first, then descend vertically at a bounded rate).
# So: SIDE_HOVER now holds an absolute site height and only fixes xy + orientation;
# SIDE_DROP then descends vertically with xy frozen on the standoff point; and every side
# phase gets an unconditional height-recovery override (if the site is below the floor,
# command PURE +z and nothing else until it is back).
# Phase budget (300-step episode): 25 + 25 + 50 + 18 + 30 + 28 + 32 + 14 = 222, leaving
# ~78 for the up-to-two aperture-checked insert retries and the lift. Successful runs
# finish at 95-127 steps; the timeouts only bind for the strugglers, and every step the
# front half wastes is a retry the back half cannot afford.
SIDE_APPROACH_Z = 0.26  # absolute site height for the reorientation + xy approach
# CL-V3 (W1-G2): 0.012 was NOT a 12 mm/step descent. W1-D measured that the solver
# realises only ~40 % of a rate-limited position request per control step, so the drop
# from SIDE_APPROACH_Z (0.26) to the pinch height (~0.130) took ~27 steps against a
# 28-step timeout -- i.e. SIDE_DROP was timing out with the site still 1-3.5 cm high, and
# the trace confirms it: at CLOSE entry, in the envs where the plate is still on the
# riser, ``gripper_to_object`` z is -0.011 to -0.035 where it should be -0.005. With the
# jaws open (+-0.0476 along the closing axis) that puts the LOWER pad at the plate's
# bottom face and the upper pad 8 cm above it, so the close bites nothing -- which is
# exactly what the aperture check reports (0.000-0.003 on every LIFT-phase failure).
SIDE_DROP_RATE = 0.030
SIDE_DROP_TOL = 0.012
SIDE_DROP_TIMEOUT = 35
# Height recovery. CL-V3 (W1-G2): the recovery used to slam the site back to
# SIDE_APPROACH_Z (0.26), which UNDOES the whole descent -- and with the trigger at 0.10
# against a pinch height of ~0.130 and a 0.105 guard, a single dip re-ran the approach and
# the env timed out in SIDE_DROP/SIDE_HOVER (17 of 45 failures ended in those two phases).
# Trigger lower (at 0.085 the hand capsule's lowest point is still 1.3 cm off the floor)
# and recover only to just above the guard, not all the way back up.
SIDE_RECOVER_Z = 0.085  # below this the side phases do nothing but climb
SIDE_RECOVER_TO = 0.125
SIDE_CLOSE_FLOOR_Z = 0.100  # floor for CLOSE/LIFT (site sits at the plate's height there)
SIDE_INSERT_RATE = 0.015
INSERT_INTEG_GAIN = 0.15
INSERT_INTEG_CLIP = 0.04
CLOSE_STEPS = 14
# The rim is 19 mm thick, so a pinch that has the plate between the pads stalls the
# aperture near 0.019; an empty close runs to ~0.000. Below this the close bit nothing.
AP_HELD_MIN = 0.012
MAX_INSERT_RETRIES = 2
HELD_TOL = 0.03
LIFT_UP = 0.10
# The plate's centre is still ~27 mm INBOARD of the riser edge at a 48 mm overhang, so a
# lift that is mostly vertical only levers the near rim up and PIVOTS the plate about the
# edge -- the centre (which is what the success test measures) goes DOWN. The lift has to
# drag the plate off the edge first, so the outward component is the load-bearing one.
# NEGATIVE RESULT (W1-G2), measured twice: LIFT_OUT 0.07 gave 0.531 -> 0.391 and, after
# the push was fixed, 0.055 gave 0.484 -> 0.453. The geometric argument for a bigger
# outward pull (the pinch is 30 mm outboard of the riser edge and the plate's centre 27 mm
# inboard, so a vertical lift pivots the plate and its CENTRE goes down) is sound, but it
# is not what is limiting this task.
LIFT_OUT = 0.03

EMA_ALPHA = 0.4
FLOOR_MIN_Z = 0.030
# Posture seeds for the side pinch (this repo's DLS, solved offline over the spawn band,
# std <= 0.23 rad per joint): from the retreat pose the unbiased solve parks 24 cm on
# the wrong side of the plate (an M4 posture trap, measured in diagnose #3: 25/32).
# Same fix as Axial-Extract: bias the null-space toward the known-good branch.
SIDE_POSTURE_POS = np.array([-0.47, 0.39, -0.13, -2.81, 1.35, 1.52, -0.85])  # heading toward +y_r
SIDE_POSTURE_NEG = np.array([0.45, 0.38, 0.11, -2.83, -1.40, 1.53, -0.71])  # heading toward -y_r
SIDE_POSTURE_WEIGHT = 0.03
# Once the arm is already ON the right branch the bias is no longer selecting anything --
# it is only adding steady-state error, and the insert is the one phase where millimetres
# matter (the pads have to straddle a 19 mm rim). W1-M measured the posture regulariser
# leaving 28-60 mm of residual at weight 0.005 against 4-11 mm at 0.0005, so the branch
# weight is held only through SIDE_HOVER/SIDE_DROP and dropped for SIDE_INSERT/CLOSE.
SIDE_POSTURE_WEIGHT_FINE = 0.002
RESET_Q_JUMP = 0.25
RESET_GTO_JUMP = 0.15


def _yaw_frame(psi: float):
  xr = np.array([np.cos(psi), np.sin(psi), 0.0])
  yr = np.array([-np.sin(psi), np.cos(psi), 0.0])
  return xr, yr


SIDE_PITCH = np.radians(20.0)


def _side_frame(heading: np.ndarray, pitch: float = SIDE_PITCH) -> np.ndarray:
  """Approach axis (site z) = ``heading`` pitched ``pitch`` DOWNWARD; closing axis (site y)
  = down, tilted accordingly. The pitch lifts the hand capsule (0.07 behind the site,
  axis along the closing direction): its lowest point moves from site - 0.10 to
  site - 0.072, which lets the site sit at the plate's mid-height for a symmetric
  closure. MEASURED (diagnose #5): 15/32 ground terminations in CLOSE because the bottom
  pad met the rim first, levered the plate and pushed the hand 2 cm down."""
  ez0 = np.array([heading[0], heading[1], 0.0])
  ez0 = ez0 / (np.linalg.norm(ez0) + 1e-9)
  ey0 = np.array([0.0, 0.0, -1.0])
  ex = np.cross(ey0, ez0)
  ez = np.cos(pitch) * ez0 + np.sin(pitch) * ey0
  ey = np.cos(pitch) * ey0 - np.sin(pitch) * ez0
  return np.column_stack([ex, ey, ez])


class EdgeGraspClassicalPolicy(ClassicalPolicyBase):
  """Push the plate to an overhang, angled side pinch on the rim, lift."""

  DEFAULT_QPOS = HOME_QPOS  # edge_grasp uses get_franka_robot_cfg (home)
  max_dq = 0.10
  orientation_weight = 0.3
  PHASE_NAMES = PHASE_NAMES
  # Command lead (base.py): gravity-sag compensation + a sustained push force
  # (measured, see stack_object.GraspTransportPolicy.lead).
  cmd_lead_max = 0.12
  # CL-V3 (W1-G2): ``cmd_ref`` is switched PER PHASE in ``_target_error``. "command"
  # builds the z target from FK(previous command), which is what sustains the pushing
  # force -- and which makes ``_guard`` inert, because the guard clamps ``err`` against
  # the ACTUAL site height while the target is built from a command that may already be
  # 0.12 rad (~6 cm) below it (this is exactly what put Tool-Pull's wrist through the
  # floor 32/32 times). The push phases need the force; the side phases need the guard.
  # A/B at n = 32: "command" everywhere 0.594, "actual" everywhere 0.312.
  cmd_ref = "command"
  cmd_ref_axes = (2,)  # z only: the full command reference made xy orbit (CPU trace)
  # W1-M's finding: the env applies ``soft_joint_pos_limit_factor`` 0.9 to the Franka but
  # the IK never sees it, so the solve can converge on a pose the actuators will not hold
  # -- which on a horizontal-wrist pinch at the edge of the workspace is most of the
  # remaining failure. MEASURED here at n = 64: 0.531 -> 0.609.
  ik_joint_limit_factor = 0.9

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_ok = np.zeros(self.num_envs, dtype=bool)
      self._psi_vec = np.zeros((self.num_envs, 2))
      self._prev_q = np.zeros((self.num_envs, 7))
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._have_prev = np.zeros(self.num_envs, dtype=bool)
      self._reseat = np.zeros(self.num_envs, dtype=np.int64)
      self._retry = np.zeros(self.num_envs, dtype=np.int64)
      self._integ = np.zeros((self.num_envs, 3))
      self._side_sign = np.zeros(self.num_envs)
    else:
      self._ema_ok[env_ids] = False
      self._have_prev[env_ids] = False
      self._reseat[env_ids] = 0
      self._retry[env_ids] = 0
      self._integ[env_ids] = 0.0
      self._side_sign[env_ids] = 0.0

  def _rewind(self, i: int) -> None:
    self._phase[i] = P_PUSH_HOVER
    self._phase_steps[i] = 0
    self._ema_ok[i] = False
    self._reseat[i] = 0
    self._retry[i] = 0
    self._integ[i] = 0.0
    self._side_sign[i] = 0.0

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    """Auto-reset = the arm teleports (> RESET_Q_JUMP rad on a joint in one step) or
    the plate re-spawns (> RESET_GTO_JUMP m). The old |joint_pos_rel| < 0.03 "at home"
    test cannot fire under the CL-V3 +-10 deg reset joint noise."""
    q = obs_i[0:7]
    g = obs_i[40:43]
    if self._have_prev[i]:
      if (
        np.max(np.abs(q - self._prev_q[i])) > RESET_Q_JUMP
        or np.linalg.norm(g - self._prev_gto[i]) > RESET_GTO_JUMP
      ):
        self._rewind(i)
    self._prev_q[i] = q
    self._prev_gto[i] = g
    self._have_prev[i] = True

  def _smooth(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not self._ema_ok[i]:
      self._ema[i] = raw
      self._ema_ok[i] = True
    else:
      self._ema[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema[i]
    return self._ema[i]

  def _smooth_o2g(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not hasattr(self, "_ema_o2g"):
      self._ema_o2g = np.zeros((self.num_envs, 3))
      self._ema_o2g_ok = np.zeros(self.num_envs, dtype=bool)
    if not self._ema_o2g_ok[i] or not self._ema_ok[i]:
      self._ema_o2g[i] = raw
      self._ema_o2g_ok[i] = True
    else:
      self._ema_o2g[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema_o2g[i]
    return self._ema_o2g[i]

  def _riser_yaw(self, i: int, obs_i: np.ndarray) -> float:
    row1 = obs_i[49:52] + obs_i[34:37]  # riser rotation matrix row 1 = (sin, cos, 0)
    v = np.array([row1[0], row1[1]])
    n = np.linalg.norm(v)
    if n > 1e-6:
      v = v / n
      if not self._ema_ok[i]:
        self._psi_vec[i] = v
      else:
        self._psi_vec[i] = 0.3 * v + 0.7 * self._psi_vec[i]
    return float(np.arctan2(self._psi_vec[i][0], self._psi_vec[i][1]))

  def _site(self, obs_i: np.ndarray):
    return self._fk(self.default_qpos + obs_i[0:9])[0]

  def _guard(self, err: np.ndarray, obs_i: np.ndarray, floor: float = FLOOR_MIN_Z) -> np.ndarray:
    z = float(self._site(obs_i)[2])
    if z + err[2] < floor:
      err = err.copy()
      err[2] = floor - z
    return err

  def _target_error(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i)
    phase = self._phase[i]
    # Per-env, per-call solver settings (base.py solves env i right after this returns).
    if phase == P_LIFT:
      # THE POSTURE SEED MUST NOT BE ON DURING THE LIFT. SIDE_POSTURE_POS/NEG are poses
      # whose gripper site sits at z 0.122 -- the pinch height -- and they are what makes
      # the horizontal-wrist approach converge at all. Keep pulling the null space toward
      # them while commanding +0.10 of lift and the posture term fights the lift itself:
      # LIFT was the largest remaining failure class (10 of 30 at n = 64), envs that had a
      # verified pinch (aperture >= 0.012) and still never got the plate's centre past
      # riser_top + 0.04. Free the null space here; the side frame (orientation) stays.
      self.damping = 0.2
      self.max_dq = 0.10
      self.cmd_lead_max = 0.12
      self.cmd_ref = "actual"
      self.posture_weight = 0.005
      self._posture_target = self.default_qpos[:7]
    elif phase >= P_SIDE_HOVER:
      self.damping = 0.3
      self.max_dq = 0.07
      self.cmd_lead_max = 0.12
      self.cmd_ref = "actual"  # the floor guard must bind here (see the class note)
      # NEGATIVE RESULT (W1-G2, n=64): dropping the branch bias to
      # SIDE_POSTURE_WEIGHT_FINE for SIDE_INSERT/CLOSE (W1-M's "0.0005 leaves 4-11 mm
      # instead of 28-60 mm" finding) measured 0.531 -> 0.297. On this task the seed is
      # not a precision term at all -- it is what keeps the solve on the reachable
      # branch, and it is still doing that during the insert. Full weight throughout.
      self.posture_weight = SIDE_POSTURE_WEIGHT
      self._posture_target = SIDE_POSTURE_POS if self._side_sign[i] >= 0 else SIDE_POSTURE_NEG
    else:
      self.posture_weight = 0.005
      self._posture_target = self.default_qpos[:7]
      self.damping = 0.2
      self.max_dq = 0.10 if phase != P_PUSH_ADVANCE else 0.05
      self.cmd_lead_max = PUSH_LEAD if phase == P_PUSH_ADVANCE else 0.12
      self.cmd_ref = "command"

    raw_gto = obs_i[40:43]  # plate - gripper
    raw_o2g = obs_i[43:46]  # target - plate
    psi = self._riser_yaw(i, obs_i)
    ee = self._site(obs_i)
    gto = self._smooth(i, raw_gto + ee) - ee
    o2g = self._smooth_o2g(i, raw_o2g)
    xr, yr = _yaw_frame(psi)
    n_out = -xr  # outward normal of the riser's robot-facing edge
    rot_goal = np.array(
      [GOAL_OFFSET[0] * xr[0] + GOAL_OFFSET[1] * yr[0],
       GOAL_OFFSET[0] * xr[1] + GOAL_OFFSET[1] * yr[1],
       GOAL_OFFSET[2]]
    )
    ledge_rel = (o2g + gto) - rot_goal  # ledge origin - gripper
    s_plate = float(np.dot((gto - ledge_rel)[:2], xr[:2]))  # riser-frame x of the plate
    overhang = -LEDGE_HALF_X - (s_plate - PLATE_R)
    plate_z_err = gto[2]  # site -> plate centre height
    far_edge_s = float(np.dot(ledge_rel[:2], xr[:2])) + LEDGE_HALF_X - FAR_EDGE_MARGIN
    # push contact point (behind the plate on the +x_r side), clamped onto the riser top

    def _contact(back: float) -> np.ndarray:
      c = gto + back * xr
      s_c = float(np.dot(c[:2], xr[:2]))
      if s_c > far_edge_s:
        c = c - (s_c - far_edge_s) * xr
      return c

    if phase == P_PUSH_HOVER:
      err = _contact(BEHIND)
      err[2] = plate_z_err + PUSH_HOVER_HEIGHT
      if np.linalg.norm(err[:2]) < PUSH_ALIGN_TOL or self._phase_steps[i] > PUSH_HOVER_TIMEOUT:
        self._phase[i] = P_PUSH_DESCEND
        self._phase_steps[i] = 0
      return self._guard(err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_PUSH_DESCEND:
      err = _contact(BEHIND)
      z_err = plate_z_err + PUSH_SITE_ABOVE_PLATE
      err[2] = max(z_err, -PUSH_DESCENT_RATE)
      if abs(z_err) < PUSH_SEAT_TOL:
        self._reseat[i] += 1
      else:
        self._reseat[i] = 0
      if self._reseat[i] >= PUSH_SEAT_SETTLE or self._phase_steps[i] > PUSH_DESCEND_TIMEOUT:
        self._phase[i] = P_PUSH_ADVANCE
        self._phase_steps[i] = 0
        self._reseat[i] = 0
      return self._guard(err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_PUSH_ADVANCE:
      if overhang >= OVERHANG_EXIT or overhang > OVERHANG_ABORT or self._phase_steps[i] > PUSH_TIMEOUT:
        self._phase[i] = P_RETREAT
        self._phase_steps[i] = 0
        self._q_cmd[i] = np.nan  # drop the lead wound up against the plate (no surge)
        err = RETREAT_BACK * xr + np.array([0.0, 0.0, RETREAT_UP])
        return self._guard(err, obs_i), _DOWN_AXIS, GRIPPER_OPEN
      along = float(np.dot(-gto[:2], xr[:2]))  # + while the pusher is behind (+x_r side)
      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        err = _contact(1.6 * BEHIND)
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        err = _contact(1.6 * BEHIND)
      else:
        remaining = max(OVERHANG_TARGET - overhang, 0.0)
        step = float(np.clip(PUSH_GAIN * remaining, PUSH_STEP_MIN, PUSH_STEP_MAX))
        err = _contact(BEHIND - step)
      # Hard height clamp: the pad must never be commanded below the plate's mid-height,
      # or the advance drives it under the rim (see PUSH_DESCENT_RATE).
      err[2] = max(plate_z_err + PUSH_SITE_ABOVE_PLATE, -PUSH_DESCENT_RATE)
      return self._guard(err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_RETREAT:
      if self._phase_steps[i] >= RETREAT_STEPS:
        self._phase[i] = P_SIDE_HOVER
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
      if self._phase_steps[i] < RETREAT_UP_STEPS:
        err = RETREAT_BACK * xr + np.array([0.0, 0.0, RETREAT_UP])
      else:
        # Travel past the near edge at altitude (hold z = plate + 0.14).
        err = -RETREAT_OUT * xr + np.array([0.0, 0.0, plate_z_err + 0.14])
        err[2] = max(err[2], 0.0)
      return self._guard(err, obs_i), _DOWN_AXIS, GRIPPER_OPEN

    # -- side pinch geometry, fresh every step --------------------------------------
    if phase >= P_SIDE_HOVER:
      # (clamp applied BEFORE ``pinch`` is built -- see the note under the phase branches)
      gto = gto.copy()
      gto[2] = max(gto[2], PLATE_ON_RISER_MIN - float(self._site(obs_i)[2]))
    pinch = gto + (PLATE_R - INSET) * n_out  # pinch point relative to the gripper
    if self._side_sign[i] == 0.0:
      # Choose the side once per attempt: keep the flange (0.2 behind the fingertips
      # along the heading) near the robot's centreline.
      p_local_y = float(self._site(obs_i)[1] + pinch[1])
      self._side_sign[i] = 1.0 if p_local_y >= 0.0 else -1.0
    heading = np.cos(ALPHA) * xr + self._side_sign[i] * np.sin(ALPHA) * yr
    rot = _side_frame(heading)

    # Unconditional height recovery: nothing else matters while the wrist is below the
    # floor the side pose needs (the hand capsule bottom is site - 0.072 with the 20 deg
    # pitch, so site 0.075 puts it 3 mm off the ground and ``ee_ground_collision`` fires).
    site_z = float(self._site(obs_i)[2])
    # DO NOT CHASE A FALLEN PLATE. Every side-phase target is expressed as a height
    # relative to the plate, so if the push shoved the plate off the riser the whole
    # approach re-aims at a plate lying on the FLOOR and the wrist follows it down:
    # diagnose #9, 15 of 20 first-terminations were ``ee_ground_collision`` in CLOSE
    # with the plate logged at z 0.008-0.073 (its height on the riser is 0.110).
    # Clamping the plate height used by the side phases (rather than returning early)
    # keeps the state machine running -- an early return stalls the phase because the
    # phase's own timeout lives inside the body it skips, which cost 0.719 -> 0.453 on
    # the first attempt at this fix.
    # Height recovery: a guarded RELATIVE +0.013 cannot beat a 5 mm/step gravity sag on
    # a horizontal wrist at the edge of the workspace, so command the full climb.
    if phase in (P_SIDE_HOVER, P_SIDE_DROP, P_SIDE_INSERT) and site_z < SIDE_RECOVER_Z:
      return np.array([0.0, 0.0, SIDE_RECOVER_TO - site_z]), rot, GRIPPER_OPEN

    if phase == P_SIDE_HOVER:
      # xy + orientation ONLY, at an absolute altitude well clear of everything.
      err = pinch - SIDE_STANDOFF * heading
      err[2] = SIDE_APPROACH_Z - site_z
      _, r_cur = self._fk(self.default_qpos + obs_i[0:9])
      oriented = float(np.dot(r_cur[:, 2], rot[:, 2])) > SIDE_HOVER_COS
      if (
        (oriented and np.linalg.norm(err[:2]) < SIDE_HOVER_TOL and abs(err[2]) < 0.05)
        or self._phase_steps[i] > SIDE_HOVER_TIMEOUT
      ):
        self._phase[i] = P_SIDE_DROP
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
      return self._guard(err, obs_i, SIDE_FLOOR_MIN_Z), rot, GRIPPER_OPEN

    if phase == P_SIDE_DROP:
      # Pure vertical descent onto the standoff point: xy is still servoed (it drifts as
      # the arm reconfigures) but the z command is rate-limited, so the wrist cannot
      # trade height for reach the way the combined waypoint did.
      err = pinch - SIDE_STANDOFF * heading
      z_target = site_z + pinch[2] + SIDE_Z_ABOVE  # absolute site height at the pinch
      z_err = z_target - site_z
      err[2] = max(z_err, -SIDE_DROP_RATE)
      if (
        (abs(z_err) < SIDE_DROP_TOL and np.linalg.norm(err[:2]) < 0.03)
        or self._phase_steps[i] > SIDE_DROP_TIMEOUT
      ):
        self._phase[i] = P_SIDE_INSERT
        self._phase_steps[i] = 0
        self._integ[i] = 0.0
      return self._guard(err, obs_i, SIDE_FLOOR_MIN_Z), rot, GRIPPER_OPEN

    if phase == P_SIDE_INSERT:
      err = pinch.copy()
      err[2] += SIDE_Z_ABOVE
      self._integ[i] = np.clip(
        self._integ[i] + INSERT_INTEG_GAIN * err, -INSERT_INTEG_CLIP, INSERT_INTEG_CLIP
      )
      radial = abs(float(np.dot(err[:2], n_out[:2])))
      if (
        (radial < SIDE_INSERT_RADIAL_TOL and abs(err[2]) < SIDE_INSERT_Z_TOL)
        or self._phase_steps[i] > SIDE_INSERT_TIMEOUT
      ):
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
      cmd = np.clip(err, -SIDE_INSERT_RATE, SIDE_INSERT_RATE) + self._integ[i]
      return self._guard(cmd, obs_i, SIDE_FLOOR_MIN_Z), rot, GRIPPER_OPEN

    if phase == P_CLOSE:
      # Ramp the site down from SIDE_Z_ABOVE to SIDE_Z_CLOSE while the jaws close.
      frac = min(1.0, self._phase_steps[i] / max(1, CLOSE_STEPS - 4))
      z_above = SIDE_Z_ABOVE + (SIDE_Z_CLOSE - SIDE_Z_ABOVE) * frac
      cmd = 0.3 * pinch + self._integ[i]
      cmd[2] = pinch[2] + z_above
      if self._phase_steps[i] >= CLOSE_STEPS:
        # The pinch is verified from the FINGER JOINTS, not from where the plate is: an
        # empty close runs the aperture to ~0.000 while ``pinch`` (a position residual)
        # still looks fine. diagnose #8: 9/22 failures closed on nothing and then "lifted".
        aperture = 0.08 + obs_i[7] + obs_i[8]
        empty = aperture < AP_HELD_MIN
        if (empty or np.linalg.norm(pinch) > HELD_TOL) and self._retry[i] < MAX_INSERT_RETRIES:
          self._retry[i] += 1
          self._phase[i] = P_SIDE_INSERT
          self._phase_steps[i] = 0
          self._integ[i] = 0.0
        else:
          self._phase[i] = P_LIFT
          self._phase_steps[i] = 0
      # Guard on the SIDE floor, not the top-down one: with the hand horizontal the
      # capsule is 7 cm below the site, so the 0.030 top-down floor is meaningless here.
      return self._guard(cmd, obs_i, SIDE_CLOSE_FLOOR_Z), rot, GRIPPER_CLOSED

    # P_LIFT: straight up and a little outward; hold (success latches).
    err = np.array([0.0, 0.0, LIFT_UP]) + LIFT_OUT * n_out
    return self._guard(err, obs_i, SIDE_CLOSE_FLOOR_Z), rot, GRIPPER_CLOSED
