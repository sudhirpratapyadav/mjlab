"""Scripted OpenLid teacher — vertical y-pinch on the knob stem, lifted round the arc.

CL-V3 REWRITE (W1-M, 2026-09-09). Same grasp as CL-V2 (pads closing along the hinge
axis on the 28 mm stem under the 44 mm cap), three things fixed after reading the
frozen-spec failures (diagnose n=32, HEAD b9b0563 + W0 init spec: 2/32).

WHAT WAS FAILING (measured)
---------------------------
1. 16/32 closed on NOTHING (aperture 0.000). Reconstructing the site in the mount frame
   at close time: the pinch centre sat 2-3 cm toward the robot (-x) from the stem
   (stem half-width 14 mm, pad half-width 9 mm). The descent was a LIMIT CYCLE, not a
   bias: gto.x swung +0.08 -> +0.02 -> -0.06 -> 0 -> +0.05 and the site z bounced
   0.15 <-> 0.24 for 100+ steps; the close fired on whichever swing satisfied the
   settle counter. Cause: a full 0.18 rad joint stride re-planned every step from the
   lagging actual joints into wrist servos that saturate at ~0.04 rad (kp 300,
   12 N m) — bang-bang. Fixed with ``step_gain`` (base.py, additive) and a smaller,
   banded integrator.
2. The mount yaw (+-15 deg) rotated the hinge axis away from world y; the old
   axis-only orientation target left the closing axis wherever the posture bias put
   it. Now the closing axis is pinned to the MOUNT's y axis (full 3x3, neutral wrist
   branch: site y = -mount y) and all arc geometry is done in the mount frame.
3. 14/32 lifted to 0.25-0.85 rad and dropped back: the old approach axis FOLLOWED the
   lid (-local z), which at 60-75 deg points back toward the robot with the hand
   beyond the knob at x 0.53-0.59 — the offline DLS check cannot reach it. A VERTICAL
   hand is reachable at every lid angle (1-3 mm residual at 0/40/60/75 deg), and it is
   geometrically valid throughout: the pads are vertical plates whose faces stay
   parallel to the stem's +-y faces under any rotation about y, the pad footprint
   (17.5 x 16.5 mm) fits inside the stem face (28 x 34 mm) at any relative angle, and
   the cap remains the stop against the pads sliding along the stem axis. So the
   approach axis stays straight down for the whole lift.

Sequence: HOVER above the stem (open) -> DESCEND (open, damped, integrator) -> CLOSE
(hold still) -> LIFT along the arc with a small angular lead and a command lead for
sustained force. Aperture (finger joints in the obs) is checked after closing: pads
that met with nothing between them re-open and re-descend. Every phase has an
unconditional step-count escape.

Only relative observations are used (gto = obs[40:43], o2g = obs[43:46]); mount yaw
from ``object_orientation`` (origin-free).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.mechanism_geometry import joint_goal_displacement

from mjlab.continual_distill.classical.base import (
  ClassicalPolicyBase,
  mount_yaw_from_obs,
  rot_z_mat,
)

# Mount-frame geometry (lid.xml): object_site (-0.09, 0, 0.05), hinge (0.10, 0, 0.04),
# axis (0,-1,0) so a NEGATIVE joint value lifts the lid; we work with phi = -q >= 0 as a
# rotation about +y that raises the near edge.
_R0 = np.array([-0.19, 0.0, 0.01])  # site - hinge at phi = 0
_MARKER = joint_goal_displacement("lid", "lid_hinge", -1.308997)
# Pinch centre up the stem from the site: stem spans local z 0.046..0.082 (cap above),
# lid panel top at 0.048; pads are +-8 mm tall about the site, so 0.018 puts them at
# 0.060..0.076 — 6 mm under the cap, 12 mm above the panel.
GRASP_UP = 0.014
HOVER = 0.09
# Two-stage descent: settle STAGE_UP above the pinch with the integrator running (no
# contact possible there), then drop the last STAGE_UP carrying the integrator. Measured
# single-stage: the arm's gravity sag (~15 mm) lands the pads' bottoms 6 mm INTO the
# lid panel's top face, friction then freezes the 17 mm x error, and the pads close
# 16 mm in front of the stem on the cap's edge.
STAGE_UP = 0.035
STAGE_TOL = 0.012
STAGE_TIMEOUT = 16
GRIPPER_OPEN = 0.0
GRIPPER_PINCH = -1.0

HOVER_TOL = 0.035
HOVER_TIMEOUT = 45  # the neutral pose is ~1 m from the knob
HOVER_LEAD = 0.28  # cmd_lead_max caps the far-field speed, not max_dq (see base.py)
DESCEND_TOL = 0.018  # gto carries +-1 cm uniform noise per axis; 0.008 never settled
DESCEND_SETTLE = 2
DESCEND_TIMEOUT = 22
DESCEND_ABORT_DIST = 0.03  # timed out further than this from the stem: re-hover
INTEG_GAIN = 0.08
INTEG_BAND = 0.05
INTEG_CLIP = 0.025
CLOSE_STEPS = 6
# Aperture (sum of the two finger joints) after closing: the stem is 28 mm, so a real
# pinch reads ~0.026; pads that met each other read ~0.0. Obs noise is +-0.01 per
# finger, hence the EMA and the wide margin.
APERTURE_MIN = 0.007  # held stem reads ~0.02 in sim (soft contact) minus noise; 0.013 fired on held lids
EMA_AP = 0.3
# Arc lead: the pinch point advanced LEAD_ANGLE round the hinge (0.28 rad = 5 cm of
# arc at r 0.19), shrinking to zero at the -75 deg stop. The chord of a 0.28 rad lead
# cuts 2 mm inside the arc; RADIAL keeps the pads pressed outward against that.
# 0.13 rad at r 0.19 is 2.5 cm of arc. Measured (v8, n=32): at 0.28 rad with
# cmd_ref="command" and a 0.35 rad lead clip — i.e. the servos at full torque — the
# median lid angle reached FELL to 0.20 rad and the lift phase aborted every ~60 steps:
# a position command that far ahead of the lid squeezes the stem out of the pads, the
# same cam-out the door showed. The budget is 250 steps and success needs 1.109 rad, so
# 0.0065 rad/step suffices; a gentle lift is free.
LEAD_ANGLE = 0.20
# RADIAL presses the hand OUTWARD along the arc radius, which pins the knob cap up
# against the pad tops — a positive mechanical stop instead of pad friction on the stem.
# It matters most at the end of the arc: past ~60 deg the lid's weight pulls along the
# stem axis rather than into the cap.
RADIAL = 1.06
# Final stage of the lift. The LIFT IS FRICTION-LIMITED: the tangent at the knob is
# almost exactly the stem axis, the pads grip the stem's +-y faces, and the pinch normal
# is perpendicular to the direction of travel — so every newton of lift comes from
# mu x ~4 N of pad squeeze, against the ~1.75 cos(phi) N the lid needs. That is a 4x
# margin in statics and none at all against an overdriven position command, which is why
# the residual failures bleed the aperture 0.047 -> 0.000 at phi 0.6-1.0. Past PUSH_PHI
# the stage therefore raises the GAINS (so the arm tracks the arc without lagging) but
# NOT the arc lead: an extra 0.06 rad of lead here measured 0.75-0.97 against 1.00.
PUSH_PHI = 0.85
PUSH_LEAD_ANGLE = 0.20
# The arc target used to be built from the NOMINAL grasp offset (GRASP_UP above
# object_site). Measured: once the lid's weight has slid the pads up the stem against the
# cap, the gripper site sits 3.5-5.0 cm above object_site, not 1.8 — so the commanded
# point was ~2 cm BELOW the hand, cancelling half of the 3.8 cm arc lead and leaving the
# 3.2-4.5 cm steady-state error the traces show at the 0.75 rad asymptote. Latch the real
# site->gripper vector in the LID's own frame at close time and carry it (slow EMA, so a
# continuing ride-up is absorbed): the target is then exactly "where this hand would be
# if it rode rigidly with the lid to phi + lead", with no grasp model at all.
# Restricting this drift to the stem axis (so the hand actively re-centres the stem in
# the pad footprint) was tried and measured WORSE — 0.72 against 0.94 at n=32 — so the
# full-vector EMA stays: the hand following the grip beats the hand fighting it.
HOLD_EMA = 0.03
# An aperture-driven force limiter (back the lead off as the aperture decays) was tried:
# CPU n=32 liked it (0.94 -> 0.97) but the GPU protocol read did not (0.898 -> 0.867 at
# n=128). Left out. NOTE for anyone sweeping here: the CPU harness ranks these parameters
# differently from the GPU — confirm every candidate with test_classical on the GPU.
HOLD_MAX = 0.13  # |hold| beyond this means the grip is gone, not riding
RADIAL_BIAS = 0.012  # metres outward along the arc radius: presses the cap onto the pads
STOP_ANGLE = 1.309  # joint range; success is latched at 1.109
# Hand tilt toward the robot follows the lid angle up to FOLLOW_MAX. Measured with a
# vertical hand (CPU probe): the lift stalls at ~38 deg because the cap, rotating with
# the lid, swings into the finger bodies 14 mm above the pads (a vertical y-pinch is
# only valid to ~25 deg of relative rotation). Fully following the lid is unreachable
# past ~60 deg (hand beyond the knob); offline DLS with the soft joint limits reaches
# the pinch at every lid angle with the tilt capped at 55 deg (<= 11 mm), which keeps
# the relative angle <= 20 deg at the -75 deg target.
FOLLOW_MAX = np.radians(75.0)
# Extra hand tilt relative to the lid. Past ~35 deg the lid's weight has a growing
# component ALONG the pad plane (g . x_lid = g sin phi), pointing toward the hinge, and
# neither the pads' friction nor the cap's flat underside opposes it — that is the
# measured cam-out at phi 0.6-1.0 (aperture 0.047 -> 0.000 over ~25 steps). Tilting the
# hand off the lid angle wedges the cap's edge against a pad corner instead of leaving
# the two faces parallel.
TILT_BIAS = 0.0
# During LIFT the pads are clamped to the stem (and slid up against the cap), so the
# hand's pitch is kinematically tied to the lid: an orientation target that differs
# from the lid angle only saturates joint 6 (measured: 0.22 rad behind its command at
# 12 N m, lift stuck at 62.5 deg). The lift therefore follows the lid exactly with a
# weak orientation term and lets position drive.
# The lift's orientation weight is NOT a free parameter. Rendered debug frames of a
# stalled lift (v9, ep01 t=84..240, identical pose for 156 steps) show why: at 0.05 the
# DLS is free to satisfy the position target with the hand pitched right over, and it
# settles with the HAND BODY lying flat on the closed lid — the arm is pressing the lid
# shut with its own wrist while pulling the knob up. That is the whole "posture plateau":
# the IK converges (1-2 mm residual, a 0.2 rad joint move asked for every step) and the
# arm is simply blocked by itself. v6, which had the base 0.3 weight here, reached
# 1.15 rad in 21/32; v7/v8, which lowered it, reached 0.4/0.2 median.
LIFT_ORI_WEIGHT = 0.30
# Lift force. Measured (v7, n=32, trace reconstruction): during a stalled lift the IK
# still solves cleanly (1-2 mm residual, a 0.18-0.29 rad joint move asked for every step
# — NOT an M4 plateau), the Cartesian error sits at a CONSTANT 5 cm, and the joint
# command is static. With ``cmd_ref = "actual"`` the IK target is FK(lagging actual
# joints) + pos_err while the command is already clipped cmd_lead_max ahead of them, so
# the step goes NEGATIVE and the command settles at "5 cm of lead" forever, whatever the
# load. ``cmd_ref = "command"`` (W1-G, base.py) anchors the target to the command, so the
# command ratchets forward until the lead clip binds and the servos apply their full
# torque. Only the LIFT uses it; the approach phases stay on "actual".
LIFT_LEAD = 0.22
LIFT_GRACE = 12  # no re-hover in the first steps of a lift (the grip settles)
# The lid's weight slides the pads UP the stem until the cap stops them (measured: the
# site rides 3.5-4.5 cm above object_site during the lift), so 'lost' must allow that.
LOST_DIST = 0.10
EMA_YAW = 0.3
EMA_ALPHA = 0.5
CMD_LEAD_MIN = 0.12
MAX_WAYPOINT = 0.18

PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "CLOSE", 3: "LIFT"}


def _rot_y(phi: float) -> np.ndarray:
  c, s = np.cos(phi), np.sin(phi)
  return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# Posture bias for the LIFT, the same mechanism `classical/axial_extract.py` documents:
# the posture term is a low-weight null-space tie-breaker that SELECTS AN IK BRANCH.
# NEUTRAL_QPOS's joint2 is -1.0, i.e. the regulariser pulls the shoulder DOWN throughout
# a motion whose whole job is to raise the hand 16 cm and extend it 11 cm. Reconstructed
# from the traces: lifts that finish take joint2 from -0.40 to +0.65 and joint4 from
# -2.90 to -1.15, while stalled lifts sit at joint2 -0.40/-0.27 for 150+ steps with the
# IK converging cleanly every step. Biasing only joint2 toward HOME's +0.3 during the
# lift (and raising the weight enough for the tie-break to bite) tips the solve into the
# branch the successes use. The approach phases keep NEUTRAL's own posture.
LIFT_POSTURE_J2 = 0.3
# ...and ramped with the lid angle: lifts that finish carry joint2 all the way to +0.65,
# so holding the tie-breaker at +0.3 stops helping exactly where the failures asymptote.
LIFT_POSTURE_J2_TOP = 0.9
LIFT_POSTURE_PHI = 1.1
LIFT_POSTURE_W = 0.004
APPROACH_POSTURE_W = 0.0005


class OpenLidClassicalPolicy(ClassicalPolicyBase):
  """Pinch the knob stem with a vertical hand and swing the lid up against gravity."""

  orientation_weight = 0.3
  ik_joint_limit_factor = 0.9  # the env's soft limits (see base.py)
  # base default 0.005 pulls toward NEUTRAL and balances the position gradient at
  # stretched postures (M4 plateau, reproduced offline from a stalled lift: 28 mm
  # residual at 0.005, 4 mm at 0.0005).
  posture_weight = 0.0005

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._psi = np.zeros(self.num_envs)
      self._psi_init = np.zeros(self.num_envs, dtype=bool)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._ap = np.full(self.num_envs, 0.08)
      self._integ = np.zeros((self.num_envs, 3))
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._attempts = np.zeros(self.num_envs, dtype=np.int64)
      self._staged = np.zeros(self.num_envs, dtype=bool)
      self._hold = np.zeros((self.num_envs, 3))
    else:
      self._psi_init[env_ids] = False
      self._ema_init[env_ids] = False
      self._ap[env_ids] = 0.08
      self._integ[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._attempts[env_ids] = 0
      self._staged[env_ids] = False
      self._hold[env_ids] = 0.0

  def _go(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._integ[i] = 0.0
    self._staged[i] = False

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    o2g = obs_i[43:46]
    psi_raw = mount_yaw_from_obs(obs_i)
    aperture_raw = float(obs_i[7] + obs_i[8]) + float(self.default_qpos[7] + self.default_qpos[8])
    if not self._psi_init[i]:
      self._psi[i] = psi_raw
      self._psi_init[i] = True
      self._ema[i] = gto_raw
      self._ema_init[i] = True
      self._ap[i] = aperture_raw
    else:
      self._psi[i] += EMA_YAW * (psi_raw - self._psi[i])
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
      self._ap[i] += EMA_AP * (aperture_raw - self._ap[i])
    psi = float(self._psi[i])
    gto = self._ema[i]
    rz = rot_z_mat(psi)

    # Lid angle from the fixed marker, in the MOUNT frame:
    # site(phi) - site(0) = _MARKER - o2g_m, and site(phi) - hinge = R_y(phi) _R0.
    d = _MARKER - rz.T @ o2g
    v = _R0 + d
    x0, z0 = _R0[0], _R0[2]
    denom = x0 * x0 + z0 * z0
    c = (x0 * v[0] + z0 * v[2]) / denom
    s = (z0 * v[0] - x0 * v[2]) / denom
    phi = float(np.arctan2(s, c))
    phi = float(np.clip(phi, -0.1, STOP_ANGLE + 0.05))
    ry = _rot_y(phi)
    up_m = ry @ np.array([0.0, 0.0, 1.0])  # lid normal, mount frame
    grasp_m = GRASP_UP * up_m  # site -> pinch centre, mount frame
    hinge_m = -(ry @ _R0)  # site -> hinge, mount frame
    grasp = gto + rz @ grasp_m  # site -> pinch centre, world
    hinge = gto + rz @ hinge_m

    # Closing axis pinned to the mount's y (neutral wrist branch); approach axis tilted
    # toward the robot by min(phi, FOLLOW_MAX) so the pads stay parallel to the stem.
    zeta = float(np.clip(phi + TILT_BIAS, 0.0, FOLLOW_MAX))
    y_ee = rz @ np.array([0.0, -1.0, 0.0])
    z_ee = rz @ np.array([-np.sin(zeta), 0.0, -np.cos(zeta)])
    target_rot = np.column_stack([np.cross(y_ee, z_ee), y_ee, z_ee])

    ph = int(self._phase[i])
    self.orientation_weight = 0.3
    self.posture_weight = APPROACH_POSTURE_W
    self._posture_target[:] = self.default_qpos[:7]
    # W1-G's gravity-sag ratchet (docs/cl_v3/logs/W1-G.md): with cmd_lead_max = 0 the
    # joint command is re-anchored to the sagged / lagging actual joints every step and
    # the site sinks ~1 cm per step on hover-then-descend motions. Keep >= 0.12 rad of
    # command lead in every phase; the pull / lift phases raise it further below.
    self.cmd_lead_max = CMD_LEAD_MIN
    gripper_a = GRIPPER_OPEN
    self.cmd_ref = "actual"
    if ph == 0:
      self.max_dq, self.step_gain, self.max_pos_err = 0.30, 0.7, 0.20
      self.cmd_lead_max = HOVER_LEAD
      pos_err = grasp + np.array([0.0, 0.0, HOVER])
      if np.linalg.norm(pos_err) > 0.25:
        target_rot = z_ee  # axis-only while far
      if np.linalg.norm(pos_err) < HOVER_TOL or self._phase_steps[i] > HOVER_TIMEOUT:
        self._go(i, 1)
    elif ph == 1:
      self.max_dq, self.step_gain, self.max_pos_err = 0.12, 0.4, 0.08
      staged = self._staged[i]
      raw = grasp if staged else grasp + np.array([0.0, 0.0, STAGE_UP])
      if np.linalg.norm(raw) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -INTEG_CLIP, INTEG_CLIP)
      pos_err = raw + self._integ[i]
      if not staged:
        if np.linalg.norm(raw) < STAGE_TOL:
          self._settle[i] += 1
        else:
          self._settle[i] = 0
        if self._settle[i] >= DESCEND_SETTLE or self._phase_steps[i] > STAGE_TIMEOUT:
          self._staged[i] = True  # integrator carried into the final drop
          self._settle[i] = 0
          self._phase_steps[i] = 0
      else:
        if np.linalg.norm(raw) < DESCEND_TOL:
          self._settle[i] += 1
        else:
          self._settle[i] = 0
        if self._settle[i] >= DESCEND_SETTLE:
          self._go(i, 2)
        elif self._phase_steps[i] > DESCEND_TIMEOUT:
          self._go(i, 2 if np.linalg.norm(raw) < DESCEND_ABORT_DIST else 0)
    elif ph == 2:
      self.max_dq, self.step_gain, self.max_pos_err = 0.06, 0.4, 0.05
      self.cmd_lead_max = 0.05  # hold still while the fingers close
      gripper_a = GRIPPER_PINCH
      pos_err = grasp + self._integ[i]
      if self._phase_steps[i] >= CLOSE_STEPS:
        if self._ap[i] < APERTURE_MIN:
          # Pads met each other: nothing between them. Re-open and re-descend.
          self._attempts[i] += 1
          self._go(i, 0)
          gripper_a = GRIPPER_OPEN
        else:
          # Latch the true site->gripper vector in the lid frame (see HOLD_EMA).
          self._hold[i] = ry.T @ (rz.T @ (-gto))
          self._go(i, 3)
    else:
      if phi > PUSH_PHI:
        self.max_dq, self.step_gain, self.max_pos_err = 0.22, 0.6, 0.14
        self.cmd_lead_max = 0.30
      else:
        self.max_dq, self.step_gain, self.max_pos_err = 0.18, 0.6, 0.10
        self.cmd_lead_max = LIFT_LEAD
      # W1-D's execution defect: with cmd_ref="actual" the position loop's fixed point is
      # "error == servo sag" and extra command lead cannot move it; referencing every
      # axis to the PREVIOUS COMMAND is what removes it.
      self.cmd_ref = "command"
      self.cmd_ref_axes = None
      self.posture_weight = LIFT_POSTURE_W
      f = float(np.clip(phi / LIFT_POSTURE_PHI, 0.0, 1.0))
      self._posture_target[1] = LIFT_POSTURE_J2 + f * (LIFT_POSTURE_J2_TOP - LIFT_POSTURE_J2)
      self.orientation_weight = LIFT_ORI_WEIGHT
      gripper_a = GRIPPER_PINCH
      if self._phase_steps[i] > LIFT_GRACE and (
        np.linalg.norm(rz.T @ (-gto)) > HOLD_MAX or self._ap[i] < APERTURE_MIN
      ):
        self._attempts[i] += 1
        self._go(i, 0)
        return grasp + np.array([0.0, 0.0, HOVER]), target_rot, GRIPPER_OPEN
      obs_hold = ry.T @ (rz.T @ (-gto))
      self._hold[i] += HOLD_EMA * (obs_hold - self._hold[i])
      ramp = min(1.0, (self._phase_steps[i] + 1) / 6.0)
      la = PUSH_LEAD_ANGLE if phi > PUSH_PHI else LEAD_ANGLE
      lead = ramp * max(0.0, min(la, STOP_ANGLE + 0.02 - phi))
      ry2 = _rot_y(phi + lead)
      radial = ry2 @ _R0
      radial = radial / max(np.linalg.norm(radial), 1e-9)
      target_m = (ry2 - ry) @ _R0 + ry2 @ self._hold[i] + RADIAL_BIAS * radial
      pos_err = gto + rz @ target_m

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_rot, gripper_a
