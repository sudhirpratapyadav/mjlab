"""Scripted RotateValve teacher — a closed-finger PROBE circling behind one spoke.

Physics refresh (2026-09-10): derive the target chord from the source hinge FK;
contact at radius 80 mm and x=-50 mm to keep the finger body away from the hub.
The historical development notes below describe earlier collision geometry.

CL-V3 REWRITE (W1-M, 2026-09-09). The pinch-and-hand-off teacher of cl25/v2 measured
15/32 on the frozen v3 spec (diagnose, HEAD b9b0563 + W0 init spec). Per engagement it
spent ALIGN ~10-45 + SEAT ~36 (timeout, hunting) + PINCH 4 steps, and most engagements
then produced 0 deg of rotation (the pinch had not acquired the spoke: "ARC1:+0deg"
runs). The productive engagements gave 30-40 deg each, so 270 deg needed 7+ cycles of
~80 steps inside a 400-step budget. The strategy is throughput-limited by design.

STRATEGY — no grasp, no hand-off, pure translation
--------------------------------------------------
The handwheel has two opposed 24 mm square spokes (r 0.02..0.11) and no rim. A closed
fingertip is a 17.5 x 30 mm block; poked into the wheel plane just BEHIND spoke A's
trailing face at radius PROBE_R it pushes the spoke by face contact — the contact
normal is tangential, so nothing pushes the probe out of the plane. The hand then
simply translates around a 6.8 cm circle: a full 270 deg of wheel rotation is 270 deg
of hand TRANSLATION, no wrist rotation at all, so the "270 deg exceeds the wrist range"
hand-off is unnecessary. The probe target is recomputed every step from the spoke's
angle (goal-marker chord, mount frame, unwrapped) with an angular lead into the spoke
(the servo pushes) and it can never overtake the spoke (the target is always just
inside its trailing face).

Why the cl25 pad push walked inboard to the hub: its target was a chord 0.9 rad ahead
on the arc, whose midpoint is 10 % inside the circle, so the hand was pulled inward
every step. Here the target sits ON the circle at the current spoke angle (+ a small
lead), so the contact radius is held.

The approach axis is tilted 60 deg down from the mount's +x (offline DLS check WITH the
env's soft joint limits: <= 8 mm round the circle at 55-65 deg; 45 deg sits on joint 6's
3.56 rad soft limit at both ends; 30 deg plateaus 25-70 mm short).
Mount yaw is read from ``object_orientation``; all geometry is in the mount frame.

Phases (each with an unconditional step-count escape): APPROACH (5 cm in front of the
wheel plane at the insertion angle) -> INSERT (+x into the plane) -> SWEEP (circle,
progress watchdog) -> RETREAT (-x out, then re-approach at the current spoke angle).

Only relative observations are used (gto = obs[40:43], o2g = obs[43:46]).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.mechanism_geometry import joint_goal_displacement

from mjlab.continual_distill.classical.base import (
  ClassicalPolicyBase,
  mount_yaw_from_obs,
  rot_z_mat,
)

_R_ARM = 0.09  # spoke-A tip radius (object_site) in the mount y-z plane
_X_TIP = -0.04  # x of the wheel plane / spoke centreline (mount frame)
_MARKER = joint_goal_displacement("valve", "valve_hinge", 4.712389)

GRIPPER_CLOSED = -1.0

PROBE_R = 0.080  # contact radius: spoke box r 0.02..0.11, hub r 0.02
PROBE_X = -0.050  # site x; pads span +-8 mm -> inside the 24 mm spoke thickness
# Probe angular offset behind the spoke centreline: spoke half 0.012 + probe half
# extent across the tangent (up to 0.015) + 3 mm margin, over PROBE_R.
EPS = 0.35
# Angular lead INTO the spoke while sweeping: commanded ~2 cm of penetration, which the
# position servos turn into a push. Closed-loop, so the probe cannot outrun the spoke.
LEAD = 0.30
OUT = 0.05  # standoff in front of the wheel plane for approach / retreat
# 60 deg, not 45: the env clamps joint 6 to its SOFT limit (0.9 x range = 3.56 rad) and
# the 45 deg pose sits on that limit at both ends of the circle (offline DLS with the
# soft limits: 30-40 mm short at -25 / 245 deg); at 55-65 deg every point is <= 8 mm
# with q6 3.0-3.5. The hand capsule then intrudes 7 mm into the wheel plane 6 cm above
# the probe, always > 100 deg away from spoke B.
TILT = np.radians(60.0)

APPROACH_TOL = 0.030
APPROACH_TIMEOUT = 30
INSERT_TOL = 0.020  # gto carries +-1 cm uniform noise per axis
INSERT_SETTLE = 2
INSERT_TIMEOUT = 20
RETREAT_TIMEOUT = 15
STALL_STEPS = 30
STALL_EPS = 0.05
DONE_ANGLE = 4.75  # success is latched at 4.512; keep pushing a little past
INTEG_GAIN = 0.08
INTEG_BAND = 0.05
INTEG_CLIP = 0.02
EMA_YAW = 0.3
EMA_ALPHA = 0.5
CMD_LEAD_MIN = 0.12
MAX_WAYPOINT = 0.20

PHASE_NAMES = {0: "APPROACH", 1: "INSERT", 2: "SWEEP", 3: "RETREAT", 4: "HOLD"}


class RotateValveClassicalPolicy(ClassicalPolicyBase):
  """Translate a closed fingertip round the wheel, pushing spoke A's trailing face."""

  orientation_weight = 0.25
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
      self._q_unwrapped = np.zeros(self.num_envs)
      self._q_prev = np.full(self.num_envs, np.nan)
      self._integ = np.zeros((self.num_envs, 3))
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._stall_q = np.zeros(self.num_envs)
      self._stall_t = np.zeros(self.num_envs, dtype=np.int64)
      self._attempts = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._psi_init[env_ids] = False
      self._q_unwrapped[env_ids] = 0.0
      self._q_prev[env_ids] = np.nan
      self._integ[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._stall_q[env_ids] = 0.0
      self._stall_t[env_ids] = 0
      self._attempts[env_ids] = 0

  def _go(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._integ[i] = 0.0
    self._stall_t[i] = 0

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    o2g = obs_i[43:46]
    psi_raw = mount_yaw_from_obs(obs_i)
    if not self._psi_init[i]:
      self._psi[i] = psi_raw
      self._psi_init[i] = True
      self._ema[i] = gto_raw
    else:
      self._psi[i] += EMA_YAW * (psi_raw - self._psi[i])
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    psi = float(self._psi[i])
    gto = self._ema[i]
    rz = rot_z_mat(psi)

    # Spoke-A angle from the fixed marker, mount frame: tip(q) - tip(0) = _MARKER - o2g_m.
    d = _MARKER - rz.T @ o2g
    ct = np.clip(d[1] / _R_ARM + 1.0, -1.0, 1.0)
    st = np.clip(d[2] / _R_ARM, -1.0, 1.0)
    q = float(np.arctan2(st, ct))
    if np.isnan(self._q_prev[i]):
      self._q_unwrapped[i] = q
    else:
      dq = (q - self._q_prev[i] + np.pi) % (2 * np.pi) - np.pi
      self._q_unwrapped[i] += dq
    self._q_prev[i] = q
    q_total = float(self._q_unwrapped[i])

    # Hinge point relative to the site (mount frame); the site is spoke A's tip.
    hinge_m = -np.array([_X_TIP, _R_ARM * np.cos(q), _R_ARM * np.sin(q)])

    def probe_m(angle: float, x: float = PROBE_X) -> np.ndarray:
      return hinge_m + np.array([x, PROBE_R * np.cos(angle), PROBE_R * np.sin(angle)])

    target_axis = rz @ np.array([np.cos(TILT), 0.0, -np.sin(TILT)])
    gripper_a = GRIPPER_CLOSED
    ph = int(self._phase[i])
    # W1-G's gravity-sag ratchet (docs/cl_v3/logs/W1-G.md): with cmd_lead_max = 0 the
    # joint command is re-anchored to the sagged / lagging actual joints every step and
    # the site sinks ~1 cm per step on hover-then-descend motions. Keep >= 0.12 rad of
    # command lead in every phase; the pull / lift phases raise it further below.
    self.cmd_lead_max = CMD_LEAD_MIN

    if ph == 0:
      self.max_dq, self.step_gain, self.max_pos_err = 0.22, 0.6, 0.12
      pos_err = gto + rz @ probe_m(q - EPS, PROBE_X - OUT)
      if np.linalg.norm(pos_err) < APPROACH_TOL or self._phase_steps[i] > APPROACH_TIMEOUT:
        self._go(i, 1)
    elif ph == 1:
      self.max_dq, self.step_gain, self.max_pos_err = 0.12, 0.4, 0.08
      raw = gto + rz @ probe_m(q - EPS)
      if np.linalg.norm(raw) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -INTEG_CLIP, INTEG_CLIP)
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < INSERT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= INSERT_SETTLE or self._phase_steps[i] > INSERT_TIMEOUT:
        self._go(i, 2)
        self._stall_q[i] = q_total
    elif ph == 2:
      self.max_dq, self.step_gain, self.max_pos_err = 0.20, 0.6, 0.12
      self.cmd_lead_max = max(CMD_LEAD_MIN, 0.20)
      if q_total >= DONE_ANGLE:
        self._go(i, 4)
        return gto + rz @ probe_m(q - EPS), target_axis, gripper_a
      self._stall_t[i] += 1
      if self._stall_t[i] >= STALL_STEPS:
        if q_total - self._stall_q[i] < STALL_EPS:
          # Not turning: retreat straight out of the wheel plane and re-approach.
          self._attempts[i] += 1
          self._go(i, 3)
          return rz @ np.array([-OUT, 0.0, 0.0]), target_axis, gripper_a
        self._stall_q[i] = q_total
        self._stall_t[i] = 0
      pos_err = gto + rz @ probe_m(q - EPS + LEAD)
    elif ph == 3:
      self.max_dq, self.step_gain, self.max_pos_err = 0.15, 0.5, 0.08
      # Pull the site OUT of the wheel plane along the mount's -x, holding y-z.
      site_m = -(rz.T @ gto)  # site - tip, mount frame
      x_now = site_m[0] + _X_TIP  # site x in the mount frame (tip is at _X_TIP)
      pos_err = rz @ np.array([(PROBE_X - OUT) - x_now, 0.0, 0.0])
      if abs(pos_err @ (rz @ np.array([1.0, 0.0, 0.0]))) < 0.012 or self._phase_steps[i] > RETREAT_TIMEOUT:
        self._go(i, 0)
    else:
      self.max_dq, self.step_gain, self.max_pos_err = 0.10, 0.4, 0.05
      pos_err = gto + rz @ probe_m(q - EPS)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_axis, gripper_a
