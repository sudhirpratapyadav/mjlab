"""Scripted FlipSwitch teacher — closed-loop PADDLE SWEEP along the toggle's arc.

CL-V3 REWRITE (W1-M, 2026-09-09). The cl25/v2 teacher (align, seat, then a 30-step
open-loop ballistic +x stroke) measured 19/32 on the frozen v3 spec (diagnose, HEAD
b9b0563 + W0 init spec). Of the 13 failures, 9 were ``ee_ground_collision``: the stroke
commands a FIXED offset (+0.16, 0, -0.03) from the hand's CURRENT position every step
for 30 steps, so a stroke that misses the toggle keeps going — the FK of the commanded
joints shows the site diving from z 0.21 to 0.01 and x 0.44 -> 0.74 until link7 hits
the floor. Several of those had first pushed the toggle PAST its OFF stop (-1.0 rad)
while seating: the pad came down on top of the tip. Four successes also hit the floor
after latching. Phase-1 had already shown the single-shot hit rate was 10-15 % and
the 0.6 was retry accumulation.

STRATEGY — never lose contact, never go open-loop
------------------------------------------------
The toggle is an inverted pendulum (50 g weight 62 mm above the pivot) that needs
only ~0.03 N m to cross centre; the position servos push far harder than that a few
mm into a target, so "servoing stalls at the balance point" is a false premise as long
as the target keeps moving ahead of the toggle. So:

* the closed fingertips are placed BELOW the tip on the OFF side (radius R_C from the
  pivot, on the shaft/weight junction, standing off S_OFF along the toggle's trailing
  direction), at contact height — not above it;
* the paddle then tracks a target on the toggle's own arc at (angle + LEAD_ANG) with
  the standoff reduced by PUSH_IN, i.e. a few mm inside the trailing face: the paddle
  pushes the toggle along its arc, every step recomputed from the observed toggle
  angle (goal-marker chord, mount frame), so contact is never lost and the hand never
  runs away. Gravity finishes the flip past centre; the sweep stops at STOP_Q (well
  past the latched +0.3736 success angle, short of the faceplate) and retreats;
* mount yaw from ``object_orientation``; the paddle geometry lives in the mount frame;
* the seat gate is SIGNED (the paddle must be on the trailing side, aligned laterally
  and radially) and every phase has an unconditional step-count escape; a sweep that
  moves nothing for 25 steps re-hovers.

The approach axis is (0.34, 0, -0.94) in the mount frame (20 deg from vertical): see the
constant for the soft-joint-limit reason it is not cl25's 33 deg.

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

_R_TIP = 0.062  # object_site (the weight) radius from the pivot
_MARKER = joint_goal_displacement("switch", "switch_hinge", 0.5235988)
# 20 deg from vertical, not cl25's 33 deg: the env clamps joint 6 to its soft limit
# (3.56 rad) and the 33 deg pose needs q6 = 3.56 at the seat point (offline DLS with the
# soft limits: on the limit), whereas 20 deg reaches seat / sweep / retreat to <= 4 mm
# with q6 3.0-3.2 and link7 >= 0.32 m.
_APPROACH_AXIS_M = np.array([0.34, 0.0, -0.94])

GRIPPER_CLOSED = -1.0

# Contact radius on the toggle: the weight cube spans r 0.049..0.075, the shaft is
# below it; the paddle footprint (+-9 mm) at 0.050 covers the shaft/weight junction —
# below the tip, so it cannot skid over the top.
R_C = 0.050
# Paddle centre standoff from the toggle axis along its trailing direction: toggle
# half-width 0.013 + finger half-width ~0.010 + 3 mm.
S_OFF = 0.026
PUSH_IN = 0.012  # standoff reduction while sweeping (commanded penetration)
LEAD_ANG = 0.30
STOP_Q = 0.45  # success latches at +0.3736; the faceplate starts at x = +0.01
HOVER_BACK = 0.06
HOVER_UP = 0.03

HOVER_TOL = 0.025
HOVER_TIMEOUT = 45
HOVER_HARD_TIMEOUT = 75
SEAT_LAT_TOL = 0.02
SEAT_RAD_TOL = 0.02
SEAT_BEHIND_MIN = 0.006  # paddle centre at least this far on the trailing side
SEAT_SETTLE = 2
SEAT_TIMEOUT = 35
SEAT_MOVING = 0.12  # rad of toggle motion during SEAT that counts as seated
INTEG_GAIN = 0.08
INTEG_BAND = 0.05
INTEG_CLIP = 0.02
SWEEP_STALL_STEPS = 25
SWEEP_STALL_EPS = 0.05
SWEEP_TIMEOUT = 60
EMA_YAW = 0.3
EMA_Q = 0.35
EMA_ALPHA = 0.5
CMD_LEAD_MIN = 0.12
MAX_WAYPOINT = 0.18

PHASE_NAMES = {0: "HOVER", 1: "SEAT", 2: "SWEEP", 3: "RETREAT"}


class FlipSwitchClassicalPolicy(ClassicalPolicyBase):
  """Seat the closed fingertips under the tip on the OFF side, sweep along the arc."""

  orientation_weight = 0.25
  ik_joint_limit_factor = 0.9  # the env's soft limits (see base.py)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._psi = np.zeros(self.num_envs)
      self._psi_init = np.zeros(self.num_envs, dtype=bool)
      self._ema = np.zeros((self.num_envs, 3))
      self._integ = np.zeros((self.num_envs, 3))
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._stall_q = np.zeros(self.num_envs)
      self._stall_t = np.zeros(self.num_envs, dtype=np.int64)
      self._attempts = np.zeros(self.num_envs, dtype=np.int64)
      self._seat_q0 = np.zeros(self.num_envs)
      self._q_ema = np.zeros(self.num_envs)
      self._q_init = np.zeros(self.num_envs, dtype=bool)
      self._stop_n = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._psi_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._stall_q[env_ids] = 0.0
      self._stall_t[env_ids] = 0
      self._attempts[env_ids] = 0
      self._seat_q0[env_ids] = 0.0
      self._q_ema[env_ids] = 0.0
      self._q_init[env_ids] = False
      self._stop_n[env_ids] = 0

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

    # Toggle angle from the fixed marker, mount frame: tip(q) - tip(0) = _MARKER - o2g_m,
    # tip(q) = R_TIP (sin q, 0, cos q) about the +y hinge.
    d = _MARKER - rz.T @ o2g
    q_raw = float(np.clip(np.arctan2(d[0] / _R_TIP, d[2] / _R_TIP + 1.0), -1.0, 1.0))
    # +-1 cm of o2g noise on a 6.2 cm radius is +-9 deg of angle: smooth it (measured
    # without the EMA: 'moving' fired on noise at the first SEAT step and the sweep
    # then drove the toggle backwards through its OFF stop in 4/32 envs).
    if not self._q_init[i]:
      self._q_ema[i] = q_raw
      self._q_init[i] = True
    else:
      self._q_ema[i] += EMA_Q * (q_raw - self._q_ema[i])
    q = float(self._q_ema[i])

    def axis(a: float) -> np.ndarray:  # toggle axis direction (mount frame)
      return np.array([np.sin(a), 0.0, np.cos(a)])

    def trail(a: float) -> np.ndarray:  # -local x: the trailing (OFF) side
      return -np.array([np.cos(a), 0.0, -np.sin(a)])

    pivot_m = -_R_TIP * axis(q)  # site -> pivot, mount frame

    def paddle_m(a: float, standoff: float) -> np.ndarray:  # site -> paddle centre
      return pivot_m + R_C * axis(a) + standoff * trail(a)

    target_axis = rz @ _APPROACH_AXIS_M
    gripper_a = GRIPPER_CLOSED
    ph = int(self._phase[i])
    # W1-G's gravity-sag ratchet (docs/cl_v3/logs/W1-G.md): with cmd_lead_max = 0 the
    # joint command is re-anchored to the sagged / lagging actual joints every step and
    # the site sinks ~1 cm per step on hover-then-descend motions. Keep >= 0.12 rad of
    # command lead in every phase; the pull / lift phases raise it further below.
    self.cmd_lead_max = CMD_LEAD_MIN

    if ph == 0:
      self.max_dq, self.step_gain, self.max_pos_err = 0.25, 0.6, 0.12
      pos_err = gto + rz @ (paddle_m(q, S_OFF + HOVER_BACK) + np.array([0.0, 0.0, HOVER_UP]))
      near = np.linalg.norm(pos_err) < HOVER_TOL
      if near or (self._phase_steps[i] > HOVER_TIMEOUT and np.linalg.norm(pos_err) < 0.06) \
          or self._phase_steps[i] > HOVER_HARD_TIMEOUT:
        self._go(i, 1)
        self._seat_q0[i] = q
    elif ph == 1:
      self.max_dq, self.step_gain, self.max_pos_err = 0.12, 0.4, 0.08
      raw = gto + rz @ paddle_m(q, S_OFF)
      if np.linalg.norm(raw) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -INTEG_CLIP, INTEG_CLIP)
      pos_err = raw + self._integ[i]
      # Signed seat gate in the toggle's own frame: where is the site relative to the
      # contact point on the toggle axis?
      site_m = -(rz.T @ gto) - pivot_m  # site relative to the pivot, mount frame
      rel = site_m - R_C * axis(q)
      behind = float(rel @ trail(q))
      radial = float(rel @ axis(q))
      lateral = float(rel[1])
      seated = (
        behind > SEAT_BEHIND_MIN
        and abs(radial) < SEAT_RAD_TOL
        and abs(lateral) < SEAT_LAT_TOL
      )
      self._settle[i] = self._settle[i] + 1 if seated else 0
      # The seat approach itself often starts the push (measured: the toggle went
      # -0.79 -> -0.41 rad while still in SEAT). A toggle that is already moving is a
      # seated paddle by definition: hand off to the sweep and keep pushing.
      moving = q > self._seat_q0[i] + SEAT_MOVING and np.linalg.norm(raw) < 0.04
      if self._settle[i] >= SEAT_SETTLE or moving:
        self._go(i, 2)
        self._stall_q[i] = q
      elif self._phase_steps[i] > SEAT_TIMEOUT:
        self._go(i, 2 if (behind > -0.01 and abs(lateral) < 0.03) else 0)
        self._stall_q[i] = q
    elif ph == 2:
      self.max_dq, self.step_gain, self.max_pos_err = 0.20, 0.6, 0.12
      self.cmd_lead_max = max(CMD_LEAD_MIN, 0.15)
      self._stop_n[i] = self._stop_n[i] + 1 if q >= STOP_Q else 0
      if self._stop_n[i] >= 2:
        self._go(i, 3)
        return gto + rz @ paddle_m(q, S_OFF + 0.05), target_axis, gripper_a
      self._stall_t[i] += 1
      if self._stall_t[i] >= SWEEP_STALL_STEPS:
        if q - self._stall_q[i] < SWEEP_STALL_EPS:
          self._attempts[i] += 1
          self._go(i, 0)
          return gto + rz @ (paddle_m(q, S_OFF + HOVER_BACK) + np.array([0.0, 0.0, HOVER_UP])), target_axis, gripper_a
        self._stall_q[i] = q
        self._stall_t[i] = 0
      if self._phase_steps[i] > SWEEP_TIMEOUT:
        self._attempts[i] += 1
        self._go(i, 0)
      a = min(q + LEAD_ANG, STOP_Q + 0.1)
      pos_err = gto + rz @ paddle_m(a, S_OFF - PUSH_IN)
    else:
      self.max_dq, self.step_gain, self.max_pos_err = 0.15, 0.5, 0.08
      pos_err = gto + rz @ paddle_m(q, S_OFF + 0.05)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_axis, gripper_a
