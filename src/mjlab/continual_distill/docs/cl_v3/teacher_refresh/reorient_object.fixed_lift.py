"""Scripted ReorientObject teacher -- stand the lying bottle upright in ONE motion (CL-V3, W1-D).

TASK
----
GSO CoQ10 packer bottle, radius 0.0150, half-length 0.0266 (30 x 53 mm, 25 g), spawned
LYING (rolled pi/2, yaw +-pi) with its axis 0.015 above the floor.  Success
(``ReorientObjectCommand``, latched over the 1000-step episode): the bottle's body z-axis
within 0.35 rad of world +z AND the bottle within 0.18 m (xy) of where it spawned.  The
predicate does not care whether the bottle is held or released: a bottle held vertical
already counts.

HISTORY (v2 0.258 / cl25 0.547)
-------------------------------
The v2 teacher (GraspTransport spine) reached alignment in ~1/3 of its failures and still
lost on drift from ~10 ``ee_ground_collision`` resets per episode; the other 2/3 never
gripped the end faces.  Two servo defects found in CL-V3 explain the collisions: the
gravity-sag ratchet (W1-G, ``cmd_lead_max = 0``) and momentum overshoot of a lead-driven
descent (W1-D, Throw); and its ``down_frame`` puts the site x-axis at the yaw although the
pads close along site **y** (measured), so its "end-face" grasp was a barrel grasp.

STRATEGY -- end-face pinch, lift, one wrist rotation
----------------------------------------------------
  0 HOVER    IK (axis-only, lead 0.12 against the sag, anti-windup lateral integrator)
             above the bottle; a joint-7 yaw servo puts the CLOSING axis (site y) along the
             bottle's axis, candidate nearest the hand's current yaw (the axis is a line).
  1 DESCEND  closed-loop IK descent, small lead, 6 mm/step, to site z 0.031: the pads
             (site -0.014 .. +0.003) cover the flat end faces over 0.017-0.030.
  2 CLOSE    closed-loop hold on the bottle's xy (absolute, base frame) until the site is
             within 5 mm, then squeeze; held check on the SMOOTHED aperture (bottle length
             0.053 -> aperture ~0.05; < 0.040 = pads met on nothing -> retry).
  3 LIFT     straight up 0.12 (lead 0.12), held check.
  4 ROTATE   full-rotation IK target: closing axis -> s * world z with s chosen so the
             bottle's +z end goes UP (s = sign of axis . closing direction at the grasp),
             approach axis -> horizontal, radially outward from the base (the family of
             frames with R y = z is free in yaw, so the hand always ends pointing away from
             the base).  Held for the rest of the episode -- success has latched.
  5 SETTLE   keep the pose.

Observations (60-D): 0:9 joint_pos_rel, 34:40 object_orientation (rows 1-2 of the
bottle's rotation matrix; body z = column 2), 40:43 gripper_to_object.  All FK on the
arm's joints (base frame); no absolute obs term is used.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  FRANKA_ACTION_SCALE,
  HOME_QPOS,
  ClassicalPolicyBase,
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

P_HOVER = 0
P_DESCEND = 1
P_CLOSE = 2
P_LIFT = 3
P_ROTATE = 4
P_SETTLE = 5
PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "CLOSE", 3: "LIFT", 4: "ROTATE", 5: "SETTLE"}

JOINT_LIMITS = np.array(
  [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
   [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]]
)

# -- bottle geometry ------------------------------------------------------------------
OBJ_AXIS_Z = 0.015           # lying bottle: axis height (collision hull max radius)
HOVER_SITE_Z = 0.15
# W1-D2: with the descent ratchet removed (cmd_ref = "command" on z, below) the site now
# LANDS on this height instead of sinking ~12 mm past it.  0.022 is where the 26/32
# successful grasps of the ratcheting teacher actually closed (measured from the diagnose
# trace: mean 0.0217, range 0.014-0.033): pads (site -0.012 .. +0.004) then span
# 0.010-0.026 across the bottle's axis height 0.015, i.e. the full chord of the end caps.
GRASP_SITE_Z = 0.020
FLOOR_MIN_Z = 0.018          # site floor (superseded by PAD_FLOOR_MIN below; kept as a backstop)
# W1-D2, MEASURED: the site floor is the WRONG guard.  The pads hang 0.012 below the site
# only while the hand is exactly vertical; the axis-only IK lets it tilt, and a tilt of th
# drops the outer pad by 0.04 sin(th) -- 7 mm at 10 deg.  The 4 remaining ground-collision
# envs had the site at 0.015-0.017 (guard satisfied) with a pad already on the floor.  The
# guard is now the true lowest pad corner, computed from the internal model.
PAD_FLOOR_MIN = 0.006  # swept on CPU at n=32 (seed 7): 0.006/0.020 -> 32/32, 0.008/0.020 -> 27/32
ALIGN_TOL = 0.012
ALIGN_SETTLE = 2
ALIGN_TIMEOUT = 60
INTEG_GAIN = 0.08
INTEG_CLIP = 0.03
INTEG_BAND = 0.04
# W1-D2, MEASURED: rate-limiting the ERROR (cmd[2] = max(z_err, -rate)) does NOT produce a
# rate-limited descent.  The DLS solve only realises ~40 % of a 10 mm request (the posture
# regulariser is a standing ~1 cm bias, M4) and step_gain halves it again, so a 10 mm/step
# request moved the site 2.1 mm/step and DESCEND timed out 6-14 cm above the bottle.
# Instead the phase carries a WAYPOINT that walks down at the rate and the servo is given
# the FULL error to it; a leash keeps the waypoint within DESCENT_LEASH of the actual site
# so the command can never run away from the arm (that runaway is what put the pads on the
# floor in the ratcheting version).
DESCENT_RATE = 0.005
DESCENT_RATE_FAST = 0.010    # until DESCENT_SLOW_BAND above the grasp height
DESCENT_SLOW_BAND = 0.04
DESCENT_LEASH = 0.020
SEAT_TOL_Z = 0.006
SEAT_TOL_X = 0.006           # along the pad width (site x)
SEAT_TOL_Y = 0.012           # along the closing axis (pads centre the bottle)
SEAT_SETTLE = 2
DESCEND_TIMEOUT = 60
CLOSE_XY_TOL = 0.005
CLOSE_DELAY = 14
CLOSE_SQUEEZE = 10
APERTURE_HELD = 0.040        # end-face: bottle length 0.053; barrel: diameter 0.030 -> see _held_min
APERTURE_EMA = 0.3
LIFT_HEIGHT = 0.18
LIFT_STEPS = 24
HELD_TOL = 0.05
ROTATE_STEPS = 80
YAW_RATE = 0.15
EMA_ALPHA = 0.35
EMA_ALPHA_LOW = 0.2
RESET_JOINT_JUMP = 0.25
RESET_OBJ_JUMP = 0.15
LEAD_HOVER = 0.12
LEAD_DESCEND = 0.08
LEAD_CLOSE = 0.08
MAX_ATTEMPTS = 8


def _object_axis(obs_i: np.ndarray) -> np.ndarray:
  """Bottle body z-axis in world frame from the rot6d observation (rows 1-2 of R)."""
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  r1 = r1 / (np.linalg.norm(r1) + 1e-9)
  r2 = r2 / (np.linalg.norm(r2) + 1e-9)
  r0 = np.cross(r1, r2)
  return np.array([r0[2], r1[2], r2[2]])


class ReorientObjectClassicalPolicy(ClassicalPolicyBase):
  """End-face pinch of the lying bottle, lift, rotate the closing axis to vertical."""

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.05
  step_gain = 0.5
  # W1-D2, MEASURED: with the default cmd_ref = "actual" and a command lead, the base
  # rebuilds the target as FK(command) + err every step, so the servo's gravity sag is
  # ADDED to the descent each step instead of staying a constant offset.  Traced on the
  # n=32 diagnose: the site sank to 0.010-0.020 against a 0.034 command and the finger
  # pads (site - 0.012) hit the floor -- ee_ground_collision in 6/6 of the failures.
  # Referencing only z to the command turns the z channel into an integrator that
  # converges on the commanded height (xy stays anchored to the actual site).
  cmd_ref = "command"
  cmd_ref_axes = (2,)
  orientation_weight = 0.5
  cmd_lead_max = LEAD_HOVER
  PHASE_NAMES = PHASE_NAMES

  # -- state -----------------------------------------------------------------------------

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      n = self.num_envs
      self._gto_ema = np.zeros((n, 3))
      self._ema_init = np.zeros(n, dtype=bool)
      self._prev_gto = np.zeros((n, 3))
      self._prev_q = np.full((n, 7), np.nan)
      self._integ = np.zeros((n, 3))
      self._settle = np.zeros(n, dtype=np.int64)
      self._yaw = np.zeros(n)
      self._yaw_ok = np.zeros(n, dtype=bool)
      self._sign = np.ones(n)
      self._mode = ["end"] * n
      self._hold = np.zeros((n, 3))
      self._closing = np.zeros(n, dtype=bool)
      self._close_k = np.zeros(n, dtype=np.int64)
      self._ap_ema = np.full(n, 0.08)
      self._ap_init = np.zeros(n, dtype=bool)
      self._rot_target = np.zeros((n, 3, 3))
      self._attempts = np.zeros(n, dtype=np.int64)
      self._zwp = np.full(n, np.nan)
      self._cur_env = 0
      self.debug = [[] for _ in range(n)]
    ids = range(self.num_envs) if env_ids is None else env_ids
    for i in ids:
      self._clear(i)
      self._prev_q[i] = np.nan
      self._attempts[i] = 0

  def _clear(self, i: int) -> None:
    self._zwp[i] = np.nan
    self._ema_init[i] = False
    self._integ[i] = 0.0
    self._settle[i] = 0
    self._yaw_ok[i] = False
    self._closing[i] = False
    self._close_k[i] = 0

  def _rewind(self, i: int, reason: str = "") -> None:
    self.debug[i].append((int(self._phase[i]), reason))
    self._attempts[i] += 1
    self._phase[i] = P_HOVER
    self._phase_steps[i] = 0
    self._clear(i)

  def _goto(self, i: int, phase: int) -> None:
    self._zwp[i] = np.nan
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._closing[i] = False
    self._close_k[i] = 0

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._gto_ema[i] = raw
      self._ema_init[i] = True
    else:
      a = EMA_ALPHA_LOW if self._phase[i] in (P_DESCEND, P_CLOSE) else EMA_ALPHA
      self._gto_ema[i] = a * raw + (1 - a) * self._gto_ema[i]
    return self._gto_ema[i]

  def _update_aperture(self, i: int, obs_i: np.ndarray) -> None:
    raw = float(self.default_qpos[7] + obs_i[7] + self.default_qpos[8] + obs_i[8])
    if not self._ap_init[i]:
      self._ap_ema[i] = raw
      self._ap_init[i] = True
    else:
      self._ap_ema[i] = APERTURE_EMA * raw + (1 - APERTURE_EMA) * self._ap_ema[i]

  def _aperture(self, i: int) -> float:
    return float(self._ap_ema[i])

  def _held_min(self, i: int) -> float:
    return APERTURE_HELD if self._mode[i] == "end" else 0.020

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    q7 = obs_i[0:7]
    raw = obs_i[40:43]
    jumped = False
    if not np.any(np.isnan(self._prev_q[i])):
      if np.max(np.abs(q7 - self._prev_q[i])) > RESET_JOINT_JUMP:
        jumped = True
    if self._ema_init[i] and np.linalg.norm(raw - self._prev_gto[i]) > RESET_OBJ_JUMP:
      jumped = True
    self._prev_q[i] = q7
    self._prev_gto[i] = raw
    if jumped:
      self._rewind(i, "reset")
      self._attempts[i] = 0

  # -- yaw servo -------------------------------------------------------------------------

  def _closing_yaw_target(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> float:
    """Closing axis along the bottle axis; the candidate (psi or psi+pi) nearest the hand's
    current closing yaw; latched per attempt.  Also latches the sign that puts the +z end
    along the closing direction (+1) or against it (-1)."""
    if not self._yaw_ok[i]:
      p, R = self._fk(q_abs)
      ax = _object_axis(obs_i)
      psi = float(np.arctan2(ax[1], ax[0]))
      # Closing yaw ~ RADIAL (from the base): end-face pinch if the bottle lies radially,
      # barrel pinch if it lies tangentially.  Bottles lying tangentially failed every
      # end-face attempt (the wrist stays at its HOME yaw and the descent stalled).
      hand = float(np.arctan2(R[1, 1], R[0, 1]))
      cands = psi + np.array([0.0, np.pi])
      modes = ["end", "end"]
      d = np.angle(np.exp(1j * (cands - hand)))
      # required q7 = q7_now - d (measured d(yaw)/dq7 = -1): keep it inside the joint-7
      # range (+-2.9 rad).  Bottles lying tangentially at large |y| needed q7 past the
      # limit for the nearer candidate, the servo saturated and every attempt missed.
      q7_req = q_abs[6] - d
      ok = np.abs(q7_req) < 2.6
      cost = np.abs(d) + np.where(ok, 0.0, 10.0)
      k = int(np.argmin(cost))
      self._yaw[i] = float(np.angle(np.exp(1j * cands[k])))
      self._mode[i] = modes[k]
      ey = np.array([np.cos(self._yaw[i]), np.sin(self._yaw[i]), 0.0])
      if self._mode[i] == "end":
        self._sign[i] = 1.0 if float(np.dot(ax, ey)) >= 0.0 else -1.0
      else:
        ex = np.cross(ey, _DOWN_AXIS)  # pad-width axis = y x z
        self._sign[i] = 1.0 if float(np.dot(ax, ex)) >= 0.0 else -1.0
      self._yaw_ok[i] = True
    return float(self._yaw[i])

  # -- action ----------------------------------------------------------------------------

  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    self._cur_env = i
    self._update_aperture(i, obs_i)
    self._detect_reset(i, obs_i)
    ph = self._phase[i]
    q_abs = self.default_qpos + obs_i[0:9]
    self.cmd_lead_max = {P_CLOSE: LEAD_CLOSE, P_DESCEND: LEAD_DESCEND}.get(ph, LEAD_HOVER)
    a = super()._act_single(i, obs_i)
    if ph <= P_LIFT:
      # joint-7 yaw servo: rotates about the (vertical) approach axis, null space of the
      # axis-only IK; measured d(closing yaw)/dq7 = -1 with the hand pointing down.
      _p, R = self._fk(q_abs)
      yaw_now = float(np.arctan2(R[1, 1], R[0, 1]))
      d_yaw = float(np.angle(np.exp(1j * (self._closing_yaw_target(i, obs_i, q_abs) - yaw_now))))
      dq7 = float(np.clip(-d_yaw, -YAW_RATE, YAW_RATE))
      q7 = float(np.clip(q_abs[6] + dq7, JOINT_LIMITS[6, 0], JOINT_LIMITS[6, 1]))
      a[6] = (q7 - self.default_qpos[6]) / FRANKA_ACTION_SCALE
      self._q_cmd[i][6] = q7
    return a

  def _integrate(self, i: int, err: np.ndarray) -> None:
    lateral = np.array([err[0], err[1], 0.0])
    if np.linalg.norm(lateral) < INTEG_BAND:
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * lateral, -INTEG_CLIP, INTEG_CLIP)

  def _pad_min_z(self) -> float:
    """Lowest corner of the two finger pads at the pose last passed to ``_fk``."""
    if not hasattr(self, "_pad_gids"):
      self._pad_gids = [self.model.geom(n).id for n in ("left_finger_pad", "right_finger_pad")]
      self._pad_hs = [self.model.geom_size[g].copy() for g in self._pad_gids]
    best = 1e9
    for g, hs in zip(self._pad_gids, self._pad_hs):
      pos = self.data.geom_xpos[g]
      R = self.data.geom_xmat[g].reshape(3, 3)
      z = float(pos[2] - abs(R[2, 0]) * hs[0] - abs(R[2, 1]) * hs[1] - abs(R[2, 2]) * hs[2])
      best = min(best, z)
    return best

  def _guard(self, err: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    z = float(self._fk(q_abs)[0][2])
    pad = self._pad_min_z()
    err = err.copy()
    if pad + err[2] < PAD_FLOOR_MIN:
      err[2] = PAD_FLOOR_MIN - pad
    if z + err[2] < FLOOR_MIN_Z:
      err[2] = FLOOR_MIN_Z - z
    return err

  def _upright_rot(self, i: int, q_abs: np.ndarray) -> np.ndarray:
    """Upright frame: closing axis (site y) -> s*z.  The approach direction is free in
    yaw; try eight horizontal directions with an offline DLS solve from the current
    posture (hold the site where it is) and keep the most reachable one -- a fixed
    'radially outward' choice was unreachable in ~1/3 of the envs and the DLS then sank
    the site 9 cm into the floor (traced)."""
    p, _R = self._fk(q_abs)
    best = None
    for k in range(8):
      th = 2 * np.pi * k / 8
      ez = np.array([np.cos(th), np.sin(th), 0.0])
      if self._mode[i] == "end":
        ey = np.array([0.0, 0.0, self._sign[i]])   # closing axis carries the bottle axis
        ex = np.cross(ey, ez)
      else:
        ex = np.array([0.0, 0.0, self._sign[i]])   # pad-width axis carries the bottle axis
        ey = np.cross(ez, ex)
      R_t = np.column_stack([ex, ey, ez])
      q = q_abs.copy()
      for _ in range(150):
        pq, _Rq = self._fk(q)
        dq = self._ik_step(q, p - pq, R_t)
        q[:7] += dq
        if np.linalg.norm(dq) < 1e-5:
          break
      pq, Rq = self._fk(q)
      e_r = 0.5 * (np.cross(Rq[:, 0], R_t[:, 0]) + np.cross(Rq[:, 1], R_t[:, 1]) + np.cross(Rq[:, 2], R_t[:, 2]))
      lim = np.sum(np.maximum(0.0, JOINT_LIMITS[:, 0] + 0.05 - q[:7]) + np.maximum(0.0, q[:7] - JOINT_LIMITS[:, 1] + 0.05))
      cost = float(np.linalg.norm(p - pq) + 0.3 * np.linalg.norm(e_r) + lim)
      if best is None or cost < best[0]:
        best = (cost, R_t)
    return best[1]

  def _target_error(self, i: int, obs_i: np.ndarray):
    q_abs = self.default_qpos + obs_i[0:9]
    gto = self._gto(i, obs_i)
    ph = self._phase[i]
    rot = _DOWN_AXIS

    if ph == P_HOVER:
      self._closing_yaw_target(i, obs_i, q_abs)
      err = np.array([gto[0], gto[1], gto[2] + (HOVER_SITE_Z - OBJ_AXIS_Z)])
      self._integrate(i, err)
      if np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < 0.04:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= ALIGN_SETTLE or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._goto(i, P_DESCEND)
      return self._guard(err + self._integ[i], q_abs), rot, GRIPPER_OPEN

    if ph == P_DESCEND:
      p_act, R = self._fk(q_abs)
      pad_slack = max(self._pad_min_z() - PAD_FLOOR_MIN, 0.0)
      z_goal = max(p_act[2] + gto[2] + (GRASP_SITE_Z - OBJ_AXIS_Z), p_act[2] - pad_slack)
      if not np.isfinite(self._zwp[i]):
        self._zwp[i] = p_act[2]
      rate = DESCENT_RATE_FAST if (self._zwp[i] - z_goal) > DESCENT_SLOW_BAND else DESCENT_RATE
      self._zwp[i] = max(z_goal, self._zwp[i] - rate, p_act[2] - DESCENT_LEASH)
      err = np.array([gto[0], gto[1], self._zwp[i] - p_act[2]])
      self._integrate(i, err)
      z_err = p_act[2] - z_goal
      cmd = err + self._integ[i]
      cmd[2] = err[2]
      e_hand = R.T @ np.array([err[0], err[1], 0.0])
      if abs(z_err) < SEAT_TOL_Z and abs(e_hand[0]) < SEAT_TOL_X and abs(e_hand[1]) < SEAT_TOL_Y:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= SEAT_SETTLE or self._phase_steps[i] > DESCEND_TIMEOUT:
        self._goto(i, P_CLOSE)
      return self._guard(cmd, q_abs), rot, GRIPPER_OPEN

    if ph == P_CLOSE:
      p, _R = self._fk(q_abs)
      if not self._closing[i]:
        self._hold[i] = np.array([p[0] + gto[0], p[1] + gto[1], GRASP_SITE_Z])
      err = self._hold[i] - p
      if not self._closing[i]:
        self._settle[i] = self._settle[i] + 1 if np.linalg.norm(err[:2]) < CLOSE_XY_TOL else 0
        if self._settle[i] >= 2 or self._phase_steps[i] >= CLOSE_DELAY:
          self._closing[i] = True
          self._close_k[i] = 0
      if self._closing[i]:
        self._close_k[i] += 1
        if self._close_k[i] >= CLOSE_SQUEEZE:
          if self._aperture(i) < self._held_min(i) and self._attempts[i] < MAX_ATTEMPTS:
            self._rewind(i, f"close_aperture {self._aperture(i):.3f}")
            return np.zeros(3), rot, GRIPPER_OPEN
          # Keep the lift above the grasp. A fresh relative upward request with
          # zero XY correction lets the posture objective pull the bottle inward.
          self._hold[i] = p.copy()
          self._hold[i, 2] += LIFT_HEIGHT
          self._goto(i, P_LIFT)
      grip = GRIPPER_CLOSED if self._closing[i] else GRIPPER_OPEN
      return self._guard(err, q_abs), rot, grip

    if ph == P_LIFT:
      if self._phase_steps[i] >= LIFT_STEPS:
        if (np.linalg.norm(gto) > HELD_TOL or self._aperture(i) < self._held_min(i)) and self._attempts[i] < MAX_ATTEMPTS:
          self._rewind(i, f"lift gto={np.round(gto, 3).tolist()} ap={self._aperture(i):.3f}")
          return np.zeros(3), rot, GRIPPER_OPEN
        self._rot_target[i] = self._upright_rot(i, q_abs)
        self._goto(i, P_ROTATE)
        return np.zeros(3), rot, GRIPPER_CLOSED
      p, _R = self._fk(q_abs)
      return self._guard(self._hold[i] - p, q_abs), rot, GRIPPER_CLOSED

    # ROTATE / SETTLE: hold the site, rotate the hand to the upright frame.
    if ph == P_ROTATE and self._phase_steps[i] >= ROTATE_STEPS:
      self._goto(i, P_SETTLE)
    p, _R = self._fk(q_abs)
    return self._guard(self._hold[i] - p, q_abs), self._rot_target[i], GRIPPER_CLOSED
