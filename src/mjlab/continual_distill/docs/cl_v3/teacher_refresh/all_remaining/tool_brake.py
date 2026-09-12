"""Scripted ToolPull teacher policy (CL-V3 rewrite, W1-G, 2026-09-09): REAL TOOL USE.

Pinch the stick's shaft near its centre of mass WITHOUT lifting it, slide the stick
across the floor until its hook bar is behind the puck, pull the puck into the near
zone, let go. A direct closed-finger push of the puck is the fallback (the puck spawns
at radial 0.62-0.71, at the edge of the top-down envelope; measured with this repo's
DLS: 0.9-2.0 cm residual out to r 0.76).

WHY THE STICK IS NOT LIFTED, AND WHY THE PINCH USED TO FAIL (measured geometry)
--------------------------------------------------------------------------------
stick.xml: shaft = cylinder r 0.011, half-length 0.13 along body +x (0.059 kg); hook bar
= cylinder r 0.011 from (0.12, 0) to (0.12, 0.07) (0.016 kg); ``object_site`` at
(-0.09, 0, 0); friction 1.0. Resting on the floor the shaft's axis is at z 0.011.
The finger pads (panda.xml FK): centre 3.7 mm below the ``gripper`` site, half-height
8.25 mm, so the pad bottom is 12 mm below the site; the lowest collidable hand geom is
~14 mm below it and ``ee_ground_collision`` terminates the episode on contact, so the
site cannot go much below ~0.027.
* A vertical pinch on a floor-resting cylinder holds against sliding out DOWNWARD only
  if mu > tan(theta), theta = the contact angle above the axis. Site 0.030 -> pad bottom
  0.018 -> theta = asin(7/11) = 40 deg, tan 0.83: with MuJoCo's max-of-the-pair friction
  rule (pad 0.3-1.5 randomised, stick 1.0) this is a coin flip -- the bimodal 20 % loss
  phase-1 measured. Site 0.027 -> theta 21 deg (tan 0.38) is much better, but the plan
  below never needs to lift at all.
* Phase-1's "squeeze ejection" is the pad's bottom EDGE meeting the dowel's upper flank
  at the finger servo's impact speed (~0.4 m/s: kp 350 / damping 10 on a 15 g finger):
  the edge contact pushes the dowel down and sideways and it pops out under a pad.
  Fix: ramp the finger command from open to closed over CLOSE_RAMP steps so the pads
  arrive at ~3 mm/step, and close 3 mm lower.
* A dowel pushed SIDEWAYS rolls (rolling torque F x 0.018 vs the bar's resisting weight
  torque 0.157 N x 0.035 = 0.0055 Nm; sliding needs 0.74 N, rolling starts at 0.3 N) --
  except toward the bar's side, where the bar digs into the floor. A PINCHED dowel is
  roll-proof (two flank contacts, torque capacity ~0.08 Nm), so once pinched the stick
  can be slid in any direction, and pulled axially with ~7.7 N of friction capacity
  against the ~1.4 N the stick + the 160 g puck need.
* The pinch holds the stick's yaw only weakly against floor friction, so the stick's
  yaw is read from the observation every step and the hand follows it.

PLAN (phases; every one has an unconditional timeout)
------------------------------------------------------
  HOVER      top-down, closing axis perpendicular to the shaft (from tool_orientation),
             settle xy over the pinch point (shaft, 2 cm ahead of the body origin = near
             the CoM) at HOVER_Z; small lateral integrator (yaw-constrained DLS bias).
  DESCEND    vertical, rate-limited, to PINCH_SITE_Z (FK height, EMA'd), seat settle.
  CLOSE      ramp the finger action +1 -> -1 over CLOSE_RAMP steps, hold, keep servoing.
  CHECK      EMA'd aperture from the finger joints: pinched if 0.012 < a < 0.034, else
             open and retry (MAX_PINCH_TRIES), then fall back to the direct push.
  CLEAR      if the puck is within 0.10 of the shaft line (stick frame v), slide -y_s so
             the bar's tip cannot clip the puck during ADVANCE.
  ADVANCE    slide +x_s until the puck's stick-frame u <= U_HOOK (0.055): the bar's -x
             face (u 0.109) is then > 1 cm behind the puck's +x rim.
  SWEEP      slide +y_s until the puck's v <= V_HOOK (0.06): the puck sits in the bar's
             span, 1 cm off the shaft.
  PULL       slide the hand along (goal - puck), clipped to +-40 deg of the stick's -x
             axis (friction keeps the puck on the bar face inside that cone), until the
             puck is within GOAL_TOL of the goal.
  RELEASE    open, rise; DONE if at goal, else the direct push (PUSH_* phases, the old
             teacher's shepherd) finishes the last centimetres from a comfortable reach.
Stick-follow check on every sliding phase: if gripper_to_tool drifts > LOST_TOL from
its value at the pinch, the stick slipped -> open and re-pinch.

OBSERVATION LAYOUT (69-D, measured against the live env)
--------------------------------------------------------
    [ 0: 9] robot_joint_pos (rel)  [ 9:18] robot_joint_vel
    [18:21] object_pos (puck, ABSOLUTE, never used)   [21:25] object_quat
    [25:28] gripper_pos (ABSOLUTE, never used)        [28:34] gripper_orientation
    [34:40] object_orientation (puck)                 [40:43] gripper_to_object (puck - gripper)
    [43:46] object_to_goal (goal - puck)              [46:52] goal_orientation_diff
    [52:55] gripper_to_tool (stick object_site - gripper)
    [55:61] tool_orientation (stick rotation matrix rows 1,2: obs[55:58] = (sin, cos, 0))
    [61:69] control_qpos_diff
The gripper's env-local position comes from FK on the arm's own joints (robot-base
frame, offset-free); every object position is gripper + a relative term.

SUCCESS (ToolPullCommand): || puck_site - (env_origin + (0.42, 0, 0.0127)) || < 0.07,
latched. Budget 600 steps (12 s). ``tool_grasped`` is tracked but does not gate success.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  HOME_QPOS,
  ClassicalPolicyBase,
  lowest_hand_z,
)

GOAL_LOCAL = np.array([0.42, 0.0, 0.0127])
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

# stick.xml body frame.
STICK_SITE_X = -0.09  # object_site along the shaft
BAR_X = 0.12  # hook bar centre-line
BAR_LEN = 0.07  # bar from y=0 to y=0.07
STICK_R = 0.011
PUCK_R = 0.0381

# Phases.
P_HOVER, P_DESCEND, P_CLOSE, P_CHECK = 0, 1, 2, 3
P_CLEAR, P_ADVANCE, P_SWEEP, P_PULL, P_RELEASE = 4, 5, 6, 7, 8
P_PUSH_HOVER, P_PUSH_DESCEND, P_PUSH, P_DONE = 9, 10, 11, 12
PHASE_NAMES = {
  0: "HOVER", 1: "DESCEND", 2: "CLOSE", 3: "CHECK", 4: "CLEAR", 5: "ADVANCE",
  6: "SWEEP", 7: "PULL", 8: "RELEASE", 9: "PUSH_HOVER", 10: "PUSH_DESCEND",
  11: "PUSH", 12: "DONE",
}

# -- pinch --------------------------------------------------------------------
PINCH_U = 0.02  # pinch point along the shaft from the body origin (CoM is at ~0.026)
HOVER_Z = 0.16
PINCH_SITE_Z = 0.030  # the empirically safe site floor (terminations seen at 0.020-0.022); pad bottom 0.018
FLOOR_MIN_Z = 0.026  # legacy site-height floor (kept for reference)
PAD_CLEARANCE = 0.012  # required clearance of the LOWEST hand point above the ground
ALIGN_TOL = 0.020  # 0.012 never settled (32/32 hover timeouts, diagnose)
ALIGN_SETTLE = 3
ALIGN_TIMEOUT = 60
HOVER_INTEG_GAIN = 0.05  # 0.15/0.04 with the 0.12 lead ORBITED the target (5-8 cm radius, CPU trace)
HOVER_INTEG_CLIP = 0.02
HOVER_INTEG_NEAR = 0.05  # integrate only within this lateral distance (no wind-up on approach)
DESCENT_RATE = 0.015
DESCENT_RATE_LOW = 0.003  # below DESCENT_SLOW_Z: keeps the servo lag (and so the command lead) near zero
DESCENT_SLOW_Z = 0.12  # a 1.5 cm/step descent let the lead build to ~5 cm and the site overshot into the floor (CPU trace)
SEAT_TOL = 0.004
SEAT_XY_TOL = 0.012
DESCEND_TIMEOUT = 60
CLOSE_RAMP = 12
CLOSE_HOLD = 8
CHECK_STEPS = 8
AP_MIN, AP_MAX = 0.012, 0.034
MAX_PINCH_TRIES = 3
LOST_TOL = 0.04

# -- sliding ------------------------------------------------------------------
SLIDE_RATE = 0.012  # m per control step (0.6 m/s)
V_CLEAR = 0.10
U_HOOK = 0.055
V_HOOK = 0.06
V_ESCAPE = 0.105  # puck slid off the bar's span
CLEAR_TIMEOUT = 30
ADVANCE_TIMEOUT = 50
SWEEP_TIMEOUT = 60
PULL_TIMEOUT = 80
PULL_CONE = np.radians(40.0)
GOAL_TOL = 0.045
RELEASE_STEPS = 15

# -- direct push fallback (the cl_v2 teacher's constants) -----------------------
# CL-V3 (W1-G2) -- the paddle has to BITE the puck. The puck is 25.4 mm tall (site z
# 0.0127 = its half-height) and the closed pads' lowest point is 12 mm below the gripper
# site, so at RIDE_Z 0.034 only the pad's bottom 3.4 mm overlapped the puck: a corner
# contact 3 mm below the puck's top face, which rides over it or tips it instead of
# pushing. 0.028 puts the pad bottom at 0.016 and gives 9.4 mm of engagement, still 16 mm
# of ground clearance (and the whole 32-env direct-push run reached P_PUSH with ZERO
# ground terminations at 0.034, so the margin is real once ``cmd_ref`` is "actual").
RIDE_Z = 0.028
PUSH_HOVER_Z = 0.16
PUSH_DESCEND_RATE = 0.012
# CL-V3 (W1-G2) -- THE PADDLE WAS BEING COMMANDED THROUGH THE PUCK.
# The pad's front face has to sit on the puck's REAR rim: puck radius 0.0381 plus the
# pad's 0.0088 half-width puts the gripper site 0.047 behind the puck's centre along the
# goal direction. The old law commanded ``gto + behind*1.8 + lead*goal_dir`` with
# behind = 0.055 and lead = min(dist, 0.09), which nets out to 0.009 BEHIND the puck's
# centre -- i.e. the site was told to go 3.8 cm PAST the contact point, straight through
# the puck, so the pad climbed onto its top face and pressed DOWN (which also raises the
# puck's own floor friction) instead of pushing. MEASURED (diagnose, 32 envs, 600 steps
# each): the puck moved 8-77 mm against the 150-220 mm it needs, `closest
# |gripper_to_object|` 0.017 (the hand is right on top of it), and the site dipped to
# 0.014-0.020, i.e. the pad dragging on the floor.
# The Group-P recipe instead: park the pad ON the contact point and command a bounded
# LEAD past it, proportional to the remaining distance.
CONTACT_BACK = 0.047  # site offset behind the puck centre along the goal direction
PUSH_LEAD_MIN = 0.015
PUSH_LEAD_MAX = 0.055  # 0.070 measured no better (0.297 -> 0.250 at n=32-64)
PUSH_LEAD_GAIN = 0.6
BEHIND = 0.070  # approach/descend standoff: land the pad clear behind the puck
BEHIND_BIAS = 1.8  # legacy, no longer used by P_PUSH
APPROACH_TOL = 0.030
PUSH_STEP = 0.09
PUSH_HOVER_TIMEOUT = 30
PUSH_DESCEND_TIMEOUT = 30
# Re-approach thresholds (Group-P recipe): a 1.8 cm paddle on a 7.6 cm disc squirts the
# puck sideways, and once the hand is level with or beside the puck, continuing to
# "push" just shoves it further off the line. Lift over and come in again from behind.
REAPPROACH_ALONG = 0.022  # hand no longer behind the puck along the goal direction
REAPPROACH_CROSS = 0.065  # hand this far off the puck-goal line

EMA_ALPHA = 0.45
RESET_Q_JUMP = 0.25
RESET_JUMP = 0.15


def closing_frame(theta: float) -> np.ndarray:
  """z down, finger-closing axis (site y) at heading ``theta``."""
  c, s = np.cos(theta), np.sin(theta)
  ey = np.array([c, s, 0.0])
  ez = np.array([0.0, 0.0, -1.0])
  ex = np.cross(ey, ez)
  return np.column_stack([ex, ey, ez])


class ToolPullClassicalPolicy(ClassicalPolicyBase):
  """Pinch the stick (no lift), slide it to hook the puck, pull; direct-push fallback."""

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.06
  orientation_weight = 0.3
  PHASE_NAMES = PHASE_NAMES
  # Command lead (base.py): without it the arm sags ~1 cm/step under gravity while
  # commanded up (measured, see stack_object.GraspTransportPolicy.lead).
  cmd_lead_max = 0.12
  # CL-V3 (W1-G, 2026-09-09) -- ``cmd_ref = "command"`` DEFEATS THE FLOOR GUARD and was
  # the whole of this task's 32/32 ``ee_ground_collision``. With it the IK target on the
  # guarded axis is FK(previous COMMAND) + err, while ``_guard`` clamps ``err`` against
  # the ACTUAL site height -- so once the command leads downward (up to cmd_lead_max =
  # 0.12 rad ~ 6 cm at the site) the target is built from a point already below the
  # floor and the clamp is meaningless. MEASURED (diagnose #4, n = 32): the site fell
  # 0.105 -> 0.067 -> 0.045 -> 0.020 in four steps against a 6 mm/step command and a
  # 0.029 guard, ending at 0.007-0.026 in every env. With "actual" the target is always
  # FK(actual) + guarded err, so the guard binds; the lead (which is what stops the
  # gravity-sag ratchet) is kept.
  cmd_ref = "actual"
  cmd_ref_axes = (2,)  # only consulted when cmd_ref == "command"
  # Skip the tool entirely and shepherd the puck with the closed gripper as a paddle
  # (the PUSH_* phases). A/B switch, default off.
  direct_push_only = False

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    n = self.num_envs
    if env_ids is None:
      self._gto = np.zeros((n, 3))
      self._gtt = np.zeros((n, 3))
      self._ema_ok = np.zeros(n, dtype=bool)
      self._psi = np.zeros((n, 2))
      self._z = np.zeros(n)
      self._z_raw = np.zeros(n)
      self._low = np.zeros(n)
      self._ap = np.zeros(n)
      self._integ = np.zeros((n, 3))
      self._settle = np.zeros(n, dtype=np.int64)
      self._tries = np.zeros(n, dtype=np.int64)
      self._gtt_ref = np.zeros((n, 3))
      self._yaw_t = np.full(n, np.nan)
      self._prev_q = np.zeros((n, 7))
      self._prev_gto = np.zeros((n, 3))
      self._have_prev = np.zeros(n, dtype=bool)
    else:
      self._ema_ok[env_ids] = False
      self._integ[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._tries[env_ids] = 0
      self._yaw_t[env_ids] = np.nan
      self._have_prev[env_ids] = False

  # -- state helpers -----------------------------------------------------------

  def _go(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._integ[i] = 0.0

  def _rewind(self, i: int) -> None:
    self._go(i, P_HOVER)
    self._yaw_t[i] = np.nan

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    q = obs_i[0:7]
    g = obs_i[40:43]
    if self._have_prev[i]:
      if (
        np.max(np.abs(q - self._prev_q[i])) > RESET_Q_JUMP
        or np.linalg.norm(g - self._prev_gto[i]) > RESET_JUMP
      ):
        self._rewind(i)
        self._tries[i] = 0
        self._ema_ok[i] = False
    self._prev_q[i] = q
    self._prev_gto[i] = g
    self._have_prev[i] = True

  def _observe(self, i: int, obs_i: np.ndarray) -> None:
    """EMA every noisy quantity this teacher uses."""
    raw_gto = obs_i[40:43]
    raw_gtt = obs_i[52:55]
    row1 = obs_i[55:58]
    v = np.array([row1[0], row1[1]])
    v = v / (np.linalg.norm(v) + 1e-9)
    q_abs = self.default_qpos + obs_i[0:9]
    ee_pos, ee_rot = self._fk(q_abs)
    z = float(ee_pos[2])
    self._z_raw[i] = z
    self._low[i] = lowest_hand_z(ee_pos, ee_rot, float(q_abs[7]))
    ap = 0.08 + obs_i[7] + obs_i[8]  # finger joints are relative to the 0.04 open default
    if not self._ema_ok[i]:
      self._gto[i] = raw_gto
      self._gtt[i] = raw_gtt
      self._psi[i] = v
      self._z[i] = z
      self._ap[i] = ap
      self._ema_ok[i] = True
    else:
      a = EMA_ALPHA
      self._gto[i] = a * raw_gto + (1 - a) * self._gto[i]
      self._gtt[i] = a * raw_gtt + (1 - a) * self._gtt[i]
      self._psi[i] = a * v + (1 - a) * self._psi[i]
      self._z[i] = 0.5 * z + 0.5 * self._z[i]
      self._ap[i] = 0.3 * ap + 0.7 * self._ap[i]

  def _stick_frame(self, i: int):
    psi = float(np.arctan2(self._psi[i][0], self._psi[i][1]))
    xs = np.array([np.cos(psi), np.sin(psi), 0.0])
    ys = np.array([-np.sin(psi), np.cos(psi), 0.0])
    return psi, xs, ys

  def _closing_heading(self, i: int, obs_i: np.ndarray, psi: float) -> float:
    """Closing axis perpendicular to the shaft; the branch nearest the current one."""
    if np.isnan(self._yaw_t[i]):
      _, r = self._fk(self.default_qpos + obs_i[0:9])
      ref = float(np.arctan2(r[1, 1], r[0, 1]))
    else:
      ref = float(self._yaw_t[i])
    base = psi + np.pi / 2
    k = np.round((ref - base) / np.pi)
    cand = base + k * np.pi
    if np.isnan(self._yaw_t[i]):
      self._yaw_t[i] = cand
    else:
      d = (cand - self._yaw_t[i] + np.pi) % (2 * np.pi) - np.pi
      self._yaw_t[i] += 0.3 * d
    return float(self._yaw_t[i])

  def _guard(self, err: np.ndarray, i: int) -> np.ndarray:
    """Clamp the commanded descent on the TRUE lowest hand point, not the site.

    With the jaws open the pad centres sit 0.0476 off the site axis, so a few degrees of
    wrist tilt drops the outer pad corner several millimetres below where a site-height
    guard believes it is -- and this teacher works with 1.4-1.8 cm of ground clearance.
    ``lowest_hand_z`` (base.py) evaluates the pad corners and the hand capsule from the
    same FK the site height comes from, so the margin below is a real clearance.
    """
    z = min(self._z[i], self._z_raw[i])  # the EMA lags on a descent: guard on the lower estimate
    low = self._low[i]
    slack = z - low  # how far the lowest hand point is below the site, this pose
    if z + err[2] < PAD_CLEARANCE + slack:
      err = err.copy()
      err[2] = PAD_CLEARANCE + slack - z
    return err

  # -- the state machine ---------------------------------------------------------

  def _target_error(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i)
    self._observe(i, obs_i)
    # THE LEAD IS ASYMMETRIC IN ITS EFFECTS, so it is switched per phase.
    # Holding a height near the floor NEEDS it: the 32-env direct-push run sat at
    # RIDE_Z for 600 steps with zero ground terminations. The DESCENT must not have it:
    # with a lead the command runs up to cmd_lead_max (~6 cm at the site) BELOW the
    # actual joints on the way down, and that stored displacement is spent after the
    # guard has already said stop. MEASURED (diagnose, cmd_ref="actual", n=32): the site
    # fell 0.088 -> 0.068 -> 0.045 -> 0.020 at 10-17 mm/step against a 6 mm/step command
    # and a 0.026 guard, 32/32 ee_ground_collision, every one of them in DESCEND.
    # With cmd_lead_max = 0 the joint command is re-anchored to the actual joints every
    # step, so nothing is stored and the guard binds on the very step it fires; the
    # gravity ratchet that costs is the one on a HOVER, and this phase is descending.
    self.cmd_lead_max = 0.12
    err, rot, grip = self._plan(i, obs_i)
    return self._guard(np.asarray(err, dtype=np.float64), i), rot, grip

  def _plan(self, i: int, obs_i: np.ndarray):
    if self.direct_push_only and self._phase[i] < P_PUSH_HOVER:
      self._go(i, P_PUSH_HOVER)
    ph = self._phase[i]
    t = self._phase_steps[i]
    gto = self._gto[i]  # puck - gripper
    gtt = self._gtt[i]  # stick site - gripper
    o2g = obs_i[43:46]  # goal - puck
    psi, xs, ys = self._stick_frame(i)
    z = self._z_raw[i]
    # Stick body origin and the puck, relative to the gripper (xy).
    body = gtt + (-STICK_SITE_X) * xs
    pinch = body + PINCH_U * xs  # the point on the shaft the pads close on
    rel = gto - body
    u = float(np.dot(rel[:2], xs[:2]))
    v = float(np.dot(rel[:2], ys[:2]))
    goal_dir = o2g[:2] / (np.linalg.norm(o2g[:2]) + 1e-8)
    dist = float(np.linalg.norm(o2g[:2]))

    # Latched success zone: stop touching anything, rise and hold.
    if dist < GOAL_TOL and ph not in (P_DONE, P_RELEASE):
      self._go(i, P_RELEASE)
      ph = P_RELEASE
      t = 0

    if ph <= P_PULL:
      rot = closing_frame(self._closing_heading(i, obs_i, psi))

    if ph == P_HOVER:
      lateral = np.array([pinch[0], pinch[1], 0.0])
      if np.linalg.norm(lateral) < HOVER_INTEG_NEAR:
        self._integ[i] = np.clip(
          self._integ[i] + HOVER_INTEG_GAIN * lateral, -HOVER_INTEG_CLIP, HOVER_INTEG_CLIP
        )
      err = np.array([pinch[0], pinch[1], HOVER_Z - z])
      if np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < 0.04:
        self._settle[i] += 1
        if self._settle[i] >= ALIGN_SETTLE:
          integ = self._integ[i].copy()
          self._go(i, P_DESCEND)
          self._integ[i] = integ  # keep the bias correction for the descent
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      if t > ALIGN_TIMEOUT:
        integ = self._integ[i].copy()
        self._go(i, P_DESCEND)
        self._integ[i] = integ
      return err + self._integ[i], rot, GRIPPER_OPEN

    if ph == P_DESCEND:
      lateral = np.array([pinch[0], pinch[1], 0.0])
      if np.linalg.norm(lateral) < HOVER_INTEG_NEAR:
        self._integ[i] = np.clip(
          self._integ[i] + HOVER_INTEG_GAIN * lateral, -HOVER_INTEG_CLIP, HOVER_INTEG_CLIP
        )
      z_err = PINCH_SITE_Z - z
      rate = DESCENT_RATE_LOW if z < DESCENT_SLOW_Z else DESCENT_RATE
      err = np.array([pinch[0], pinch[1], max(z_err, -rate)])
      if abs(z_err) < SEAT_TOL and np.linalg.norm(pinch[:2]) < SEAT_XY_TOL:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          integ = self._integ[i].copy()
          self._go(i, P_CLOSE)
          self._integ[i] = integ
      else:
        self._settle[i] = 0
      if t > DESCEND_TIMEOUT:
        integ = self._integ[i].copy()
        self._go(i, P_CLOSE)
        self._integ[i] = integ
      return err + self._integ[i], rot, GRIPPER_OPEN

    if ph == P_CLOSE:
      # Ramp the finger command so the pads arrive at ~3 mm/step, not at impact speed.
      grip = float(np.clip(1.0 - 2.0 * (t + 1) / CLOSE_RAMP, -1.0, 1.0))
      err = np.array([0.5 * pinch[0], 0.5 * pinch[1], PINCH_SITE_Z - z])
      if t >= CLOSE_RAMP + CLOSE_HOLD:
        self._go(i, P_CHECK)
      return err + self._integ[i], rot, grip

    if ph == P_CHECK:
      err = np.array([0.0, 0.0, PINCH_SITE_Z - z])
      if t >= CHECK_STEPS:
        if AP_MIN < self._ap[i] < AP_MAX:
          self._gtt_ref[i] = gtt.copy()
          self._go(i, P_CLEAR if v < V_CLEAR else P_ADVANCE)
        else:
          self._tries[i] += 1
          if self._tries[i] >= MAX_PINCH_TRIES:
            self._go(i, P_PUSH_HOVER)
          else:
            self._rewind(i)
          return np.array([0.0, 0.0, 0.05]), rot, GRIPPER_OPEN
      return err, rot, GRIPPER_CLOSED

    if ph in (P_CLEAR, P_ADVANCE, P_SWEEP, P_PULL):
      # Stick-follow check: the pinch slipped if the stick moved relative to the hand.
      if np.linalg.norm(gtt - self._gtt_ref[i]) > LOST_TOL:
        self._tries[i] += 1
        if self._tries[i] >= MAX_PINCH_TRIES:
          self._go(i, P_PUSH_HOVER)
        else:
          self._rewind(i)
        return np.array([0.0, 0.0, 0.06]), rot, GRIPPER_OPEN

      if ph == P_CLEAR:
        d = ys * min(max(V_CLEAR - v, 0.0), SLIDE_RATE) * -1.0
        if v >= V_CLEAR - 0.003 or t > CLEAR_TIMEOUT:
          self._go(i, P_ADVANCE)
      elif ph == P_ADVANCE:
        d = xs * min(max(u - U_HOOK, 0.0), SLIDE_RATE)
        if u <= U_HOOK + 0.003 or t > ADVANCE_TIMEOUT:
          self._go(i, P_SWEEP)
      elif ph == P_SWEEP:
        d = ys * min(max(v - V_HOOK, 0.0), SLIDE_RATE)
        if v <= V_HOOK + 0.005 or t > SWEEP_TIMEOUT:
          self._go(i, P_PULL)
      else:  # P_PULL
        # Pull along the goal direction, inside the friction cone about -x_s.
        back = -xs[:2]
        ang = float(np.arctan2(back[0] * goal_dir[1] - back[1] * goal_dir[0], np.dot(back, goal_dir)))
        ang = float(np.clip(ang, -PULL_CONE, PULL_CONE))
        c, s = np.cos(ang), np.sin(ang)
        pull = np.array([c * back[0] - s * back[1], s * back[0] + c * back[1]])
        d = np.array([pull[0], pull[1], 0.0]) * min(dist, SLIDE_RATE)
        if v > V_ESCAPE or u > BAR_X + 0.02 or t > PULL_TIMEOUT:
          self._go(i, P_RELEASE)
      err = np.array([d[0], d[1], PINCH_SITE_Z - z])
      return err, rot, GRIPPER_CLOSED

    if ph == P_RELEASE:
      if t >= RELEASE_STEPS:
        self._go(i, P_DONE if dist < GOAL_TOL else P_PUSH_HOVER)
      return np.array([0.0, 0.0, 0.10 - min(z, 0.10) + 0.02]), _DOWN_AXIS, GRIPPER_OPEN

    # -- direct push fallback (cl_v2 shepherd) ---------------------------------------
    behind = np.array([-goal_dir[0] * BEHIND, -goal_dir[1] * BEHIND, 0.0])
    if ph == P_PUSH_HOVER:
      err = gto + behind
      err[2] = PUSH_HOVER_Z - z
      if (np.linalg.norm(err[:2]) < APPROACH_TOL and t > 6) or t > PUSH_HOVER_TIMEOUT:
        self._go(i, P_PUSH_DESCEND)
      return err, _DOWN_AXIS, GRIPPER_CLOSED
    if ph == P_PUSH_DESCEND:
      err = gto + behind
      err[2] = np.clip(RIDE_Z - z, -PUSH_DESCEND_RATE, PUSH_DESCEND_RATE)
      if abs(RIDE_Z - z) < 0.012 or t > PUSH_DESCEND_TIMEOUT:
        self._go(i, P_PUSH)
      return err, _DOWN_AXIS, GRIPPER_CLOSED
    if ph == P_PUSH:
      along = float(np.dot(gto[:2], goal_dir))
      cross = float(np.linalg.norm(gto[:2] - along * goal_dir))
      # ``along`` is 0.047 when the pad is correctly seated on the puck's rear rim;
      # below REAPPROACH_ALONG the hand has climbed onto or past the puck.
      if (along < REAPPROACH_ALONG or cross > REAPPROACH_CROSS) and dist > GOAL_TOL:
        # Lift over the puck and come back in from behind, on the corrected line.
        self._go(i, P_PUSH_HOVER)
        return np.array([0.0, 0.0, PUSH_HOVER_Z - z]), _DOWN_AXIS, GRIPPER_CLOSED
      lead = float(np.clip(PUSH_LEAD_GAIN * dist, PUSH_LEAD_MIN, PUSH_LEAD_MAX))
      off = lead - CONTACT_BACK
      err = gto + np.array([goal_dir[0] * off, goal_dir[1] * off, 0.0])
      err[2] = np.clip(RIDE_Z - z, -PUSH_DESCEND_RATE, PUSH_DESCEND_RATE)
      return err, _DOWN_AXIS, GRIPPER_CLOSED
    # P_DONE: hold high, hands off.
    return np.array([0.0, 0.0, max(0.16 - z, 0.0)]), _DOWN_AXIS, GRIPPER_OPEN
