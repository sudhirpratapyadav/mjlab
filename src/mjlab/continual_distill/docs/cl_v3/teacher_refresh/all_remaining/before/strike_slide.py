"""Scripted StrikeSlide teacher -- accelerate the puck in contact and let it go (CL-V3, W1-D2).

TASK
----
Regulation ice-hockey puck (r 0.0381, half-height 0.0127, 0.16 kg) spawned at x 0.32-0.44,
y +-0.15; goal band x 0.88-1.05, y +-0.18, i.e. 0.42-0.90 m of slide, deliberately beyond
the arm's ~0.85 m stretch.  Success (``PushingCommand``): ||puck - goal|| < 0.08, latched
over the 200-step (4.0 s) episode.

DECISION D5 (applied, see ``config/franka/env_cfgs.py``)
--------------------------------------------------------
The task was impossible as specified: MuJoCo combines pair friction by MAX unless a geom
has ``priority``, so the terrain plane's default 1.0 defeated puck.xml's documented 0.4 and
the puck-floor pair ran at mu_eff = 1.02-1.07 (measured), needing a 2.9-4.2 m/s launch
against an arm whose best flat end-effector speed anywhere is 1.8 m/s.  The Strike scene
now gives the puck geom ``priority = 1`` at an ice-like mu = 0.04, chosen so the goal band
needs a 0.57-0.84 m/s launch -- inside what a contact launch actually delivers at the
puck's spawn radius (0.77-0.93 m/s measured).  Tool-Pull's puck is untouched.

WHAT DOES NOT WORK (measured, W1-D + W1-D2)
--------------------------------------------
* A golf swing / punch at the puck -- the previous version of this file (0.008 at n=128)
  and W1-D's planar-swing prototype.  Re-run at n=6 on the CL-V3 spec: 38 of 42
  episodes ended in ``ee_ground_collision`` (the fast arc dips 1-2 cm below its planned
  height; static-sag compensation does not remove it), only 1 of 6 envs made contact at
  all, and that hit left the puck 37 deg off line.  A free-flying impact between a
  17 mm pad face and a 76 mm disc is a knife edge in both speed and direction.
* The old site-height floor guard (``site z >= 0.030``).  The pads hang 0.012 below the
  site only while the hand is exactly vertical; the axis-only IK lets it tilt and a tilt
  of th drops the outer pad by 0.04 sin(th) -- 7 mm at 10 deg.  This file guards the true
  lowest pad corner instead, computed from the internal model.

STRATEGY -- a shuffleboard launch, not an impact
--------------------------------------------------
The puck leaves at the speed of whatever is touching it, so bring it up to speed IN
CONTACT and then stop: no impact, no restitution lottery, and the direction is set by the
pusher's path rather than by where on the rim the blow lands.

  0 HOVER    above the seat point, one puck-radius + a pad behind the puck along the
             goal line, hand down, fingers CLOSED (the closed pads are the paddle).
  1 DESCEND  waypoint descent (rate-limited waypoint + full error to it -- the DLS solve
             realises only ~40 % of a rate-limited ERROR, so rate-limiting the error does
             not rate-limit the descent) to site z 0.020: the pads then span 0.008-0.024
             across the puck's 0-0.0254 rim, with 8 mm of floor clearance.
  2 SEAT     close the last centimetre onto the rim at that height.
  3 PUSH     a waypoint walks along the goal line, its speed ramped to v_target over
             ACCEL_STEPS and then held, with a leash so the command can never run away
             from the arm; lateral error to the puck's centre is servoed out every step.
             v_target = sqrt(2 mu_eff g d) * SPEED_CAL, d read once from object_to_goal.
  4 RETREAT  up and back the moment the ramp is done -- the puck is now faster than the
             hand and separates cleanly.
  5 HOLD     station-keep so a sliding puck is never touched again.

Observations (60-D): 0:9 joint_pos_rel, 40:43 gripper_to_object (puck - gripper),
43:46 object_to_goal (goal - puck).  Relative terms only.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0
YAW_RATE = 0.15              # rad/step on the joint-7 servo that keeps the wedge square

P_HOVER = 0
P_DESCEND = 1
P_SEAT = 2
P_PUSH = 3
P_RETREAT = 4
P_HOLD = 5
PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "SEAT", 3: "PUSH", 4: "RETREAT", 5: "HOLD"}

PUCK_RADIUS = 0.0381
PUCK_HALF_H = 0.0127

# Pad geometry MEASURED on the panda XML in the SITE frame (fingers closed): the pads
# span site x +-0.0088, site y +-0.0152, and hang 0.0119 below the site.  With the hand
# vertical and site x along the push direction the paddle is therefore a flat 30 x 16 mm
# face standing 8.8 mm in front of the site -- so the site-to-puck-centre distance AT
# CONTACT is 0.0381 + 0.0088 = 0.047.  The first version's SEAT_STANDOFF of 0.045 was
# INSIDE the puck and the seat phase bulldozed it (traced: puck at 1.46 m/s in SEAT).
RIDE_SITE_Z = 0.028          # pads span 0.016-0.032 across the puck's 0-0.0254 rim.  A
                             # push at that height cannot tip the puck: tipping needs
                             # F > m g R / h = 2.4 N, the push itself is ~1.0 N.
PAD_FLOOR_MIN = 0.014        # guard on the true lowest pad corner (not on the site)
HOVER_Z = 0.13
# THE WEDGE (W1-D3).  W1-D2's residual was the paddle: one flat 17.6 mm pad face against a
# 76.2 mm disc.  A flat face or a corner touching a cylinder always pushes THROUGH the
# centre, so it applies no torque -- but the line of action is the RADIUS to the contact
# point, so a lateral offset e launches the puck asin(e/R) off line (5 mm -> 7.5 deg ->
# 0.079 m over a 0.6 m slide, against an 0.08 m test) AND the offset GROWS during the push,
# because a single off-centre contact squirts the puck sideways.  Measured direction errors
# of 20-49 deg say the offset reached 13-29 mm.
# Opening the jaws to WEDGE_HALF per side turns the two pads into a symmetric V.  The puck
# then makes TWO contacts whose radial normals are mirror images: the lateral components
# cancel, the offset is self-correcting instead of self-amplifying, and -- with two
# contacts on a 2-DOF planar body -- the puck's position relative to the hand is fully
# determined, so it travels at EXACTLY the hand's velocity and leaves with exactly the
# hand's speed and direction.  D5's mu = 0.04 puck (priority 1, so every puck contact runs
# at 0.04, the pads included) is what keeps the wedge from binding.
WEDGE_HALF = 0.030           # per-finger opening: pad inner faces at +-0.030 vs a 0.0381
                             # puck radius -> contacts at asin(0.030/0.0381) = 51.9 deg
                             # off the goal line, each contributing cos = 0.617 forward
GRIPPER_WEDGE = WEDGE_HALF / 0.04 - 1.0   # finger target = 0.04 * (action + 1)
CONTACT_STANDOFF = 0.0323    # site -> puck centre with both pad edges on the rim:
                             # 0.0088 (pad front face ahead of the site)
                             # + sqrt(0.0381^2 - 0.030^2)
STANDOFF = 0.045             # hover/descend: the pad edges then sit 0.047 from the puck
                             # centre, 9 mm clear of its 0.0381 rim
SEAT_STANDOFF = 0.0353       # 3 mm short of contact
Z_RATE = 0.004               # cap on the commanded height change per step once low

ALIGN_TOL = 0.015
ALIGN_TIMEOUT = 60
# The hand starts at (r 0.67, z 0.47) and the seat point is INSIDE it at r ~0.27, so the
# straight-line error is 0.35 lateral by 0.35 down.  Traced: descending on that diagonal
# put the site at z 0.038 while still 8 cm short laterally -- the closed pads ploughed
# into the puck and shoved it 4 cm backwards before HOVER had converged.  The hand now
# only descends once it is inside HOVER_LATERAL_GATE of the seat point.
HOVER_LATERAL_GATE = 0.08
DESCENT_RATE = 0.006
DESCENT_RATE_FAST = 0.012
DESCENT_SLOW_BAND = 0.04
DESCENT_LEASH = 0.020
DESCEND_TIMEOUT = 55
SEAT_TOL = 0.006
SEAT_TIMEOUT = 30

# PUSH.  v_target (m/s) -> per-step waypoint advance v_target * DT.
DT = 0.02
ACCEL_STEPS = 16             # ramp 0 -> v_target
HOLD_STEPS = 14              # steps at constant v_target before letting go
PUSH_MAX_DQ = 0.20           # never binding: the per-step command increment is ~0.06 rad
APPROACH_MAX_DQ = 0.12
# Height loop.  With z referenced to the previous COMMAND the height channel is an
# INTEGRATOR, and the arm answers a command with a ~5-step lag, so a per-step gain near 1
# is unstable: traced at Z_GAIN 0.5 the site limit-cycled between 0.014 and 0.042 (command
# 0.024-0.078) and clipped the floor on every trough.  0.15 is inside the delay margin.
Z_GAIN = 0.15
PUSH_TIMEOUT = 40
MU_EFF = 0.04                # measured on the D5 scene: slide = v^2 / (2 mu g)
G = 9.81
SPEED_CAL = 1.0              # correction on the ballistic law (measured below)
# The command is driven OVERDRIVE times faster than the speed we actually want, and the
# phase ends the moment the MEASURED site speed reaches the target -- open-loop timing
# cannot work here because the arm's radial speed depends on where in its workspace the
# push happens (0.3 m/s of spread between the near and far ends of the same push).
OVERDRIVE = 1.35
SPEED_WIN = 5                # steps in the finite-difference speed estimate (FK on the
                             # observed joints: +-0.01 rad -> ~4 mm -> 0.06 m/s at 5 steps)
# The finite difference over SPEED_WIN steps estimates the speed at the MIDDLE of the
# window, and the hand is still accelerating, so the instantaneous speed at the release
# decision is higher than the estimate by about a*SPEED_WIN*DT/2.  Releasing on
# v_meas >= v_target therefore launched the puck 10-25 % fast (measured: radial landing
# error +0.14 to +0.43 m).  The threshold carries the correction.
RELEASE_FRAC = 0.88
PUSH_LAT_GAIN = 0.0         # the wedge self-centres; a lateral servo on the noisy live
                            # puck observation (+-0.01 m) only fights it
V_MIN, V_MAX = 0.5, 2.2

RETREAT_STEPS = 24
RETREAT_UP = 0.14
RETREAT_BACK = 0.10

INTEG_GAIN = 0.10
INTEG_CLIP = 0.03
INTEG_BAND = 0.05
EMA_ALPHA = 0.35
# W1-D4.  The seat's lateral accuracy is the whole game (measured, see below), and the
# only estimate of the puck's position is `gripper_to_object`, whose observation noise is
# 0.010-0.013 m PER AXIS.  An EMA at alpha 0.35 averages ~3 samples, so it leaves ~6-7 mm
# of lateral sd -- a third of the wedge's +-0.030 capture basin.  The puck does not move
# until the pads touch it, so the derived ABSOLUTE puck position (FK(observed q) + the
# relative observation) can be averaged over EVERY approach step instead: 40-60 samples
# take the sd to ~1.5 mm.  The mean is frozen the moment the pads could touch the puck.
RESET_JUMP = 0.12
LEAD_APPROACH = 0.12
LEAD_PUSH = 0.60             # the servo may lag the command freely along the push
MAX_WAYPOINT = 0.30


class StrikeSlideClassicalPolicy(ClassicalPolicyBase):
  """Seat the closed pads on the puck's rim, accelerate it along the goal line, let go."""

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.12
  orientation_weight = 0.2
  # M4, MEASURED (W1-D2, physics-free sweep of the base DLS): with the default
  # posture_weight 0.005 the solve does not converge on low, near targets -- asked for
  # (r 0.24-0.36, z 0.02-0.03) top-down it stops 2.6-3.3 cm short in radius and ~1 cm
  # high, EVERY time, because the regulariser pulls toward HOME_QPOS and nothing balances
  # it.  With posture_weight 0 the same solve lands on the target to 1e-3.  The seat point
  # for this task is one puck-radius INSIDE the puck (r 0.27-0.39), i.e. exactly where the
  # bias bites: the hand stalled at r 0.36 and never touched the puck.  Damping (0.2)
  # still conditions the solve; redundancy is resolved minimum-norm.
  posture_weight = 0.0
  cmd_lead_max = LEAD_APPROACH
  # Reference the z channel to the previous COMMAND (W1-G's additive knob): with the
  # default "actual" the servo's gravity sag is re-added to the target every step and a
  # descent ratchets into the floor (measured on Reorient: the site sank 12 mm past its
  # command).  Referencing z to the command turns it into an integrator on the height.
  cmd_ref = "command"
  cmd_ref_axes = (2,)
  PHASE_NAMES = PHASE_NAMES

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    n = self.num_envs
    if env_ids is None:
      self._dir = np.zeros((n, 3))
      self._dir_ok = np.zeros(n, dtype=bool)
      self._req = np.zeros(n)
      self._v_target = np.zeros(n)
      self._settle = np.zeros(n, dtype=np.int64)
      self._integ = np.zeros((n, 3))
      self._zwp = np.full(n, np.nan)
      self._wp = np.zeros((n, 3))
      self._wp_ok = np.zeros(n, dtype=bool)
      self._v = np.zeros(n)
      self._prev_gto = np.zeros((n, 3))
      self._anchor = np.zeros((n, 3))
      self._puck0 = np.zeros((n, 3))
      self._pmean = np.zeros((n, 3))
      self._pn = np.zeros(n, dtype=np.int64)
      self._phist = np.zeros((n, SPEED_WIN + 1, 3))
      self._phist_k = np.zeros(n, dtype=np.int64)
      self._yaw_target = np.zeros(n)
    ids = range(n) if env_ids is None else env_ids
    for i in ids:
      self._clear(i)

  def _clear(self, i: int) -> None:
    self._dir_ok[i] = False
    self._settle[i] = 0
    self._integ[i] = 0.0
    self._zwp[i] = np.nan
    self._wp_ok[i] = False
    self._v[i] = 0.0
    self._pn[i] = 0
    self._pmean[i] = 0.0

  def _goto(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._zwp[i] = np.nan
    self._wp_ok[i] = False

  def _rewind(self, i: int) -> None:
    """The env auto-reset underneath us (ground collision / out of bounds)."""
    self._phase[i] = P_HOVER
    self._phase_steps[i] = 0
    self._clear(i)

  # -- helpers ---------------------------------------------------------------------------

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

  def _guard(self, err: np.ndarray, q_abs: np.ndarray | None = None) -> np.ndarray:
    # The base builds the z target from FK(previous COMMAND) (cmd_ref = "command"), so a
    # guard evaluated only at the ACTUAL pose lets the command sit a sag below the floor.
    # Bound both references.
    pad = self._pad_min_z()
    if q_abs is not None and self.cmd_lead_max > 0.0:
      i = self._guard_env
      if not np.any(np.isnan(self._q_cmd[i])):
        q_ref = q_abs.copy()
        q_ref[:7] = np.clip(self._q_cmd[i], q_abs[:7] - self.cmd_lead_max,
                            q_abs[:7] + self.cmd_lead_max)
        self._fk(q_ref)
        pad = min(pad, self._pad_min_z())
        self._fk(q_abs)
    err = err.copy()
    if pad + err[2] < PAD_FLOOR_MIN:
      err[2] = PAD_FLOOR_MIN - pad
    n = float(np.linalg.norm(err))
    if n > MAX_WAYPOINT:
      err = err * (MAX_WAYPOINT / n)
    return err

  def _puck_abs(self, i: int, obs_i: np.ndarray, p_act: np.ndarray, live: bool) -> np.ndarray:
    """Puck centre in the base frame, averaged over every approach step.

    ``gripper_to_object`` carries 10-13 mm of observation noise per axis, but the puck is
    STATIC until the pads reach it, so ``FK(q_obs) + gto`` is the same quantity measured
    afresh every step and its mean converges as 1/sqrt(n).  ``live`` is False once the
    hand is close enough to disturb the puck: the mean is then frozen.
    """
    if live:
      self._pn[i] += 1
      self._pmean[i] += (p_act + obs_i[40:43] - self._pmean[i]) / float(self._pn[i])
    elif self._pn[i] == 0:
      self._pmean[i] = p_act + obs_i[40:43]
      self._pn[i] = 1
    return self._pmean[i]

  def _latch(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    if not self._dir_ok[i]:
      o2g = obs_i[43:46]
      d = float(np.linalg.norm(o2g[:2]))
      self._req[i] = d
      self._dir[i] = np.array([o2g[0] / (d + 1e-9), o2g[1] / (d + 1e-9), 0.0])
      # Ballistic requirement: the puck decelerates at mu g, so v^2 = 2 mu g d.
      v = float(np.sqrt(2.0 * MU_EFF * G * d)) * SPEED_CAL
      self._v_target[i] = float(np.clip(v, V_MIN, V_MAX))
      self._dir_ok[i] = True
    return self._dir[i]

  # -- action ----------------------------------------------------------------------------

  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    a = super()._act_single(i, obs_i)
    # Joint-7 servo (the axis-only IK leaves the roll about the approach axis free, and
    # d(closing yaw)/dq7 = -1 with the hand down): the wedge only works square to the goal
    # line -- a rotated V puts one pad ahead of the other and the two normals stop
    # cancelling.
    q_abs = self.default_qpos + obs_i[0:9]
    _p, R = self._fk(q_abs)
    yaw_now = float(np.arctan2(R[1, 1], R[0, 1]))
    d_yaw = float(np.angle(np.exp(2j * (self._yaw_target[i] - yaw_now))) / 2.0)
    dq7 = float(np.clip(-d_yaw, -YAW_RATE, YAW_RATE))
    q7 = float(np.clip(q_abs[6] + dq7, -2.8973, 2.8973))
    a[6] = (q7 - self.default_qpos[6]) / 0.04
    self._q_cmd[i][6] = q7
    return a

  def _target_error(self, i: int, obs_i: np.ndarray):
    q_abs = self.default_qpos + obs_i[0:9]
    raw = obs_i[40:43]
    if self._pn[i] > 0 and np.linalg.norm(raw - self._prev_gto[i]) > RESET_JUMP:
      self._rewind(i)
    self._prev_gto[i] = raw
    self._guard_env = i
    self.cmd_ref_axes = (2,)
    self.max_dq = APPROACH_MAX_DQ
    d = self._latch(i, obs_i)
    self._yaw_target[i] = float(np.arctan2(d[1], d[0])) + np.pi / 2.0
    p_act, _R = self._fk(q_abs)
    ph = self._phase[i]
    self.cmd_lead_max = LEAD_PUSH if ph == P_PUSH else LEAD_APPROACH
    # Puck centre in the base frame (relative obs + own FK: no scene-origin offset),
    # averaged over the whole approach while the puck is still untouched.
    puck = self._puck_abs(i, obs_i, p_act, live=(ph == P_HOVER))

    if ph == P_HOVER:
      tgt = puck - d * STANDOFF
      err = np.array([tgt[0] - p_act[0], tgt[1] - p_act[1], HOVER_Z - p_act[2]])
      lat = np.array([err[0], err[1], 0.0])
      if np.linalg.norm(lat) > HOVER_LATERAL_GATE:
        err[2] = max(err[2], 0.0)  # travel at altitude; never descend across the puck
      if np.linalg.norm(lat) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * lat, -INTEG_CLIP, INTEG_CLIP)
      if np.linalg.norm(lat) < ALIGN_TOL and abs(err[2]) < 0.04:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= 2 or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._puck0[i] = puck.copy()  # frozen mean: chasing the LIVE (noisy) puck while
        self._goto(i, P_DESCEND)      # descending let the hand bulldoze it (traced)
      return self._guard(err + self._integ[i], q_abs), _DOWN_AXIS, GRIPPER_WEDGE

    if ph == P_DESCEND:
      pad_slack = max(self._pad_min_z() - PAD_FLOOR_MIN, 0.0)
      z_goal = max(RIDE_SITE_Z, p_act[2] - pad_slack)
      if not np.isfinite(self._zwp[i]):
        self._zwp[i] = p_act[2]
      rate = DESCENT_RATE_FAST if (self._zwp[i] - z_goal) > DESCENT_SLOW_BAND else DESCENT_RATE
      self._zwp[i] = max(z_goal, self._zwp[i] - rate, p_act[2] - DESCENT_LEASH)
      tgt = self._puck0[i] - d * STANDOFF
      err = np.array([tgt[0] - p_act[0], tgt[1] - p_act[1], self._zwp[i] - p_act[2]])
      if abs(p_act[2] - RIDE_SITE_Z) < SEAT_TOL or self._phase_steps[i] > DESCEND_TIMEOUT:
        self._goto(i, P_SEAT)
      return self._guard(err, q_abs), _DOWN_AXIS, GRIPPER_WEDGE

    if ph == P_SEAT:
      tgt = self._puck0[i] - d * SEAT_STANDOFF
      err = np.array([tgt[0] - p_act[0], tgt[1] - p_act[1],
                      float(np.clip(Z_GAIN * (RIDE_SITE_Z - p_act[2]), -Z_RATE, Z_RATE))])
      if np.linalg.norm(err[:2]) < SEAT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= 2 or self._phase_steps[i] > SEAT_TIMEOUT:
        self._phist[i, :] = p_act
        self._goto(i, P_PUSH)
      return self._guard(err, q_abs), _DOWN_AXIS, GRIPPER_WEDGE

    if ph == P_PUSH:
      # MEASURED, and this is the whole difficulty of the phase: a Cartesian waypoint held
      # 5-10 cm ahead of the hand does drive it to ~1 m/s, but the base scales the joint
      # step UNIFORMLY down to max_dq, so the small "hold your height" component is scaled
      # by the same factor as the big along-track one and vanishes -- the site dipped to
      # 0.014 against a 0.026/0.030 command (identically at both, i.e. it is not a
      # proportional sag) and ee_ground_collision fired in 4-6 of every 6 episodes.
      # Instead the command is stepped along the path ONE control step at a time
      # (v * dt ~ 20 mm, about 0.05 rad of joint motion: never clipped), referenced to the
      # previous command on ALL axes, with a large lead so the servo may lag as far behind
      # as it needs to.  The hand then follows a genuinely FLAT path at the commanded
      # speed, and the lag it carries is along-track, not downward.
      self.cmd_ref_axes = None
      self.cmd_lead_max = LEAD_PUSH
      self.max_dq = PUSH_MAX_DQ
      k = self._phase_steps[i]
      self._phist[i, min(k, SPEED_WIN)] = p_act
      # Release on the MEASURED site speed along the push direction, not on a step count.
      if k >= SPEED_WIN:
        v_meas = float(np.dot(p_act - self._phist[i, 0], d)) / (SPEED_WIN * DT)
        self._phist[i, :SPEED_WIN] = self._phist[i, 1:]
      else:
        v_meas = 0.0
      frac = min(1.0, (k + 1) / ACCEL_STEPS)
      self._v[i] = self._v_target[i] * OVERDRIVE * frac
      z_corr = float(np.clip(Z_GAIN * (RIDE_SITE_Z - p_act[2]), -Z_RATE, Z_RATE))
      err = d * (self._v[i] * DT) + np.array([0.0, 0.0, z_corr])
      # Keep the pad face centred on the puck: the paddle is 30 mm wide against a 76 mm
      # disc, and an off-centre contact torques it off line (measured direction errors of
      # 20-49 deg on the worst hits).
      perp = np.array([-d[1], d[0], 0.0])
      lat = float(np.dot(puck - p_act, perp))
      err = err + perp * (PUSH_LAT_GAIN * lat)
      if v_meas >= RELEASE_FRAC * self._v_target[i] or self._phase_steps[i] > PUSH_TIMEOUT:
        self._goto(i, P_RETREAT)
      return self._guard(err, q_abs), _DOWN_AXIS, GRIPPER_WEDGE

    if ph == P_RETREAT:
      if self._phase_steps[i] >= RETREAT_STEPS:
        self._goto(i, P_HOLD)
      err = -d * RETREAT_BACK + np.array([0.0, 0.0, RETREAT_UP])
      return err, _DOWN_AXIS, GRIPPER_WEDGE

    return np.zeros(3), _DOWN_AXIS, GRIPPER_WEDGE
