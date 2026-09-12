"""Scripted OpenDoor teacher — FRONT PINCH on the bar pull, dragged round the hinge arc.

CL-V3 REWRITE (W1-M, 2026-09-09). Strategy first, then constants.

WHY THE CL-V2 SIDE PINCH COULD NOT WORK ON THE FROZEN v3 SPEC (measured, diagnose n=32,
HEAD b9b0563 + W0 init spec: 0/32; 21/32 "mechanism_not_moved", 8/32 never seated;
n=128 baseline 0.070)
------------------------------------------------------------------------------------
The v2 teacher came in from the leaf's FREE-EDGE side with the finger-closing axis along
the pull tangent, i.e. the hand capsule (r 0.04, half-length 0.06) lay BESIDE the leaf.
Reconstructing the gripper site in the leaf frame from the trace: in every failing env
the site parked at y = 0.30-0.33 (the leaf's free edge) and never reached the bar at
y = 0.24-0.26 — the carcass's free-side wall (y 0.305-0.323, x >= 0.015) is only 4.5 cm
from the bar and the capsule fouls on it. The mount yaw (+-15 deg) is a second, smaller
problem: the old code assumed the leaf faces +x.

The budget is NOT the problem (decision D4 measured, not needed): envs that did seat
swung the door at 0.55-0.70 rad per 25 steps (~1.3 rad/s), so 90 deg needs ~60 of the
150 steps.

A TOP HOOK WAS TRIED AND IS GEOMETRICALLY DEAD (measured 0/32, 6/32, 0/32 over three
iterations + render frames): closed fingertips lowered from above into the 25 mm slot
between bar and leaf. The hand capsule sits 7 cm up the approach axis with r 0.04, so
a vertical hand intrudes 2.7 cm into the leaf face and the approach must tilt >= 23 deg
into the leaf; but a tilted 17.5 x 16.5 mm pad has an x-footprint of
2 (8.75 cos a + 8.25 sin a) = 23-24 mm in the 25 mm slot, so the tips catch the bar's
top or the leaf face (friction 1.0) on nearly every descent, and the tilt then pushes
the leaf into its stop. Do not retry it with this hand.

STRATEGY — pinch the bar from the front, pull with the hand yawing
-------------------------------------------------------------------
Approach along the leaf normal with the fingers OPEN, pads closing along the leaf's
WIDTH so they land on the bar's +-y faces (bar 20 mm; open aperture 80 mm, so +-3 cm
of lateral tolerance; the bar is 16 cm tall, so +-6 cm vertically), pinch centre at the
bar's mid-depth, then drag along the hinge arc with a small angular lead while the full
3x3 hand target yaws with the leaf. The pull is along the approach axis, i.e. carried by
pad friction: on the CL-V2 door that is fine — the leaf's inertia about the hinge is
~0.08 kg m^2 and the damping 0.05 N m s, so the pull needs ~0.5 N, against ~7 N of
friction hold (finger servo kp 350 N/m x 10 mm of squeeze x 2 pads x mu 1.0). The cl25
"cam-out at 0.000" was measured on a 14 kg slab needing ~17 N; the physics changed
with the asset. At theta = 0 the pose is essentially NEUTRAL (hand pointing +x, site y
= -world y), so the IK starts near the answer.

Mount yaw from ``object_orientation``; hinge angle from the goal chord (yaw invariant).
Aperture (finger joints in the obs) is checked after closing: pads that met each other
re-open and re-approach. Every phase has an unconditional step-count escape; the pull
has a progress watchdog. ``posture_weight`` is lowered (base default 0.005 pulls toward
NEUTRAL and, at stretched postures, balances the position gradient: an M4 plateau
reproduced offline on the lid/valve — 28-60 mm residual at 0.005, 4-11 mm at 0.0005).

Only relative observations are used (gto = obs[40:43], o2g = obs[43:46]).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  ClassicalPolicyBase,
  mount_yaw_from_obs,
  rot_z_mat,
)

# Leaf-frame geometry at hinge angle 0 (door.xml, mount frame == leaf frame at 0).
_SITE_FROM_HINGE = np.array([-0.040, 0.250, 0.0])
_HINGE_FROM_SITE = -_SITE_FROM_HINGE
_R0_LEN = float(np.linalg.norm(_SITE_FROM_HINGE))  # 0.2532
# Pinch centre relative to object_site: the bar spans x -0.065..-0.045 (site at -0.040);
# the pads are 16.5 mm long along the approach, so centring them at -0.056 clamps
# x -0.064..-0.048 — the bar's full depth with 1-3 mm to spare each side.
# Pinch depth. The seat error is one-sided: it converges to 2-4 cm and hunts, and what
# it runs out of is DEPTH along the approach, so the window is placed to tolerate a
# short seat. At leaf x -0.052 the tilted pads span x -0.064..-0.040, overlapping the
# bar (-0.065..-0.045) over 19 mm and poking 5 mm into the 25 mm slot behind it; a 1 cm
# short seat still clamps 17 mm of bar and the pad tips stay 20 mm off the leaf face.
PINCH_FROM_SITE = np.array([-0.012, 0.0, 0.0])
# Each failed close adds this much extra depth to the next attempt (capped), so a
# systematically short seat corrects itself instead of repeating.
RETRY_DEPTH = 0.012
RETRY_DEPTH_MAX = 0.028
# Two-stage seat: settle STAGE_BACK short of the pinch along the approach with the
# integrator running, then close the gap carrying it (the seat otherwise stalls
# 1-2 cm short and the pads close in front of the bar: aperture ~0 in 24/32).
STAGE_BACK = 0.03
STAGE_TOL = 0.012
STAGE_TIMEOUT = 26
# 60 deg down from the leaf normal: offline DLS with the env soft limits reaches the pinch
# at 0/30/60/90 deg to 2-3 mm; at 15-45 deg joints 4 and 6 sit on their limits at 0-30 deg
# (measured live: hand stuck 7 cm above the bar with q4 -2.92, q6 3.56). The pads stay
# perpendicular to the leaf width, so the pinch itself is unchanged.
TILT = np.radians(60.0)
# Roll of the hand about its own approach axis. The finger actuator is a spring about the
# closed position (force = 350 N/m x aperture), so the normal force the pads apply is
# PROPORTIONAL TO THE THICKNESS of what is between them: 20 mm of bar across the flats
# gives ~3.5 N per pad, but 28.3 mm across the diagonal gives ~4.95 N — 41 % more normal
# force, hence 41 % more friction, on a pull that is friction-limited and cams out.
# Rolling the hand 45 deg puts the pads on the bar's corners instead of its faces.
ROLL = np.radians(0.0)  # 30 and 45 deg measured 0.19 / 0.31 against 0.72 at n=32 — the rolled wrist costs more in the seat than the extra squeeze buys

GRIPPER_OPEN = 0.0
GRIPPER_PINCH = -1.0

PRE_BACK = 0.10  # pre-grasp standoff in front of the pinch point along the leaf normal
# The approach is the budget: measured (v8, n=32) PRE burned its full 40-step timeout in
# 32/32 and SEAT 27-47 more, so the first CLOSE landed at step 68-92 of 150 and the pull
# got 8-25 steps. cmd_lead_max is what caps the far-field speed (the command can never be
# more than that many rad ahead of the lagging actual joints), so PRE raises it instead of
# raising max_dq, which is not the binding constraint.
# PRE hands over EARLY on purpose: it only has to get the hand into the corridor in front
# of the bar, and SEAT's far-gain branch closes the rest faster than PRE does. Measured
# n=32: 0.72 at 0.050, 0.78 at 0.090, 0.81 at 0.090 with a longer seat window.
PRE_TOL = 0.090
PRE_TIMEOUT = 26  # measured (v11): 45 pushed the first pull to step 102-140 of 150
PRE_LEAD = 0.28
# The seat converges to 2-4 cm and then hunts (measured v9: |gto| 0.02-0.05 for 30+
# steps), so an isotropic 12 mm settle tolerance is unreachable and every env burned its
# timeout; see SEAT_ALONG_TOL below for the gate that replaced it.
SEAT_SETTLE = 2
SEAT_TIMEOUT = 32
SEAT_FAR = 0.06  # above this the seat keeps the approach gains, not the fine ones
# The seat gate is anisotropic: the pads are 80 mm open on a 20 mm bar (+-30 mm of
# lateral slack, +-60 mm vertically) but only ~+-10 mm of slack ALONG the approach.
# Gating the whole vector at 12 mm is unreachable — it timed out in 32/32 — while
# gating it at 30 mm closes in front of the bar. Gate the two directions separately.
SEAT_ALONG_TOL = 0.012
SEAT_LAT_TOL = 0.030
# The timeout-close gate is what sets GRIP DURABILITY, and grip durability is the whole
# task: measured n=32, a pull whose pads land square on the bar holds the aperture at
# 0.020-0.058 for 30+ steps and swings the leaf 0.033 rad/step (0.95 rad in 30 steps),
# while a pull whose pads are off in DEPTH bleeds the aperture 0.031 -> 0.006 in 14 steps
# and gets 0.23 rad. The pads' x-footprint is 23.4 mm on a 20 mm bar, so 22 mm of depth
# error leaves no overlap at all — closing from there was buying attempts that could not
# work. Depth is gated hard; lateral offset is cheap (the pads open to 80 mm).
SEAT_ALONG_ABORT = 0.014
SEAT_LAT_ABORT = 0.038
# Measured (v10, n=32): loosening the settle tolerance to 18 mm ISOTROPICALLY and closing
# from up to 45 mm out turned 16/32 into "mechanism_not_moved" — the pads close in front
# of the 20 mm bar. Depth is what has to be tight, not lateral offset.
INTEG_GAIN = 0.10
INTEG_BAND = 0.06
INTEG_CLIP = 0.03
CLOSE_STEPS = 5
# Aperture after closing on the 20 mm bar reads ~0.015-0.02 (soft contact); pads that
# met each other read ~0. Obs noise +-0.01 per finger, hence the EMA and the margin.
APERTURE_MIN = 0.006
EMA_AP = 0.3
# Arc lead while pulling. The pull is carried by pad friction, and the lead sets the
# force: a 0.20 rad lead is 5 cm of arc, and a position command 5 cm ahead of the leaf is
# answered by the servos with tens of newtons of shear across the pads. Measured (v8,
# n=32): the aperture decays monotonically 0.031 -> 0.002 over ~20 pull steps in EVERY
# env — the bar cams out of the pinch — while the leaf swings 0.05 rad/step. 0.10 rad
# (2.5 cm) still gives ~0.025 rad/step, which reaches 1.52 rad in ~60 of the 150 steps.
# 0.13 rad = 3.3 cm of arc. Measured (v9, n=32): at this scale the aperture HOLDS at
# 0.022-0.031 for a 100-step pull and the leaf reaches 1.50 rad; at 0.20 rad it decayed
# 0.031 -> 0.002 in 20 steps (cam-out). An adaptive lead (v11) drifted to its 0.24 cap
# and brought the cam-out back — a fixed, gentle lead is the right answer.
# Pull rate is GRIP-limited, and the trade is measured on both sides: at a 0.13-0.17 rad
# lead the aperture holds 0.021-0.031 for an 83-step pull but the leaf only swings
# 0.015-0.019 rad/step, so 1.52 rad needs 80-100 steps on top of a 50-65 step approach —
# right at the 150-step budget, which is exactly why the SR sat at 0.19-0.25. At a 0.20+
# rad lead the leaf swings 0.05 rad/step (0.87 rad in 24 steps) and THEN cams out. The
# faster branch wins if — and only if — re-gripping is cheap, so the pull is now
# deliberately aggressive and a cam-out returns to the SEAT stage point (~15 steps), not
# to PRE (~45).
LEAD_ANGLE = 0.14
# A rate-limited pull (advance a commanded angle at a fixed rate, clamped to a maximum
# lead) was tried and lost: 0.50-0.66 against 0.72 for the fixed lead at n=32 on the CPU
# harness, across rates 0.016-0.030 rad/step. Do not re-try it without new geometry.
REGRIP_DIST = 0.06  # cam-out with the hand still this close to the bar: re-seat, not re-approach
# The arc target used to be built from the NOMINAL pinch offset. The bar creeps in the
# pads under the pull (that IS the cam-out), so the commanded point drifts behind the
# hand and eats the lead. Latch the real site->gripper vector in the LEAF frame at close
# time and carry it with a slow EMA: the target becomes "where this hand would be if it
# rode rigidly with the leaf to theta + lead", with no grasp model in it.
HOLD_EMA = 0.03
DONE_ANGLE = 1.56  # 1.65 measured slightly worse (0.625 vs 0.656, n=32 GPU)  # success needs theta > 1.4708 (latched)
# The hand LEADS the pinch by the arc lead, so |pinch| in steady state is a large part of
# the lead itself: LOST_DIST must sit above it or the pull aborts on its own command.
# theta is derived from the goal chord, and d(theta)/d|o2g| is 5.6 rad/m: the +-1 cm of
# observation noise on object_to_goal is +-0.056 rad of leaf angle = +-1.4 cm on the
# commanded pinch. Measured (v13): 7/32 pulls aborted "lost" after 15-20 steps with the
# aperture still reading 0.021-0.030, i.e. on noise. theta is EMA'd and `lost` now has to
# hold for LOST_STREAK consecutive steps.
# An aperture-driven force limiter was tried here too and measured worse (0.72 -> 0.56 at
# n=32, and no threshold in 0.010-0.032 recovered it): a healthy door pull already reads
# only 0.021-0.034, so the limiter throttles pulls that were never in trouble.
# Pull servo gains, separate from the arc lead. The lead sets the FORCE (and the pull is
# friction-limited, so it must stay small); these set how crisply the hand tracks the
# commanded arc point. Sweepable.
PULL_MAXDQ = 0.20
PULL_GAIN = 0.6
PULL_MAXPOS = 0.10
PULL_CMDLEAD = 0.20
LOST_DIST = 0.11
LOST_STREAK = 3
EMA_THETA = 0.35
PULL_GRACE = 12  # no abort for this many steps after CLOSE (the grip settles)
STALL_STEPS = 25
STALL_EPS = 0.03
# The bar is STATIC until the pull starts, so its pose should be estimated with a running
# MEAN over every approach step, not an EMA. gripper_to_object carries ~10-13 mm of noise
# per axis and an EMA at alpha 0.4 averages ~4 samples (6-7 mm of sigma left); a mean over
# the 40-60 approach steps leaves ~1.5 mm. Depth accuracy at close time is what decides
# whether the pinch holds for 30 steps or bleeds out in 14 (see SEAT_ALONG_ABORT), so this
# goes straight at the dominant failure. The average is taken on the object position in
# the ROBOT BASE frame — FK of the observed joints plus gto — which is static and free of
# the per-env scene origin, unlike the absolute gripper_pos observation term. It freezes
# at CLOSE, after which the bar moves and the EMA takes over.
MEAN_SKIP = 4   # let the seat settle before averaging (the hand must be nearly still)
MEAN_MIN = 6    # samples before the mean is trusted
EMA_YAW = 0.3
EMA_ALPHA = 0.4
CMD_LEAD_MIN = 0.12
MAX_WAYPOINT = 0.20

PHASE_NAMES = {0: "PRE", 1: "SEAT", 2: "CLOSE", 3: "PULL", 4: "HOLD"}


class OpenDoorClassicalPolicy(ClassicalPolicyBase):
  """Pinch the bar pull from the front and drag it round the hinge arc."""

  orientation_weight = 0.3
  posture_weight = 0.0005
  ik_joint_limit_factor = 0.9  # the env's soft limits (see base.py)

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._psi = np.zeros(self.num_envs)
      self._psi_init = np.zeros(self.num_envs, dtype=bool)
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ap = np.full(self.num_envs, 0.08)
      self._integ = np.zeros((self.num_envs, 3))
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._stall_theta = np.zeros(self.num_envs)
      self._stall_t = np.zeros(self.num_envs, dtype=np.int64)
      self._attempts = np.zeros(self.num_envs, dtype=np.int64)
      self._staged = np.zeros(self.num_envs, dtype=bool)
      self._hold = np.zeros((self.num_envs, 3))
      self._obj_sum = np.zeros((self.num_envs, 3))
      self._obj_n = np.zeros(self.num_envs, dtype=np.int64)
      self._psi_sum = np.zeros(self.num_envs)
      self._psi_n = np.zeros(self.num_envs, dtype=np.int64)
      self._cmd_theta = np.zeros(self.num_envs)
      self._theta = np.zeros(self.num_envs)
      self._theta_init = np.zeros(self.num_envs, dtype=bool)
      self._lost_t = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._psi_init[env_ids] = False
      self._ap[env_ids] = 0.08
      self._integ[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._stall_theta[env_ids] = 0.0
      self._stall_t[env_ids] = 0
      self._attempts[env_ids] = 0
      self._staged[env_ids] = False
      self._hold[env_ids] = 0.0
      self._obj_sum[env_ids] = 0.0
      self._obj_n[env_ids] = 0
      self._psi_sum[env_ids] = 0.0
      self._psi_n[env_ids] = 0
      self._cmd_theta[env_ids] = 0.0
      self._theta_init[env_ids] = False
      self._lost_t[env_ids] = 0

  def _go(self, i: int, phase: int) -> None:
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._integ[i] = 0.0
    self._stall_t[i] = 0
    self._staged[i] = False
    self._lost_t[i] = 0

  @staticmethod
  def _frame(leaf_rot: np.ndarray) -> np.ndarray:
    """Full 3x3 EE target: approach along the leaf normal (tilted TILT down), closing
    axis along -leaf_y (the neutral wrist branch)."""
    z_ee = leaf_rot @ np.array([np.cos(TILT), 0.0, -np.sin(TILT)])
    y_ee = -(leaf_rot @ np.array([0.0, 1.0, 0.0]))
    y_ee = y_ee - (y_ee @ z_ee) * z_ee
    y_ee /= np.linalg.norm(y_ee)
    if ROLL != 0.0:
      x_ee = np.cross(y_ee, z_ee)
      c, s = np.cos(ROLL), np.sin(ROLL)
      y_ee = c * y_ee + s * x_ee
      y_ee /= np.linalg.norm(y_ee)
    return np.column_stack([np.cross(y_ee, z_ee), y_ee, z_ee])

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    o2g = obs_i[43:46]
    psi_raw = mount_yaw_from_obs(obs_i)
    aperture_raw = float(obs_i[7] + obs_i[8]) + float(self.default_qpos[7] + self.default_qpos[8])
    if not self._psi_init[i]:
      self._psi_init[i] = True
      self._gto_ema[i] = gto_raw
      self._ap[i] = aperture_raw
      self._psi_sum[i] = 0.0
      self._psi_n[i] = 0
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
      self._ap[i] += EMA_AP * (aperture_raw - self._ap[i])
    # The mount never moves during an episode: plain running mean, all episode.
    self._psi_sum[i] += psi_raw
    self._psi_n[i] += 1
    psi = float(self._psi_sum[i] / self._psi_n[i])

    # Static-target running mean (see EMA_YAW comment above), in the robot base frame.
    ee_now, _ = self._fk(self.default_qpos + obs_i[0:9])
    if int(self._phase[i]) == 1 and self._phase_steps[i] >= MEAN_SKIP:
      self._obj_sum[i] += ee_now + gto_raw
      self._obj_n[i] += 1
    if self._obj_n[i] >= MEAN_MIN and int(self._phase[i]) <= 2:
      gto = (self._obj_sum[i] / self._obj_n[i]) - ee_now
    else:
      gto = self._gto_ema[i]

    # Hinge angle from the chord to the goal marker (the handle at 90 deg).
    chord = np.clip(np.linalg.norm(o2g) / (2.0 * _R0_LEN), 0.0, 1.0)
    theta_raw = float(np.pi / 2.0 - 2.0 * np.arcsin(chord))
    if not self._theta_init[i]:
      self._theta[i] = theta_raw
      self._theta_init[i] = True
    else:
      self._theta[i] += EMA_THETA * (theta_raw - self._theta[i])
    theta = float(self._theta[i])

    leaf_rot = rot_z_mat(psi + theta)
    x_leaf = leaf_rot @ np.array([1.0, 0.0, 0.0])
    depth = min(RETRY_DEPTH * float(self._attempts[i]), RETRY_DEPTH_MAX)
    pinch = gto + leaf_rot @ (PINCH_FROM_SITE + np.array([-depth, 0.0, 0.0]))
    hinge = gto + leaf_rot @ _HINGE_FROM_SITE  # site -> hinge (world)
    target_rot = self._frame(leaf_rot)
    gripper_a = GRIPPER_OPEN

    ph = int(self._phase[i])
    self.cmd_ref = "actual"
    # W1-G's gravity-sag ratchet (docs/cl_v3/logs/W1-G.md): keep >= 0.12 rad of command
    # lead in every phase; the pull raises it.
    self.cmd_lead_max = CMD_LEAD_MIN
    if ph == 0:
      self.max_dq, self.step_gain, self.max_pos_err = 0.30, 0.7, 0.18
      self.cmd_lead_max = PRE_LEAD
      pos_err = pinch - PRE_BACK * x_leaf
      if np.linalg.norm(pos_err) > 0.25:
        target_rot = target_rot[:, 2]  # axis-only while far: a full 3x3 makes IK crawl
      if np.linalg.norm(pos_err) < PRE_TOL or self._phase_steps[i] > PRE_TIMEOUT:
        self._go(i, 1)
    elif ph == 1:
      staged = self._staged[i]
      approach = target_rot[:, 2]
      raw = pinch if staged else pinch - STAGE_BACK * approach
      if np.linalg.norm(raw) > SEAT_FAR:
        # PRE hands over at its timeout from wherever it got to; crawling the last 10-30
        # cm at the fine-positioning gains is what made the FIRST seat miss its gate in
        # 7/8 envs (measured v12), costing a whole 60-step approach cycle.
        self.max_dq, self.step_gain, self.max_pos_err = 0.25, 0.55, 0.12
        self.cmd_lead_max = 0.20
      else:
        self.max_dq, self.step_gain, self.max_pos_err = 0.12, 0.4, 0.08
      along = float(raw @ approach)
      lat = float(np.linalg.norm(raw - along * approach))
      if np.linalg.norm(raw) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -INTEG_CLIP, INTEG_CLIP)
      pos_err = raw + self._integ[i]
      if not staged:
        if np.linalg.norm(raw) < STAGE_TOL:
          self._settle[i] += 1
        else:
          self._settle[i] = 0
        if self._settle[i] >= SEAT_SETTLE or self._phase_steps[i] > STAGE_TIMEOUT:
          self._staged[i] = True
          self._settle[i] = 0
          self._phase_steps[i] = 0
      else:
        if abs(along) < SEAT_ALONG_TOL and lat < SEAT_LAT_TOL:
          self._settle[i] += 1
        else:
          self._settle[i] = 0
        if self._settle[i] >= SEAT_SETTLE:
          self._go(i, 2)
        elif self._phase_steps[i] > SEAT_TIMEOUT:
          ok = abs(along) < SEAT_ALONG_ABORT and lat < SEAT_LAT_ABORT
          self._go(i, 2 if ok else 0)
    elif ph == 2:
      self.max_dq, self.step_gain, self.max_pos_err = 0.06, 0.4, 0.05
      self.cmd_lead_max = 0.05
      gripper_a = GRIPPER_PINCH
      pos_err = pinch + self._integ[i]
      if self._phase_steps[i] >= CLOSE_STEPS:
        if self._ap[i] < APERTURE_MIN:
          # Pads met with nothing between them. The hand is still at the bar, so re-seat
          # from the 3 cm stage point (~15 steps) instead of re-running PRE (~45), and
          # go a little deeper each time (the miss is a depth shortfall).
          self._attempts[i] += 1
          self._go(i, 1)
          gripper_a = GRIPPER_OPEN
        else:
          self._hold[i] = leaf_rot.T @ (-gto)
          self._cmd_theta[i] = theta
          self._go(i, 3)
          self._stall_theta[i] = theta
    elif ph == 3:
      # Gentle pull: the grip is friction-only, so the command must not run far ahead of
      # the leaf (see LEAD_ANGLE). max_pos_err is the per-step Cartesian bite.
      self.max_dq = PULL_MAXDQ
      self.step_gain = PULL_GAIN
      self.max_pos_err = PULL_MAXPOS
      self.cmd_lead_max = PULL_CMDLEAD
      # W1-D's "error == servo sag" fixed point is real, but its cure (cmd_ref="command",
      # which ratchets the command forward until the lead clip binds) is WRONG here and
      # measured so: 0.438 -> 0.094 at n=32. This grip is friction-only, so the sustained
      # full-torque pull it produces is exactly the cam-out. The door instead removes the
      # sag term from the target by latching the true hold vector (see HOLD_EMA).
      gripper_a = GRIPPER_PINCH
      self._stall_t[i] += 1
      if np.linalg.norm(pinch) > LOST_DIST:
        self._lost_t[i] += 1
      else:
        self._lost_t[i] = 0
      lost = self._lost_t[i] >= LOST_STREAK
      # Measured (v12): a pull with the pads closed on air (aperture 0.000) moves the
      # leaf 0.06-0.15 rad and then nothing, but the stall watchdog still cost 25 steps
      # before it fired — 40 steps per empty attempt. A real grip reads 0.021-0.034 for
      # a 100-step pull, so an empty aperture past the grace is decisive on its own.
      empty = self._ap[i] < APERTURE_MIN
      stalled = False
      if self._stall_t[i] >= STALL_STEPS:
        stalled = theta - self._stall_theta[i] < STALL_EPS
        if not stalled:
          self._stall_theta[i] = theta
          self._stall_t[i] = 0
      if self._phase_steps[i] > PULL_GRACE and (lost or stalled or empty):
        self._attempts[i] += 1
        if np.linalg.norm(pinch) < REGRIP_DIST:
          self._go(i, 1)  # cheap re-seat from the stage point, fingers open
          return pinch - STAGE_BACK * target_rot[:, 2], target_rot, GRIPPER_OPEN
        self._go(i, 0)
        return pinch - PRE_BACK * x_leaf, target_rot, GRIPPER_OPEN
      self._hold[i] += HOLD_EMA * (leaf_rot.T @ (-gto) - self._hold[i])
      lead = max(0.0, min(LEAD_ANGLE, DONE_ANGLE - theta))
      lead_rot = rot_z_mat(psi + theta + lead)
      pos_err = gto + (lead_rot - leaf_rot) @ _SITE_FROM_HINGE + lead_rot @ self._hold[i]
      target_rot = self._frame(rot_z_mat(psi + theta + 0.5 * lead))
      if theta >= DONE_ANGLE:
        self._go(i, 4)
    else:
      # HOLD used to freeze at the pinch. theta is a chord estimate carrying +-0.056 rad
      # of observation noise, so an optimistic estimate at DONE_ANGLE stops the pull a few
      # mrad short of the 1.4708 rad success latch. Keep a small lead instead and press
      # gently into the leaf's own 1.5708 rad stop, which costs nothing.
      self.max_dq, self.step_gain, self.max_pos_err = 0.12, 0.5, 0.06
      self.cmd_lead_max = 0.15
      gripper_a = GRIPPER_PINCH
      hold_rot = rot_z_mat(psi + theta + 0.06)
      pos_err = gto + (hold_rot - leaf_rot) @ _SITE_FROM_HINGE + hold_rot @ self._hold[i]

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_rot, gripper_a
