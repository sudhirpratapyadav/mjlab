"""Scripted ThrowToBin teacher -- a real underhand THROW (CL-V3, W1-D).

TASK
----
Same scene as Place-In-Container (46 mm cube, moulded basket, ``PlaceInContainerCommand``
unmodified) but the basket is spawned at radial 0.78-0.91 m -- beyond the arm's static
reach BY DESIGN -- so the cube has to be launched, not carried.  Success (latched over the
250-step / 5 s episode): cube centre within 0.0585 m laterally of the basket's interior
site, below its rim (0.093) and above its floor (-0.020), and moving slower than 0.12 m/s.

HISTORY (why the old teacher scored 0.000 and what changed)
-----------------------------------------------------------
cl25 W3-c / cl_v2: grasp, then reach toward the bin and drop.  Measured: the reach
saturates 27-40 cm short of the bin for every sampled position (phase-1 LOGS, W0-b's
diagnose dump: 18/32 envs ended holding the cube 0.27 m away, 13/32 hit the floor while
stretching).  A drop is impossible; release timing was never the binding question.

CL-V3 feasibility measurements (W1-D, CPU sim, sim-state velocities, logs/W1-D.md):
  * joint-space bang-bang swings reach 1.8-3.0 m/s at the gripper site (shoulder 3.2
    rad/s, elbow 2.0-2.7, wrist 5.3, base yaw 2.5 rad/s) -- far above the paper estimate
    from the XML damping;
  * the ballistic requirement for a release at (r 0.50, z 0.42) with a 40 deg launch to a
    bin at 0.78-0.91 m is only 1.0-1.35 m/s (2.1-2.3 m/s would be needed from the floor at
    45 deg);
  * a joint-space RAMP (constant joint rates, PD tracking) reproduces the planned site
    speed to within ~3 % -- the launch speed is a controllable quantity.
So the throw is feasible with margin; precision is the whole problem (basket inner
half-span 0.0815 minus the cube's 0.023 = 0.0585 m).

STRATEGY -- grasp, wind up, planar swing, release on the planned posture
------------------------------------------------------------------------
  0 HOVER    top-down above the cube, hand yaw aligned to the cube's faces (the pads must
             meet two FACES, not edges: an edge grasp rolls the cube off a pad at release
             and flicks it sideways -- measured).  Lateral integrator against the ~2 cm
             DLS bias so the hover actually converges instead of timing out.
  1 DESCEND  to a deep grasp (site 0.038 -> pads 0.024-0.041 on a 0-0.045 cube), slow
             final approach (no overshoot into the floor), lateral integrator carried.
  2 CLOSE    hold the pose, squeeze.  HELD CHECK BY APERTURE: a 46 mm cube keeps the
             aperture >= 0.040; if the fingers closed to < 0.035 the pads met on nothing
             (measured: envs that "held" the cube by |gripper_to_object| alone dropped it
             during the wind-up) -> reopen and retry.
  3 CLIMB    straight up.  Bin position (base frame) is accumulated from
             FK(joints) + gripper_to_object + object_to_goal.
  4 WINDUP   joint-space move to the wind-up posture: arm plane at the bin's azimuth
             (joint 1), planar joints 3/5 at zero, hand closing axis ACROSS the plane so
             the cube leaves forward between the pads (q7 = +pi/4), pitch joints 2/4/6 at
             release posture minus rates x swing time.  The cube's offset from the site is
             measured here in the hand frame (it rides 1-3 cm below the site).
  5 SWING    constant joint rates (a ramp the PD tracks) toward and through the release
             posture; the gripper opens on the control step in which the OBSERVED joints
             cross the release posture.  Rates come from the CUBE's Jacobian at the release
             posture (site Jacobian minus [offset]x angular Jacobian), so the hand's pitch
             rate times the grasp offset -- up to 0.14 m/s, ~5 cm of range -- is planned
             for instead of scattering the landing.
  6 FOLLOW   the ramp continues three steps (the hand stays ahead of the cube), then holds.

Planar swing geometry (internal model): release posture with the site at (REL_R, REL_Z)
in the arm plane and the hand pointing down; rates on joints 2/4/6 from a weighted
least-norm solve (weights spare the elbow, whose plateau is ~2 rad/s); required speed from
the ballistic formula for the cube's release point and the bin's interior site.

Observations (60-D layout): 0:9 joint_pos_rel, 34:40 object_orientation (rows 1-2 of the
cube's rotation matrix), 40:43 gripper_to_object, 43:46 object_to_goal.  All FK is on the
arm's own joints (base frame); absolute obs terms carry the per-env origin and are never
used.
"""

from __future__ import annotations

import mujoco
import numpy as np

from mjlab.continual_distill.classical.base import (
  FRANKA_ACTION_SCALE,
  HOME_QPOS,
  ClassicalPolicyBase,
)

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0
# Partial close for the wind-up / swing: finger ctrl 0.016 -> 7 mm interference per side on
# the 46 mm cube = ~2.5 N per pad (4x the swing load at mu 0.3).  A full 8 N squeeze made the
# pads drag the cube sideways as they separated (measured: cube-velocity azimuth errors of
# 3-8 deg with the site's azimuth exact).
GRIPPER_HOLD = -0.6
GRIPPER_ZERO = -0.425        # finger ctrl 0.023 = the cube half-width: zero PD force on the pads
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
DT = 0.02
G = 9.81

P_HOVER = 0
P_DESCEND = 1
P_CLOSE = 2
P_CLIMB = 3
P_WINDUP = 4
P_SWING = 5
P_FOLLOW = 6
PHASE_NAMES = {0: "HOVER", 1: "DESCEND", 2: "CLOSE", 3: "CLIMB", 4: "WINDUP", 5: "SWING", 6: "FOLLOW"}

JOINT_LIMITS = np.array(
  [[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698],
   [-2.8973, 2.8973], [-0.0175, 3.7525], [-2.8973, 2.8973]]
)
PITCH_JOINTS = [1, 3, 5]  # joints 2, 4, 6: parallel axes when joints 3 and 5 are zero

# -- grasp (46 mm cube, centre rests at 0.0226) --------------------------------------
OBJ_CENTER_Z = 0.0226
HOVER_SITE_Z = 0.15          # pads (site - 0.0138) clear the cube top (0.045) by 9 cm
GRASP_SITE_Z = 0.030         # W1-D2: with the descent ratchet removed the site LANDS on this
                             # height instead of sinking ~12 mm past it, so the old 0.036 put the
                             # pads on the cube's top third and the swing load rotated it out
                             # (measured n=16 CPU: 0.036 -> 11/16 released, 0.032 -> 16/16).
                             # pads 0.020-0.036 on the cube's 0-0.045 face (was 0.045:
                             # a top-1.5 cm grasp rotated out of the pads under the swing load)
FLOOR_MIN_Z = 0.030          # site floor: lowest pad 1.38 cm below the site (measured)
ALIGN_TOL = 0.012
ALIGN_SETTLE = 2
ALIGN_TIMEOUT = 35           # a failed grasp costs a whole attempt out of a 250-step
                             # budget; leaner approach timeouts fit a third attempt in
HOVER_INTEG_GAIN = 0.08      # 0.25 limit-cycled +-3 cm through the servo lag (traced)
HOVER_INTEG_CLIP = 0.03
HOVER_INTEG_BAND = 0.04      # anti-windup: integrate the lateral error only inside this
                             # (a saturated +-5 cm bias during the 30 cm approach stretched
                             # the arm, sagged the hover to z 0.05 and tilted the hand 10-15 deg)
DESCENT_RATE_FAST = 0.025
DESCENT_RATE_SLOW = 0.008
DESCENT_SLOW_BAND = 0.03
SEAT_TOL_Z = 0.006
SEAT_TOL_X = 0.006           # along the pad width (EE x): the pads cannot centre the cube here
SEAT_TOL_Y = 0.012           # along the closing axis (EE y): closing pads centre the cube
SEAT_SETTLE = 2
DESCEND_TIMEOUT = 32
DESCEND_SITE_Z = 0.045       # joint-space descent target (pads at 0.031); the CLOSE hold takes the
                             # last 1.1 cm with an absolute IK target
DESCEND_DQ = 0.04            # rad/step of the joint-space descent
DESCEND_MODE = "ik"          # "ik": closed-loop IK descent; "js": open-loop joint move
DESCENT_LEASH = 0.020        # the waypoint may not run further than this below the site
DESCENT_RATE = 0.006         # m/step of the IK descent: slow enough that momentum cannot carry
                             # the site past the floor guard
LEAD_DESCEND = 0.05
LEAD_HOVER = 0.12            # command lead per phase: gravity compensation in the hover / climb,
LEAD_CLOSE = 0.08            # smaller in the hold (a 0.12 lead drove the descent 1.5 cm past the guard)
CLOSE_STEPS = 22
CLOSE_DELAY = 10
CLOSE_XY_TOL = 0.005         # squeeze as soon as the hold has the site this close to the cube xy
#             # hold at depth first: fingers close in ~3 steps, faster than the
                             # hand can descend from a timed-out descent (top-edge grasps measured)
APERTURE_HELD = 0.030        # 46 mm cube -> aperture ~0.046; closed on nothing ~0.006-0.014.
                             # MEASURED on GPU at n=128: 0.035 rejected good grasps and cost
                             # a third of the throws (0.516); 0.030 -> 0.586; 0.024 (too
                             # permissive, passes half-grasps into the swing) -> 0.500.
APERTURE_EMA = 0.3
CLIMB_STEPS = 12
CLIMB_ERR = 0.12
HELD_TOL = 0.05
YAW_RATE = 0.15              # rad/step on joint 7 for the face-alignment servo
SPIN_FREE = 0                # W1-D4: spend the swing's redundancy on w2+w4+w6 = 0
APPROACH_MEAN = 1            # W1-D4: aim the approach at the RUNNING MEAN of the derived
                             # absolute cube position instead of a 3-sample EMA
EMA_ALPHA = 0.35
EMA_ALPHA_LOW = 0.2
RESET_JOINT_JUMP = 0.25      # rad in one step: only an env reset does that
RESET_OBJ_JUMP = 0.15

# -- swing ------------------------------------------------------------------------------
REL_R = 0.50                 # release: site radial / height in the arm plane (base frame)
REL_Z = 0.42
LAUNCH_EL = np.radians(48.0)  # 40 deg left only ~2.6 cm of near-rim clearance for far bins (computed
                              # from logged releases); a tumbling cube's corners sweep +-3.2 cm
LAND_Z = 0.027               # cube centre resting on the basket floor (env-local z)
AIM_SHORT = 0.02             # aim this much short of the interior site: landings averaged 1.8 cm
                             # long and a cube that slides to the far wall fails the 0.0585 test
Q7_RELEASE = 0.7853          # closing axis across the arm plane (pads left/right of the cube)
SWING_T = 26                 # steps of ramp from wind-up to release posture (18 left the
                             # arm still accelerating: measured release/required speed 0.90)
KICK = 1.0                   # command lead in ramp steps (PD lag ~1 step, measured)
RELEASE_LEAD = 0.5           # open when the crossing is within this fraction of a step
SPEED_GAIN = 1.16            # W1-D3, GPU n=128 with the ringing removed and the release
                             # closed on the predicted landing point: 1.08 -> 0.617,
                             # 1.00 -> 0.641, 1.16 -> 0.656.  A faster ramp is now free --
                             # the predicate cuts the throw at the right range whatever the
                             # ramp does, so the gain only has to guarantee the range is
                             # REACHABLE before the ramp runs out.
                             # (W1-D2, superseded) MEASURED landing errors, n=16 CPU: SWING_T 18 / gain 1.0 gave
                             # release/required 0.90 and a radial landing error of -0.088 m
                             # (mean) with sd 0.153; SWING_T 26 / gain 1.12 gives 1.07 and
                             # -0.009 +- 0.068
# -- closed-loop release (W1-D3) --------------------------------------------------------
# W1-D2 measured the residual: the arm's swing-tracking ratio varies 0.73-1.27 env to env,
# and range goes as v^2, so a release fired on a PLANNED posture scatters the landing by
# 0.068-0.15 m against an 0.0585 m test.  The fix is to stop trusting the plan at the
# release: fit the CUBE's own observed position over a short window, propagate it
# ballistically to the goal's height, and open the fingers on the step whose predicted
# landing point is closest to the goal -- whatever posture the arm happens to be in.
RELEASE_MODE = "predict"     # "predict" = closed loop; "posture" = the legacy crossing test
VEL_FIT_N = 11               # samples of object_pos (obs 18:21) in the velocity fit.  MEASURED
                             # noise on that term is ~0.010-0.013 m per axis per step (bigger
                             # than the +-0.01 the cfg asks for), so a QUADRATIC fit's endpoint
                             # derivative scatters by 0.25-0.30 m/s -- useless.  A LINEAR fit
                             # over 11 samples gives the window-mean rate to ~0.05 m/s, and
                             # once the ramp-in below has removed the ringing the swing rate is
                             # constant, so the window mean IS the current velocity.
RELEASE_STEP_LEAD = 1.0      # steps of lead: the cube leaves at the END of the step whose
                             # action opens the fingers, one step of ramp later than the
                             # observation the predicate saw
SWING_EXTRA = 16             # steps the ramp may continue PAST the planned release posture
                             # while the predicate waits for the range (slow envs)
# THE RINGING (W1-D3, MEASURED from a swing dump: cube speed 1.24 -> 0.81 -> 0.95 -> 0.63 ->
# 1.03 m/s over the last 20 steps of a single swing, i.e. +-30 % at ~5 Hz).  Two causes, both
# a STEP in the joint command at the swing start:
#  (a) ``_plan_swing`` re-derives the wind-up posture from the refined bin estimate and the
#      measured grasp offset, but the arm has already settled on the OLD one -- traced: joint
#      6 commanded +0.360 at the last wind-up step and +0.623 on the first swing step, a
#      0.26 rad jump that put 0.25 rad of tracking error on a wrist that saturates at ~0.04.
#      Fixed by re-settling on the NEW wind-up posture before the ramp starts (RE_SETTLE).
#  (b) even without that, ``KICK`` puts the whole 1+KICK steps of ramp on the first command.
#      RAMP_IN eases the rate in linearly instead, which the PD can follow.
RAMP_IN = 6                  # steps over which the swing rate eases in from 1/RAMP_IN to 1
RE_SETTLE = 4                # wind-up steps to re-settle on the re-planned posture
W_MAX = (2.5, 1.8, 4.0)      # rate weights (rad/s) for joints 2/4/6: spare the elbow (its plateau ~2 rad/s)
V_MIN, V_MAX = 0.6, 2.2
WINDUP_DQ = 0.05             # rad/step joint-space move to the wind-up posture (0.07 shook shallow grasps out)
SHALLOW_OFFSET = -0.024      # cube centre this far below the site after CLIMB = top-edge grasp: retry
WINDUP_SETTLE = 8
WINDUP_TIMEOUT = 70
OFFSET_SAMPLES = 6
FOLLOW_STEPS = 3


def _ballistic_speed(R: float, h: float, el: float) -> float:
  """Launch speed for horizontal range R with a drop h (landing below release)."""
  c, s, t = np.cos(el), np.sin(el), np.tan(el)
  denom = 2 * c * c * (R * t + h)
  return float(np.sqrt(max(G * R * R / max(denom, 1e-6), 0.0)))


def _cube_yaw(obs_i: np.ndarray) -> float:
  """Yaw of one cube face normal (mod pi/2) from object_orientation (rows 1-2 of R)."""
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  r1 = r1 / (np.linalg.norm(r1) + 1e-9)
  r2 = r2 / (np.linalg.norm(r2) + 1e-9)
  r0 = np.cross(r1, r2)
  xaxis = np.array([r0[0], r1[0], r2[0]])  # body x-axis in world
  return float(np.arctan2(xaxis[1], xaxis[0]))


def _down_frame_closing(closing_yaw: float) -> np.ndarray:
  """EE rotation: approach axis (EE z) straight down, CLOSING axis (EE y, measured: the
  pads sit at +-0.048 along EE y) at ``closing_yaw`` in the world xy-plane."""
  c, s = np.cos(closing_yaw), np.sin(closing_yaw)
  ey = np.array([c, s, 0.0])
  ez = np.array([0.0, 0.0, -1.0])
  ex = np.cross(ey, ez)
  return np.column_stack([ex, ey, ez])


class ThrowToBinClassicalPolicy(ClassicalPolicyBase):
  """Grasp the cube, wind up, swing in the bin's azimuth plane, release on the planned posture."""

  DEFAULT_QPOS = HOME_QPOS
  # Tracking, not planning, limits the grasp: max_dq 0.07 (3.5 rad/s strides re-planned
  # every step from the LAGGING joints) saturated the 12 N m wrist actuators, the hand
  # tilted 10-15 deg and the hover sagged to z 0.05 (measured; the physics-free IK loop
  # converged in 15 steps).  Half-stride commands (step_gain) damp that.
  max_dq = 0.05
  step_gain = 0.5
  orientation_weight = 0.5  # keep the hand vertical: a tilt lowers one pad by 0.048*sin(tilt)
  # Gravity-sag ratchet (W1-G, logs/W1-G.md): with cmd_lead_max = 0 the joint command is
  # re-anchored to the sagged ACTUAL joints every step, so joint 2 never gets the offset
  # that holds the arm against gravity -- traced here as the descent sinking 3.5 mm/step
  # through the target with the commanded z error already positive.  A bounded command
  # lead (integrate from the previous command) is sustained servo force.
  cmd_lead_max = 0.12
  # W1-D2, MEASURED (same defect found and fixed on Reorient): with cmd_ref = "actual" the
  # base rebuilds the target as FK(previous COMMAND) + err every step, so the sag the
  # command carries is ADDED to the descent each step -- the site sank ~12 mm past its own
  # commanded grasp height and the pads reached the floor.  Referencing z to the command
  # makes the height channel an integrator that converges on the commanded height.
  cmd_ref = "command"
  cmd_ref_axes = (2,)
  PAD_FLOOR_MIN = 0.008
  PHASE_NAMES = PHASE_NAMES

  def __init__(self, num_envs: int):
    super().__init__(num_envs)
    tf = np.arange(VEL_FIT_N, dtype=np.float64) * DT
    self._vfit = np.linalg.pinv(np.stack([np.ones_like(tf), tf], axis=1))
    self._vfit_t = float(tf[-1])
    q = self._solve_release_posture()
    self._q_rel_base = q
    self._jp_rel, self._jr_rel = self._jac7(q)
    _p, self._R_rel = self._fk7(q)

  # -- planning on the internal model ----------------------------------------------

  def _fk7(self, q7: np.ndarray):
    q = self.default_qpos.copy()
    q[:7] = q7
    return self._fk(q)

  def _jac7(self, q7: np.ndarray):
    self._fk7(q7)
    jp = np.zeros((3, self.model.nv))
    jr = np.zeros((3, self.model.nv))
    mujoco.mj_jacSite(self.model, self.data, jp, jr, self._site_id)
    return jp[:, self._arm_dofadr].copy(), jr[:, self._arm_dofadr].copy()

  def _solve_release_posture(self) -> np.ndarray:
    """Planar posture (q1=q3=q5=0) with the site at (REL_R, 0, REL_Z), hand down."""
    p_des = np.array([REL_R, 0.0, REL_Z])
    down = np.array([0.0, 0.0, -1.0])
    q = np.array([0.0, 0.6, 0.0, -2.0, 0.0, 2.6, Q7_RELEASE])
    for _ in range(300):
      p, R = self._fk7(q)
      jp, jr = self._jac7(q)
      e = np.concatenate([p_des - p, np.cross(R[:, 2], down)])
      J = np.vstack([jp[:, PITCH_JOINTS], jr[:, PITCH_JOINTS]])
      dq = np.linalg.solve(J.T @ J + 1e-4 * np.eye(3), J.T @ e)
      q[PITCH_JOINTS] += np.clip(dq, -0.2, 0.2)
      q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    p, _ = self._fk7(q)
    assert np.linalg.norm(p - p_des) < 0.005, (p, p_des)
    return q

  def _rates_for(self, v_xz: np.ndarray, r_off_world: np.ndarray) -> np.ndarray:
    """Joint rates (7,) on joints 2/4/6 giving the CUBE the planar velocity v_xz at the
    release posture (arm plane at azimuth 0).  Weighted least-norm."""
    skew = np.array(
      [[0, -r_off_world[2], r_off_world[1]],
       [r_off_world[2], 0, -r_off_world[0]],
       [-r_off_world[1], r_off_world[0], 0]]
    )
    jc = self._jp_rel - skew @ self._jr_rel
    Jp = jc[[0, 2]][:, PITCH_JOINTS]
    if SPIN_FREE:
      # W1-D4.  Joints 2/4/6 have PARALLEL axes at this posture, so the hand's angular
      # rate in the swing plane is exactly w2 + w4 + w6.  The planar problem is 2
      # equations in 3 unknowns; the existing weighted least-norm spends the redundancy
      # on sparing the elbow, but the third equation w2 + w4 + w6 = 0 buys two things at
      # once: the cube leaves with ZERO SPIN (a tumbling cube in a basket lands and
      # skips off a wall), and with omega = 0 the omega x r term that put up to 0.14 m/s
      # of scatter on the release (W1-D, the cube rides 0.8-3.2 cm below the site and the
      # hand pitches at ~4.7 rad/s) disappears identically -- the cube leaves at exactly
      # the site's velocity.
      A = np.vstack([Jp, np.ones((1, 3))])
      b = np.array([v_xz[0], v_xz[1], 0.0])
      w = np.linalg.lstsq(A, b, rcond=None)[0]
      om = np.zeros(7)
      om[PITCH_JOINTS] = w
      return om
    Winv = np.diag(np.asarray(W_MAX, dtype=np.float64) ** 2)
    w = Winv @ Jp.T @ np.linalg.solve(Jp @ Winv @ Jp.T + 1e-9 * np.eye(2), v_xz)
    om = np.zeros(7)
    om[PITCH_JOINTS] = w
    return om

  # -- state ---------------------------------------------------------------------------

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      n = self.num_envs
      self._gto_ema = np.zeros((n, 3))
      self._pmean = np.zeros((n, 3))
      self._pn = np.zeros(n, dtype=np.int64)
      self._ema_init = np.zeros(n, dtype=bool)
      self._prev_gto = np.zeros((n, 3))
      self._prev_q = np.full((n, 7), np.nan)
      self._integ = np.zeros((n, 3))
      self._settle = np.zeros(n, dtype=np.int64)
      self._yaw = np.zeros(n)
      self._yaw_ok = np.zeros(n, dtype=bool)
      self._bin_acc = np.zeros((n, 3))
      self._bin_n = np.zeros(n, dtype=np.int64)
      self._off_acc = np.zeros((n, 3))
      self._off_n = np.zeros(n, dtype=np.int64)
      self._q_w = np.zeros((n, 7))
      self._q_rel = np.zeros((n, 7))
      self._omega = np.zeros((n, 7))
      self._q_cmd_js = np.full((n, 7), np.nan)
      self._swing_k = np.zeros(n, dtype=np.int64)
      self._v_req = np.zeros(n)
      self._hold = np.zeros((n, 3))
      self._q_grasp = np.zeros((n, 7))
      self._closing = np.zeros(n, dtype=bool)
      self._close_k = np.zeros(n, dtype=np.int64)
      self._ap_ema = np.full(n, 0.08)
      self._ap_init = np.zeros(n, dtype=bool)
      self._zwp = np.full(n, np.nan)
      self._cp_hist = np.zeros((n, VEL_FIT_N, 3))
      self._cp_n = np.zeros(n, dtype=np.int64)
      self._goal_acc = np.zeros((n, 3))
      self._goal_n = np.zeros(n, dtype=np.int64)
      self._r_prev = np.full(n, np.nan)
      self._dbg = np.zeros((n, 6))
      self._replanned = np.zeros(n, dtype=bool)
      self._cur_env = 0
      self.debug = [[] for _ in range(n)]  # (phase, reason) of every rewind, for diagnosis
    ids = range(self.num_envs) if env_ids is None else env_ids
    for i in ids:
      self._clear(i)
      self._prev_q[i] = np.nan

  def _clear(self, i: int) -> None:
    self._closing[i] = False
    self._close_k[i] = 0
    self._ema_init[i] = False
    self._pn[i] = 0
    self._pmean[i] = 0.0
    self._integ[i] = 0.0
    self._settle[i] = 0
    self._yaw_ok[i] = False
    self._bin_acc[i] = 0.0
    self._bin_n[i] = 0
    self._off_acc[i] = 0.0
    self._off_n[i] = 0
    self._q_cmd_js[i] = np.nan
    self._swing_k[i] = 0
    self._zwp[i] = np.nan
    self._cp_n[i] = 0
    self._goal_acc[i] = 0.0
    self._goal_n[i] = 0
    self._r_prev[i] = np.nan
    self._replanned[i] = False

  def _rewind(self, i: int, reason: str = "") -> None:
    self.debug[i].append((int(self._phase[i]), reason))
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
      # heavier smoothing near the grasp (obs noise +-1 cm per axis vs a 7 mm seat)
      a = EMA_ALPHA_LOW if self._phase[i] == P_DESCEND and -raw[2] < 0.05 else EMA_ALPHA
      self._gto_ema[i] = a * raw + (1 - a) * self._gto_ema[i]
    if APPROACH_MEAN and self._phase[i] in (P_HOVER, P_DESCEND):
      # W1-D4.  ``gripper_to_object`` carries 10-13 mm of observation noise PER AXIS
      # (measured by W1-D3), and an EMA at alpha 0.35 averages about three samples, so
      # the approach is aimed with ~6 mm of lateral sd against a 46 mm cube in an 80 mm
      # jaw.  The cube does not move until the pads touch it, so FK(q_obs) + gto is the
      # same static quantity measured afresh every step: its running mean over the 20-40
      # hover steps converges as 1/sqrt(n).  Only the FK term is re-read live, and its
      # own noise (+-0.01 rad of joint noise, a few mm at the site) is what is left.
      # The mean is frozen at the end of HOVER -- from DESCEND on the pads can reach the
      # cube, and a polluted mean is worse than a noisy one (measured on Strike-Slide).
      p_act, _R = self._fk(self.default_qpos + obs_i[0:9])
      if self._phase[i] == P_HOVER:
        self._pn[i] += 1
        self._pmean[i] += (p_act + raw - self._pmean[i]) / float(self._pn[i])
      elif self._pn[i] == 0:
        self._pmean[i] = p_act + raw
        self._pn[i] = 1
      return self._pmean[i] - p_act
    return self._gto_ema[i]

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    """An env reset (ee_ground_collision / out-of-bounds) teleports the arm to the noisy
    home pose and respawns the cube: a joint jump > 0.25 rad in one control step, or a
    cube jump > 0.15 m, never happens otherwise."""
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

  # -- action ----------------------------------------------------------------------------

  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    self._cur_env = i
    # The goal (basket interior site) is static within an episode (D1 removed the
    # mid-episode resample), so averaging obs 18:21 + obs 43:46 over the whole episode
    # drives the +-0.01 m observation noise on it to nothing before the swing needs it.
    self._goal_acc[i] += obs_i[18:21] + obs_i[43:46]
    self._goal_n[i] += 1
    self._update_aperture(i, obs_i)
    self._detect_reset(i, obs_i)
    ph = self._phase[i]
    q_abs = self.default_qpos + obs_i[0:9]
    if ph == P_DESCEND and DESCEND_MODE == "js":
      return self._descend_js(i, obs_i, q_abs)
    if ph < P_WINDUP:
      self.cmd_lead_max = {P_CLOSE: LEAD_CLOSE, P_DESCEND: LEAD_DESCEND}.get(ph, LEAD_HOVER)
      a = super()._act_single(i, obs_i)
      # Yaw servo on joint 7 (rotates about the approach axis: null space of the
      # axis-only IK, measured d(closing yaw)/dq7 = -1 with the hand pointing down).
      # A yaw-CONSTRAINED DLS target was measured to wreck the position tracking
      # (hover sagging to z 0.05 with a 10-16 deg tilt, pads into the floor).
      _p, R = self._fk(q_abs)
      yaw_now = float(np.arctan2(R[1, 1], R[0, 1]))
      d_yaw = float(np.angle(np.exp(4j * (self._closing_yaw_target(i, obs_i, q_abs) - yaw_now))) / 4.0)
      dq7 = float(np.clip(-d_yaw, -YAW_RATE, YAW_RATE))
      q7 = float(np.clip(q_abs[6] + dq7, JOINT_LIMITS[6, 0], JOINT_LIMITS[6, 1]))
      a[6] = (q7 - self.default_qpos[6]) / FRANKA_ACTION_SCALE
      self._q_cmd[i][6] = q7  # keep the lead integrator consistent with the yaw servo
      return a
    if ph == P_WINDUP:
      return self._windup(i, obs_i, q_abs)
    if ph == P_SWING:
      return self._swing(i, obs_i, q_abs)
    return self._follow(i)

  def _aperture(self, obs_i: np.ndarray) -> float:
    """Smoothed finger aperture. The raw finger-joint observations carry +-0.01 m of
    noise EACH (+-2 cm on a 4.6 cm aperture), so a single read against the 3.5 cm
    threshold rejected good grasps; the EMA is updated once per step in _act_single."""
    return float(self._ap_ema[self._cur_env])

  def _update_aperture(self, i: int, obs_i: np.ndarray) -> None:
    raw = float(self.default_qpos[7] + obs_i[7] + self.default_qpos[8] + obs_i[8])
    if not self._ap_init[i]:
      self._ap_ema[i] = raw
      self._ap_init[i] = True
    else:
      self._ap_ema[i] = APERTURE_EMA * raw + (1 - APERTURE_EMA) * self._ap_ema[i]

  def _closing_yaw_target(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> float:
    """Cube face yaw (mod pi/2) nearest the hand's current closing-axis yaw; latched."""
    if not self._yaw_ok[i]:
      _p, R = self._fk(q_abs)
      hand = float(np.arctan2(R[1, 1], R[0, 1]))
      base = _cube_yaw(obs_i)
      cands = base + np.arange(-4, 5) * (np.pi / 2)
      d = np.angle(np.exp(1j * (cands - hand)))
      self._yaw[i] = float(cands[np.argmin(np.abs(d))])
      self._yaw_ok[i] = True
    return float(self._yaw[i])

  def _target_error(self, i: int, obs_i: np.ndarray):
    q_abs = self.default_qpos + obs_i[0:9]
    gto = self._gto(i, obs_i)
    ph = self._phase[i]
    rot = _DOWN_AXIS  # axis-only: yaw is handled by the joint-7 servo in _act_single

    if ph == P_HOVER:
      err = np.array([gto[0], gto[1], gto[2] + (HOVER_SITE_Z - OBJ_CENTER_Z)])
      self._integrate(i, err)
      if np.linalg.norm(err[:2]) < ALIGN_TOL and abs(err[2]) < 0.04:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= ALIGN_SETTLE or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._plan_descend(i, obs_i, q_abs)
        self._goto(i, P_DESCEND)
      return self._guard(err + self._integ[i], q_abs), rot, GRIPPER_OPEN

    if ph == P_DESCEND:  # closed-loop IK descent (DESCEND_MODE == "ik")
      # Rate-limiting the ERROR does not rate-limit the descent: the DLS solve realises
      # only ~40 % of a small request (measured), so the phase carries a WAYPOINT that
      # walks down at the rate, leashed to the actual site, and the servo is given the
      # full error to it.
      p_act, _Rd = self._fk(q_abs)
      err = np.array([gto[0], gto[1], gto[2] + (GRASP_SITE_Z - OBJ_CENTER_Z)])
      pad_slack = max(self._pad_min_z() - self.PAD_FLOOR_MIN, 0.0)
      z_goal = max(p_act[2] + err[2], p_act[2] - pad_slack)
      if not np.isfinite(self._zwp[i]):
        self._zwp[i] = p_act[2]
      self._zwp[i] = max(z_goal, self._zwp[i] - DESCENT_RATE, p_act[2] - DESCENT_LEASH)
      self._integrate(i, err)
      z_err = p_act[2] - z_goal
      cmd = err + self._integ[i]
      cmd[2] = self._zwp[i] - p_act[2]
      err = np.array([err[0], err[1], z_err])
      _p, R = self._fk(q_abs)
      e_hand = R.T @ np.array([err[0], err[1], 0.0])
      if abs(z_err) < SEAT_TOL_Z and abs(e_hand[0]) < SEAT_TOL_X and abs(e_hand[1]) < SEAT_TOL_Y:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= SEAT_SETTLE or self._phase_steps[i] > DESCEND_TIMEOUT:
        self._goto(i, P_CLOSE)
      return self._guard(cmd, q_abs), rot, GRIPPER_OPEN

    if ph == P_CLOSE:
      # Hold an ABSOLUTE pose while the fingers squeeze: a zero error means "stay where
      # you are now", which follows the gravity sag (traced: the site sank 1.8 cm in 10
      # steps of CLOSE).  The hold point is the site xy at the CLOSE transition and the
      # grasp height, both in the base frame from FK.
      # Closed-loop fine positioning on the CUBE (absolute target in the base frame,
      # refreshed from the smoothed relative observation) before the squeeze; the
      # joint-space descent lands within ~1-3 cm, the hold closes that gap.
      p, _R = self._fk(q_abs)
      if not self._closing[i]:
        self._hold[i] = np.array([p[0] + gto[0], p[1] + gto[1], GRASP_SITE_Z])
      err = self._hold[i] - p
      xy_ok = np.linalg.norm(err[:2]) < CLOSE_XY_TOL
      if not self._closing[i]:
        self._settle[i] = self._settle[i] + 1 if xy_ok else 0
        if self._settle[i] >= 2 or self._phase_steps[i] >= CLOSE_DELAY:
          self._closing[i] = True
          self._close_k[i] = 0
      if self._closing[i]:
        self._close_k[i] += 1
        if self._close_k[i] >= CLOSE_STEPS - CLOSE_DELAY:
          if self._aperture(obs_i) < APERTURE_HELD:
            self._rewind(i, f"close_aperture {self._aperture(obs_i):.3f}")  # pads met on nothing
            return np.zeros(3), rot, GRIPPER_OPEN
          self._goto(i, P_CLIMB)
      grip = GRIPPER_CLOSED if self._closing[i] else GRIPPER_OPEN
      return self._guard(err, q_abs), rot, grip

    # P_CLIMB: straight up; accumulate the bin position in the base frame.
    p, _R = self._fk(q_abs)
    self._bin_acc[i] += p + obs_i[40:43] + obs_i[43:46]
    self._bin_n[i] += 1
    if self._phase_steps[i] >= CLIMB_STEPS:
      if (np.linalg.norm(gto) > HELD_TOL or self._aperture(obs_i) < APERTURE_HELD
          or gto[2] < SHALLOW_OFFSET):
        self._rewind(i, f"climb gto={np.round(gto,3).tolist()} ap={self._aperture(obs_i):.3f}")
        return np.zeros(3), rot, GRIPPER_OPEN
      self._plan_windup(i)
      self._goto(i, P_WINDUP)
      return np.zeros(3), rot, GRIPPER_CLOSED
    return np.array([0.0, 0.0, CLIMB_ERR]), rot, GRIPPER_CLOSED

  def _plan_descend(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> None:
    """Offline IK (internal model) for the grasp posture: site above the cube centre at
    DESCEND_SITE_Z, hand down, yaw from the face servo.  A joint-space ramp to it tracks
    to ~1 cm, where the lead-driven IK descent overshot the floor guard by 1.5 cm."""
    p_now, _R = self._fk(q_abs)
    gto = self._gto(i, obs_i)
    target = p_now + np.array([gto[0], gto[1], gto[2] + (DESCEND_SITE_Z - OBJ_CENTER_Z)])
    q = q_abs.copy()
    for _ in range(200):
      p, _R = self._fk(q)
      dq = self._ik_step(q, target - p, _DOWN_AXIS)
      q[:7] += dq
      if np.linalg.norm(dq) < 1e-5:
        break
    q7 = q[6]
    _p, R = self._fk(q)
    yaw_now = float(np.arctan2(R[1, 1], R[0, 1]))
    d_yaw = float(np.angle(np.exp(4j * (self._closing_yaw_target(i, obs_i, q_abs) - yaw_now))) / 4.0)
    q[6] = float(np.clip(q7 - d_yaw, JOINT_LIMITS[6, 0], JOINT_LIMITS[6, 1]))
    last = self._q_cmd[i]
    self._q_grasp[i] = np.clip(q[:7], JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    self._q_cmd_js[i] = q_abs[:7] if np.any(np.isnan(last)) else last.copy()

  def _descend_js(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    d = self._q_grasp[i] - self._q_cmd_js[i]
    peak = float(np.max(np.abs(d)))
    if peak > DESCEND_DQ:
      d = d * (DESCEND_DQ / peak)
    self._q_cmd_js[i] += d
    self._q_cmd[i] = self._q_cmd_js[i].copy()  # keep the IK lead integrator continuous
    arrived = peak < 1e-3 and self._phase_steps[i] > 3
    if arrived:
      self._settle[i] += 1
    if self._settle[i] >= SEAT_SETTLE or self._phase_steps[i] > DESCEND_TIMEOUT:
      self._goto(i, P_CLOSE)
    return self._js_action(self._q_cmd_js[i], GRIPPER_OPEN)

  def _integrate(self, i: int, err: np.ndarray) -> None:
    lateral = np.array([err[0], err[1], 0.0])
    if np.linalg.norm(lateral) < HOVER_INTEG_BAND:
      self._integ[i] = np.clip(
        self._integ[i] + HOVER_INTEG_GAIN * lateral, -HOVER_INTEG_CLIP, HOVER_INTEG_CLIP
      )

  def _pad_min_z(self) -> float:
    """Lowest corner of the two finger pads at the pose last passed to ``_fk``.  The pads
    hang 0.0119 below the site only while the hand is exactly vertical; a tilt of th drops
    the outer pad by 0.04 sin(th) (7 mm at 10 deg), which a site-height guard cannot see."""
    if not hasattr(self, "_pad_gids"):
      self._pad_gids = [self.model.geom(n).id for n in ("left_finger_pad", "right_finger_pad")]
      self._pad_hs = [self.model.geom_size[g].copy() for g in self._pad_gids]
    best = 1e9
    for g, hs in zip(self._pad_gids, self._pad_hs):
      pos = self.data.geom_xpos[g]
      R = self.data.geom_xmat[g].reshape(3, 3)
      best = min(best, float(pos[2] - abs(R[2, 0]) * hs[0] - abs(R[2, 1]) * hs[1]
                             - abs(R[2, 2]) * hs[2]))
    return best

  def _guard(self, err: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    z = float(self._fk(q_abs)[0][2])
    pad = self._pad_min_z()
    err = err.copy()
    if pad + err[2] < self.PAD_FLOOR_MIN:
      err[2] = self.PAD_FLOOR_MIN - pad
    if z + err[2] < FLOOR_MIN_Z:
      err[2] = FLOOR_MIN_Z - z
    return err

  # -- wind-up / swing -------------------------------------------------------------------

  def _bin_estimate(self, i: int) -> np.ndarray:
    return self._bin_acc[i] / max(1, self._bin_n[i])

  def _plan_windup(self, i: int) -> None:
    b = self._bin_estimate(i)
    phi = float(np.arctan2(b[1], b[0]))
    D = float(np.hypot(b[0], b[1]))
    v = float(np.clip(_ballistic_speed(D - REL_R, REL_Z - LAND_Z, LAUNCH_EL), V_MIN, V_MAX))
    q_rel = self._q_rel_base.copy()
    q_rel[0] = phi
    self._q_rel[i] = q_rel
    self._v_req[i] = v
    om = self._rates_for(v * np.array([np.cos(LAUNCH_EL), np.sin(LAUNCH_EL)]), np.zeros(3))
    self._omega[i] = SPEED_GAIN * om
    self._q_w[i] = np.clip(
      q_rel - self._omega[i] * DT * SWING_T, JOINT_LIMITS[:, 0] + 0.02, JOINT_LIMITS[:, 1] - 0.02
    )
    self._q_cmd_js[i] = np.nan
    self._settle[i] = 0
    self._off_acc[i] = 0.0
    self._off_n[i] = 0
    self._replanned[i] = False

  def _plan_swing(self, i: int) -> None:
    """Re-plan at the swing start with the refined bin estimate and the measured cube
    offset (hand frame -> world at the release posture)."""
    b = self._bin_estimate(i)
    phi = float(np.arctan2(b[1], b[0]))
    D = float(np.hypot(b[0], b[1]))
    off_hand = self._off_acc[i] / max(1, self._off_n[i]) if self._off_n[i] > 0 else np.zeros(3)
    off_world0 = self._R_rel @ off_hand  # arm plane at azimuth 0
    # cube release point in the arm plane: site (REL_R, REL_Z) + offset
    R = D - AIM_SHORT - (REL_R + off_world0[0])
    h = (REL_Z + off_world0[2]) - LAND_Z
    v = float(np.clip(_ballistic_speed(R, h, LAUNCH_EL), V_MIN, V_MAX))
    q_rel = self._q_rel_base.copy()
    q_rel[0] = phi
    self._q_rel[i] = q_rel
    self._v_req[i] = v
    om = self._rates_for(v * np.array([np.cos(LAUNCH_EL), np.sin(LAUNCH_EL)]), off_world0)
    self._omega[i] = SPEED_GAIN * om
    self._q_w[i] = q_rel - self._omega[i] * DT * SWING_T

  def _windup(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    if np.isnan(self._q_cmd_js[i, 0]):
      last = self._q_cmd[i]
      self._q_cmd_js[i] = q_abs[:7] if np.any(np.isnan(last)) else last.copy()
    d = self._q_w[i] - self._q_cmd_js[i]
    peak = float(np.max(np.abs(d)))
    if peak > WINDUP_DQ:
      d = d * (WINDUP_DQ / peak)
    self._q_cmd_js[i] += d
    if peak < 1e-3:
      self._settle[i] += 1
      if self._settle[i] > 2:  # hand at rest: measure the cube offset in the hand frame
        p, R = self._fk(q_abs)
        self._off_acc[i] += R.T @ obs_i[40:43]
        self._off_n[i] += 1
    if self._replanned[i]:
      # Second stage: the ramp may only start from the posture the arm is actually AT.
      if peak < 1e-3 or self._settle[i] >= RE_SETTLE or self._phase_steps[i] > WINDUP_TIMEOUT + 20:
        self._goto(i, P_SWING)
        self._swing_k[i] = 0
      return self._js_action(self._q_cmd_js[i], GRIPPER_CLOSED)
    if self._settle[i] >= WINDUP_SETTLE + OFFSET_SAMPLES or self._phase_steps[i] > WINDUP_TIMEOUT:
      if np.linalg.norm(obs_i[40:43]) > HELD_TOL or self._aperture(obs_i) < APERTURE_HELD:
        self._rewind(i, f"windup gto={np.round(obs_i[40:43],3).tolist()} ap={self._aperture(obs_i):.3f}")
        return self._js_action(self._q_cmd_js[i], GRIPPER_OPEN)
      self._plan_swing(i)
      self._replanned[i] = True
      self._settle[i] = 0
    return self._js_action(self._q_cmd_js[i], GRIPPER_CLOSED)

  @staticmethod
  def _ramp_s(k: int) -> float:
    """Ramp position (in units of DT * omega) after control step ``k``, with the rate eased
    in linearly over RAMP_IN steps.  A pure ``k + 1 + KICK`` puts 2 steps of ramp on the
    first command and rings the arm at ~5 Hz for the rest of the swing (measured)."""
    n = k + 1
    if n <= RAMP_IN:
      s = n * (n + 1) / (2.0 * RAMP_IN)
    else:
      s = (RAMP_IN + 1) / 2.0 + (n - RAMP_IN)
    return s + KICK * min(1.0, n / float(RAMP_IN))

  def _cube_state(self, i: int, obs_i: np.ndarray):
    """Smoothed cube position + velocity at NOW, from a quadratic fit of the last
    VEL_FIT_N ``object_pos`` samples.  The term is absolute (it carries the per-env scene
    origin) but the origin is constant, so it cancels both in the derivative and in the
    difference against the accumulated goal -- the relative-observation rule is kept."""
    h = self._cp_hist[i]
    n = int(self._cp_n[i])
    if n < VEL_FIT_N:
      h[n] = obs_i[18:21]
      self._cp_n[i] = n + 1
      return None
    h[:-1] = h[1:]
    h[-1] = obs_i[18:21]
    c = self._vfit @ h  # rows: constant and linear coefficients (3,) each
    return c[0] + c[1] * self._vfit_t, c[1]

  def _predicted_range(self, i: int, obs_i: np.ndarray):
    """(range the cube would fly if released NOW, range it has to fly), both measured
    along the horizontal direction from the cube to the goal.  Ballistic propagation to
    the goal's own height; the goal is the episode-mean of obs 18:21 + obs 43:46."""
    st = self._cube_state(i, obs_i)
    if st is None:
      return None
    p, u = st
    d = self._goal_acc[i] / max(1, self._goal_n[i]) - p
    nd = float(np.linalg.norm(d[:2]))
    if nd < 1e-6:
      return None
    drop = max(-float(d[2]), 0.0)
    tf = (float(u[2]) + float(np.sqrt(max(u[2] ** 2 + 2.0 * G * drop, 0.0)))) / G
    r_now = float(np.dot(u[:2], d[:2] / nd)) * tf
    self._dbg[i] = (r_now, nd - AIM_SHORT, u[0], u[1], u[2], tf)
    return r_now, nd - AIM_SHORT

  def _swing(self, i: int, obs_i: np.ndarray, q_abs: np.ndarray) -> np.ndarray:
    k = self._swing_k[i]
    self._swing_k[i] += 1
    self._q_cmd_js[i] = self._q_w[i] + self._omega[i] * DT * self._ramp_s(k)
    om = self._omega[i]
    n = float(np.linalg.norm(om)) + 1e-9
    s = float(np.dot(q_abs[:7] - self._q_rel[i], om / n))  # progress along the ramp (rad)
    if RELEASE_MODE == "predict":
      pr = self._predicted_range(i, obs_i)
      fire = False
      if pr is not None:
        r_now, r_need = pr
        dr = 0.0 if np.isnan(self._r_prev[i]) else r_now - self._r_prev[i]
        self._r_prev[i] = r_now
        # Half-open interval on the STEP the landing point crosses the goal: the ramp
        # adds ~2-3 cm of range per control step, so firing on "already past" biases the
        # throw long by half a step.  The lead also covers the one step of ramp the cube
        # still rides after the opening action is issued.
        fire = (r_now + RELEASE_STEP_LEAD * max(dr, 0.0)) >= r_need
      release = fire or k > SWING_T + SWING_EXTRA
    else:
      release = s >= -RELEASE_LEAD * n * DT or k > SWING_T + 12
    if release:
      self._goto(i, P_FOLLOW)
      # Zero-force release: finger target AT the cube width for this one step, so the PD
      # force on the pads vanishes without moving them (opening under an 8 N squeeze let
      # the lagging second finger kick the cube 3-10 deg sideways -- measured); the
      # fingers open fully from the next step.
      return self._js_action(self._q_cmd_js[i], GRIPPER_ZERO)
    return self._js_action(self._q_cmd_js[i], GRIPPER_CLOSED)

  def _follow(self, i: int) -> np.ndarray:
    if self._phase_steps[i] < FOLLOW_STEPS:
      k = self._swing_k[i]
      self._swing_k[i] += 1
      self._q_cmd_js[i] = self._q_w[i] + self._omega[i] * DT * self._ramp_s(k)
    return self._js_action(self._q_cmd_js[i], GRIPPER_OPEN)

  def _js_action(self, q7: np.ndarray, grip: float) -> np.ndarray:
    a = np.zeros(8, dtype=np.float32)
    q7 = np.clip(q7, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    a[:7] = (q7 - self.default_qpos[:7]) / FRANKA_ACTION_SCALE
    a[7] = grip
    return a
