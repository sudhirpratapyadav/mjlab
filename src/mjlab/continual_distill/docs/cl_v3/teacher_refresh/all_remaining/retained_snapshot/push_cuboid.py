"""Scripted Push-Cuboid teacher: the closed gripper as a flat paddle (CL-V3, W1-P).

This file also hosts ``PlanarPushPolicy``, the planar-transport machinery shared with
``drag_pull.py`` (same carton, goal toward the base) and ``cage_drag.py`` (open
fingers caging a cube). History: the phase-1 / cl_v2 "contact-point servo" teacher and
its seven-strategy graveyard are described in ``docs/cl25/phase_1/LOGS.md`` and the
analysis that retired it in ``docs/cl_v3/logs/W1-P.md``. The short version:

* MEASURED (panda.xml FK, site frame, z = approach axis): the finger-closing axis is
  the site **y** axis; the two ``*_finger_pad`` boxes are the only collidable finger
  geometry and, closed, form one paddle x ∈ ±0.0087, y ∈ ±0.0152, z ∈ [-0.0046,
  +0.0119] (1.19 cm BELOW the site). The hand capsule's lowest point is 3.0 cm ABOVE
  the site, so the pads always touch the floor first.
* MEASURED (W0-b trace): the old teacher commanded the site 0.028 above the box centre
  and got 0.023 (gravity sag is a permanent offset in the base-class loop, and the
  ±1 cm observation noise on z went straight into the command); the pad's bottom edge
  therefore sat at ~0.026 against a box top at 0.030 — a top-edge point contact that
  tips / climbs the box (rises of up to 2.6 cm) instead of sliding it. That is the
  "friction-limited push stall" of the phase-1 handover.

Strategy (docs/cl_v3/PLAN.md §4 group P, verified here):
  1. paddle at the object's mid-height, z servoed ABSOLUTELY from the arm's own FK
     (base frame — no scene offset, so it is legal) with an integral term for the sag;
  2. contact face chosen from ``object_orientation`` (obs 34:37 is the second row of
     the rotation matrix → yaw = atan2(obs[34], obs[35])): the face whose outward normal
     is most opposite to the goal direction, with hysteresis; the face's own half-extent
     gives the contact plane;
  3. contact point where the goal line through the box centre exits that face (push
     through the centre of friction → no torque) plus a proportional lateral offset that
     steers the box square to the goal line; if the pad leaves the face, RESEAT;
  4. command = contact point + lead along the goal line, lead ∝ remaining distance,
     with a small command integrator (``cmd_lead_max``) for sustained force;
  5. stop on a FILTERED distance inside the tolerance and HOLD in light contact
     (success latches) instead of backing off.

Observation layout (60-D): 0:9 joint_pos_rel | 34:40 object rot6d | 40:43
gripper_to_object | 43:46 object_to_goal. Only relative terms + joint angles are used.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# Measured pad geometry (site frame, closed fingers). See module docstring.
PAD_FACE_X = 0.0087  # paddle contact face offset along site x (closed pads)
PAD_BELOW_SITE = 0.0037  # pad vertical centre below the site
PAD_BOTTOM_BELOW_SITE = 0.0119  # lowest collidable point below the site
PAD_INNER_Y_OFFSET = 0.0  # open pad inner face sits at y = ±q_finger exactly
PAD_HALF_WIDTH_CLOSED = 0.0152  # closed paddle: half width across the push direction
PAD_HALF_WIDTH_OPEN = 0.0087  # one open pad's inner face is 1.74 cm wide

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])  # axis-only IK target (yaw free)

# Phase ids (also exported for diagnose.py).
P_APPROACH, P_DESCEND, P_ENTER, P_PUSH, P_HOLD, P_RESEAT_UP, P_RESEAT_SIDE = range(7)
PHASE_NAMES = ["APPROACH", "DESCEND", "ENTER", "PUSH", "HOLD", "RESEAT_UP", "RESEAT_SIDE"]


def down_frame(yaw: float) -> np.ndarray:
  """Full EE rotation: site z straight down, site x at ``yaw`` in the world xy-plane
  (so the finger axis, site y, is at yaw - 90°)."""
  c, s = np.cos(yaw), np.sin(yaw)
  ex = np.array([c, s, 0.0])
  ez = np.array([0.0, 0.0, -1.0])
  ey = np.cross(ez, ex)
  return np.column_stack([ex, ey, ez])


def _wrap(a: float) -> float:
  return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _wrap_half(a: float) -> float:
  """Wrap to (-pi/2, pi/2]: the paddle / finger pair is symmetric under a 180° yaw."""
  return float((a + np.pi / 2) % np.pi - np.pi / 2)


class PlanarPushPolicy(ClassicalPolicyBase):
  """Push a box-shaped object on the floor to a goal with the gripper as a paddle."""

  DEFAULT_QPOS = HOME_QPOS
  max_dq = 0.15
  step_clip_mode = "uniform"
  ik_iters = 30
  # MEASURED (offline DLS probe, full-frame top-down target at z = 0.0225 over the
  # workspace): the base class's posture_weight 0.005 leaves a 13-35 mm position
  # residual (the HOME pull balances the position term at a low, extended pose — the
  # phase-1 "M4" stationary point, here on every seat). 0.001 -> 9 mm, 0.0003 -> 2.6 mm
  # worst case, 0 -> 0; 0.0003 keeps the redundancy regularised.
  posture_weight = 0.0003

  # ---- task geometry (override per task) ----------------------------------
  OBJ_HALF = np.array([0.0365, 0.0446, 0.0150])  # body-frame half extents (x, y, z)
  SUCCESS_TOL = 0.02  # task's success_threshold (3-D)
  CAGE = False  # False: closed paddle along site x; True: open pads, push along site y
  GRIPPER_ACTION = -1.0
  PAD_HALF_W = PAD_HALF_WIDTH_CLOSED  # half width of the contact patch across the push

  # ---- heights (ABSOLUTE site z in the base frame; the object rests on the floor)
  # MEASURED (v21): APPROACH used to hand over at a site height of ~0.14 and DESCEND,
  # which runs at the slow push max_dq, only manages ~5 mm/step -- it hit its cap with
  # the arm still 4 cm high, PUSH then began mid-descent and drove the pads into the
  # floor (41 % floor terminations). The traverse height is lowered to just clear the
  # box, the hand-over waits for it, and the upper half of the descent keeps the fast
  # free-space rate.
  HOVER_Z = 0.050  # traverse height (pads at 0.038: 8 mm over the 0.030 box)
  DESCEND_START_XY = 0.10  # xy error below which the traverse already drops to RIDE_Z
  # MEASURED (v12 trace, n = 32): at RIDE_Z 0.026 the pad's bottom edge rides 1.41 cm
  # over the floor and 59 % of envs terminated on `ee_ground_collision` -- the sensor is
  # the SUBTREE of link7, i.e. the finger pads, and the loaded arm tracks 1-2 cm below
  # its commanded height. The pad face is 1.65 cm tall and the carton 3.0 cm, so the
  # site can sit anywhere in 0.026-0.034 and still put a flat face on the box; taking
  # the top of that window buys 2.1 cm of floor clearance for 0.9 cm of engagement.
  # Tipping is not the limit: the contact centroid is then 0.0256 above the floor
  # against a 0.0365 m half-depth (ratio 0.70; the box tips only above 1.0).
  RIDE_Z = 0.030  # pad spans z 0.0201..0.0366 -> the carton's top 0.9 cm (box 0..0.030)
  FLOOR_MIN_Z = 0.022  # the commanded site height is never allowed below this
  # Emergency xy gate (on the FILTERED height): the pads reach the floor at a site
  # height of 0.0119, so 0.024 / 0.019 leaves 12 / 7 mm of pad clearance and sits a
  # clear 8 mm below the ride band, where the height servo normally holds the arm.
  PAD_CLEAR_MIN = 0.005  # lowest pad corner is never commanded below this
  PAD_CLEAR_THROTTLE = 0.006  # ... the xy command is throttled below this ...
  PAD_CLEAR_FREEZE = 0.003  # ... and frozen below this
  XY_THROTTLE_Z = 0.026
  XY_FREEZE_Z = 0.019
  XY_STEP_MAX = 0.050
  PUSH_STEP_MAX = 0.090  # ... but a loaded push may command the full lead  # largest single xy waypoint below XY_STEP_LOW_Z
  XY_STEP_LOW_Z = 0.055
  Z_ALPHA = 0.35  # EMA on the FK height, for the phase logic and the gate above
  RISE_RATE = 0.022
  # MEASURED at n = 128, the number that matters: saw-tooth 0.352 / 0.383 / 0.383 on
  # Push-Cuboid / Drag-Pull / Cage-Drag against 0.344 / 0.430 / 0.461 for climb-and-hold.
  # It looked like a clear win at n = 32 on the pinned seed (0.531 vs 0.375) and was not.
  PUSH_CYCLE = False  # saw-tooth; False = climb to the cap and hold there
  PUSH_CYCLE_AMP = 0.006
  PUSH_CYCLE_RESET = 0.004  # bounded downward step that restarts the cycle (unused when PUSH_CYCLE is False)
  PUSH_CREEP = 0.0009  # m/step of upward creep of the height reference while pushing
  REF_LAG_MAX = 0.055  # the reference never sits more than this BELOW the arm ...
  REF_LEAD_MAX = 0.035  # ... nor this far above it (no wind-up on a blocked hand)
  DESCENT_RATE_FAR = 0.025  # m per step of commanded descent above DESCENT_SLOW_Z
  # MEASURED (v12): the last 2 cm of the descent ran at 5-7 mm/step of ACTUAL motion and
  # the arm carried straight through the ride height into the floor in 2-3 steps (the
  # integrator, at 1 mm/step, cannot arrest that). 4 mm/step costs ~5 steps and settles.
  DESCENT_RATE_NEAR = 0.006  # ... below DESCENT_SLOW_Z
  DESCENT_SLOW_Z = 0.045
  # Height integrator (per step, anti-windup band below). MEASURED: the command lead
  # alone leaves the site 1-2 cm below an absolute z target (the IK re-solves the
  # absolute target from the observed joints, so the command converges to the IK
  # solution and the PD sits at solution - sag); the integrator held 0.022-0.025 for a
  # 0.0235 target with a bias of 0.012-0.020 (CPU trace).
  # MEASURED (env trace): with KI 0.3 the site oscillated 0.021 <-> 0.031 (1 cm p-p,
  # ~15-step period) and the box moved ONLY while the pad rode up the face: a pad that
  # rubs down the face drags it down with pad friction (mu up to 1.5) and pins the box
  # to the floor (j2/j4 at 30-44 Nm on a stationary box). The sag is learned once at
  # the settled hover and only trimmed here.
  Z_KI = 0.06
  Z_INT_BAND = 0.03  # integrate only inside this band (no wind-up on the transient)
  Z_BIAS_MIN = 0.0  # gravity sag is downward: never command below the ride height
  Z_BIAS_MAX = 0.035
  APPROACH_MAX_DQ = 0.30  # free-space joint rate; max_dq (0.15) once near the object
  APPROACH_LEAD = 0.20  # free-space command lead (rad); cmd_lead_max once near the object
  YAW_FREE_DIST = 0.10  # xy error above which the approach is yaw-free (axis-only IK)
  YAW_RATE_FREE = 0.10  # rad per step for the yaw ramp before contact
  APPROACH_DESCENT_RATE = 0.050  # m per step of commanded descent while translating
  OVER_MARGIN = 0.045  # pad this far inside the face plane (3 steps running) -> RESEAT_UP
  OVER_STEPS = 5
  LATERAL_MARGIN = 0.030  # |s_t| beyond h_t + this -> the pad has slipped off the box

  # ---- push law -------------------------------------------------------------
  STANDOFF = 0.004  # pad-to-face clearance while seating
  # MEASURED: a 4.5 cm lead put the arm permanently mid-way along a joint-space
  # interpolation and dipped the site ~1 cm below the ride height (floor touches in
  # the first push steps); 2.5 cm halves the dip and still moves the carton briskly.
  LEAD_GAIN = 0.8
  LEAD_MIN = 0.030
  # MEASURED (v31): the box moves on only 60 % of push steps and averages 3.9 mm/step,
  # against a p75 of 7.3 -- the median episode needs 24 cm of travel and only ~60 steps
  # are left after the (torque-limited, ~35-45 step) approach. The lead IS the push
  # force: a command driven deep into the box keeps the forward joint error, and hence
  # the contact force, saturated for as long as the box resists. It only applies far
  # from the goal; inside AIM_DIRECT_DIST the 0.8 * dist law takes over and bounds it.
  LEAD_MAX = 0.070
  LEAD_NEAR = 0.035  # lead floor until the box is within STOP band + LEAD_NEAR_DIST
  LEAD_NEAR_DIST = 0.03
  Z_FF_PER_LEAD = 0.25  # MEASURED ~1 cm site dip per 4 cm of pending push lead
  LEAD_RAMP_STEPS = 5
  # Steps of the box's own velocity added to the push waypoint. MEASURED: 3.0 cost
  # 0.531 -> 0.344 on the pinned n = 32 draw and it is left OFF. `_o2g_v` is an EMA of a
  # +-1 cm observation, so its ~2 mm/step of noise times three steps is 6 mm of waypoint
  # jitter -- the same size as the standoff it is meant to remove.
  VEL_FF_STEPS = 0.0
  # ---- GLUE PUSH: a sustained command OFFSET, not a chased absolute waypoint -------
  # MEASURED (W1-P3, claim3_push trace, 10 422 push steps, TRUE pad-to-carton clearance
  # recomputed from the SIM object pose rather than the teacher's estimate):
  #   clearance band      share of PUSH   P(box moves)   box mm/step   site mm/step
  #     [-2, +8] mm          36 %          0.68-0.93       5.2-9.8       6.4-9.7
  #     > 12 mm              38 %          0.14-0.20       1.3-2.2       4.4-5.2
  #     < -5 mm (jammed)     12 %          0.04            0.4           1.0
  # In the good band the box travels at the ARM's own speed: the transport rate is not
  # friction-limited or impulse-limited, it is simply the fraction of push steps spent in
  # contact. The loss is a slow DRIFT, not a shoot-away: only 3.7 % of good steps jump
  # straight to > 12 mm, but the median clearance grows +0.4 mm/step inside the band, and
  # once it is out, it closes at only 1.2 mm/step median although the site is moving
  # 4.4 mm/step. I.e. ~4 of the site's 6.4 mm/step go SIDEWAYS, not into the box.
  # The cause is the waypoint: `seat + xhat*(standoff+lead)` is an ABSOLUTE point rebuilt
  # every step from a lagging estimate of a moving box, whose lateral part jumps with the
  # face normal, the steer offset and the estimate noise. The arm is force-saturated
  # during a push (joints 1-2 are kp 1000 against a 100 N m forcerange, so the 0.12 rad
  # command clip saturates them), so it travels at a FIXED speed in whatever direction
  # `pos_err` points: every millimetre of lateral waypoint is a millimetre not pushed.
  # So while pushing the command becomes a pure sustained offset from where the pad IS:
  # forward along the pad's own normal by GLUE_ADVANCE (enough to keep the servo
  # saturated, which is what holds the pad on the face — the box is kept in contact by
  # force, not by predicting where it will be), plus a rate-limited lateral term that
  # re-centres the pad on the face. No box-position feedback enters the along-push axis
  # at all, so estimate lag can no longer make the pad fall behind.
  GLUE_PUSH = True
  GLUE_ADVANCE = 0.050  # m of commanded penetration per step (saturates the servo)
  GLUE_LAT_CONTACT = 0.006  # lateral re-centring allowed while the pad is loaded
  GLUE_LAT_FREE = 0.030  # ... and while it is clear of the box
  GLUE_CONTACT_GAP = 0.010  # estimated normal gap above which the pad counts as free
  # ---- FORK: the pusher is a two-point contact, not one narrow paddle ---------------
  # The gripper action is not binary. ``JointPositionAction`` gives the finger actuator
  # ``target = 0.04 + 0.04 * a`` (default offset 0.04, FRANKA_ACTION_SCALE 0.04, clipped
  # to the 0..0.04 ctrlrange), so any aperture is directly commandable.
  # GEOMETRY (measured, site frame): each pad box is centred at y = +-(q + 0.0076) with
  # half-extents (0.0088, 0.0076, 0.0082). CLOSED the two pads merge into one paddle
  # 30.4 mm wide; at q = 0.020 they are two patches spanning |y| = 20.0-35.2 mm, i.e. an
  # effective pusher 70.4 mm wide with a 40 mm hole in the middle.
  # WHY IT MATTERS: the carton's face is 73 or 89 mm wide. A 30 mm paddle puts the whole
  # contact resultant within +-15 mm of the box centre, so any lateral error is a torque
  # and the box yaws off the pad -- which is what the crab / lateral-slip re-seats in
  # this file exist to recover from, at 20-43 steps an episode. Two patches 55 mm apart
  # resist that yaw by differential normal force instead: the box self-aligns flush
  # against the fork and stays there. The carton (73 mm on its short axis) cannot pass
  # through the 40 mm gap, so the fork also captures it laterally.
  FORK = False
  FORK_APERTURE = 0.020  # commanded finger position (m); pads then span |y| 20-35.2 mm
  FORK_PAD_EDGE = 0.0152  # pad outer edge = q + this; inner edge = q
  AIM_ALWAYS = False  # hold the pusher on the goal line at every distance (see below)
  GOAL_MEAN = True  # goal error from the averaged static goal, not the lag-corrected EMA
  # MEASURED (CPU traces): a pad placed off the face centre cannot be re-positioned
  # once loaded (it sticks to the face, pad friction up to 1.5), and a single sticking
  # contact point pushing a box is unstable: the box rotated straight through
  # alignment (phi -11° -> +51°) until the face flipped. Both the "exit point of the
  # goal line" contact and an explicit proportional offset did this. So: the paddle is
  # FLUSH and CENTRED on the face (CoM inside the pad width -> line contact, no
  # rotation), pushes along its OWN normal, and steers by yawing slowly (above).
  # STEERING. The paddle is kept FLUSH with the contact face (a yawed paddle touches on
  # an edge -- a single sticking point, which is the unstable contact the phase-1 /
  # cl_v2 graveyard measured: the box rotated straight through alignment and the face
  # flipped). The steering input is instead the pad's LATERAL OFFSET on the face: a
  # flush pad narrower than the face (3.04 cm against 7.3-8.9 cm) puts the contact
  # resultant at the PAD's centre, so an offset `e` applies a moment F*e about the box
  # centre and turns the box. Because the box turns quasi-statically -- the yaw RATE, not
  # the yaw, is what `e` commands -- a pure proportional law overshoots (measured); the
  # damping term on the box's own measured yaw rate is what makes it settle.
  STEER_GAIN = 0.030  # m of pad offset per rad of face-vs-goal error
  STEER_DAMP = 0.25  # m per (rad/step) of measured box yaw rate
  STEER_MAX = 0.020  # capped again per face by (half-width - pad half-width - 3 mm)
  STEER_YAW_ALPHA = 0.4
  RESEAT_Z = 0.048  # lift the pads above the box top (0.030 + 0.012 + margin) to reseat
  STOP_FRAC = 0.5  # STOP_TOL = STOP_FRAC * SUCCESS_TOL on the filtered distance
  RESUME_MARGIN = 0.006
  FACE_HYST = np.deg2rad(10.0)
  AIM_DIRECT_DIST = 0.060  # inside this the pad is aimed down the goal line, not flush
  AIM_ST_MAX = 0.028
  AIM_RESEAT_ST = 0.030  # crab beyond which a spent segment needs the orbit, not a re-lay
  RESEAT_HOLDOFF_SPEED = 0.0018  # box speed (m/step) above which no re-seat is allowed  # pad crab across the box that ends a latched segment
  AIM_PASSED = 0.004  # along-track remainder below which the segment is spent
  # Steering = yawing the flush paddle toward the goal line slowly enough that the box
  # follows it (the CoM must keep projecting inside the 3 cm pad width: the hand is never
  # commanded more than YAW_LAG_MAX ahead of the box's face).
  YAW_RATE = 0.035  # rad per step toward the goal direction
  YAW_LAG_MAX = 0.22  # rad the hand may lead the box's contact-face normal
  YAW_FREEZE_DIST = 0.03
  REVERSE_STEPS = 4  # goal behind the paddle for this long -> go around (RESEAT_UP)
  # Aim point for the heading = goal + LOOKAHEAD along the start->goal line, so the
  # heading command stays well-conditioned next to the goal (pure pursuit of the goal
  # itself spins the heading demand as the box comes abeam: MEASURED near-miss 0.032).
  # MEASURED (v18): with the aim point 6 cm BEYOND the goal the pursuit drove the box to
  # that point, so every env that came within 2-4 cm was then pushed back out to 0.1-0.5
  # (env 0: 0.027 at t = 110, 0.092 at t = 149). With the support-plane contact the pad
  # simply circles the box as the bearing swings, so the aim point can sit on the goal;
  # a small lookahead is kept only to keep the heading well conditioned at contact.
  LOOKAHEAD = 0.015
  CAGE_ALIGN_TOL = 0.008  # xy error the cage must hold before dropping past the cube top
  CAGE_GATE_Z = 0.062  # ... and the height it waits at until it does

  # ---- filters ----------------------------------------------------------------
  # `gripper_to_object` / `object_to_goal` both carry +-0.01 m of uniform noise per axis
  # (5.8 mm rms). OBJ_ALPHA 0.3 takes the contact geometry to 2.6 mm rms and, because
  # the hand and the box move TOGETHER once the pad is loaded, costs almost no lag.
  OBJ_ALPHA = 0.3
  GOAL_ALPHA = 0.25
  GOAL_LAG_STEPS = 3.0  # ~(1 - GOAL_ALPHA) / GOAL_ALPHA: the EMA's lag, removed above
  RESET_JUMP = 0.12

  # ---- phase caps (unconditional) ----------------------------------------------
  APPROACH_CAP = 40
  DESCEND_CAP = 10
  ENTER_CAP = 20
  RESEAT_UP_STEPS = 7
  RESEAT_SIDE_STEPS = 14
  ORBIT_R = 0.080  # orbit radius: box half-diagonal 0.058 + 2.2 cm clearance
  ORBIT_RATE = 0.40  # rad per step around the box
  ORBIT_TOL = 0.25  # bearing error at which the orbit hands back to PUSH

  # Command integrator ON, with a cap that leaves the shoulder its gravity torque.
  # MEASURED: (a) with q_des = q_obs + Δ (base default) the ±0.01 rad observation noise
  # on robot_joint_pos goes straight into every joint command and rings the
  # underdamped wrist (kp 300, kv 2): the physical site height bounced 0.016 <-> 0.034
  # between consecutive steps with CONSTANT commanded actions (GPU trace, env 0
  # t=120-149; the W0-b trace of the old teacher has the same ±1 cm spread). With the
  # pads 1.2 cm below the site that jitter is the floor-collision rate. (b) a 0.03 rad
  # cap starves the shoulder of gravity torque (kp*0.03 = 30 Nm) and the arm sinks
  # during the approach. 0.12 rad clears the shoulder's tracking lag under gravity and
  # still bounds a blocked push to kp*0.12.
  cmd_lead_max = 0.12

  # -- state --------------------------------------------------------------------
  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    n = self.num_envs
    if env_ids is None:
      self._obj_f = np.full((n, 3), np.nan)
      self._goal_f = np.full((n, 3), np.nan)
      self._gto_f = np.zeros((n, 3))
      self._obj_m = np.full((n, 3), np.nan)  # running MEAN of the static object pose
      self._goal_m = np.full((n, 3), np.nan)  # ... and of the (always static) goal
      self._obj_n = np.zeros(n)
      self._aim_dir = np.full((n, 2), np.nan)  # latched endgame push direction
      self._oyaw = np.full(n, np.nan)   # filtered object yaw
      self._oyaw_d = np.zeros(n)        # ... and its per-step rate (CCW positive)
      self._o2g_f = np.zeros((n, 3))
      self._o2g_v = np.zeros((n, 3))
      self._z_bias = np.zeros(n)
      self._yaw_cmd = np.full(n, np.nan)
      self._face = np.full(n, -1, dtype=np.int64)
      self._timer = np.zeros(n, dtype=np.int64)
      self._side = np.ones(n)
      self._site_prev = np.full((n, 3), np.nan)
      self._z_prev = np.full(n, np.nan)
      self._z_f = np.full(n, np.nan)
      self._z_ref = np.full(n, np.nan)
      self._over = np.zeros(n, dtype=np.int64)
      self._rev = np.zeros(n, dtype=np.int64)
      self._line = np.full((n, 2), np.nan)
      self.dbg = np.zeros((n, 8))
    else:
      self._obj_f[env_ids] = np.nan
      self._goal_f[env_ids] = np.nan
      self._obj_m[env_ids] = np.nan
      self._goal_m[env_ids] = np.nan
      self._obj_n[env_ids] = 0.0
      self._z_bias[env_ids] = 0.0
      self._yaw_cmd[env_ids] = np.nan
      self._face[env_ids] = -1
      self._timer[env_ids] = 0
      self._site_prev[env_ids] = np.nan
      self._over[env_ids] = 0
      self._line[env_ids] = np.nan

  def _reinit(self, i: int) -> None:
    self._phase[i] = P_APPROACH
    self._phase_steps[i] = 0
    self._obj_f[i] = np.nan
    self._goal_f[i] = np.nan
    self._obj_m[i] = np.nan
    self._goal_m[i] = np.nan
    self._obj_n[i] = 0.0
    self._o2g_v[i] = 0.0
    self._z_bias[i] = 0.0
    self._yaw_cmd[i] = np.nan
    self._face[i] = -1
    self._timer[i] = 0
    self._q_cmd[i] = np.nan
    self._site_prev[i] = np.nan
    self._z_f[i] = np.nan
    self._z_ref[i] = np.nan
    self._aim_dir[i] = np.nan
    self._oyaw[i] = np.nan
    self._oyaw_d[i] = 0.0
    self._over[i] = 0
    self._line[i] = np.nan

  def _set_phase(self, i: int, p: int) -> None:
    self._phase[i] = p
    self._phase_steps[i] = 0

  def _rev_ok(self, i: int) -> bool:
    self._rev[i] += 1
    if self._rev[i] >= self.REVERSE_STEPS:
      self._rev[i] = 0
      return True
    return False

  # -- geometry helpers -----------------------------------------------------------
  def _object_yaw(self, obs_i: np.ndarray) -> float:
    # rot6d = rows 1 and 2 of the body rotation matrix; row 1 of Rz(yaw) = (sin, cos, 0)
    return float(np.arctan2(obs_i[34], obs_i[35]))

  def _faces(self, yaw: float):
    """Outward normals (world xy), half-extent along the normal, half-extent across."""
    c, s = np.cos(yaw), np.sin(yaw)
    ex = np.array([c, s])
    ey = np.array([-s, c])
    hx, hy = float(self.OBJ_HALF[0]), float(self.OBJ_HALF[1])
    return [(ex, hx, hy), (-ex, hx, hy), (ey, hy, hx), (-ey, hy, hx)]

  def _pick_face(self, i: int, faces, u: np.ndarray) -> int:
    dots = np.array([float(n @ u) for n, _, _ in faces])  # most negative = best
    best = int(np.argmin(dots))
    cur = int(self._face[i])
    if cur < 0:
      return best
    # hysteresis: keep the current face unless the best is clearly better
    a_cur = np.arccos(np.clip(-dots[cur], -1.0, 1.0))
    a_best = np.arccos(np.clip(-dots[best], -1.0, 1.0))
    return best if a_cur - a_best > self.FACE_HYST else cur

  # -- main -------------------------------------------------------------------------
  def _target_error(self, i: int, obs_i: np.ndarray):
    q_abs = self.default_qpos + obs_i[0:9]
    p_site, R_site = self._fk(q_abs)
    gto = obs_i[40:43]
    o2g = obs_i[43:46]

    # RELATIVE estimates. MEASURED (v13 trace, n = 32): the +-0.01 rad observation noise
    # on the nine joint angles is ~1 cm of FK position noise at the site, and the old
    # estimator put that noise into the CONTACT geometry (it filtered `p_site + gto` and
    # then measured the pad against it, so the arm's own FK error no longer cancelled).
    # The consequence is visible in the trace: at an estimated flush contact only 47 % of
    # push steps moved the box at all, while at an estimated 3-10 mm of PENETRATION 63 %
    # did, at 1.9 mm/step — i.e. the teacher had to over-drive by the size of its own
    # position error before it was really touching. Here the object centre is carried as
    # ``p_site + gto_f``, with ``gto_f`` a filter of the RELATIVE observation alone, so
    # every contact quantity below (seat, s_n, s_t, and pos_err itself) is a difference
    # in which ``p_site`` cancels exactly and only the 1 cm noise of `gripper_to_object`
    # survives -- and that one is filtered. `p_site` is still used for the ABSOLUTE
    # height servo (where the noise cancels in the command) and for reset detection.
    obj_raw = p_site + gto
    site_jump = (
      not np.isnan(self._site_prev[i, 0])
      and np.linalg.norm(p_site - self._site_prev[i]) > 0.10
    )
    self._site_prev[i] = p_site
    if np.isnan(self._obj_f[i, 0]):
      self._obj_f[i] = obj_raw
      self._gto_f[i] = gto
      self._o2g_f[i] = o2g
      self._o2g_v[i] = 0.0
    elif site_jump or np.linalg.norm(obj_raw[:2] - self._obj_f[i, :2]) > self.RESET_JUMP:
      # The env auto-reset in place (floor collision in the train cfg): the hand jumps
      # back to HOME (> 10 cm in one step) and/or the object re-spawns. Start over.
      self._reinit(i)
      self._site_prev[i] = p_site
      self._obj_f[i] = obj_raw
      self._gto_f[i] = gto
      self._o2g_f[i] = o2g
      self._o2g_v[i] = 0.0
    else:
      # gto moves with the HAND, so it is only filtered once the hand is slow (from
      # DESCEND on); in APPROACH the raw term is used and a lag would be the larger error.
      a = 1.0 if int(self._phase[i]) == P_APPROACH else self.OBJ_ALPHA
      self._obj_f[i] = (1 - 0.5) * self._obj_f[i] + 0.5 * obj_raw
      self._gto_f[i] = (1 - a) * self._gto_f[i] + a * gto
      prev = self._o2g_f[i].copy()
      g = self.GOAL_ALPHA
      self._o2g_f[i] = (1 - g) * prev + g * o2g
      self._o2g_v[i] = 0.7 * self._o2g_v[i] + 0.3 * (self._o2g_f[i] - prev)
    # Object centre and goal, both expressed in the base frame through the CURRENT site
    # so that p_site cancels out of every difference taken below.
    # THE OBJECT IS STATIC UNTIL THE PAD TOUCHES IT, SO AVERAGE IT.
    # `gripper_to_object` carries +-0.01 m of uniform noise per axis (5.8 mm rms) and the
    # nine joint angles carry +-0.01 rad (~5 mm rms at the site), so a single-sample
    # estimate of where the carton IS is good to ~8 mm and the 0.3 EMA -- three effective
    # samples -- to ~4 mm. That error is committed at the instant the seat is chosen and
    # is exactly a pad seated off the face centre, which is what the crab / lateral-slip
    # re-seats then spend 20-40 steps recovering. But `p_site + gto` is a measurement of a
    # quantity that does NOT move during the approach: a RUNNING MEAN over the ~40
    # approach steps takes it to ~1.2 mm. (Credit: W1-S measured the same mechanism on
    # Strike-Slide, 0.758 -> 0.926.)
    # It is used two ways: as the ABSOLUTE seat anchor while the box is still untouched --
    # the base class rebuilds the IK target as FK(q_obs) + pos_err with the SAME FK the
    # teacher used, so an absolute target cancels the FK noise exactly -- and as the seed
    # for the relative filter at the moment PUSH starts, so the first contact is made with
    # a 1 mm estimate instead of a 6 mm one.
    if int(self._phase[i]) in (P_APPROACH, P_DESCEND):
      if np.isnan(self._obj_m[i, 0]):
        self._obj_m[i] = obj_raw
        self._goal_m[i] = obj_raw + o2g
        self._obj_n[i] = 1.0
      else:
        self._obj_n[i] += 1.0
        self._obj_m[i] += (obj_raw - self._obj_m[i]) / self._obj_n[i]
        self._goal_m[i] += (obj_raw + o2g - self._goal_m[i]) / self._obj_n[i]
      self._gto_f[i] = self._obj_m[i] - p_site  # seed the relative filter, every step
    c = p_site[:2] + self._gto_f[i, :2]
    # The EMA on object_to_goal lags a moving box by ~(1-g)/g steps; the velocity term
    # takes that lag out so the stop test is not biased high while the box is running.
    # THE GOAL IS STATIC FOR THE WHOLE EPISODE, so it gets the same treatment: its
    # running mean over the approach is good to ~1.5 mm, and the goal error is then
    # `goal_mean - (p_site + gto_f)` -- an UNLAGGED difference of two good estimates,
    # because `gto_f` follows the box with the hand. The EMA-plus-velocity form below
    # needed `_o2g_v * 3` to cancel the filter's lag on a running box, and `_o2g_v` is a
    # difference of +-1 cm observations: three steps of it is ~6 mm of noise injected
    # straight into the stop test and the heading demand. It is kept only as the fallback
    # for an env whose approach was cut short (no mean yet).
    if self.GOAL_MEAN and not np.isnan(self._goal_m[i, 0]):
      o2g_hat = self._goal_m[i] - (p_site + self._gto_f[i])
    else:
      o2g_hat = self._o2g_f[i] + self._o2g_v[i] * self.GOAL_LAG_STEPS
    to_goal = o2g_hat[:2]
    dist = float(np.linalg.norm(o2g_hat))  # 3-D: the success predicate is 3-D
    d_xy = float(np.linalg.norm(to_goal))
    if np.isnan(self._line[i, 0]) and d_xy > 1e-6:
      self._line[i] = to_goal / d_xy  # the start->goal line, fixed per episode
    aim = o2g_hat[:2] + self._line[i] * self.LOOKAHEAD
    u = aim / (np.linalg.norm(aim) + 1e-9)  # heading demand (unit, world xy)

    # THE CONTACT PLANE IS THE BOX'S SUPPORT PLANE ALONG THE GOAL LINE, not one of its
    # four faces.
    # MEASURED (v16/v17, n = 32): picking the nearest of the four faces quantises the
    # push direction to the face normals, so it is up to 45 deg (+ 15 deg of hysteresis)
    # off the goal line -- |phi| ran 8-36 deg through PUSH -- and every time the goal
    # direction rotated past a face boundary the pad had to be lifted over the box and
    # brought round (RESEAT_UP took 11 % of all steps). Both attempts to close that
    # angle failed: yawing the paddle off flush (v16) turns the line contact into an
    # edge contact, and offsetting the pad on the face (v17) did rotate the box, but at
    # ~1 deg/step and only after the pad had crabbed 2 cm sideways across a loaded face.
    # The success predicate is POSITION ONLY -- the box's orientation is free -- and a
    # flat pusher drives a box into face-flush alignment with ITSELF. So the paddle is
    # aimed along the goal line and pressed into whatever the box presents; the box
    # aligns to the pad and then travels straight down the line. `n` is the outward
    # normal of that virtual face (exactly -u), `h_n` / `h_t` are the box's support
    # half-widths along and across it, and `phi` is zero by construction.
    yaw_obj = self._object_yaw(obs_i)
    self._face[i] = 0
    ca, sa = np.cos(yaw_obj), np.sin(yaw_obj)
    e_x = np.array([ca, sa])
    e_y = np.array([-sa, ca])
    hx, hy = float(self.OBJ_HALF[0]), float(self.OBJ_HALF[1])
    # THE PADDLE PRESSES FLUSH ON A REAL FACE, and the box is turned under it.
    # MEASURED (v26, n = 32): aiming the flat pad straight down the goal line and
    # letting the box "align itself to the pusher" does not happen fast enough -- the
    # median flush error stayed at 24 deg for the whole push, and transport is 5x worse
    # off flush (0-7 deg: 57 % of steps move the box at 1.8 mm/step; 25-46 deg: 29 % at
    # 0.36). So the contact face is the box face whose outward normal is most opposed to
    # the goal direction (paddle flush -> line contact -> the box slides), and the
    # residual direction error `phi` is worked off by OFFSETTING the pad on that face:
    # a flush pad narrower than the face puts the contact resultant at the pad's centre,
    # so an offset e applies a moment F*e about the box centre. MEASURED (v16/v17): that
    # turns the box at ~1.2 deg/step for a 1-1.5 cm offset, which closes a 25 deg error
    # in ~20 steps while the push continues.
    # ... but only while there is real distance to cover. A face normal is quantised to
    # 90 deg steps, so it is up to 45 deg off the goal line, and the steering only turns
    # the box at ~1 deg/step -- fine over a 30 cm transport, useless over the last 3 cm.
    # MEASURED: flush-face pushing (v27) transports 43 % faster than aiming down the goal
    # line (v26) -- 3.8 vs 2.65 mm/step, flush error 0.4 vs 23.8 deg -- and got 23 of 32
    # envs inside 10 cm against 19; but aiming down the goal line put 7 envs inside 2 cm
    # against 4. So: flush while far, aimed while close.
    # ENDGAME: LATCHED STRAIGHT SEGMENTS, NOT CONTINUOUS PURSUIT.
    # MEASURED (v29 traces, envs 0/1/7): inside ~5 cm the continuous re-aim becomes a
    # pure-pursuit limit cycle. The box cannot turn as fast as the bearing to the goal
    # swings (its travel direction is inside the pad's friction cone, up to ~30 deg off
    # the commanded one), so it ORBITS: the true goal error froze at 0.033 for 15+ steps
    # while the box kept moving 3-8 mm/step tangentially, and the pad crabbed sideways
    # across it (s_t 0.026 -> 0.065) until it slid off and `reverse` forced a reseat.
    # Inside AIM_DIRECT_DIST the push direction is therefore LATCHED when the segment
    # starts and held: one straight shot at the goal. It is re-latched (through a reseat,
    # so the pad starts centred again) when the pad has crabbed off centre or the box has
    # run past the goal along the latched line.
    # CAGE-DRAG NEVER AIMS OFF FLUSH. The cage's clearance is the cube's support ALONG
    # the finger axis against the 80 mm open gap: 46 mm (17 mm per side) when the finger
    # axis is on a cube face normal, but 65 mm (7.5 mm per side) on the diagonal. Yawing
    # to the goal line inside the endgame therefore squeezes the caged cube against a pad
    # from the OUTSIDE and drives the finger joint shut -- MEASURED (n = 32): 4 of 32
    # episodes had their minimum aperture pushed to 0.018-0.054 against the task's 0.055
    # floor and were silently voided, two of them after reaching 0.012 and 0.028 of the
    # 0.03 goal tolerance. Cage-Drag keeps the flush face at every distance; it can
    # afford the quantised direction (0.03 tolerance, 200-step budget).
    # AIM_ALWAYS: with the FORK the pusher is wider than the box's face, so an off-flush
    # box touches ONE pad, is turned by that torque and is caught flush by the other --
    # it SELF-ALIGNS to the pusher. The 30 mm closed paddle could not do this (an
    # off-flush contact is a single corner point and the box pivots on it, which is the
    # v26 measurement "the median flush error stayed at 24 deg"), which is why the flush
    # face had to be tracked and the push direction was quantised to the box's four face
    # normals. With the fork the paddle can simply be held on the goal line at every
    # distance: no face quantisation, no flush/aimed switch, no orbit to change face.
    aimed = (self.AIM_ALWAYS or dist < self.AIM_DIRECT_DIST) and not self.CAGE
    if aimed:
      if dist >= self.AIM_DIRECT_DIST:
        self._aim_dir[i] = u  # re-aim continuously while far (no limit cycle out here)
      elif int(self._phase[i]) != P_PUSH or np.isnan(self._aim_dir[i, 0]):
        self._aim_dir[i] = u  # ... and latch one straight segment for the endgame
      n = -self._aim_dir[i]
      self._face[i] = -1
    if not aimed:
      faces = self._faces(yaw_obj)
      fi = self._pick_face(i, faces, u)
      self._face[i] = fi
      n = faces[fi][0]
    t = np.array([-n[1], n[0]])
    h_t = hx * abs(float(t @ e_x)) + hy * abs(float(t @ e_y))
    # Half width of the contact patch across the push direction, and the lateral offsets
    # at which the pad ACTUALLY has material (the fork has a hole in the middle).
    if self.FORK:
      q_f = max(float(q_abs[7]), float(q_abs[8]))
      pad_hw = q_f + self.FORK_PAD_EDGE
      pad_offs = (-pad_hw, -q_f, q_f, pad_hw)
    else:
      pad_hw = float(self.PAD_HALF_W)
      pad_offs = (-pad_hw, 0.0, pad_hw)

    def _entry(off: float) -> float:
      """Distance from the box centre, along +n, at which a ray offset ``off`` across
      the push direction first meets the box (+inf if it misses it)."""
      lo, hi = -1e9, 1e9
      for ax, h in ((e_x, hx), (e_y, hy)):
        b = float(n @ ax)
        a = float(t @ ax)
        if abs(b) < 1e-9:
          if abs(off * a) > h:
            return float("inf")
          continue
        l1 = (h - off * a) / b
        l2 = (-h - off * a) / b
        lo = max(lo, min(l1, l2))
        hi = min(hi, max(l1, l2))
      return hi if hi >= lo else float("inf")

    # THE CONTACT DISTANCE IS THE RAY-BOX ENTRY, NOT THE SUPPORT DISTANCE.
    # MEASURED (v19, n = 32): with the support distance (hx|n.ex| + hy|n.ey|) the
    # controller believed contact happened at the plane TANGENT TO A CORNER, but the
    # surface the pad actually meets is behind it -- for a 0.0365 x 0.0446 carton at 30
    # deg to the push direction the two differ by 1.2 cm. The pad therefore spent the
    # whole of its 1-2 cm lead just reaching the box, the commanded penetration went to
    # zero at the moment of contact, and the arm stopped pressing: |q_cmd - q| collapsed
    # to 0.005-0.03 rad and the box did not move on 83 % of the steps inside 3 cm of the
    # goal (against 15 % at 6-12 cm, where the box happened to be closer to flush).
    # That is the "friction-limited push stall" the whole graveyard is about: it was
    # never friction, it was a geometry error that removed the push force.
    h_cands = [_entry(o) for o in pad_offs]
    h_cands = [v for v in h_cands if np.isfinite(v)]
    h_n = min(h_cands) if h_cands else h_t
    # ... and the gap test uses the entry distance under the pad WHERE IT ACTUALLY IS.
    phi = float(np.arctan2(-n[0] * u[1] + n[1] * u[0], float(-n @ u)))
    # Box yaw rate (CCW positive), filtered: the derivative term of the steering law.
    if np.isnan(self._oyaw[i]):
      self._oyaw[i] = yaw_obj
    else:
      d_oy = _wrap(yaw_obj - self._oyaw[i])
      self._oyaw[i] = _wrap(self._oyaw[i] + self.STEER_YAW_ALPHA * d_oy)
      self._oyaw_d[i] = (1 - self.STEER_YAW_ALPHA) * self._oyaw_d[i] + (
        self.STEER_YAW_ALPHA * self.STEER_YAW_ALPHA * d_oy
      )
    # Box yaw rate (CCW positive), filtered: the derivative term of the steering law. A
    # pure proportional offset overshot alignment in the cl_v2 graveyard because the
    # offset commands the yaw RATE, not the yaw.
    if np.isnan(self._oyaw[i]):
      self._oyaw[i] = yaw_obj
    else:
      d_oy = _wrap(yaw_obj - self._oyaw[i])
      self._oyaw[i] = _wrap(self._oyaw[i] + self.STEER_YAW_ALPHA * d_oy)
      self._oyaw_d[i] = (1 - self.STEER_YAW_ALPHA) * self._oyaw_d[i] + (
        self.STEER_YAW_ALPHA * self.STEER_YAW_ALPHA * d_oy
      )
    # M = +F*e turns the box CCW for e > 0 (r x F with r = n h_n + t e, F = -F n), and
    # phi > 0 puts the goal CCW of the push direction, so the sign is direct.
    steer_cap = min(self.STEER_MAX, max(0.0, h_t - pad_hw - 0.004))
    steer = float(
      np.clip(
        self.STEER_GAIN * phi - self.STEER_DAMP * float(self._oyaw_d[i]),
        -steer_cap,
        steer_cap,
      )
    )
    if int(self._phase[i]) != P_PUSH or aimed:
      steer = 0.0  # seat on the face centre; steer only once the pad is loaded
    p_contact = c + n * h_n + t * steer

    # Hand yaw. The paddle normal (site x; the finger axis, site y, for the cage) starts
    # flush with the contact face and turns toward the goal line at YAW_RATE, never more
    # than YAW_LAG_MAX ahead of the box's face so the box follows the paddle.
    yaw_site_cur = float(np.arctan2(R_site[1, 0], R_site[0, 0]))
    cage_off = np.pi / 2 if self.CAGE else 0.0
    yaw_goal = float(np.arctan2(u[1], u[0])) + cage_off
    yaw_face = float(np.arctan2(-n[1], -n[0])) + cage_off
    if np.isnan(self._yaw_cmd[i]):
      # start from where the hand IS (the yaw-free approach leaves it wherever the
      # redundancy put it) and ramp to the face normal before contact
      self._yaw_cmd[i] = yaw_site_cur + _wrap_half(yaw_face - yaw_site_cur) * 0.0
    if dist > self.YAW_FREEZE_DIST or self._phase[i] < P_PUSH:
      # ALWAYS the face normal: the paddle tracks the box's face and the goal direction
      # is served by the lateral contact offset (see STEER_GAIN), not by yawing into an
      # edge contact.
      tgt = yaw_face
      rate = self.YAW_RATE_FREE if self._phase[i] < P_PUSH else self.YAW_RATE
      d_yaw = _wrap_half(tgt - self._yaw_cmd[i])
      d_yaw = float(np.clip(d_yaw, -rate, rate))
      lead_face = _wrap_half(self._yaw_cmd[i] + d_yaw - yaw_face)
      if abs(lead_face) > self.YAW_LAG_MAX:
        d_yaw -= lead_face - float(np.sign(lead_face)) * self.YAW_LAG_MAX
      self._yaw_cmd[i] += d_yaw
    yaw_cmd = float(self._yaw_cmd[i])
    target_rot = down_frame(yaw_cmd)
    yaw_free = False
    # Paddle normal in the world xy-plane (the push direction is along it). The closed
    # paddle (and the open finger PAIR) is symmetric about the site, so ``_wrap_half``
    # only pins the yaw command modulo pi and the +axis branch is meaningless: the push
    # direction must come from GEOMETRY — the branch that points from the site into the
    # contact face.
    # MEASURED (v11 trace, n = 32): taking the +axis branch unconditionally left 52 % of
    # all push steps with the paddle normal 168-173 deg away from the face normal; the
    # `reverse` test below then read those as "the goal is behind the hand" and the env
    # spent the whole episode in a HOLD / RESEAT_UP / DESCEND loop. 17 of 32 envs never
    # moved the box by even 1 mm (SR 0.000, and 65 % of push steps stalled at EVERY pad
    # friction, which is why the stall looked friction-independent).
    if self.CAGE:
      base_dir = np.array([np.sin(yaw_cmd), -np.cos(yaw_cmd)])  # site y (finger axis)
    else:
      base_dir = np.array([np.cos(yaw_cmd), np.sin(yaw_cmd)])  # site x (paddle normal)
    xhat = base_dir if float(base_dir @ (-n)) >= 0.0 else -base_dir

    # Paddle offset from the site along u (positive = paddle ahead of the site).
    if self.CAGE:
      # which finger trails (sits at -xhat)? site y = (sin yaw, -cos yaw)
      site_y = np.array([np.sin(yaw_cmd), -np.cos(yaw_cmd)])
      if float(site_y @ xhat) > 0:
        q_trail = float(q_abs[8])  # right finger (-y) trails
      else:
        q_trail = float(q_abs[7])  # left finger (+y) trails
      pad_ahead = -max(q_trail, 0.02)
    else:
      pad_ahead = PAD_FACE_X
    # Site xy that puts the paddle face on the contact point (minus standoff).
    seat = p_contact - xhat * (pad_ahead + self.STANDOFF)

    # Pad position relative to the face frame (for the reseat checks).
    pad_xy = p_site[:2] + xhat * pad_ahead
    s_n = float((pad_xy - c) @ n)  # > h_act: pad short of the box
    s_t = float((pad_xy - c) @ t)
    _ha = [_entry(s_t + o) for o in pad_offs]
    _ha = [v for v in _ha if np.isfinite(v)]
    h_act = min(_ha) if _ha else h_n  # entry distance under the pad where it now is
    # The paddle can only push the face it is BEHIND. If the pad has ended up past the
    # box centre along the face normal (the box drifted abeam, or ran past the goal and
    # the face flipped), a straight move to the seat would sweep the box the wrong way:
    # the hand must go AROUND (RESEAT_UP over the box), never push through. This is a
    # position test, not a heading test (see the xhat note above).
    reverse = s_n < -0.025

    z_site = float(p_site[2])
    # TRUE LOWEST PAD CORNER, not the site. The pads hang 1.19 cm below the site and are
    # 3.0 cm wide (8.0-11.0 cm across, fingers open), so a few degrees of wrist tilt puts
    # a corner several mm lower again -- MEASURED: `ee_ground_collision` (the whole link7
    # subtree against the terrain) fires at site heights of 0.013-0.019, well above the
    # 0.0119 the site-only geometry predicts. The guard is applied to the corner.
    hy = pad_hw if not self.CAGE else max(float(q_abs[7]), float(q_abs[8])) + 0.0152
    dz_low = 1e9
    for sx in (-PAD_FACE_X, PAD_FACE_X):
      for sy in (-hy, hy):
        dz_low = min(dz_low, float(R_site[2, 0] * sx + R_site[2, 1] * sy + R_site[2, 2] * PAD_BOTTOM_BELOW_SITE))
    pad_low = z_site + dz_low
    # Filtered height: the raw FK height carries ~1 cm of joint-noise, which must not
    # reach any THRESHOLD test (the command keeps the raw value: see the z servo below,
    # where the absolute target makes the noise cancel exactly).
    if np.isnan(self._z_f[i]):
      self._z_f[i] = z_site
    else:
      self._z_f[i] += self.Z_ALPHA * (z_site - self._z_f[i])
    z_f = float(self._z_f[i])
    z_ff = 0.0
    stop_tol = self.STOP_FRAC * self.SUCCESS_TOL
    ph = int(self._phase[i])
    steps = int(self._phase_steps[i])
    target_xy = seat.copy()
    z_des = self.RIDE_Z + (0.004 if int(self._phase[i]) in (P_PUSH, P_HOLD) else 0.0)
    hold = False
    cause = 0.0  # diagnostic: what triggered a reseat / hold this step
    lead_dbg = 0.0
    err_xy_dbg = 0.0

    if ph == P_APPROACH:
      # CAGE ENTRY IS TOP-DOWN, NOT FROM THE SIDE.
      # The cage seat works out to roughly the cube's own centre (the site sits
      # q_trail - h_n - STANDOFF ~ 8 mm ahead of it), so the pads straddle the cube if
      # the hand simply comes straight down on it: the open inner gap is 80 mm against a
      # <= 65 mm cube support, i.e. >= 7.5 mm of clearance per side even at 45 deg yaw.
      # MEASURED (n = 32): the old side-entry swung the pads 6.5 cm laterally at the
      # ride height and 69-94 % of episodes ended on `ee_ground_collision`, 19 of them
      # during that sweep or the descent into it. A centred vertical descent is the same
      # motion Lift-Cube uses at 1.000, and it is gated on alignment below.
      err_xy = np.linalg.norm(target_xy - p_site[:2])
      err_xy_dbg = float(err_xy)
      # Descend WHILE closing in. MEASURED (v28): APPROACH hit its 30-step cap in every
      # env (the arm needs ~30 steps just to fly from HOME to the seat) and DESCEND then
      # cost another 35 -- 82 of the 150 steps went on overhead against 60 of pushing.
      # The traverse only has to clear the 3 cm box, so the last 10 cm of the traverse
      # is flown at the ride height and DESCEND is left with the final settle.
      z_des = self.HOVER_Z if err_xy > self.DESCEND_START_XY else self.RIDE_Z
      cause = 1.0 if s_n <= h_n - 0.005 else 0.0
      if err_xy > self.YAW_FREE_DIST:
        yaw_free = True  # axis-only IK while far: the yaw joint no longer throttles the step
        self._yaw_cmd[i] = yaw_site_cur  # ramp starts from wherever the hand is
      # MEASURED (v16): APPROACH + DESCEND cost a median 53 of the 150 steps. The seat
      # is 4 mm BEHIND the contact face, so the last 3 cm of the traverse is safe to fly
      # while already descending -- as long as the pad is not over the box.
      if (
        err_xy < 0.030 and z_f < self.RIDE_Z + 0.012
      ) or steps >= self.APPROACH_CAP:
        # preset the height integrator with the gravity sag measured at the hover
        self._z_bias[i] = float(np.clip(z_des - z_f, self.Z_BIAS_MIN, self.Z_BIAS_MAX))
        self._set_phase(i, P_DESCEND)
    elif ph == P_DESCEND:
      err_xy = np.linalg.norm(target_xy - p_site[:2])
      if self.CAGE and err_xy > self.CAGE_ALIGN_TOL:
        # do not drop the open pads past the top of the cube until they are centred on
        # it: off centre they land ON the cube and knock it out of the cage
        z_des = max(z_des, self.CAGE_GATE_Z)
      # sag-aware exit: the ARM is at the ride height (not the command) and has
      # SETTLED (the integrator has learned the sag); MEASURED: exiting while still
      # descending put the pad on the floor in the first push steps (28 % of envs)
      dz_step = abs(z_f - self._z_prev[i]) if not np.isnan(self._z_prev[i]) else 1.0
      # MEASURED (v12): the old `or z_site < z_des + 0.001` bypass let DESCEND exit
      # while the arm was still falling at 5-7 mm/step, and PUSH then started with 1.5 cm
      # of downward momentum -- 59 % floor terminations, most of them 1-3 steps into
      # PUSH. Only leave DESCEND once the arm has SETTLED (or is already too low to gain
      # anything by waiting, or the unconditional cap fires).
      # The height reference is rate-limited and absolute now, so the arm no longer
      # arrives with momentum and the "has settled" test (which cost ~15 steps waiting
      # for the FILTERED height to go quiet) is redundant.
      if (
        (err_xy < 0.025 and z_f < z_des + 0.005)
        or z_f < z_des - 0.004
        or steps >= self.DESCEND_CAP
      ):
        self._set_phase(i, P_PUSH)
    elif ph == P_ENTER:
      target_xy = seat
      err_xy = np.linalg.norm(target_xy - p_site[:2])
      if err_xy < 0.010 or steps >= self.ENTER_CAP:
        self._set_phase(i, P_PUSH)  # (unused for the top-down cage entry)
    elif ph in (P_PUSH, P_HOLD):
      if ph == P_PUSH and dist < stop_tol:
        self._set_phase(i, P_HOLD)
        ph = P_HOLD
      elif ph == P_HOLD and dist > stop_tol + self.RESUME_MARGIN:
        self._set_phase(i, P_PUSH)
        ph = P_PUSH
      if ph == P_HOLD:
        hold = True
        target_xy = p_site[:2].copy()
      else:
        # pad over / inside the box (persistently, beyond noise + filter lag)
        self._over[i] = (
          self._over[i] + 1
          if (s_n < h_act - self.OVER_MARGIN and abs(s_t) < h_t)
          else 0
        )
        # IF THE BOX IS MOVING, THE PUSH IS WORKING -- never re-seat.
        # MEASURED (v34/v35): the reseat tests fired ~6 times per episode and the orbit
        # took a third of every episode, while the effective transport rate over the
        # whole episode stayed at 1.8 mm/step against the ~2.8 needed for the p90 spawn.
        # All three tests (crab, penetration, reversal) are geometry read through a
        # lagging estimate of a box that is being pushed; the one unambiguous signal
        # that the contact is good is that the goal error is actually falling.
        # `_o2g_v` is the per-step velocity of object_to_goal, i.e. the box's own speed.
        # MEASURED (W1-P3): `if moving: pass` fell through the WHOLE elif chain, so on
        # every step where the box was actually moving the push law below never ran and
        # `target_xy` stayed at the bare `seat` set above -- the pad was commanded to
        # just touch the face with NO lead, i.e. the arm was told to stop pressing at
        # exactly the moment the push was working. `lead_dbg` is 0 on the MEDIAN push
        # step of the v51 trace and < 40 mm on 74 % of them. That is the whole
        # "impulsive rather than continuous push": whack -> box moves -> `moving` -> the
        # command collapses to the seat -> the box coasts away and stops -> `moving`
        # clears -> the lead comes back -> whack. It is a control-flow bug, not physics.
        # `moving` must suppress the RE-SEAT tests only, and hand to the push law.
        # ... but the SEGMENT-SPENT test is not one of them: it fires when the latched
        # endgame line has done its job (the box ran past the goal along it), which is
        # most likely to be true precisely WHILE the box is moving. Gating it on
        # `moving` lets the endgame push the box straight through the goal and out.
        moving = float(np.linalg.norm(self._o2g_v[i][:2])) > self.RESEAT_HOLDOFF_SPEED
        # Of the two segment-spent conditions only the second is independent of whether
        # the push is working: "the box has run PAST the goal along the latched line"
        # is a fact about the line, and it is most likely to become true exactly while
        # the box is moving. The CRAB condition is a "the contact has degraded" test
        # like the other three and stays gated on `moving` -- MEASURED (W1-P3, v55-v58,
        # n = 32): ungating it too fires the orbit continuously through the endgame and
        # took RESEAT_SIDE from 17 to 41-44 steps an episode.
        if aimed and float(to_goal @ self._aim_dir[i]) < self.AIM_PASSED:
          # The straight segment is spent. If the pad is still squarely behind the box
          # a fresh line can simply be laid from where it stands -- the seat target moves
          # with the new direction and the pad walks across the face under load. Only a
          # pad that has crabbed off, or ended up in front of the box, needs the orbit
          # (which costs 5-14 steps and, inside the last few centimetres, more than the
          # travel it is protecting).
          # MEASURED at n = 128: re-laying the line in place when the pad is still well
          # placed (rather than always orbiting) reads better at n = 32 on the pinned
          # seed and is worth 0.344 -> 0.297 over 128. Always orbit.
          cause = 6.0
          self._aim_dir[i] = np.nan
          self._set_phase(i, P_RESEAT_SIDE)
          self._timer[i] = self.RESEAT_SIDE_STEPS
        elif not moving and aimed and abs(s_t) > self.AIM_ST_MAX:
          cause = 6.0
          self._aim_dir[i] = np.nan
          self._set_phase(i, P_RESEAT_SIDE)
          self._timer[i] = self.RESEAT_SIDE_STEPS
        # ... or persistently inside the box -> lift and re-descend behind the face
        elif not moving and self._over[i] >= self.OVER_STEPS:
          cause = 2.0
          self._over[i] = 0
          self._set_phase(i, P_RESEAT_SIDE)
          self._timer[i] = self.RESEAT_SIDE_STEPS
        # pad slipped off the face laterally (or the contact face changed): lift over
        # the box and come back around behind the new seat
        elif not moving and abs(s_t) > h_t + self.LATERAL_MARGIN:
          cause = 3.0
          self._set_phase(i, P_RESEAT_SIDE)
          self._timer[i] = self.RESEAT_SIDE_STEPS
        elif not moving and reverse:
          cause = 4.0
          if self._rev_ok(i):
            self._set_phase(i, P_RESEAT_SIDE)
            self._timer[i] = self.RESEAT_SIDE_STEPS
          else:
            target_xy = p_site[:2].copy()  # hold until the reversal is confirmed
        else:
          self._rev[i] = 0
          # NO TAPER NEAR THE GOAL. MEASURED (v19/v25): the box needs ~3 cm of commanded
          # penetration before it slides at all; with the lead tapered to 0.5 * dist it
          # fell to 1.2-1.5 cm inside 3 cm of the goal and the box then moved on only
          # 17-24 % of steps (median 0.0 mm/step) -- every near miss in the trace sat
          # frozen at a true goal error of 0.023-0.029 for the last 20-40 steps. Success
          # LATCHES and the box only travels 4-8 mm/step, so it is sampled inside the
          # 2 cm ball for several steps even if the push then carries it through.
          lead = float(np.clip(self.LEAD_GAIN * dist, self.LEAD_MIN, self.LEAD_MAX))
          lead *= min(1.0, (steps + 1) / self.LEAD_RAMP_STEPS)
          lead = max(lead, self.LEAD_MIN)
          # Do not lean on the box until the pads are at the ride height: a lead applied
          # mid-descent is what carried them into the floor (v21).
          lead *= float(np.clip((self.RIDE_Z + 0.016 - z_f) / 0.016, 0.0, 1.0))
          lead_dbg = float(lead)
          z_ff = self.Z_FF_PER_LEAD * lead
          # VELOCITY FEED-FORWARD. The contact is intermittent: MEASURED, the box moves
          # on only ~50 % of push steps but at 6-11 mm/step when it does, and the best
          # band is a SMALL POSITIVE gap (pad just short of the surface, 77 % moving) --
          # i.e. the box shoots away under the impulse, the pad spends 2-3 steps catching
          # up, and that duty cycle halves the mean rate. `-_o2g_v` is the box's own
          # velocity (the goal is static), so leading the seat by a few steps of it keeps
          # the pad with the box instead of chasing it.
          v_box = -self._o2g_v[i][:2]
          target_xy = seat + xhat * (self.STANDOFF + lead) + v_box * self.VEL_FF_STEPS
          if self.GLUE_PUSH:
            # See GLUE_PUSH above: a sustained offset from the pad's CURRENT position,
            # forward along the pad normal, with the lateral term rate-limited so the
            # saturated arm spends its speed on the box and not on crabbing across it.
            adv = self.GLUE_ADVANCE * min(1.0, (steps + 1) / self.LEAD_RAMP_STEPS)
            adv *= float(np.clip((self.RIDE_Z + 0.016 - z_f) / 0.016, 0.0, 1.0))
            lat_cap = (
              self.GLUE_LAT_FREE
              if (s_n - h_act) > self.GLUE_CONTACT_GAP
              else self.GLUE_LAT_CONTACT
            )
            lat = float(np.clip(steer - s_t, -lat_cap, lat_cap))
            target_xy = p_site[:2] + xhat * adv + t * lat
            lead_dbg = float(adv)
            z_ff = self.Z_FF_PER_LEAD * adv
    elif ph == P_RESEAT_UP:
      z_des = self.RESEAT_Z
      # Climb clear of the box BEFORE translating: the reseat target can be 10-15 cm
      # away (the pad has to be brought round the box) and flying that at ride height
      # is what put the pads on the floor in v20.
      if z_f < self.RESEAT_Z - 0.010:
        target_xy = p_site[:2].copy()
        cause = 5.0
      else:
        target_xy = seat - xhat * 0.02
      self._timer[i] -= 1
      if (self._timer[i] <= 0 and z_f > self.RESEAT_Z - 0.015) or (
        self._timer[i] <= self.RESEAT_UP_STEPS - 4
        and np.linalg.norm(target_xy - p_site[:2]) < 0.015
        and z_f > self.RESEAT_Z - 0.015
      ) or steps >= 2 * self.RESEAT_UP_STEPS:
        self._set_phase(i, P_DESCEND)
    elif ph == P_RESEAT_SIDE:
      # ORBIT THE BOX AT RIDE HEIGHT -- do not lift.
      # MEASURED (v30-v33): re-seating by RESEAT_UP (lift 7-10 steps) + DESCEND (up to
      # 10 more, and the descent is torque-limited to ~11 mm/step) cost 33-35 % of every
      # episode, against 28-35 % actually spent pushing; the effective transport rate
      # over the whole episode was ~1.7 mm/step and the median episode needs 24 cm. The
      # pad never has to cross the box: swinging it round on a circle of radius
      # ORBIT_R (> the box's 5.8 cm half-diagonal) keeps it clear the whole way, stays
      # at the ride height so nothing has to be re-descended, and hands straight back to
      # PUSH.
      rel = pad_xy - c
      rn = float(np.linalg.norm(rel))
      b_now = float(np.arctan2(rel[1], rel[0])) if rn > 1e-6 else 0.0
      seat_rel = p_contact - c  # the contact point is on the far side of the box from u
      b_tgt = float(np.arctan2(seat_rel[1], seat_rel[0]))
      d_b = _wrap(b_tgt - b_now)
      b_cmd = b_now + float(np.clip(d_b, -self.ORBIT_RATE, self.ORBIT_RATE))
      # Swing at whatever radius the pad is ALREADY at (just clear of the surface),
      # never pulling it out to a fixed radius and pushing it back in: MEASURED (v34)
      # a fixed 8 cm orbit turned the reseat into 35 % of the episode by itself.
      r_orb = float(np.clip(rn, h_act + 0.014, self.ORBIT_R))
      target_xy = c + r_orb * np.array([np.cos(b_cmd), np.sin(b_cmd)]) - xhat * pad_ahead
      self._timer[i] -= 1
      if abs(d_b) < self.ORBIT_TOL or self._timer[i] <= 0:
        self._set_phase(i, P_PUSH)

    # Height servo (absolute, base frame) with an anti-windup integrator for the PD sag.
    fast = ph in (P_APPROACH, P_DESCEND) and z_f > self.RIDE_Z + 0.010
    self.max_dq = self.APPROACH_MAX_DQ if fast else type(self).max_dq
    self.cmd_lead_max = self.APPROACH_LEAD if fast else type(self).cmd_lead_max
    # HEIGHT: an ABSOLUTE, RATE-LIMITED REFERENCE, never re-anchored to the arm.
    # MEASURED (v15 trace, env 2): the previous form set the commanded drop RELATIVE to
    # the measured height (`z_err = -4 mm` per step below DESCENT_SLOW_Z). Because the
    # base class rebuilds the IK target as FK(q_obs) + pos_err, that makes the command
    # chase the arm's own fall: the target is always 4 mm under wherever the arm has got
    # to, so a fall is never opposed and the site dropped 8-11 mm/step straight through
    # the ride height into the floor. Here `_z_ref` is a reference height that moves at a
    # bounded rate toward `z_des` on its own clock; the IK target then works out to
    # `_z_ref + bias`, an absolute height that a falling arm is pulled back up to. The
    # reference is still clamped to a band around the arm so it cannot wind away from a
    # blocked hand, and the sag integrator now runs against the moving reference, so the
    # gravity offset is learned DURING the descent instead of after it.
    if np.isnan(self._z_ref[i]):
      self._z_ref[i] = z_f
    # The rate belongs to the HEIGHT, not to the phase. MEASURED (Cage-Drag, n = 32):
    # since APPROACH now drops to the ride height itself over the last 10 cm of the
    # traverse, keeping APPROACH_DESCENT_RATE all the way down let the reference fall
    # 5 cm/step through the last centimetre; the site went 0.029 -> 0.018 in one step and
    # the pads reached the floor. 94 % of Cage-Drag episodes terminated that way, 23 of
    # them still in APPROACH or DESCEND.
    if self._z_ref[i] < self.DESCENT_SLOW_Z:
      rate_dn = self.DESCENT_RATE_NEAR
    elif ph == P_APPROACH:
      rate_dn = self.APPROACH_DESCENT_RATE
    else:
      rate_dn = self.DESCENT_RATE_FAR
    d_ref = float(np.clip(z_des - self._z_ref[i], -rate_dn, self.RISE_RATE))
    # NEVER LET THE PAD RUB DOWN THE FACE WHILE PUSHING.
    # MEASURED (v36, 1111 in-contact push steps): the box's motion is governed by the
    # pad's VERTICAL slip direction, not by pad friction (which is flat across the
    # randomised 0.3-1.5 range). With the site descending faster than 1.5 mm/step the
    # box moved on 35 % of steps at a median 0.11 mm/step; with the site RISING faster
    # than 1.5 mm/step, 74 % of steps at 4.67 mm/step -- a 40x difference. The mechanism
    # is a wedge: a pad sliding down a vertical face drags the box down with it, which
    # adds mu_pad * N to the floor's normal load, and for mu_floor * mu_pad >= 1 no
    # horizontal force can slide the box at all. So during PUSH the height reference is
    # a RATCHET (it may rise, never fall) and creeps gently upward while at or below the
    # ride height, which keeps the pad's slip unloading the box instead of pinning it.
    if ph in (P_PUSH, P_HOLD):
      # SAW-TOOTH, not a hold. The measured 40x difference is between a RISING pad and a
      # falling one, and a rise cannot be sustained (8 mm of climb lifts the pad's bottom
      # edge clear of the 3 cm carton). So the reference climbs slowly through the
      # contact band and then resets in a single step: the pad spends ~5 steps in 6
      # unloading the box and one re-seating down it.
      if self.PUSH_CYCLE and self._z_ref[i] >= self.RIDE_Z + self.PUSH_CYCLE_AMP:
        # Reset the cycle by a BOUNDED step, not straight back to the ride height: a
        # 7 mm downward jump in the reference is followed by an arm that overshoots it,
        # and MEASURED (v45, n = 128) 14 % of episodes were lost to
        # `ee_ground_collision` at site heights of 0.0147-0.0171, almost all of them on
        # this reset inside PUSH.
        self._z_ref[i] = max(self.RIDE_Z, z_f - self.PUSH_CYCLE_RESET)
        d_ref = 0.0
      else:
        d_ref = max(d_ref, self.PUSH_CREEP)
    self._z_ref[i] = float(
      np.clip(self._z_ref[i] + d_ref, z_f - self.REF_LAG_MAX, z_f + self.REF_LEAD_MAX)
    )
    self._z_ref[i] = max(self._z_ref[i], self.FLOOR_MIN_Z)
    if ph in (P_PUSH, P_HOLD):
      # The ratchet needs a CEILING: without one the creep walked the reference to
      # 0.040-0.048 (v37), which lifts the pad's bottom edge above the carton's 0.030
      # top and the contact is simply lost (only 25 % of push steps moved the box).
      self._z_ref[i] = min(self._z_ref[i], self.RIDE_Z + self.PUSH_CYCLE_AMP)
    z_ref = float(self._z_ref[i])
    e_ref = z_ref - z_f  # tracking error of the ARM against the reference
    if abs(e_ref) < self.Z_INT_BAND and z_ref < 0.06:
      self._z_bias[i] = float(
        np.clip(self._z_bias[i] + self.Z_KI * e_ref, self.Z_BIAS_MIN, self.Z_BIAS_MAX)
      )
    z_err = (z_ref - z_site) + self._z_bias[i] + z_ff
    if pad_low < self.PAD_CLEAR_MIN:  # emergency: a pad corner at the floor, climb
      z_err = max(z_err, (self.PAD_CLEAR_MIN - pad_low) + 0.006)

    pos_err = np.array([target_xy[0] - p_site[0], target_xy[1] - p_site[1], z_err])
    # ONE WAYPOINT AT A TIME NEAR THE FLOOR. `pos_err` is the whole REMAINING error and
    # the base class solves IK for all of it, then scales the joint stride uniformly by
    # `max_dq`: for a large delta the site follows a chord of the joint-space path
    # instead of the commanded straight line, and the chord dips. MEASURED (v22): a
    # 6-7 cm lateral waypoint taken at the ride height (a re-descent after a reseat, or
    # a long push lead) pulled the site down 7 mm/step and put the pads on the floor --
    # 21 of 32 envs. Capping the waypoint costs no speed, because the arm cannot cover
    # more than ~2.5 cm in a control step anyway, and it keeps the IK target next to the
    # current pose where the linearisation is good.
    # In PUSH the waypoint IS the push force (the command is driven into the box and the
    # contact holds the arm back), so the cap there is loose; everywhere else it is the
    # floor guard described above.
    step_cap = self.PUSH_STEP_MAX if ph in (P_PUSH, P_HOLD) else self.XY_STEP_MAX
    if z_f < self.XY_STEP_LOW_Z:
      nxy = float(np.linalg.norm(pos_err[:2]))
      if nxy > step_cap:
        pos_err[:2] *= step_cap / nxy
    # HEIGHT HAS PRIORITY OVER THE PUSH, but only as an EMERGENCY. `max_dq` scales the
    # whole joint-space stride uniformly, so a 3-4 cm xy lead can consume the authority
    # that should be holding the pads up (v12: the arm sank 5 mm/step into the floor).
    # MEASURED (v14 trace) the first version of this gate keyed on the RAW FK height with
    # thresholds inside the normal operating band (0.028-0.034 against a 0.032 ride
    # height and ~1 cm of FK noise): it fired on roughly half of all push steps, zeroing
    # or 0.3-scaling the push command, and the site then crawled 2 mm/step with the pad
    # in contact -- the dominant stall. It now keys on the FILTERED height and only well
    # below the ride band, where the pads really are close to the floor.
    if not hold:
      if pad_low < self.PAD_CLEAR_FREEZE:
        pos_err[:2] = 0.0
      elif pad_low < self.PAD_CLEAR_THROTTLE:
        pos_err[:2] *= 0.4
    if hold:
      pos_err[:2] = 0.0
    self._z_prev[i] = z_f
    self.dbg[i] = (
      s_n - h_act,
      s_t,
      dist,
      err_xy_dbg,
      float(np.linalg.norm(pos_err[:2])),
      z_site,
      cause,
      lead_dbg,
    )
    grip = self.GRIPPER_ACTION
    if self.FORK:
      # target = 0.04 + 0.04 * a  ->  a = (q_target - 0.04) / 0.04
      grip = (self.FORK_APERTURE - 0.04) / 0.04
    if yaw_free:
      return pos_err, _DOWN_AXIS, grip
    return pos_err, target_rot, grip


class PushCuboidClassicalPolicy(PlanarPushPolicy):
  """Push the 97 g gelatin box to a 2 cm goal on the floor with the closed paddle."""

  OBJ_HALF = np.array([0.0365, 0.0446, 0.0150])
  SUCCESS_TOL = 0.02
  CAGE = False
  GRIPPER_ACTION = -1.0
