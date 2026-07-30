"""Scripted OpenDoor teacher policy — SIDE PINCH on the handle bar, arc pull.

MEASURED GEOMETRY (driving the hinge through its range and reading
object_site in the robot frame, env-origin removed):

     hinge   handle (x, y) rel. base   planar |p|   3D from shoulder
       0     (+0.447, +0.010)            0.447          0.669
      20     (+0.262, -0.036)            0.264          0.563
      40     (+0.103, -0.144)            0.177          0.528
      60     (-0.009, -0.299)            0.299          0.580
      90     (-0.063, -0.579)            0.583          0.766

Circle fit: hinge centre (0.4873, -0.5395), radius 0.5514, handle polar angle
94.16 deg at closed and INCREASING 1:1 with the hinge angle. So the panel
rotates about +z and _OPEN_SIGN = +1.

TWO THINGS THIS MEASUREMENT SETTLES
-----------------------------------
1. Opening is a PULL, not a push. The handle moves in -x (toward the robot)
   for the whole sweep. A face push from the robot's side drives the panel
   into its range="0 90" LOWER stop; measured directly — a near-hinge face
   push reached a mean of 0.1 deg over 32 episodes. And the robot cannot get
   behind the panel to push the other way: the static ``door_body`` barrier
   geom (size 0.01 x 0.5 x 0.8) walls off the far side across the full door.
   Pulling the handle is the only mechanically available option.

2. Reach is NOT the binding constraint. The handle stays 0.53-0.77 m from the
   shoulder, inside the Franka's ~0.85 m envelope, for the entire 0-90 deg
   sweep. What does get hard is the middle of the sweep, where the handle
   passes within 0.18 m of the base axis (shoulder singularity), and the last
   ~20 deg, where it sits behind the base and the arm must reach backwards.

WHY THE PINCH IS ORIENTED THE WAY IT IS
---------------------------------------
The previous teacher "caged" the bar with the finger-closing axis along y and
the approach axis along x, then pulled along -x — i.e. it pulled ALONG the
approach axis, straight out through the open finger gap. That is pure cam-out
and it measured 0.000: alignment at closure was fine (~0.02-0.03, inside the
2cm bar) but the approach-axis error grew to 0.13 within ~8 steps.

Here the wrist is rotated 90 deg about the vertical: the fingers approach
along the door's TANGENT-PERPENDICULAR and close along the PULL direction, so
the bar is pinched between two pad faces that are NORMAL to the pull. The load
is then carried face-on by the pads instead of trying to squeeze the bar
sideways out of the gap, and the pull direction is the one direction the pinch
CAN resist. Fingers command a real pinch (-1.0) on a 2cm bar, not a loose cage.

Success needs |90 deg - theta| < 0.1 rad, i.e. theta >= 84.3 deg -- essentially
the full stop, with no partial credit.

MEASURED RESULT: STILL 0.000. Two independent 32-episode runs of this exact
code gave 0.031 and 0.000; the lower figure is the honest one, so this is NOT
an improvement over the old teacher's 0.000 -- it is a different failure with
a better-understood cause.

What DID change is the underlying mechanics, and that is the useful part:
mean peak door angle rose from 2.5 deg to 12.0 deg, and individual envs reach
80-90 deg (a full open) where the old cage-grasp teacher never exceeded ~7.
So the side pinch genuinely retains the bar sometimes -- the cam-out that
defeated the previous strategy is fixed. What is not fixed is RELIABILITY:
26 of 32 envs never pass 10 deg.

The binding constraint is throughput against the 150-step budget. A holding
pull advances only ~1-2 deg per 25 steps, and the initial approach alone
costs ~40 steps (measured), so 84.3 deg is reachable only when the very first
seat holds for essentially the whole episode. Every attempt to buy speed made
it worse, each measured: PULL_DTHETA 22->40 deg (mean 6.7 deg), approach
max_dq 0.30->0.40 with max_pos_err 0.12->0.25 (mean 1.4 deg), drag max_dq
0.26->0.34 with cmd_lead 0.35->0.55 (neutral), and a stall watchdog (mean
2.4 deg). The pinch is retention-limited: anything that hurries it loses the
bar sooner. Note also the run-to-run variance is large enough that 8-env
samples are useless here -- two identical configs sampled 0.125 and 0.250 --
so only 32-episode runs should be trusted.

CONCLUSION: this task needs a learned teacher, or a task-side change that is
out of scope. Success requires 84.3 of 90 deg with no partial credit, while
the handle sweeps 1.1m through the shoulder singularity and ends up behind
the robot base. A scripted single-grasp pull is the wrong shape of solution;
what would plausibly work is a ratcheting multi-regrasp pull (like the one
that rescued Rotate-Valve here), but the door's 150-step budget -- a third of
the valve's 400 -- does not fit more than one engagement.

Only relative observations are used (gto = obs[40:43] handle - gripper,
o2g = obs[43:46] goal - handle); per-env scene-origin offsets cancel.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Handle radius vector from the hinge at theta = 0 (robot frame), MEASURED.
_R0 = np.array([-0.040, 0.551, 0.0])
_R0_LEN = float(np.linalg.norm(_R0))  # 0.5514
_OPEN_SIGN = 1.0

GRIPPER_OPEN = 0.0
GRIPPER_PINCH = -1.0

STANDOFF = 0.11  # pre-grasp standoff along the approach axis
# The bar is 2cm square and 16cm tall, protruding 4cm toward the robot. Aim
# the pinch at the middle of that protrusion so both pads land on the bar
# rather than on the panel behind it.
BAR_OUT = 0.02  # how far out along the protrusion to centre the pinch
ALIGN_TOL = 0.05
SEAT_TOL = 0.028  # bar is 2cm; the pads must straddle it before closing
SEAT_SETTLE = 2
CLOSE_STEPS = 5  # let the fingers actually close before loading the grasp
EMA_ALPHA = 0.4
INTEG_GAIN = 0.18
PULL_DTHETA = np.radians(22.0)  # arc-waypoint advance per control step
SLIP_DIST = 0.10  # pad this far off the bar => the pinch is gone, re-approach
MAX_WAYPOINT = 0.22
# Timeouts, in steps. These are a LAST RESORT, not a schedule: cutting them to
# 22/18 to save budget measured WORSE (mean 5 deg vs 28 deg) because the hand
# then pinches on empty air before it has reached the bar. Seating correctly
# matters more than seating early.
ALIGN_TIMEOUT = 35
SEAT_TIMEOUT = 30
# Stall watchdog: steps of drag allowed without STALL_EPS of angular progress
# before the grasp is abandoned and re-approached. Borrowed from what fixed
# Rotate-Valve, where jammed engagements deadlocked for 300+ steps because the
# progress-keyed handoff gate could never be reached from a standstill.
STALL_STEPS = 25
STALL_EPS = np.radians(1.5)


def _rot_z(theta: float) -> np.ndarray:
  c, s = np.cos(theta), np.sin(theta)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class OpenDoorClassicalPolicy(ClassicalPolicyBase):
  """Pinch the handle bar across the pull axis and drag it round the arc."""

  max_dq = 0.15
  orientation_weight = 0.25

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      # True once we have pinched at least once: a later seat is a RE-seat and
      # keeps the fingers closed rather than paying for a full re-approach.
      self._reseat = np.zeros(self.num_envs, dtype=bool)
      # Stall watchdog state (see the drag phase).
      self._stall_theta = np.zeros(self.num_envs)
      self._stall_t = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._reseat[env_ids] = False
      self._stall_theta[env_ids] = 0.0
      self._stall_t[env_ids] = 0

  def _grasp_frame(self, theta: float):
    """Return (approach_dir, grasp_rot, bar_offset) at door angle ``theta``.

    ``approach_dir`` is the unit vector the hand travels ALONG to reach the
    bar (EE z-axis). ``grasp_rot`` puts the finger-closing axis (EE y) along
    the local pull tangent, so the pads sandwich the bar normal to the pull.
    """
    rot = _rot_z(_OPEN_SIGN * theta)
    r_vec = rot @ _R0
    u = r_vec / np.linalg.norm(r_vec)  # hinge -> handle
    # Tangent of INCREASING door angle (the pull direction): z_hat x u.
    tangent = np.array([-u[1], u[0], 0.0])
    # Fingers CLOSE along the pull tangent -> the pads are normal to the pull.
    y_ee = tangent
    # Approach along the panel's outward normal, from the robot's side. The
    # bar protrudes toward the robot, so the hand comes in along -u x z ...
    # concretely the outward normal on the robot side is -tangent rotated:
    # the panel face normal is +/-tangent's perpendicular in-plane, i.e. u.
    # Approach ALONG -u (from the free edge inboard) keeps the wrist clear of
    # the panel and lets the pads close around the bar's 2cm thickness.
    z_ee = -u
    x_ee = np.cross(y_ee, z_ee)
    x_ee = x_ee / np.linalg.norm(x_ee)
    grasp_rot = np.column_stack([x_ee, y_ee, z_ee])
    # Centre the pinch on the protruding part of the bar (toward the robot,
    # i.e. along the panel normal on the robot's side = -tangent side).
    bar_offset = -tangent * BAR_OUT
    return z_ee, grasp_rot, bar_offset

  def _target_error(self, i: int, obs_i: np.ndarray):
    ph = int(self._phase[i])
    self.max_dq = 0.30 if ph == 0 else (0.14 if ph < 3 else 0.26)
    # Sustained command lead while dragging: the hinge resists and commanding
    # from the lagging actual joints caps the available servo force.
    self.cmd_lead_max = 0.35 if ph >= 3 else 0.0
    self.max_pos_err = 0.12 if ph == 0 else (0.08 if ph < 3 else 0.18)

    gto_raw = obs_i[40:43]
    o2g = obs_i[43:46]
    if not self._ema_init[i]:
      self._gto_ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    gto = self._gto_ema[i]

    # Door angle from the chord to the goal: the goal marker is the handle at
    # the 90 deg target, so |o2g| = 2 r sin((pi/2 - theta)/2). object_quat
    # reports the STATIC door base, not the swinging panel, so it is useless
    # for recovering theta.
    chord = np.clip(np.linalg.norm(o2g) / (2.0 * _R0_LEN), 0.0, 1.0)
    theta = np.pi / 2.0 - 2.0 * np.arcsin(chord)

    approach, grasp_rot, bar_off = self._grasp_frame(theta)
    seat = gto + bar_off  # site -> desired pinch centre on the bar

    gripper_a = GRIPPER_OPEN
    target_rot = grasp_rot

    if ph == 0:
      # Stand off along the approach axis with the fingers open.
      pos_err = seat - STANDOFF * approach
      if np.linalg.norm(pos_err) > 0.18:
        target_rot = approach  # axis-only while far: a full 3x3 makes IK crawl
      if np.linalg.norm(pos_err) < ALIGN_TOL or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif ph == 1:
      # Slide in until the open pads straddle the bar. Integral action nulls
      # the DLS steady-state bias, which is 2-3.6cm laterally and would
      # otherwise park the hand just short of the bar forever.
      raw = seat
      # On a RE-seat the pads stay pinched: the bar is only a few cm away and
      # a closed pad shoved back against it re-loads the grasp immediately,
      # whereas reopening means paying the 5-step close again.
      if self._reseat[i]:
        gripper_a = GRIPPER_PINCH
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.05, 0.05)
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < SEAT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= SEAT_SETTLE or (
        self._phase_steps[i] > SEAT_TIMEOUT and np.linalg.norm(raw) < 0.07
      ):
        # Already pinched -> straight back to the drag; otherwise close first.
        self._phase[i] = 3 if self._reseat[i] else 2
        self._phase_steps[i] = 0
    elif ph == 2:
      # CLOSE on the bar and hold still while the fingers travel. Closing
      # early (while still moving in) was what defeated the previous attempt.
      gripper_a = GRIPPER_PINCH
      pos_err = seat + self._integ[i]
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
        self._reseat[i] = True
    else:
      # Drag along the hinge arc: aim at where the bar will be a further
      # PULL_DTHETA round, keeping the pinch closed and the pads normal to
      # the pull.
      gripper_a = GRIPPER_PINCH
      # Slip check on RAW obs (the EMA lags during a fast drag). This fires on
      # nearly every drag frame, which LOOKS like a bug -- the waypoint is an
      # arc stride ahead of the bar, so the residual reflects the commanded
      # lead as much as real slippage. It is nevertheless load-bearing:
      # suppressing it for the first 6 drag steps measured WORSE (6 deg mean
      # vs 34). The bounce back to phase 0 is not a failure mode, it IS the
      # working cycle -- each re-approach re-seats the pinch and ratchets the
      # panel a few more degrees, which is how the successful envs get round.
      # NOTE: a stall watchdog here (the trick that rescued Rotate-Valve, where
      # jammed engagements deadlocked for 300+ steps) measured WORSE on the
      # door: 2.4 deg mean vs 12.0. The difference is that the valve either
      # turns freely or is hard-jammed, whereas the door's drag is genuinely
      # slow -- a working pull advances only ~1-2 deg per 25 steps, which is
      # indistinguishable from a stall, so the watchdog kept aborting grasps
      # that were in fact working. Deliberately absent.
      if np.linalg.norm(gto_raw + bar_off) > SLIP_DIST:
        # Lost the bar. Re-seat via phase 1 with the fingers STILL PINCHED and
        # the integral preserved, NOT via phase 0 with the hand reopened. A
        # full re-approach costs ~40 steps of a 150-step budget (measured: a
        # failing env spent 40 steps reaching the bar and 30 more seating), so
        # bouncing to phase 0 on every slip guarantees the episode expires
        # around 2-3 deg. Re-closing on the bar from a few cm away is cheap.
        self._phase[i] = 1
        self._settle[i] = 0
        return seat + self._integ[i], grasp_rot, GRIPPER_PINCH
      rot_now = _rot_z(_OPEN_SIGN * theta) @ _R0
      rot_next = _rot_z(_OPEN_SIGN * (theta + PULL_DTHETA)) @ _R0
      step = rot_next - rot_now
      step[2] = 0.0  # the arc is horizontal
      # NOTE: over-weighting the z residual here (the trick open_drawer needs)
      # measured WORSE, 6 deg mean vs 28 deg. The drawer hook hangs the pads
      # BELOW the site so height is the retention axis; this pinch sandwiches
      # a 16cm-tall bar between vertical pads, where a few cm of height error
      # is harmless and a stiff z term just fights the arc. Left un-weighted.
      # Ramping the arc term in over the first few drag frames also measured
      # worse (8 deg mean vs 28). The pinch either holds from the first frame
      # or it never holds; easing in just spends budget.
      pos_err = seat + step + self._integ[i]
      _, target_rot, _ = self._grasp_frame(theta + PULL_DTHETA)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_rot, gripper_a
