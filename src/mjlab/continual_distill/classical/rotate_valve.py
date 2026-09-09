"""Scripted RotateValve teacher — spoke PINCH walked round the arc, with HANDOFF.

MEASURED (2026-09-09, n=128, HEAD 1127d12): 0.523 baseline, 0.484 after adding
SEAT_HARD_TIMEOUT (see below) -- not a clear improvement, within this task's
own documented run-to-run noise. Earlier readings: 0.469 and 0.344 on two
independent 32-episode runs, up from 0.031 for the previous pad-push teacher.
The spread is itself a property of the task, which is very sensitive to the
spawn pose (x in [0.50, 0.70], y in [-0.1, 0.1]). Mean peak swept angle
218-255 deg against the ~258.5 deg actually needed (270 minus the 11.5 deg
success threshold) in the most recent instrumented runs.

SECOND MECHANISM (2026-09-09): phase 1 (DROP/seat)'s timeout is NOT actually
unconditional -- unlike ALIGN_TIMEOUT (phase 0) and STALL_STEPS (phase 2,
below), SEAT_TIMEOUT only fires when ALSO norm(raw) < 0.10, so an engagement
whose DLS lateral bias never drops under 10cm can sit in DROP indefinitely.
Instrumented: DROP consumes ~38-40% of the 400-step budget on average, MORE
than the productive ARC_FOLLOW phase (~31-34%). Added SEAT_HARD_TIMEOUT, an
unconditional cap mirroring ALIGN_TIMEOUT's pattern (see its definition
below for the full writeup and why falling through with an imprecise seat is
safe). Did not clearly move the aggregate SR -- kept anyway as a genuine
correctness fix, not a tuned constant. See docs/cl25/phase_1/LOGS.md
(2026-09-09, W2-b) for the full instrumented trail.

WHAT CHANGED, AND WHY THE PAD PUSH FAILED
-----------------------------------------
The previous version kept the fingers CLOSED and used the pad as a face-normal
pusher. It seated fine but could not RETAIN: traced live, the pad ends up
parked against the HUB with the valve jittering +-5 deg while the EE does not
move at all (one env sat at (0.467, -0.044, 0.178) from t=200 to t=400). The
spoke is a 2.4cm bar and a face push has nothing resisting the pad sliding
along it, so under the 0.08 hinge damping the contact walks inboard until the
moment arm vanishes. The LOST_TOL re-seat never fires, because the pad IS
still on the arc -- just at the useless end of it.

Now the fingers actually PINCH the spoke at PINCH_FRAC of its length,
straddling it along the direction of travel, so the driving load is carried
face-on between two pads and the spoke cannot walk out circumferentially.
Friction is only asked to resist RADIAL sliding, so the domain-randomised
fingertip friction (as low as 0.3) is not on the critical path.

THE OTHER HALF OF THE FIX: THE STALL WATCHDOG
---------------------------------------------
The handoff gate keys on SWEPT ANGLE, so an engagement that jams makes no
progress and therefore can never reach the gate -- it deadlocks for the rest
of the episode. That was the single biggest loss: envs frozen at a perfectly
constant angle (e.g. exactly 85.81 deg) for 300+ steps. STALL_STEPS forces a
handoff on lack of progress, which removed every frozen env. Both parts were
needed; the pinch alone measured only ~0.125.


Geometry (valve.xml, verified live): hinge about world +x (the mount has no yaw
randomisation), range [0, 360 deg], damping 0.08. Two OPPOSED spokes; the tracked
one (``object_site``, spoke A) has its tip at body-local (-0.04, 0.09, 0) — radius
0.09 in the y-z plane. Spoke B is the same arm at 180 deg. The goal marker is
pinned at base_site + (0, -0.09, 0.09), a fixed world point per episode. Mount x
in [0.50, 0.70], y in [-0.1, 0.1]; hub at z = 0.18.

Success: hinge reaches +270 deg, shortfall-only, threshold 0.2 rad, latched.

WHY THIS ONE IS DIFFERENT. The task was designed around the Franka wrist: 270 deg
exceeds joint 7's range from any single grasp, so a grasping policy must release and
re-engage — and this teacher now does grasp, so it genuinely needs that cycle. The spoke tip stays between
z = 0.09 and z = 0.27, safely above the ground plane through the entire sweep, so
there is no floor conflict either. What DOES force a handoff is arm geometry: past
roughly 180 deg the pad is driven behind the mount plate, where the forearm fouls
it. At that point the policy hands off to the OPPOSITE spoke, which by construction
sits at a comfortable angle whenever spoke A has become awkward, and continues the
same push. That is the multi-phase engage / rotate / release / re-engage structure,
just realised with pushes rather than grasps.

Strategy: pinch the spoke and walk it around its CIRCULAR ARC, recomputed every step
by rotating the CURRENT radius vector a further LEAD_ANGLE about +x.

Only relative terms are used (gto = obs[40:43], o2g = obs[43:46]); per-env
scene-origin offsets cancel. The hinge angle is recovered from o2g against the
fixed marker, NOT from the noisy rot6d — a sign flip in the reconstructed axis
drives the valve backwards (this cost turn_lever ~40 points of success before it
was removed there).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

_R_ARM = 0.09  # spoke tip radius in the y-z plane
_X_OFF = -0.04  # constant x offset of the tip from the hinge (on the axis)
_MARKER_OFF = np.array([0.0, -0.09, 0.09])

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GRIPPER_CLOSED = -1.0

FACE_STANDOFF = 0.014
CONTACT_FRAC = 0.78
HOVER = 0.09
ALIGN_TOL = 0.055
CONTACT_TOL = 0.030
EMA_ALPHA = 0.4
INTEG_GAIN = 0.2
LEAD_ANGLE = 0.9  # rad of arc to lead by (POSITIVE: the valve turns 0 -> +270)
SETTLE = 2
LOST_TOL = 0.14
MAX_WAYPOINT = 0.20

# --- pinch retention -------------------------------------------------------
# The pad-push above seats reliably but does not RETAIN: measured, the pad
# parks against the hub and the valve jitters +-5 deg for 200+ steps while the
# EE does not move at all (one traced env sat at (0.467, -0.044, 0.178) from
# t=200 to t=400). The spoke is a 2.4cm bar and a face push has nothing to
# resist the pad sliding along it, so under the 0.08 hinge damping the contact
# walks inboard to the hub, where the moment arm goes to zero and the push
# stops doing any work. LOST_TOL never fires because the pad IS still near the
# arc -- it is just near the useless end of it.
#
# So: actually PINCH the spoke. The fingers straddle it along the direction of
# travel, which is the direction the load acts in, so the spoke is trapped
# between two pad faces instead of being pushed off one. This does lean on
# fingertip friction to a degree, which domain randomisation drives as low as
# 0.3 -- but only to resist sliding ALONG the spoke (radially); the driving
# load itself is carried face-on by the pads and is friction-free.
GRIPPER_PINCH = -1.0
GRIPPER_OPEN = 0.0
PINCH_FRAC = 0.80  # grab this far out along the spoke (tip is at frac 1.0)
PINCH_CLOSE_STEPS = 4  # steps to let the fingers close before loading
PINCH_LOST = 0.10  # site this far from the intended pinch point => re-seat
# Hard timeouts. 150 steps for 270 deg leaves no slack, and the DLS solve keeps a
# few-cm steady-state bias at this low, far-out pose, so a pure tolerance gate can
# park the arm in an approach phase for the whole episode.
ALIGN_TIMEOUT = 45
SEAT_TIMEOUT = 35
# SEAT_TIMEOUT above is NOT actually unconditional: it only fires when ALSO
# norm(raw) < 0.10, so an engagement whose lateral bias never drops under 10cm
# (INTEG_GAIN's clip at +-0.06 is not always enough to null it) can sit in
# DROP indefinitely -- this is the same deadlock class as the ARC_FOLLOW stall
# below, just missing its own unconditional escape. Instrumented: DROP eats
# ~39-40% of the 400-step budget on average (more than ARC_FOLLOW's ~31-33%),
# vs an intended ~35-step-per-engagement budget, and individual engagements
# were traced sitting in DROP for 200+ steps. SEAT_HARD_TIMEOUT is the
# unconditional cap ALIGN_TIMEOUT already has; falling through to PINCH_CLOSE
# and then ARC_FOLLOW with a still-imprecise seat is safe because
# ARC_FOLLOW's own PINCH_LOST check re-seats it immediately if the pinch was
# not actually acquired -- it does not risk a bad physical action, only a
# faster failure-and-retry instead of a silent multi-hundred-step stall.
SEAT_HARD_TIMEOUT = 60
# Hand off to the opposite spoke once the pushed spoke has gone this far round:
# beyond here the pad would be driven behind the mount plate.
# Hand off well BEFORE the arm becomes awkward. 2.6 rad (~150 deg) was too
# late: traced envs froze at a perfectly constant angle (one sat at exactly
# 85.81 deg from t=80 to t=400) with the arm still pressing -- the wrist had
# run out of travel and the forearm was fouling, but the handoff gate had not
# been reached, so nothing ever re-triggered. ~100 deg per engagement means
# three engagements cover the 270 deg target with margin.
HANDOFF_ANGLE = 1.40  # rad (~80 deg)
# Stall watchdog. The handoff gate keys on swept angle, so an engagement that
# jams makes NO progress and therefore never reaches the gate -- it deadlocks
# for the rest of the episode. Force a handoff if the unwrapped angle has not
# advanced over this many steps.
# 25 steps measured best. 40 lets a jammed engagement burn budget (0.125);
# 15 abandons engagements that were still turning (0.375); 25 gave 0.625 on
# the same 8-env sample.
STALL_STEPS = 25
STALL_EPS = 0.05  # rad of progress required within STALL_STEPS


def _rot_x(v: np.ndarray, th: float) -> np.ndarray:
  c, s = np.cos(th), np.sin(th)
  return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])


class RotateValveClassicalPolicy(ClassicalPolicyBase):
  """Walk a closed fingertip around the valve's arc, handing off between spokes."""

  max_dq = 0.17
  orientation_weight = 0.2

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      # Which spoke the pad is currently pushing: 0 = the tracked spoke A,
      # 1 = the opposite spoke B (same arm, +180 deg).
      self._spoke = np.zeros(self.num_envs, dtype=np.int64)
      # Unwrapped hinge angle. The marker-based estimate is wrapped to (-pi, pi],
      # but the target is +270 deg, so a wrapped angle folds back past 90 deg and
      # the arc-follower silently reverses direction. Accumulating wrapped
      # INCREMENTS is what makes a >180 deg sweep trackable at all.
      self._theta_unwrapped = np.zeros(self.num_envs)
      self._theta_prev = np.full(self.num_envs, np.nan)
      self._engage_theta = np.zeros(self.num_envs)
      self._stall_theta = np.zeros(self.num_envs)
      self._stall_t = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._spoke[env_ids] = 0
      self._theta_unwrapped[env_ids] = 0.0
      self._theta_prev[env_ids] = np.nan
      self._engage_theta[env_ids] = 0.0
      self._stall_theta[env_ids] = 0.0
      self._stall_t[env_ids] = 0

  def _arc_state(self, gto: np.ndarray, o2g: np.ndarray, spoke: int):
    """Return (hinge_rel, theta, radius_vec, face_normal) for the pushed spoke.

    ``theta`` is spoke A's absolute hinge angle, recovered from the fixed goal
    marker: tip(theta) - tip(0) = _MARKER_OFF - o2g. With
    tip(0) - hinge = (_X_OFF, _R_ARM, 0) rotated by theta about +x,
    d_y = R(cos t - 1) and d_z = R sin t. ``radius_vec`` and ``face_normal``
    describe the spoke actually being pushed (A, or B at +180 deg).
    """
    d = _MARKER_OFF - o2g
    ct = np.clip(d[1] / _R_ARM + 1.0, -1.0, 1.0)
    st = np.clip(d[2] / _R_ARM, -1.0, 1.0)
    theta = float(np.arctan2(st, ct))

    # gto tracks spoke A's tip. The pushed spoke's tip is A's, rotated 180 deg
    # about the hinge if we are on spoke B.
    r_a = np.array([0.0, _R_ARM * np.cos(theta), _R_ARM * np.sin(theta)])
    if spoke == 0:
      radius_vec = r_a
      tip_rel = gto
    else:
      radius_vec = -r_a
      # tip_B = hinge - r_a = (tip_A - r_a - x_off) - r_a
      tip_rel = gto - np.array([_X_OFF, 0.0, 0.0]) - 2.0 * r_a + np.array(
        [_X_OFF, 0.0, 0.0]
      )
    hinge_rel = gto - np.array([_X_OFF, 0.0, 0.0]) - r_a
    # Face normal: the spoke's local +z, rotated by theta (flipped on spoke B so
    # we always push on the face that leads the POSITIVE direction of travel).
    n_a = np.array([0.0, -np.sin(theta), np.cos(theta)])
    face_normal = n_a if spoke == 0 else -n_a
    # The pad must sit BEHIND the spoke relative to its direction of travel, so
    # that closing on the spoke drives it forward. Positive rotation about +x
    # gives tangent = +axis x r, so the standoff direction is -tangent. (Standing
    # on the +tangent side instead puts the pad in the spoke's way and produces
    # exactly the observed failure: solid contact, zero rotation.)
    tangent = np.cross(np.array([1.0, 0.0, 0.0]), radius_vec)
    if np.dot(face_normal, tangent) > 0:
      face_normal = -face_normal
    return hinge_rel, theta, radius_vec, face_normal, tip_rel

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = obs_i[43:46]

    # Unwrapped total progress: theta from the marker is wrapped to (-pi, pi], but
    # the target is +270 deg. Track how far the valve has actually turned by
    # accumulating wrapped increments.
    hinge_rel, theta, radius_vec, face_normal, _tip = self._arc_state(
      gto, o2g, int(self._spoke[i])
    )
    # Unwrap: accumulate the wrapped increment so total progress past 180 deg is
    # tracked correctly (see the note in reset()).
    if np.isnan(self._theta_prev[i]):
      self._theta_unwrapped[i] = theta
    else:
      d_th = theta - self._theta_prev[i]
      d_th = (d_th + np.pi) % (2 * np.pi) - np.pi
      self._theta_unwrapped[i] += d_th
    self._theta_prev[i] = theta

    contact = (
      hinge_rel
      + np.array([_X_OFF, 0.0, 0.0])
      + radius_vec * CONTACT_FRAC
      + face_normal * FACE_STANDOFF
    )

    # The PINCH point: on the spoke itself, not standing off its face.
    pinch = hinge_rel + np.array([_X_OFF, 0.0, 0.0]) + radius_vec * PINCH_FRAC

    gripper_a = GRIPPER_PINCH

    if self._phase[i] == 0:
      # Approach with the fingers OPEN, standing off along the spoke's face
      # normal so the open gap comes down across the spoke rather than into it.
      gripper_a = GRIPPER_OPEN
      pos_err = pinch + face_normal * HOVER
      perp = pos_err - np.dot(pos_err, face_normal) * face_normal
      if np.linalg.norm(perp) < ALIGN_TOL or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Drop onto the spoke, still open. Integral action nulls the DLS
      # steady-state bias, which is a few cm at this low, far-out pose and
      # would otherwise leave the gap hovering beside the spoke forever.
      gripper_a = GRIPPER_OPEN
      raw = pinch
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.06, 0.06)
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < CONTACT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if (
        self._settle[i] >= SETTLE
        or (self._phase_steps[i] > SEAT_TIMEOUT and np.linalg.norm(raw) < 0.10)
        or self._phase_steps[i] > SEAT_HARD_TIMEOUT
      ):
        self._phase[i] = 3
        self._phase_steps[i] = 0
    elif self._phase[i] == 3:
      # Close on the spoke and hold still while the fingers travel.
      pos_err = pinch + self._integ[i]
      if self._phase_steps[i] >= PINCH_CLOSE_STEPS:
        self._phase[i] = 2
        self._phase_steps[i] = 0
    else:
      # Arc-follow in the POSITIVE direction.
      lead_r = _rot_x(radius_vec, LEAD_ANGLE)
      target = (
        hinge_rel + np.array([_X_OFF, 0.0, 0.0]) + lead_r * PINCH_FRAC
      )
      pos_err = target + self._integ[i]
      # HANDOFF: once the pushed spoke has swung round to where the forearm would
      # foul the mount plate, release and switch to the opposite spoke, then
      # re-engage from phase 0. This is the "release, reposition, re-engage" cycle.
      # Handoff on TOTAL swept angle since the last engagement, using the
      # unwrapped estimate.
      swung = float(self._theta_unwrapped[i]) - float(self._engage_theta[i])
      # Stall watchdog (see STALL_STEPS): treat "no progress for a while" as a
      # reason to hand off, exactly like having swung far enough.
      self._stall_t[i] += 1
      stalled = False
      if self._stall_t[i] >= STALL_STEPS:
        stalled = (
          abs(float(self._theta_unwrapped[i]) - float(self._stall_theta[i]))
          < STALL_EPS
        )
        self._stall_theta[i] = self._theta_unwrapped[i]
        self._stall_t[i] = 0
      if abs(swung) > HANDOFF_ANGLE or stalled:
        self._spoke[i] = 1 - self._spoke[i]
        self._engage_theta[i] = self._theta_unwrapped[i]
        self._stall_theta[i] = self._theta_unwrapped[i]
        self._stall_t[i] = 0
        self._phase[i] = 0
        self._phase_steps[i] = 0
        self._settle[i] = 0
        self._integ[i] = 0.0
      elif np.linalg.norm(pinch) > PINCH_LOST:
        # Lost the spoke -> re-seat. Go back to the OPEN approach (phase 0),
        # not straight to the drop: once the fingers have been dragged off,
        # they are usually on the wrong side of the spoke and closing again
        # from there just jams against it.
        self._phase[i] = 0
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = pinch

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, _DOWN_AXIS, gripper_a
