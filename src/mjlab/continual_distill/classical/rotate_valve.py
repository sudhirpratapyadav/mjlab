"""Scripted RotateValve teacher — continuous arc-push, with a spoke HANDOFF.

Geometry (valve.xml, verified live): hinge about world +x (the mount has no yaw
randomisation), range [0, 360 deg], damping 0.08. Two OPPOSED spokes; the tracked
one (``object_site``, spoke A) has its tip at body-local (-0.04, 0.09, 0) — radius
0.09 in the y-z plane. Spoke B is the same arm at 180 deg. The goal marker is
pinned at base_site + (0, -0.09, 0.09), a fixed world point per episode. Mount x
in [0.50, 0.70], y in [-0.1, 0.1]; hub at z = 0.18.

Success: hinge reaches +270 deg, shortfall-only, threshold 0.2 rad, latched.

WHY THIS ONE IS DIFFERENT — and why the intended regrasp turns out to be avoidable.
The task was designed around the Franka wrist: 270 deg exceeds joint 7's range from
any single GRASP, so a grasping policy must release and re-engage. But this teacher
does not grasp at all (see below), so the wrist never accumulates rotation — the
whole arm walks the pad around the circle instead. The spoke tip stays between
z = 0.09 and z = 0.27, safely above the ground plane through the entire sweep, so
there is no floor conflict either. What DOES force a handoff is arm geometry: past
roughly 180 deg the pad is driven behind the mount plate, where the forearm fouls
it. At that point the policy hands off to the OPPOSITE spoke, which by construction
sits at a comfortable angle whenever spoke A has become awkward, and continues the
same push. That is the multi-phase engage / rotate / release / re-engage structure,
just realised with pushes rather than grasps.

Strategy, as in turn_lever and open_drawer: fingers stay CLOSED and the pad is a
face-normal pusher driven around the tip's CIRCULAR ARC, recomputed every step by
rotating the CURRENT radius vector a further LEAD_ANGLE about +x. Nothing depends
on grip, so the domain-randomised fingertip friction (as low as 0.3) is irrelevant.

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
# Hard timeouts. 150 steps for 270 deg leaves no slack, and the DLS solve keeps a
# few-cm steady-state bias at this low, far-out pose, so a pure tolerance gate can
# park the arm in an approach phase for the whole episode.
ALIGN_TIMEOUT = 45
SEAT_TIMEOUT = 35
# Hand off to the opposite spoke once the pushed spoke has gone this far round:
# beyond here the pad would be driven behind the mount plate.
HANDOFF_ANGLE = 2.6  # rad (~150 deg)


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
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._spoke[env_ids] = 0
      self._theta_unwrapped[env_ids] = 0.0
      self._theta_prev[env_ids] = np.nan
      self._engage_theta[env_ids] = 0.0

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

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      pos_err = contact + face_normal * HOVER
      perp = pos_err - np.dot(pos_err, face_normal) * face_normal
      if np.linalg.norm(perp) < ALIGN_TOL or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      raw = contact
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.06, 0.06)
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < CONTACT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= SETTLE or (
        self._phase_steps[i] > SEAT_TIMEOUT and np.linalg.norm(raw) < 0.10
      ):
        self._phase[i] = 2
        self._phase_steps[i] = 0
    else:
      # Arc-follow in the POSITIVE direction.
      lead_r = _rot_x(radius_vec, LEAD_ANGLE)
      lead_n = _rot_x(face_normal, LEAD_ANGLE)
      target = (
        hinge_rel
        + np.array([_X_OFF, 0.0, 0.0])
        + lead_r * CONTACT_FRAC
        + lead_n * FACE_STANDOFF
      )
      pos_err = target + self._integ[i]
      # HANDOFF: once the pushed spoke has swung round to where the forearm would
      # foul the mount plate, release and switch to the opposite spoke, then
      # re-engage from phase 0. This is the "release, reposition, re-engage" cycle.
      # Handoff on TOTAL swept angle since the last engagement, using the
      # unwrapped estimate.
      swung = float(self._theta_unwrapped[i]) - float(self._engage_theta[i])
      if abs(swung) > HANDOFF_ANGLE:
        self._spoke[i] = 1 - self._spoke[i]
        self._engage_theta[i] = self._theta_unwrapped[i]
        self._phase[i] = 0
        self._phase_steps[i] = 0
        self._settle[i] = 0
        self._integ[i] = 0.0
      elif np.linalg.norm(contact) > LOST_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = contact

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, _DOWN_AXIS, gripper_a
