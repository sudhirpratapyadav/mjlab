"""Scripted OpenLid teacher — arc-push on the lip, continuing what gravity starts.

Geometry (lid.xml, verified live): hinge at box-local (0.10, 0, 0.04), axis world
+y (the mount has no yaw randomisation), range [-75 deg, 0]. The grab lip
(``object_site``) is at box-local (-0.09, 0, 0.05), so the lip's radius vector
from the hinge is (-0.19, 0, 0.01) — length 0.19 m, pointing back toward the
robot. The goal marker is pinned at base_site + (0.10, 0, 0.12), a FIXED world
point per episode. Measured: lip at z = 0.15 when closed, mount x in [0.50, 0.70].

MEASURED FACT that shapes the whole design: this is a DROP-DOWN flap, not a
lift-up lid. Rotating -75 deg about +y swings the lip DOWN and back under the
box, so gravity ASSISTS rather than opposes. A do-nothing policy (zero actions,
measured over 8 envs x 150 steps) lets the lid fall from 0 to -0.735 rad and stall
there — short of the -1.309 rad target, whose 0.2 rad shortfall threshold needs
<= -1.109. So the free-fall does roughly two thirds of the work and the policy's
real job is the LAST ~0.37 rad, driving the lip further round its arc while the
panel is already hanging steeply and its own weight torque has gone slack.

Strategy: same friction-free principle as open_drawer and turn_lever — keep the
fingers CLOSED and use the pad as a pusher on the lip's outward face, driven along
the lip's CIRCULAR ARC about the hinge. Nothing depends on grip, which matters
because domain randomisation drives fingertip friction as low as 0.3. There is no
grasp to lose and no regrasp: one continuous arc-follow. The waypoint is
recomputed every step by rotating the lip's CURRENT radius vector a further
LEAD_ANGLE about +y, so the pad tracks the true arc instead of chasing a fixed
point that would drag it off the panel.

Only relative terms are used (gto = obs[40:43], o2g = obs[43:46]); per-env
scene-origin offsets cancel.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Lip radius vector from the hinge at theta = 0 (x, z components; y is on-axis).
_R0 = np.array([-0.19, 0.0, 0.01])
_R_LEN = float(np.linalg.norm(_R0))
# Goal-marker offset from base_site (viz-only marker, but a fixed world anchor we
# use to recover the hinge angle from a relative observation).
_MARKER_OFF = np.array([0.10, 0.0, 0.12])

# EE z-axis straight down. The pad's broad face then points down/along the arc for
# the steeply-hanging panel, which is where the useful push direction lies.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GRIPPER_CLOSED = -1.0

FACE_STANDOFF = 0.030  # outboard of the lip along its outward face normal
CONTACT_FRAC = 0.95  # push near the lip (max lever arm about the hinge)
HOVER = 0.09
ALIGN_TOL = 0.055
CONTACT_TOL = 0.075
EMA_ALPHA = 0.4
INTEG_GAIN = 0.2
LEAD_ANGLE = 0.9  # rad of arc to lead by, recomputed from the CURRENT angle
SETTLE = 2
LOST_TOL = 0.22
MAX_WAYPOINT = 0.20
# Let gravity do its free-fall first; pushing into a lid that is still swinging
# just fights it. Measured: the fall completes well inside 40 steps.
FALL_WAIT = 25


def _rot_y(v: np.ndarray, th: float) -> np.ndarray:
  c, s = np.cos(th), np.sin(th)
  return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


class OpenLidClassicalPolicy(ClassicalPolicyBase):
  """Push the lid lip around its hinge arc past where gravity stalls it."""

  max_dq = 0.16
  orientation_weight = 0.2

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0

  def _arc_state(self, gto: np.ndarray, o2g: np.ndarray):
    """Recover (hinge_rel, theta, radius_vec, face_normal) from relative obs.

    lip(theta) - lip(0) = _MARKER_OFF - o2g, because the marker is a fixed world
    point and o2g = marker - lip(theta). Given lip(0) - hinge = _R0, solving the
    planar rotation for theta is a 2x2 least-squares on the x-z components.
    """
    d = _MARKER_OFF - o2g  # lip(theta) - lip(0)
    # lip(theta) - hinge = R(theta) _R0  =>  target = _R0 + d
    v = _R0 + d
    # theta from atan2 of the rotation taking _R0 to v (about +y).
    # R(t)[x,z] = [x c + z s, -x s + z c]
    x0, z0 = _R0[0], _R0[2]
    vx, vz = v[0], v[2]
    denom = x0 * x0 + z0 * z0
    c = (x0 * vx + z0 * vz) / denom
    s = (z0 * vx - x0 * vz) / denom
    theta = float(np.arctan2(s, c))

    radius_vec = _rot_y(_R0, theta)
    hinge_rel = gto - radius_vec
    # Outward face normal of the lip: the panel's local +z, rotated by theta.
    face_normal = _rot_y(np.array([0.0, 0.0, 1.0]), theta)
    return hinge_rel, theta, radius_vec, face_normal

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = obs_i[43:46]

    hinge_rel, theta, radius_vec, face_normal = self._arc_state(gto, o2g)
    contact = hinge_rel + radius_vec * CONTACT_FRAC + face_normal * FACE_STANDOFF

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Hold clear while gravity drops the flap, then align above the contact.
      pos_err = contact + face_normal * HOVER
      perp = pos_err - np.dot(pos_err, face_normal) * face_normal
      if self._phase_steps[i] > FALL_WAIT and np.linalg.norm(perp) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      raw = contact
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.06, 0.06)
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < CONTACT_TOL:
        self._settle[i] += 1
        if self._settle[i] >= SETTLE:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
    else:
      lead_r = _rot_y(radius_vec, -LEAD_ANGLE)
      lead_n = _rot_y(face_normal, -LEAD_ANGLE)
      target = hinge_rel + lead_r * CONTACT_FRAC + lead_n * FACE_STANDOFF
      pos_err = target + self._integ[i]
      if np.linalg.norm(contact) > LOST_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = contact

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, _DOWN_AXIS, gripper_a
