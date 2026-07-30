"""Scripted TurnLever teacher — tangential FACE-PUSH around the hinge arc.

Geometry (lever.xml, verified against a live env): the mount is a mocap body with
NO yaw randomisation (``pose_range.yaw`` is (0, 0) and the reset event writes no
rotation), so the hinge axis is exactly world +x — pointing away from the robot.
The lever arm is a box extending +y from the hinge, grasp site (``object_site``)
at body-local (-0.05, 0.12, 0): radius 0.12 in the y-z plane. Mount spawns at
x in [0.50, 0.70], y in [-0.1, 0.1]; the handle site sits at z = 0.18 (measured),
radial 0.46-0.51.

Success: hinge driven to -90 deg, shortfall-only (overshoot counts), threshold
0.15 rad, latched over the episode.

Strategy — NOT a grasp. Domain randomisation drives fingertip friction as low as
0.3, so pinching the bar and rotating the wrist is fragile; and -90 deg of wrist
rotation about the approach axis eats most of joint 7's range. Instead the fingers
stay CLOSED and the pad is driven around the grasp point's CIRCULAR ARC as a pure
face-normal pusher. Rotation is -90 deg about +x (y -> -z, z -> +y), so the bar's
local +z face stays perpendicular to the tangential direction for the entire
sweep: one continuous friction-free push, no regrasp.

Everything is built from ``gripper_to_object`` (obs[40:43]) plus the KNOWN world
hinge axis, so per-env scene-origin offsets cancel. The handle's current angle is
recovered from the geometry of that one relative vector (see ``_arc_state``); the
noisy rot6d observation is deliberately NOT used — with +-1cm noise its
reconstructed axis occasionally flipped sign and drove the lever the wrong way.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# object_site in the handle body frame (lever.xml).
_P_LOCAL = np.array([-0.05, 0.12, 0.0])
_R_ARM = 0.12  # arc radius of the grasp site in the y-z plane
_X_OFF = -0.05  # constant x offset of the site from the hinge (on the axis)

_AXIS = np.array([1.0, 0.0, 0.0])  # hinge axis, world (never yaws)

# EE z-axis points straight DOWN. With the site z-axis down the finger-closing
# axis is world y, so the closed pad presents a flat face to the bar.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GRIPPER_CLOSED = -1.0

FACE_STANDOFF = 0.018  # push this far outboard of the bar centre along its face
CONTACT_FRAC = 0.90  # contact this far out along the arm (fraction of _R_ARM)
HOVER = 0.09
ALIGN_TOL = 0.05
CONTACT_TOL = 0.04
EMA_ALPHA = 0.4
INTEG_GAIN = 0.2
LEAD_ANGLE = 0.60  # rad of arc to lead by; recomputed from the CURRENT angle
SETTLE = 2
# Hard cap on how far the commanded waypoint may sit from the gripper. Without
# it a momentarily bad angle estimate produces a waypoint metres away and the
# arm flings itself out of the workspace (observed: gripper-object distance 0.9m).
MAX_WAYPOINT = 0.20


class TurnLeverClassicalPolicy(ClassicalPolicyBase):
  """Push the lever tip around its hinge arc with a closed fingertip."""

  max_dq = 0.13
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

  # -- geometry -------------------------------------------------------------

  def _arc_state(self, gto: np.ndarray, o2g: np.ndarray):
    """Return (hinge_rel, theta, radius_vec, tangent, face_normal).

    The hinge lies on the world +x line through the site, offset by _X_OFF in x
    and by the arm radius in the y-z plane. The arm's ANGLE is not directly
    observable from ``gto`` alone (that needs the hinge position), so we take it
    from the *goal marker*: ``object_to_goal`` (obs[43:46]) is
    goal_marker - site, and the marker is pinned at base_site + (0, -0.12, -0.12)
    — i.e. a FIXED world point per episode. Subtracting gives the site's
    displacement from its theta = 0 pose, from which theta follows directly.
    """
    # Site position relative to its theta=0 pose (world), offset-free:
    #   site(theta) - site(0) = (goal - site(0)) - (goal - site) = marker_off - o2g
    marker_off = np.array([0.0, -0.12, -0.12])
    d = marker_off - o2g  # site(theta) - site(0), in world y-z
    # site(0) - hinge = (X_OFF, R, 0); rotating by theta about +x:
    #   y = R cos t, z = R sin t  => d_y = R(cos t - 1), d_z = R sin t
    ct = np.clip(d[1] / _R_ARM + 1.0, -1.0, 1.0)
    st = np.clip(d[2] / _R_ARM, -1.0, 1.0)
    theta = float(np.arctan2(st, ct))

    radius_vec = np.array([0.0, _R_ARM * np.cos(theta), _R_ARM * np.sin(theta)])
    hinge_rel = gto - np.array([_X_OFF, 0.0, 0.0]) - radius_vec
    tangent = -np.cross(_AXIS, radius_vec)
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    # Bar's local +z face normal, rotated by theta about +x.
    face_normal = np.array([0.0, -np.sin(theta), np.cos(theta)])
    return hinge_rel, theta, radius_vec, tangent, face_normal

  @staticmethod
  def _rot_x(v: np.ndarray, th: float) -> np.ndarray:
    c, s = np.cos(th), np.sin(th)
    return np.array([v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]])

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = obs_i[43:46]

    hinge_rel, theta, radius_vec, _tan, face_normal = self._arc_state(gto, o2g)

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
      if np.linalg.norm(perp) < ALIGN_TOL:
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
      # Arc-follow: rotate the CURRENT radius vector a further LEAD_ANGLE in the
      # -x sense and aim there. Recomputed every step, so this tracks the true
      # circular path rather than converging on a fixed goal point.
      lead_r = self._rot_x(radius_vec, -LEAD_ANGLE)
      lead_n = self._rot_x(face_normal, -LEAD_ANGLE)
      target = (
        hinge_rel
        + np.array([_X_OFF, 0.0, 0.0])
        + lead_r * CONTACT_FRAC
        + lead_n * FACE_STANDOFF
      )
      pos_err = target + self._integ[i]
      # Re-seat only if the pad has genuinely lost the bar.
      if np.linalg.norm(contact) > 0.14:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = contact

    n = np.linalg.norm(pos_err)
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, _DOWN_AXIS, gripper_a
