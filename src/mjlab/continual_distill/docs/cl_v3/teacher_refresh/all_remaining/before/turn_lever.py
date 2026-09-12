"""Scripted TurnLever teacher — tangential FACE-PUSH around the hinge arc.

Physics refresh (2026-09-10): approach at 45 degrees below mount +x, with
the finger closing axis along mount y. This keeps the wrist in front of the hub
during the downward sweep. Historical development notes follow.

Geometry (lever.xml, verified against a live env): the mount is a mocap body. CL-V3
gives it a yaw band (+-0.26 rad about the facing-the-robot pose) and a 3 cm height
band, so the hinge axis is the MOUNT's +x, not world +x. The mount yaw psi is read
from ``object_orientation`` (obs[34:40] = rows 1-2 of the ROOT body's rotation
matrix, and the root body of the lever entity is the mocap mount itself, the same
``root_link_quat_w`` the command rotates its goal marker with), EMA-smoothed, and the
whole teacher below runs in the MOUNT frame: ``gto`` / ``o2g`` are rotated by -psi on
the way in and the waypoint by +psi on the way out. At psi = 0 this is bit-identical
to the cl25/v2 teacher. Before this, a yawed mount rotated the goal marker (and hence
the recovered hinge angle and the whole contact arc) by up to 15 deg while the teacher
still pushed along world axes: v3 baseline 0.906 (128) vs v2 0.922.
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

from mjlab.continual_distill.classical.base import (
  ClassicalPolicyBase,
  mount_yaw_from_obs,
  rot_z,
  rot_z_mat,
  yaw_down_rot,
)

# object_site in the handle body frame (lever.xml).
_P_LOCAL = np.array([-0.05, 0.12, 0.0])
_R_ARM = 0.12  # arc radius of the grasp site in the y-z plane
_X_OFF = -0.05  # constant x offset of the site from the hinge (on the axis)

_AXIS = np.array([1.0, 0.0, 0.0])  # hinge axis in the MOUNT frame (world +x at yaw 0)

# EE approach axis: DOWN, pitched 15 deg BACK toward the robot (CL-V3, W1-N). The v3
# diagnose put the three residual classes of this teacher on one geometry: the hand
# capsule (r 0.04, half-length 0.06 along the finger-closing axis, 0.07 up the approach
# axis) and the link7 flange sit in the bar's own y-z plane, 6 cm in front of the 30 cm
# back plate and directly over the hub at the end of the arc. `ee_lever_collision`
# (link7) was true for 75-97 % of the last 40 steps in every "never moved" env and the
# two 0.01-rad near-misses parked with the wrist on the hub; successes never touched
# the fixture with link7 before success. A CPU probe of the same envs showed the
# successes' approach axis pitched 12-16 deg back, the stalls 0-6 deg forward. Pitching
# the target axis back moves the capsule/flange ~2.5-4 cm toward the robot, off the
# plate and hub; the pad still pushes DOWN on the bar's face (its lower front edge
# leads). Yaw stays free. Two pinch alternatives were probed and rejected on the CPU
# model before this (see logs/W1-N.md): closing across the bar's thickness puts the
# capsule's long axis through the plate (measured: the site parks 6 cm short), and a
# horizontal wrist-roll pinch is unreachable at the mount height (joints 0.5-1.3 rad
# past their limits along the arc).
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
# Consecutive in-tolerance steps required before phase 1 -> 2 (and, after a re-seat,
# phase 1 -> 2 again). Was 2. Once RESEAT_TOL (below) started firing correctly, a new
# residual appeared: recovered episodes still fell short of the target within the
# 150-step budget. Instrumented (2026-09-08): a typical recovering episode re-seats
# 3-6 times, and each re-seat pays this settle delay on top of the re-approach itself
# -- at SETTLE=2 that is 6-12+ wasted steps per episode just re-confirming contact
# already re-confirmed the step before. Dropping to 1 measured 0.891->0.953 (128) with
# no regression in the direct (non-recovery) path, because CONTACT_TOL (0.04) is tight
# enough on its own that a single in-tolerance step is not a false positive here.
SETTLE = 1
# Hard cap on how far the commanded waypoint may sit from the gripper. Without
# it a momentarily bad angle estimate produces a waypoint metres away and the
# arm flings itself out of the workspace (observed: gripper-object distance 0.9m).
MAX_WAYPOINT = 0.20

# Re-seat gate for the arc-follow phase (phase 2). See the "STALL" note on
# `_target_error`'s phase-2 branch: this was 0.14 and it silently deadlocked a large
# fraction of episodes, never triggering. 0.06 is calibrated directly off instrumented
# traces, not an arbitrary retune -- see that note for the numbers.
RESEAT_TOL = 0.06
# CL-V3 (W1-N): hand yaw is no longer free. 16/19 failures of the lead-only teacher had a
# NEGATIVE mount yaw (9/12 for the v2 teacher): with yaw free the closing axis -- which is
# also the hand capsule's long axis -- follows the shoulder yaw, and for a mount on the
# robot's right with psi < 0 that lies in the bar's own plane, so the capsule ends up on the
# hub / bar at the end of the arc (link7 contact in 7/19). Full-frame target: down, closing
# axis rotated HAND_YAW_OFF from the bar's line. Bounded at ~17 deg so the capsule's far end
# (0.06 sin a + 0.04) stays clear of the back plate 6 cm behind the bar.
HAND_YAW_OFF = 0.30
APPROACH_TILT = 45.0


class TurnLeverClassicalPolicy(ClassicalPolicyBase):
  """Push the lever tip around its hinge arc with a closed fingertip."""

  max_dq = 0.13
  orientation_weight = 0.2
  # Sustained command lead (W1-G's gravity-sag ratchet, logs/W1-G.md): with 0 the joint
  # command is re-anchored to the sagged actual joints every step, so a seat / hold from
  # a stretched pose sinks and a push against the bar's damping relaxes. Bounded lead =
  # gravity compensation + a push that does not fade.
  cmd_lead_max = 0.10

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._yaw = np.zeros(self.num_envs)
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._yaw[env_ids] = 0.0

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
    # Mount yaw (CL-V3 yaw band): smoothed the same way as gto, latched at first sight.
    yaw_raw = mount_yaw_from_obs(obs_i)
    if not self._ema_init[i]:
      self._yaw[i] = yaw_raw
    else:
      self._yaw[i] = EMA_ALPHA * yaw_raw + (1 - EMA_ALPHA) * self._yaw[i]
    psi = float(self._yaw[i])
    # Everything below is in the MOUNT frame (hinge axis = +x, arc in the y-z plane).
    gto_raw = rot_z(obs_i[40:43], -psi)
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = rot_z(obs_i[43:46], -psi)

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
      # STALL / re-seat gate. Instrumented finding (2026-09-08): the gate used to be
      # `norm(contact) > 0.14`, and it essentially never fired. Traced per-step: a
      # failing env's pad drifts off the bar's pushing face (almost entirely a +z
      # climb, e.g. contact=[+0.02,+0.01,+0.12] by ~12 steps into phase 2) and then
      # SITS at a stable ~0.10-0.13 contact-norm equilibrium for the rest of the
      # episode with jv frozen to 4 decimal places -- genuinely zero force, not slow
      # progress. That equilibrium never crosses 0.14, so the gate designed to catch
      # "lost the bar" never caught the one failure mode that actually occurs.
      # Contrast with envs that succeed: during the real accelerating push (jv
      # racing from 0 to past -90deg) contact-norm stays under ~0.05 the whole time;
      # it only grows past 0.08 AFTER the joint has already hit its hard stop and
      # success is already latched, so tightening this does not cost those episodes
      # anything. 0.06 sits between the "actively pushing" band (<0.05) and the
      # "silently stuck" plateau (>=0.10), so it now actually fires while the
      # episode still has budget left to recover.
      if np.linalg.norm(contact) > RESEAT_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = contact

    n = np.linalg.norm(pos_err)
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    # Back to world axes for the IK; full-frame hand target (see HAND_YAW_OFF).
    rotation = yaw_down_rot(psi + HAND_YAW_OFF)
    if APPROACH_TILT:
      tilt = np.deg2rad(APPROACH_TILT)
      approach = np.array([np.cos(tilt), 0, -np.sin(tilt)])
      closing = np.array([0, 1, 0])
      rotation = rot_z_mat(psi) @ np.column_stack([np.cross(closing, approach), closing, approach])
    return rot_z(pos_err, psi), rotation, gripper_a
