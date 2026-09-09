"""Scripted PushFlap teacher -- arc-push a handle-less flap around its hinge.

BLOCKED BY A BENCHMARK BUG, NOT A TEACHER PROBLEM -- read this first
------------------------------------------------------------------------
``flap.xml``'s ``flap_hinge`` joint is written ``range="-1.4 0"``, and every
comment near it (this file, ``PushFlapCommandCfg``, the XML header) treats
that as RADIANS -- "short of the hinge's own -1.4 hard stop". It is not.
MuJoCo's compiler defaults to DEGREES (``mujoco.MjSpec().compiler.degree ==
True``) and flap.xml never overrides it, so the COMPILED joint range is
``[-0.0244, 0]`` radians (-1.4 DEGREES), confirmed directly off the live
model (``model.jnt_range`` after ``load_env_cfg``). Compare every sibling
asset in the same family, all written correctly in degrees matching
MuJoCo's default: ``door.xml`` ``range="0 90"``, ``lid.xml`` ``range="-75
0"``, ``lever.xml`` ``range="-90 0"``, ``valve.xml`` ``range="0 360"``.
flap.xml is the one asset that wrote a radian-scaled number into a
degree-interpreted field. The command target (-1.2217 rad, -70 deg) is
therefore roughly 50x further than the hinge can physically travel -- NO
policy, scripted or learned, can reach it as this asset currently ships.
Reported as a blocker in ``STATUS.md``; full mechanism account, including
what was ruled out before finding this, is in ``LOGS.md``. Per the Phase-1
rules this is a benchmark fix, not something to fold into the teacher, so
this file is written against the task's INTENDED geometry (as if the range
bug were fixed) rather than hacked around it.

GEOMETRY (flap.xml, verified live against the running sim)
------------------------------------------------------------
The mount (``flap_base``) is a mocap body with NO yaw randomisation, so the
hinge is exactly world +z, sitting AT the mount origin (the ``handle`` body
that carries the joint has no local offset). The panel is a thin box
(half-extents 0.012 x 0.11 x 0.15) hanging off the hinge along local +y at
rest; the push point (``object_site``) is at handle-local (-0.012, 0.18, 0) --
i.e. radius vector from the hinge at theta=0 is
    _R0 = (-0.012, 0.18, 0.0),  |_R0| = 0.1804
on the panel's -x face (the face whose outward normal points toward the
robot). ``base_site`` (on the STATIC mount body, not the rotating one) sits
at mount-local (-0.012, 0.12, 0), and the (viz-only) goal marker is pinned at
``base_site + goal_marker_offset`` = mount-local (0.098, 0.04, 0) -- a FIXED
world point per episode, independent of the joint angle.

THE SIGN TRAP (read PushFlapCommandCfg before touching this file)
---------------------------------------------------------------------
target_value = -1.2217 rad (-70 deg). The joint range is meant to be (-1.4,
0) and there is no handle, so the ONLY reachable direction is negative: a +x
force applied at the push point's +y lever arm gives r x F = (0, L, 0) x
(F, 0, 0) = (0, 0, -LF), i.e. NEGATIVE hinge torque. A positive target would
need pulling the handle-less panel toward the robot -- unreachable by
face-pushing.

STRATEGY -- arc-push, not a grasp
-----------------------------------
Same principle as turn_lever/open_lid/slide_window/open_drawer: keep the
fingers CLOSED and drive the pad past the push point's CURRENT surface,
re-deriving the contact point every step from the LIVE hinge angle estimate
(not a lead/projected future pose -- an earlier version of this file tried
that and it is worth recording why it is wrong: see LOGS.md). No orientation
objective is used (``orientation_weight = 0.0``, the same move
``reach_target.py`` makes for its far targets): the mount sits far enough out
that the arc's far end lives in the any-orientation reach band, not the
tighter top-down band (``workspace.py``), so holding a constant approach axis
the way the shorter-reach mechanism teachers do is not reachable across this
arc's whole span.

The hinge angle theta is recovered from ``object_to_goal`` (obs[43:46])
against the FIXED marker -- not from the noisy rot6d orientation observation
(CONTEXT.md sec 5 and turn_lever's own docstring both flag a sign-flipped
rot6d axis as a real, costly bug elsewhere in this codebase). Only relative
terms are used (gto = obs[40:43], o2g = obs[43:46]); per-env scene-origin
offsets cancel in both.

No FALL_WAIT is needed here (unlike open_lid): push_flap runs GRAVITY-OFF
(mechanism task, deliberate -- see CONTEXT.md sec 6), so there is no free
drop to wait out before engaging.

Observation layout (60 dims, ``push_flap`` task -- see push_flap_env_cfg.py):
  0:9    robot_joint_pos (relative to NEUTRAL_QPOS -- this task uses
         get_franka_robot_cfg_neutral, the door/drawer/lever/lid group)
  9:18   robot_joint_vel
  18:21  object_pos      (push-point site, world, ABSOLUTE -- unused)
  21:25  object_quat
  25:28  gripper_pos     (absolute, unused)
  28:34  gripper_orientation (rot6d, unused)
  34:40  object_orientation  (rot6d, unused -- see sign-flip warning above)
  40:43  gripper_to_object vector (push point - gripper site)      -- USED
  43:46  object_to_goal vector (fixed marker - push point)         -- USED
  46:52  goal_orientation_diff (rot6d, unused)
  52:60  control_qpos_diff (unused)
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# object_site in the handle-body frame at theta=0 (flap.xml).
_R0 = np.array([-0.012, 0.18, 0.0])
# Outward face normal (the panel's -x face, robot-facing) at theta=0.
_N0 = np.array([-1.0, 0.0, 0.0])
# target_pos (fixed marker) - site(0), derived from base_site + goal_marker_offset
# vs. object_site's own local position -- see module docstring geometry section.
_MARKER_OFF = np.array([0.11, -0.14, 0.0])

GRIPPER_CLOSED = -1.0

FACE_STANDOFF = 0.030  # outboard of the panel along its face normal (hover/seat)
PUSH_DEPTH = 0.12  # inboard of the CURRENT live surface along -face_normal (push)
CONTACT_FRAC = 0.92  # push near the site (good lever arm about the hinge)
HOVER = 0.09
ALIGN_TOL = 0.055
CONTACT_TOL = 0.045
EMA_ALPHA = 0.4
INTEG_GAIN = 0.2
SETTLE = 2
LOST_TOL = 0.20
MAX_WAYPOINT = 0.20


def _rot_z(v: np.ndarray, th: float) -> np.ndarray:
  c, s = np.cos(th), np.sin(th)
  return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


class PushFlapClassicalPolicy(ClassicalPolicyBase):
  """Push the handle-less flap around its hinge arc with a closed fingertip."""

  max_dq = 0.15
  # No orientation objective -- see the module docstring.
  orientation_weight = 0.0

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
    """Return (hinge_rel, theta, radius_vec, face_normal), all LIVE.

    site(theta) - site(0) = _MARKER_OFF - o2g, because the marker is a fixed
    world point and o2g = marker - site(theta). Given site(0) - hinge = _R0,
    solving the planar (x-y, about +z) rotation for theta is a 2x2
    least-squares on the x-y components -- mirrors open_lid's x-z solve about
    +y and turn_lever's y-z solve about +x, just the third axis pair.
    """
    d = _MARKER_OFF - o2g  # site(theta) - site(0)
    v = _R0 + d  # = R_z(theta) @ _R0
    x0, y0 = _R0[0], _R0[1]
    vx, vy = v[0], v[1]
    denom = x0 * x0 + y0 * y0
    c = (x0 * vx + y0 * vy) / denom
    s = (x0 * vy - y0 * vx) / denom
    theta = float(np.arctan2(s, c))

    radius_vec = _rot_z(_R0, theta)
    hinge_rel = gto - radius_vec
    face_normal = _rot_z(_N0, theta)
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

    hinge_rel, _theta, radius_vec, face_normal = self._arc_state(gto, o2g)
    contact = hinge_rel + radius_vec * CONTACT_FRAC + face_normal * FACE_STANDOFF

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Align above/behind the contact point (no free-fall wait: gravity is
      # OFF for this task, see module docstring).
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
      # PUSH: aim past the CURRENT live surface along -face_normal. Re-derived
      # every step from the live hinge angle, so the target always tracks
      # wherever the panel genuinely is -- see the module docstring for why a
      # projected future/lead pose is wrong for this mechanism.
      target = hinge_rel + radius_vec * CONTACT_FRAC - face_normal * PUSH_DEPTH
      pos_err = target
      if np.linalg.norm(contact) > LOST_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        pos_err = contact

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, None, gripper_a
