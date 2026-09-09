"""Scripted OpenLid teacher — pinch the lifting knob and swing it up round the arc.

WHY THIS IS A REWRITE
---------------------
The cl25 asset hinged the lid with `axis="0 1 0"`, which turned the -75 deg target into
a DROP-DOWN flap. Measured on HEAD (and identically on the cl25 XML):

* with zero actions the lid free-fell to the -1.309 rad stop in ~0.25 s, so the task
  was winnable by doing nothing — `verify_task` G3 fails it on joint drift, and the
  cl25 teacher's 1.000 was mostly gravity's;
* from about -27 deg onward the lid swept THROUGH the box (nothing collides: every
  mechanism geom is `contype="2" conaffinity="1"`, and 2 & 1 = 0), which also put the
  contact point inside the carcass where the arm cannot follow it. Adding a friction
  stay to fix the first problem exposed the second: the teacher then stalled at
  0.30-0.89 rad against a 1.109 rad cutoff.

CL-V2 reverses the axis to `(0,-1,0)`, so the same -75 deg target LIFTS the lid up and
back over the hinge, the way a real chest lid opens. That is what
`open_lid_env_cfg.py` and `OpenLidCommand` have always claimed the task is — "gravity
opposes the motion throughout and the lid falls shut if released". Measured after the
change: zero-action drift 0.0014 rad over a whole 3 s episode, no interpenetration, and
the swept drop below the mount falls 0.158 -> 0.070 m.

STRATEGY
--------
A lift cannot be a push, so this is the one task in the cabinet family that needs a
real grasp: pinch the turned lifting knob (40 mm across, gripper opens 80 mm) with the
pads closing along WORLD Y, then follow the hinge arc upward.

Why world y: the hinge axis IS world y, so a y-closing pinch is INVARIANT under the
lid's rotation — the wrist only tilts in the x-z plane as the lid comes up and never
twists about the approach axis. World y is also the closing axis a Franka top-down pose
already presents, so the IK starts near the answer.

An earlier version grasped a 30 x 100 mm lifting BATTEN with the pads closing along the
lid's local x, which forced a 90 deg wrist twist to reach the grasp and another 35 deg
during the lift. Measured 0.000 at n=32: every env reached the lift phase, none moved
the lid past 0.13 rad, and several pressed it INTO its closed stop.

Only relative terms are used (gto = obs[40:43], o2g = obs[43:46]); per-env scene-origin
offsets cancel.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Lip radius vector from the hinge at theta = 0 (x, z; y is on-axis): object_site is at
# (-0.09, 0, 0.05) and the hinge at (0.10, 0, 0.04) in lid_base frame.
_R0 = np.array([-0.19, 0.0, 0.01])
# Goal-marker offset from base_site. Viz only, but it is a fixed world anchor, so it
# lets the hinge angle be recovered from a purely relative observation.
_MARKER_OFF = np.array([0.10, 0.0, 0.12])

GRIPPER_OPEN = 0.0
GRIPPER_PINCH = -1.0

# From object_site UP the lid's local z to the centre of the pinch. The knob spans local
# z 0.044..0.096 (the site is at 0.050) and the lid panel's top face is at 0.048, so the
# pads have a 48 mm window to land in. The collision pads span -0.0118..+0.0046 along
# the approach axis from the gripper site (measured from panda.xml), so a site 0.025
# above the site plane centres them on the stem just under the cap.
# That margin is the point: measured, the descent settles ~8 mm below its target, and
# at GRASP_UP = 0.014 that put the pads' lower faces THROUGH the lid panel, so the hand
# pressed the lid 0.05 rad past its closed stop instead of gripping the knob.
GRASP_UP = 0.020
# CL-V2 RE-DERIVATION of cl25's FACE_STANDOFF (0.030), which described a pusher resting
# on the lid's outward face. This teacher grasps instead, so the constant that replaces
# it is a pre-descent stand-off along the lid's outward normal: it must clear the knob's
# cap (which tops out 0.046 above object_site) plus the pad, i.e. > 0.06. 0.08 leaves
# 20 mm and no more — the descent runs at 0.18 rad/step and every 10 mm of hover costs
# ~3 of the 250 steps.
HOVER = 0.08
# FULL-NORM tolerances, not perpendicular-only. Measured (CPU trace, n=2): the chest now
# sits at mount z = 0.10, so the neutral start pose is ~1.0 m from the knob and the
# reach alone costs 60+ of the 150 steps. An in-plane-only align test let phase 0 exit
# at |gto| = 0.59 and the 30-step descent timeout then closed the fingers on air at
# |gto| = 0.18; the hand spent the rest of the episode pressing the closed pads DOWN on
# the knob (lid joint driven +0.24 rad past its closed stop) instead of gripping it.
ALIGN_TOL = 0.055  # |gripper -> hover point| before the descent starts
CONTACT_TOL = 0.020  # pads straddling the 40 mm knob
CLOSE_TOL = 0.035  # never close the fingers further than this from the knob
SETTLE = 3
CLOSE_STEPS = 6  # let the fingers actually close before loading the grasp
EMA_ALPHA = 0.4
# Integral gain on the descent, with anti-windup: it only integrates once the hand is
# already close (0.10 m), because integrating a 0.3 m approach error winds up ~0.09 m of
# bias and the hand then overshoots the knob by 100 mm and hunts.
INTEG_GAIN = 0.15
INTEG_BAND = 0.10
# Arc lead while lifting: a FORCE BIAS, not a reachable waypoint. This is the single
# most sensitive constant in the file — measured at n=16 on CPU, with the end-of-travel
# margin moving with it: 0.35/0.15 -> 0.000 (stalls at -1.01 rad), 0.55/0.15 -> 0.000
# (-1.10), 0.70/0.30 -> **0.125**, 0.95/0.50 -> 0.063. Too small and the pinch never
# loads enough to finish the last 20 deg; too large and it prises the knob out of the
# pads.
LEAD_ANGLE = 0.70
LOST_TOL = 0.10  # pads this far off the knob => the pinch is gone, re-seat
MAX_WAYPOINT = 0.18
PHASE_TIMEOUT = 45  # only ever fires together with the CLOSE_TOL distance test


def _rot_y(v: np.ndarray, th: float) -> np.ndarray:
  """Rotate v about +y by th."""
  c, s = np.cos(th), np.sin(th)
  return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])


class OpenLidClassicalPolicy(ClassicalPolicyBase):
  """Pinch the lifting knob and swing the lid up against gravity."""

  max_dq = 0.16
  orientation_weight = 0.3

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._pinched = np.zeros(self.num_envs, dtype=bool)
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._pinched[env_ids] = False

  def _arc_state(self, gto: np.ndarray, o2g: np.ndarray):
    """Recover (hinge_rel, theta, radius_vec, local_x, local_z) from relative obs.

    lip(theta) - lip(0) = _MARKER_OFF - o2g, because the marker is a fixed world point
    and o2g = marker - lip(theta). Given lip(0) - hinge = _R0, solving the planar
    rotation for theta is a 2x2 least-squares on the x-z components. ``theta`` comes
    out as the rotation about +y, which is +|lid_hinge| for the CL-V2 axis sign.
    """
    d = _MARKER_OFF - o2g  # lip(theta) - lip(0)
    v = _R0 + d  # lip(theta) - hinge
    x0, z0 = _R0[0], _R0[2]
    denom = x0 * x0 + z0 * z0
    c = (x0 * v[0] + z0 * v[2]) / denom
    s = (z0 * v[0] - x0 * v[2]) / denom
    theta = float(np.arctan2(s, c))
    radius_vec = _rot_y(_R0, theta)
    hinge_rel = gto - radius_vec
    local_x = _rot_y(np.array([1.0, 0.0, 0.0]), theta)
    local_z = _rot_y(np.array([0.0, 0.0, 1.0]), theta)
    return hinge_rel, theta, radius_vec, local_x, local_z

  @staticmethod
  def _grasp_rot(local_z: np.ndarray) -> np.ndarray:
    """AXIS-ONLY orientation target: align the EE approach axis with the lid's normal.

    Deliberately not a full 3x3. The hinge axis is world y, so the closing axis needs
    to stay world y for the whole lift — and a Franka top-down pose already presents
    world y as its closing axis (see open_drawer.py), so leaving the yaw FREE lands on
    the right grasp while keeping a DOF for the solver. A full frame fixes the yaw to
    one of the two equivalent signs and, when it picks the far one, asks the wrist for
    a 180 deg roll it then spends the whole episode fighting: measured 0.000, with the
    hand pressing the lid 0.16 rad past its closed stop instead of lifting.
    """
    return -local_z / (np.linalg.norm(local_z) + 1e-9)

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = obs_i[43:46]

    hinge_rel, theta, radius_vec, local_x, local_z = self._arc_state(gto, o2g)
    grasp_off = radius_vec + GRASP_UP * local_z  # hinge -> pinch centre
    grasp = hinge_rel + grasp_off  # site -> pinch centre
    target_rot = self._grasp_rot(local_z)

    ph = int(self._phase[i])
    # Sustained command lead while lifting: the hinge carries the lid's weight, and
    # commanding from the lagging actual joints caps the available servo force. The
    # approach is deliberately brisk — the reach is ~1.0 m and the lift needs the rest
    # of the budget.
    self.cmd_lead_max = 0.30 if ph == 3 else 0.0
    # Loosen the orientation term while lifting: near the -75 deg target the approach
    # axis is nearly horizontal, and a stiff orientation term there fights the arc.
    self.orientation_weight = 0.15 if ph == 3 else 0.3
    self.max_dq = 0.22 if ph == 3 else (0.34 if ph == 0 else 0.18)
    self.max_pos_err = 0.20 if ph == 0 else (0.12 if ph == 3 else 0.08)
    if ph == 0:
      gripper_a = GRIPPER_OPEN
      pos_err = grasp + HOVER * local_z
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif ph == 1:
      # Descend onto the knob. On a RE-seat keep the pads closed: the knob is only a
      # few cm away and reopening costs the whole close again.
      # A re-seat keeps the pads closed only while the knob is still within reach of
      # them; further out than that, reopen and grasp again properly.
      if self._pinched[i] and np.linalg.norm(grasp) > 0.06:
        self._pinched[i] = False
      gripper_a = GRIPPER_PINCH if self._pinched[i] else GRIPPER_OPEN
      raw = grasp
      if np.linalg.norm(raw) < INTEG_BAND:
        self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.05, 0.05)
      else:
        self._integ[i] = 0.0
      pos_err = raw + self._integ[i]
      if np.linalg.norm(raw) < CONTACT_TOL:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      timed_out = (self._phase_steps[i] > PHASE_TIMEOUT
                   and np.linalg.norm(raw) < CLOSE_TOL)
      if self._settle[i] >= SETTLE or timed_out:
        self._phase[i] = 3 if self._pinched[i] else 2
        self._phase_steps[i] = 0
    elif ph == 2:
      gripper_a = GRIPPER_PINCH
      pos_err = grasp + self._integ[i]
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
        self._pinched[i] = True
    else:
      gripper_a = GRIPPER_PINCH
      if np.linalg.norm(grasp) > LOST_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        return grasp + self._integ[i], target_rot, gripper_a
      # Ramp the arc lead in over the first few steps: a full lead applied the instant
      # the fingers stop closing jerks the knob out of a grasp that has not loaded yet.
      ramp = min(1.0, (self._phase_steps[i] + 1) / 8.0)
      lead = ramp * min(LEAD_ANGLE, max(1.308997 + 0.30 - theta, 0.0))
      lead_off = _rot_y(grasp_off, lead)
      pos_err = hinge_rel + lead_off + self._integ[i]
      # Orientation tracks the lid's CURRENT angle, not the lead: commanding the wrist
      # a third of a radian ahead of the object it is holding pries the pinch open.
      target_rot = self._grasp_rot(local_z)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, target_rot, gripper_a
