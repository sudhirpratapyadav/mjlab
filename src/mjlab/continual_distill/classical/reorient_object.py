"""Scripted ReorientObject teacher policy -- stand a lying cylinder upright.

TASK
----
The cylinder is squat: radius 0.02, half-length 0.02 (a 4cm x 4cm puck-like billet).
It spawns LYING DOWN (rolled 90deg about x, then a uniformly random yaw), so its body
z-axis is horizontal with an arbitrary bearing.

SUCCESS (ReorientObjectCommand) -- the only ANGULAR predicate in the suite
--------------------------------------------------------------------------
    acos( dot(object_body_z, world_z) ) < 0.35 rad   (~20 deg)
  AND
    ||object_xy - target_pos_xy|| < 0.18             (anti-fling drift bound)
Position is otherwise irrelevant. So this is a pure wrist-rotation task -- do NOT
transport the object anywhere; every centimetre of travel spends drift budget for no
gain.

STRATEGY
--------
1. Recover the cylinder's axis direction from the observation. ``object_orientation``
   (obs[34:40]) is ``mat.flatten()[3:]``, i.e. ROWS 1 and 2 of the body rotation
   matrix -- row 0 is missing but recoverable as ``cross(row1, row2)`` because the
   matrix is orthonormal. The body z-axis is then column 2.
2. Approach top-down with the finger-closing axis (EE x) aligned with that horizontal
   cylinder axis, so the pads land on the two FLAT circular end faces. That is the
   only grasp that gives control of the roll: gripping the curved side lets the
   cylinder spin freely between the pads.
3. Close, lift a few centimetres, then rotate the EE 90deg so the closing axis swings
   from horizontal to VERTICAL. The cylinder is rigidly held between the pads, so its
   axis follows the closing axis and ends up along world +z -- upright.
4. Lower back onto (nearly) the same spot and release, so the drift term stays small.

KNOWN ENVIRONMENT BUG THAT CAPS THIS TASK
-----------------------------------------
``ReorientObjectCommand._resample_command`` sets ``target_pos`` by reading
``self._object_pos()`` immediately after ``_spawn_object`` writes the new pose to the
sim. Forward kinematics has not re-run at that point, so ``site_pos_w`` still holds
the PREVIOUS episode's cylinder position. The drift bound is therefore measured from
a stale point that is a random 0.05-0.35m away from where the cylinder actually is
(measured: 0.054, 0.111, 0.206, 0.333 across four envs on one reset). In the envs
where that stale offset already exceeds ``max_drift`` = 0.18, success is unreachable
no matter how perfectly the cylinder is stood up. The identical bug is documented and
worked around analytically in ``PlaceInContainerCommand``; this command was not
given the same fix. This file cannot fix it (commands.py is owned elsewhere), so the
policy simply minimises its own drift and accepts the cap.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.stack_object import (
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  P_CARRY,
  P_DONE,
  P_LIFT,
  GraspTransportPolicy,
  down_frame,
)

# Steps spent rotating the wrist from the top-down grasp to the upright hold, and
# then holding. The rotation is driven through the same DLS solve as everything else,
# so it needs time to converge; releasing mid-rotation drops the cylinder on its side.
ROTATE_STEPS = 55
SETTLE_STEPS = 45

# How far to lift before rotating. Enough that the swinging cylinder end clears the
# ground (its half-length is 0.02) without wasting drift budget on lateral motion.
ROTATE_LIFT = 0.10


def _object_axis(obs_i: np.ndarray) -> np.ndarray:
  """Cylinder body z-axis in world frame, from the rot6d observation.

  obs[34:40] is rows 1 and 2 of the body rotation matrix (``mat.flatten()[3:]``).
  Row 0 is recovered by orthonormality; the body z-axis is column 2.
  """
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  r1 = r1 / (np.linalg.norm(r1) + 1e-9)
  r2 = r2 / (np.linalg.norm(r2) + 1e-9)
  r0 = np.cross(r1, r2)
  mat = np.stack([r0, r1, r2], axis=0)
  return mat[:, 2]


class ReorientObjectClassicalPolicy(GraspTransportPolicy):
  """Grasp the lying cylinder across its flat ends, then roll the wrist upright."""

  gto_slice = slice(40, 43)  # 60-D layout
  o2g_slice = slice(43, 46)

  hover_height = 0.12
  align_tol = 0.020
  descend_tol = 0.012
  close_steps = 14
  max_dq = 0.08
  # The lying cylinder's axis is 2cm above the ground (its radius) and the lowest
  # fingertip geom is 1.24cm below the ``gripper`` site, so grasping at the axis
  # leaves only 8mm before ``ee_ground_collision`` terminates the episode. Grasp 8mm
  # high: the flat end faces are 4cm across, so the pads still land well inside them.
  grasp_z_offset = 0.008

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._yaw = np.zeros(self.num_envs)
      self._yaw_ok = np.zeros(self.num_envs, dtype=bool)
    else:
      self._yaw_ok[env_ids] = False

  def _on_reset(self, i: int) -> None:
    # The cylinder respawns with a fresh random yaw, so the latched grasp bearing
    # from the previous episode is meaningless.
    self._yaw_ok[i] = False

  def _grasp_bearing(self, i: int, obs_i: np.ndarray) -> float:
    """Latched world-frame bearing of the cylinder's (horizontal) axis.

    Latched at first sight and then frozen: once the fingers touch, the cylinder can
    roll a little, and chasing a rotating target with the wrist shoves it away.
    """
    if not self._yaw_ok[i]:
      ax = _object_axis(obs_i)
      yaw = float(np.arctan2(ax[1], ax[0]))
      # The cylinder's axis is a LINE, so +ax and -ax describe the same grasp and we
      # are free to pick either. The choice matters for the upright hold: that pose
      # puts the approach axis at (-cos yaw, -sin yaw, 0), and with yaw near +-pi the
      # wrist has to reach around and point back OUT along +x, which the IK cannot
      # place -- measured position error 0.31m at yaw=+-pi versus <0.03m for
      # |yaw| < pi/2, and 0.31m of drift blows the 0.18 bound on its own.
      # Folding yaw into (-pi/2, pi/2] keeps the approach axis pointing back toward
      # the robot base, where the arm is comfortable, for every spawn orientation.
      if yaw > np.pi / 2:
        yaw -= np.pi
      elif yaw <= -np.pi / 2:
        yaw += np.pi
      self._yaw[i] = yaw
      self._yaw_ok[i] = True
    return float(self._yaw[i])

  def _approach_rot(self, i: int, obs_i: np.ndarray):
    # Closing axis along the cylinder's axis -> pads on the flat end faces.
    return down_frame(self._grasp_bearing(i, obs_i))

  def _upright_rot(self, i: int) -> np.ndarray:
    """EE frame with the CLOSING axis pointing along world +z.

    Holding the cylinder in this frame puts its axis vertical, which is exactly the
    success condition. The approach axis is swung into the horizontal plane, pointing
    back toward the robot's grasp bearing, which keeps the elbow away from the floor.
    """
    yaw = float(self._yaw[i])
    ex = np.array([0.0, 0.0, 1.0])  # closing axis -> world up
    ez = np.array([-np.cos(yaw), -np.sin(yaw), 0.0])  # approach axis, horizontal
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez])

  def _plan(self, i: int, obs_i: np.ndarray):
    """Replace the spine's carry/place tail with a rotate-in-place tail."""
    self._detect_reset(i, obs_i)  # branches below may not reach super()
    ph = self._phase[i]

    if ph == P_CARRY:
      # Hold altitude and rotate the wrist to the upright frame. Position error is
      # only the small residual lift; all the work is in the orientation term.
      if self._phase_steps[i] >= ROTATE_STEPS:
        self._phase[i] = P_DONE
        self._phase_steps[i] = 0
      return (
        np.array([0.0, 0.0, 0.02]),
        self._upright_rot(i),
        GRIPPER_CLOSED,
      )

    if ph == P_DONE:
      # Lower a little, open, and hold. Success can latch any time the cylinder is
      # within 20deg of upright, so the settle window is generous.
      s = self._phase_steps[i]
      if s < 18:
        return np.array([0.0, 0.0, -0.04]), self._upright_rot(i), GRIPPER_CLOSED
      if s < 18 + SETTLE_STEPS:
        return np.zeros(3), self._upright_rot(i), GRIPPER_OPEN
      # Retreat straight up, staying clear of the standing cylinder.
      return np.array([0.0, 0.0, 0.08]), self._upright_rot(i), GRIPPER_OPEN

    if ph == P_LIFT:  # use the shorter rotate-lift, not the transport lift
      if self._phase_steps[i] >= 20:
        self._phase[i] = P_CARRY
        self._phase_steps[i] = 0
      return (
        np.array([0.0, 0.0, ROTATE_LIFT]),
        self._approach_rot(i, obs_i),
        GRIPPER_CLOSED,
      )

    return super()._plan(i, obs_i)
