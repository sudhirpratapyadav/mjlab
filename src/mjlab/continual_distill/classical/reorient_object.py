"""Scripted ReorientObject teacher policy -- stand a lying cylinder upright.

TASK
----
The cylinder is squat: radius 0.02, half-length 0.02 (a 4cm x 4cm billet). It spawns
LYING DOWN (rolled 90deg about x, then a uniformly random yaw), so its body z-axis is
horizontal with an arbitrary bearing, and its axis sits 0.02 above the ground.

SUCCESS (ReorientObjectCommand) -- the only ANGULAR predicate in the suite
--------------------------------------------------------------------------
    acos( dot(object_body_z, world_z) ) < angle_threshold   (0.35 rad, ~20 deg)
  AND
    ||object_xy - target_pos_xy|| < max_drift               (0.18, anti-fling)
Position is otherwise irrelevant, so this is a pure reorientation: do NOT transport the
object anywhere, every centimetre of travel spends drift budget for no gain.

WHAT WAS ACTUALLY WRONG (measured, not guessed)
-----------------------------------------------
Two earlier diagnoses of this teacher were both wrong, and it is worth recording why so
they are not re-derived a third time:

  * "The stale-anchor env bug caps the task." That bug was real and is fixed (d8e5115);
    re-measuring afterwards gave the same score, so it was never the limiter.
  * "The drift bound binds." It does not. Instrumented over a full episode, whenever the
    angular criterion was met at all the drift was 0.048-0.134, comfortably inside 0.18.

The actual limiter is ``ee_ground_collision``. Instrumented over 600 steps x 8 envs, the
old teacher took **45 ground-collision terminations, 34 of them in P_DESCEND**, at a
gripper site height of 0.012-0.021. Every one of those auto-resets the env under a state
machine that the harness never tells about it, so the arm spends the episode "lifting"
and "rotating" a cylinder it never touched. When an env did survive to the rotate phase
the strategy worked perfectly -- best axis alignment 0.99 in exactly those envs. So the
strategy was never the problem; staying off the floor long enough to run it was.

The old ``floor_min_z = 0.022`` is the bug. Measured directly on the Franka model, over
the genuinely collidable link7-subtree geoms (``hand_capsule``, ``left_finger_pad``,
``right_finger_pad`` -- every other subtree geom has contype=conaffinity=0 and is visual
only), the lowest pad sits **1.4cm below the ``gripper`` site**. A site clamped at 0.022
therefore puts pad material at 0.008, and the descent's transient overshoot closes that
in one step. The guard has to sit at 0.014 + a real margin.

STRATEGY
--------
1. Recover the cylinder's axis from ``object_orientation`` (obs[34:40] = rows 1 and 2 of
   its body rotation matrix; row 0 by orthonormality, body z-axis is column 2), so the
   approach adapts to the random spawn yaw.
2. Approach top-down with the finger-closing axis ALONG that axis, so the pads land on
   the two flat circular end faces. Rotating the wrist then carries the cylinder's axis
   with the closing axis, straight to vertical.
3. Lift a few centimetres, then rotate the wrist 90deg so the closing axis swings from
   horizontal to world +z.
4. Lower back onto nearly the same spot and release, so the drift term stays small.

A BARREL GRASP WAS TRIED AND IS WORSE -- record, so it is not re-tried
---------------------------------------------------------------------
The obvious hypothesis from "geometric engagement beats friction" is that closing across
the curved barrel (closing axis perpendicular to the cylinder axis) should cage the
object better than pinching two flat end faces in a friction-randomised grip. It was
implemented and measured head-to-head on the same 32 episode-instances: barrel grasp
0.469, end-face grasp 0.531. The end-face grasp wins, and the reason is that it is the
grasp whose geometry makes the reorientation trivial -- with the pads on the end faces
the cylinder's axis IS the closing axis, so the wrist rotation that stands it up cannot
change the grasp. Under a barrel grasp the cylinder is free to spin about its own axis
between the pads during the rotation, and a 4cm-diameter barrel offers the pads a curved
surface rather than a flat one. The friction argument is real but it is not what
dominates here; losing control of the roll is.
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

# Steps spent rotating the wrist to the upright hold, and then holding. The rotation is
# driven through the same DLS solve as everything else, so it needs time to converge;
# releasing mid-rotation drops the cylinder on its side.
ROTATE_STEPS = 55
SETTLE_STEPS = 45

# How far to lift before rotating. Enough that the swinging end clears the ground (the
# cylinder's half-length is 0.02) without spending drift budget on lateral motion.
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

  gto_slice = slice(40, 43)
  o2g_slice = slice(43, 46)

  hover_height = 0.12
  align_tol = 0.020
  descend_tol = 0.012
  close_steps = 14

  # A phase-dependent rate limit (brisk while high, gentle while low) was tried here on
  # the theory that it would cut the ground collisions further, the way it fixed
  # push_button. It does the opposite: FAST_DQ = 0.14 above z = 0.10 measured 0.188
  # against 0.531 for this flat 0.08 on the same 32 episode-instances. This task has no
  # travel leg worth accelerating -- the cylinder is reoriented in place, so the whole
  # episode is "acting" -- and the larger stride simply throws the wrist into the floor
  # on entry to the descent. Keep it flat.
  max_dq = 0.08

  # See the module docstring: the lowest COLLIDABLE link7-subtree geom is 1.4cm below
  # the ``gripper`` site (measured), so the base class's 0.022 leaves pad material at
  # 0.008 and the descent's overshoot trips ``ee_ground_collision``. This keeps 1.6cm of
  # pad clearance, which is what the collision count actually responds to. Measured
  # head-to-head: 0.030 -> 0.531, 0.034 -> 0.500, and the old 0.022 -> 0.125. The
  # collision count over 2 episodes x 16 envs falls from the old teacher's rate to 304
  # at 0.030 and 278 at 0.034, so pushing the guard higher keeps trading fewer
  # collisions for a shorter reach; 0.030 is the measured optimum of that trade.
  floor_min_z = 0.030

  # The lying cylinder's axis is 0.02 above the ground and the lowest pad is 1.4cm below
  # the ``gripper`` site, so grasping at the axis would need the site at 0.02 -- below
  # the guard. Grasp 8mm high instead: the flat end faces are 4cm across, so the pads
  # still land well inside them.
  grasp_z_offset = 0.008

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._yaw = np.zeros(self.num_envs)
      self._yaw_ok = np.zeros(self.num_envs, dtype=bool)
    else:
      self._yaw_ok[env_ids] = False

  def _on_reset(self, i: int) -> None:
    # The cylinder respawns with a fresh random yaw, so the latched grasp bearing from
    # the previous episode is meaningless.
    self._yaw_ok[i] = False

  def _grasp_bearing(self, i: int, obs_i: np.ndarray) -> float:
    """Latched world-frame bearing of the cylinder's (horizontal) axis.

    Latched at first sight and then frozen: once the fingers touch, the cylinder can
    roll a little, and chasing a rotating target with the wrist shoves it away.
    """
    if not self._yaw_ok[i]:
      ax = _object_axis(obs_i)
      yaw = float(np.arctan2(ax[1], ax[0]))
      # The axis is a LINE, so +ax and -ax describe the same grasp and either may be
      # chosen. Folding into (-pi/2, pi/2] keeps the wrist working in front of the base
      # rather than reaching around, where the IK cannot place the upright hold.
      if yaw > np.pi / 2:
        yaw -= np.pi
      elif yaw <= -np.pi / 2:
        yaw += np.pi
      self._yaw[i] = yaw
      self._yaw_ok[i] = True
    return float(self._yaw[i])

  def _approach_rot(self, i: int, obs_i: np.ndarray):
    # Closing axis ACROSS the barrel -> pads on the curved side (see docstring).
    return down_frame(self._grasp_bearing(i, obs_i))

  def _upright_rot(self, i: int) -> np.ndarray:
    """EE frame with the CLOSING axis pointing along world +z.

    The grasp puts the pads on the cylinder's two flat end faces, so the cylinder's axis
    IS the closing axis. Holding the wrist in this frame therefore puts that axis
    vertical, which is exactly the success condition. The approach axis is swung into
    the horizontal plane pointing back along the grasp bearing, which keeps the elbow
    away from the floor.
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
      # Hold altitude and rotate the wrist to the upright frame. Position error is only
      # the small residual lift; all the work is in the orientation term.
      if self._phase_steps[i] >= ROTATE_STEPS:
        self._phase[i] = P_DONE
        self._phase_steps[i] = 0
      return np.array([0.0, 0.0, 0.02]), self._upright_rot(i), GRIPPER_CLOSED

    if ph == P_DONE:
      # Lower a little, open, and hold. Success latches any time the cylinder is within
      # the angular threshold, so the settle window is generous.
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
