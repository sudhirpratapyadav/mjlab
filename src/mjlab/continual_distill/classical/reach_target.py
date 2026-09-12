"""Scripted ReachTarget teacher policy.

The simplest Class A skill: drive the ``gripper`` SITE onto a sampled 3D target.
There is no object and no grasp, so the whole policy is one servo law — feed
``gripper_to_target`` straight through as the position error and let the base
class's DLS IK do the rest.

Observation layout is the 38-D reach layout (see the teacher brief):
  [ 0: 9] robot_joint_pos (relative to HOME_QPOS)
  [ 9:18] robot_joint_vel
  [18:21] gripper_pos           <- absolute, unused
  [21:27] gripper_orientation
  [27:30] gripper_to_target     <- USE THIS (target - gripper site)
  [30:38] control_qpos_diff

Note the success metric is measured at the SITE, not the fingertips, so no tool
offset is applied. Targets are sampled in x 0.4-0.7, y +-0.25, z 0.15-0.5 with a
5cm success threshold, and success is latched over the episode.

Orientation is left unconstrained (position-only IK): far targets (x up to 0.7)
are only reachable with the wrist tilted, and any orientation objective makes
them infeasible.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

# Fingers held open. The reach task has no contact, but an open hand keeps the
# fingertips away from the ground on low targets (z=0.15).
GRIPPER_OPEN = 1.0

# Smoothing on the (noisy, +-1cm) target vector. The final approach has to hold
# inside a 5cm ball, and raw obs noise alone is ~1.7cm rms in 3D.
EMA_ALPHA = 0.35


class ReachTargetClassicalPolicy(ClassicalPolicyBase):
  """Servo the gripper site onto the reach target."""

  DEFAULT_QPOS = HOME_QPOS  # reach env uses get_franka_robot_cfg (home)
  legacy_layout = "reach"
  max_dq = 0.08
  # A pure position task: no orientation term at all, so the solver spends every
  # DOF on getting the site to the point.
  orientation_weight = 0.0
  # Small posture pull keeps the arm out of silly configurations without
  # fighting the position term at the far edge of the box.
  posture_weight = 0.003

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
    else:
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0

  def _target_error(self, i: int, obs_i: np.ndarray):
    g2t_raw = obs_i[27:30]  # target - gripper site (relative: offsets cancel)
    if not self._ema_init[i]:
      self._ema[i] = g2t_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * g2t_raw + (1 - EMA_ALPHA) * self._ema[i]
    err = self._ema[i]

    # Integral action nulls the DLS steady-state bias. Without it the solver
    # parks ~3cm short of far targets, which is inside the 5cm ball only
    # sometimes; with it the site converges to the point.
    dist = float(np.linalg.norm(err))
    if dist < 0.12:
      self._integ[i] = np.clip(self._integ[i] + 0.15 * err, -0.08, 0.08)
    else:
      self._integ[i] *= 0.9

    return err + self._integ[i], None, GRIPPER_OPEN
