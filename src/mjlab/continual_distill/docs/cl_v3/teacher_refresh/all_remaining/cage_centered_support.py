"""Cage a cube from above, then transport it with both fingers always open.

The physical aperture must remain above 69.5 mm throughout the episode. Aligning
above the object before descending avoids sweeping the outside of an open pad
through the cube, which can close a finger despite an open actuator command.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  HOME_QPOS,
  ClassicalPolicyBase,
  lowest_hand_z,
)
from mjlab.continual_distill.classical.push_cuboid import down_frame

GRIPPER_OPEN = 1.0


class CageDragClassicalPolicy(ClassicalPolicyBase):
  """Center, lower, and translate an open cage; never squeeze the cube."""

  DEFAULT_QPOS = HOME_QPOS
  posture_weight = 0.0003
  orientation_weight = 0.5
  max_dq = 0.10
  step_gain = 0.6
  cmd_lead_max = 0.10
  gravity_compensation = True
  cmd_ref = "actual"
  cmd_ref_axes = (2,)
  ik_joint_limit_factor = 0.95
  PHASE_NAMES = {0: "ALIGN", 1: "LOWER", 2: "TRANSPORT", 3: "HOLD"}

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      n = self.num_envs
      self._object = np.full((n, 3), np.nan)
      self._goal = np.zeros((n, 3))
      self._count = np.zeros(n)
      self._yaw = np.full(n, np.nan)
      self._direction = np.zeros((n, 2))
      self._previous = np.full((n, 7), np.nan)
      self._z = np.full(n, np.nan)
      self._settle = np.zeros(n, dtype=int)
    else:
      self._object[env_ids] = np.nan
      self._count[env_ids] = 0
      self._yaw[env_ids] = np.nan
      self._previous[env_ids] = np.nan
      self._z[env_ids] = np.nan
      self._settle[env_ids] = 0

  def _target_error(self, i, obs_i):
    q = self.default_qpos + obs_i[:9]
    if (
      np.isfinite(self._previous[i]).all()
      and np.max(np.abs(q[:7] - self._previous[i])) > 0.3
    ):
      self.reset([i])
    self._previous[i] = q[:7]
    p, rotation = self._fk(q)
    raw = p + obs_i[40:43]
    goal = raw + obs_i[43:46]
    phase = self._phase[i]
    self._count[i] += 1
    alpha = 1 / self._count[i] if phase < 2 else 0.35
    if not np.isfinite(self._object[i]).all():
      self._object[i] = raw
      self._goal[i] = goal
    else:
      self._object[i] += alpha * (raw - self._object[i])
      self._goal[i] += (goal - self._goal[i]) / self._count[i]
    if np.isnan(self._yaw[i]):
      u = obs_i[43:45]
      u = u / max(np.linalg.norm(u), 1e-6)
      self._direction[i] = u
      yaw = np.arctan2(u[1], u[0]) + np.pi / 2
      current = np.arctan2(rotation[1, 0], rotation[0, 0])
      self._yaw[i] = current + (yaw - current + np.pi / 2) % np.pi - np.pi / 2
    target_rot = down_frame(self._yaw[i])
    target = self._object[i].copy()
    target[2] = 0.10
    if phase == 0:
      aligned = np.linalg.norm(target[:2] - p[:2]) < 0.012
      oriented = np.trace(rotation.T @ target_rot) > 2.98
      self._settle[i] = (
        self._settle[i] + 1 if aligned and oriented and p[2] < 0.12 else 0
      )
      if self._settle[i] >= 3:
        self._phase[i] = 1
        self._phase_steps[i] = 0
        self._z[i] = p[2]
      elif self._phase_steps[i] > 85:
        self._phase_steps[i] = 0
        self._yaw[i] = np.nan
    elif phase == 1:
      self._z[i] = max(0.033, self._z[i] - 0.004)
      target[2] = self._z[i]
      if p[2] < 0.040 and np.linalg.norm(target[:2] - p[:2]) < 0.012:
        self._phase[i] = 2
        self._phase_steps[i] = 0
      elif self._phase_steps[i] > 60:
        self._phase[i] = 0
        self._phase_steps[i] = 0
    else:
      delta = self._goal[i][:2] - self._object[i][:2]
      distance = np.linalg.norm(delta)
      # A short lead loads the trailing inner pad; the front pad bounds overshoot.
      target[:2] += delta / max(distance, 1e-6) * min(0.030, distance)
      target[2] = 0.033
      if distance < 0.020:
        self._phase[i] = 3
        target[:2] = self._object[i][:2]
      elif phase == 3:
        self._phase[i] = 2
    error = target - p
    clearance = lowest_hand_z(p, rotation, max(q[7], q[8]))
    error[2] = max(error[2], 0.005 - clearance)
    return error, target_rot, GRIPPER_OPEN
