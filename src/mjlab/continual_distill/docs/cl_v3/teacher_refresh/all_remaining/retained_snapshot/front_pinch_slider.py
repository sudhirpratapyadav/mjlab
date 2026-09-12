"""Front grasps for sliding mechanisms with a real collidable palm/finger body.

Approach the exposed bar along its normal, close across its thin dimension, then
translate while retaining the measured grasp. No insertion into undersized gaps.
"""
from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import (
  ClassicalPolicyBase, mount_yaw_from_obs, rot_z_mat,
)

PHASE_NAMES = {0: 'APPROACH', 1: 'SEAT', 2: 'CLOSE', 3: 'TRANSPORT'}


class FrontPinchSliderPolicy(ClassicalPolicyBase):
  orientation_weight = 0.3
  posture_weight = 0.0005
  ik_joint_limit_factor = 0.9
  cmd_lead_max = 0.12
  max_dq = 0.18
  step_gain = 0.5
  max_pos_err = 0.10
  TILT = 0.0
  VERTICAL_BAR = False
  FRONT = 0.065
  SEAT_X = -0.006
  PULL_STEP = 0.025
  CLOSE_STEPS = 12
  LOST_TOL = 0.055

  def reset(self, env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      self._estimate = np.zeros((self.num_envs, 3))
      self._samples = np.zeros(self.num_envs, dtype=int)
      self._psi = np.zeros(self.num_envs)
      self._hold = np.zeros((self.num_envs, 3))
      self._integral = np.zeros((self.num_envs, 3))
      self._last_object = np.zeros((self.num_envs, 3))
    else:
      self._samples[env_ids] = 0
      self._integral[env_ids] = 0

  def _go(self, i, phase):
    self._phase[i] = phase
    self._phase_steps[i] = 0
    self._integral[i] = 0

  def _target_error(self, i, obs):
    q = self.default_qpos + obs[:9]
    ee, _ = self._fk(q)
    raw_object = ee + obs[40:43]
    psi = mount_yaw_from_obs(obs)
    if self._samples[i] == 0:
      self._estimate[i] = raw_object
      self._psi[i] = psi
    else:
      self._psi[i] += 0.25 * (psi - self._psi[i])
    rz = rot_z_mat(self._psi[i])
    if self._phase[i] == 0:
      # The bar is static before contact; average independent noisy observations.
      self._samples[i] += 1
      self._estimate[i] += (raw_object - self._estimate[i]) / self._samples[i]
    else:
      self._estimate[i] += 0.45 * (raw_object - self._estimate[i])
    gto = self._estimate[i] - ee
    angle = np.deg2rad(self.TILT)
    c, s = np.cos(angle), np.sin(angle)
    approach = np.array([c, 0, -s])
    closing = np.array([0, 1, 0]) if self.VERTICAL_BAR else np.array([s, 0, c])
    rotation = rz @ np.column_stack([np.cross(closing, approach), closing, approach])
    seat_offset = rz @ np.array([self.SEAT_X, 0, 0])
    seat = gto + seat_offset
    ph = int(self._phase[i])
    grip = 1.0 if ph < 2 else -1.0
    if ph == 0:
      error = seat - rz[:, 0] * self.FRONT
      if np.linalg.norm(error) < 0.018 or self._phase_steps[i] > 60:
        self._go(i, 1)
    elif ph == 1:
      error = seat
      self._integral[i] = np.clip(self._integral[i] + .12 * error, -.025, .025)
      error = error + self._integral[i]
      if np.linalg.norm(seat) < .012 or self._phase_steps[i] > 35:
        self._go(i, 2)
    elif ph == 2:
      error = seat + self._integral[i]
      if self._phase_steps[i] >= self.CLOSE_STEPS:
        self._hold[i] = gto
        self._go(i, 3)
    else:
      # Retain the captured offset, with a bounded translational lead.
      goal = obs[43:46]
      length = np.linalg.norm(goal)
      lead = goal * min(1., self.PULL_STEP / max(length, 1e-8))
      error = gto - self._hold[i] + lead
      if np.linalg.norm(gto - self._hold[i]) > self.LOST_TOL:
        self._samples[i] = 0
        self._go(i, 0)
    return error, rotation, grip


class DrawerFrontPinchPolicy(FrontPinchSliderPolicy):
  pass


class WindowFrontPinchPolicy(FrontPinchSliderPolicy):
  VERTICAL_BAR = True
  TILT = 55.0
