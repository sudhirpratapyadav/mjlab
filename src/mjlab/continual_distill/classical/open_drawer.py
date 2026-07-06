"""Scripted OpenDrawer teacher policy.

Drawer front faces the robot (-x); the handle is a horizontal bar (16cm wide,
2cm thick) ~3cm in front of the drawer panel at z=0.5. The slide joint opens
along -x. Strategy: approach horizontally with the finger-closure axis along
world z (fingers straddle the bar from above/below — a top-down grasp is
infeasible because the rear finger hits the panel), grasp, then pull along
object_to_goal. If the grip slips, reopen and re-engage.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Approach along +x with slight downward tilt; fingers close along world z.
_Z_COL = np.array([0.97, 0.0, -0.243])
_Z_COL = _Z_COL / np.linalg.norm(_Z_COL)
_Y_COL_RAW = np.array([0.0, 0.0, 1.0])
_X_COL = np.cross(_Y_COL_RAW, _Z_COL)
_X_COL /= np.linalg.norm(_X_COL)
_Y_COL = np.cross(_Z_COL, _X_COL)
_ROT_GRASP = np.column_stack([_X_COL, _Y_COL, _Z_COL])

STANDOFF_X = 0.12
# Fingertips sit ~6cm beyond the gripper site along the approach axis; the
# site stops short of the handle by this vector so the tips land ON the bar.
TIP_VEC = 0.06 * _Z_COL
ALIGN_TOL = 0.06
ENGAGE_TOL = 0.03
ENGAGE_SETTLE = 2
INTEG_GAIN = 0.3  # integral action nulls the ~2-3cm DLS steady-state bias
EMA_ALPHA = 0.4  # smooth the noisy gto (obs noise ~±1.4cm)
CLOSE_STEPS = 8
GOAL_TOL = 0.02
PULL_STEP = 0.10
SLIP_DIST = 0.07  # handle this far from the fingertips while pulling => slipped
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.0


class OpenDrawerClassicalPolicy(ClassicalPolicyBase):
  """Grasp the drawer bar and pull it open along object_to_goal."""

  max_dq = 0.15  # 150-step budget: brisk approach + 0.23m pull

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
    else:
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]  # handle - gripper
    o2g = obs_i[43:46]  # goal - handle
    if not self._ema_init[i]:
      self._gto_ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    gto = self._gto_ema[i]

    gripper_a = GRIPPER_OPEN
    if self._phase[i] == 0:
      # Stand off in front of the handle at handle height, fingers open.
      pos_err = gto + np.array([-STANDOFF_X, 0.0, 0.0])
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Move in until the fingertips reach the bar; integral action nulls
      # the DLS steady-state bias that otherwise leaves the grasp off target.
      raw_err = gto - TIP_VEC
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw_err, -0.06, 0.06)
      pos_err = raw_err + self._integ[i]
      if np.linalg.norm(raw_err) < ENGAGE_TOL:
        self._settle[i] += 1
        if self._settle[i] >= ENGAGE_SETTLE:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
    elif self._phase[i] == 2:
      # Hold (keeping the integral correction) and close on the bar.
      pos_err = gto - TIP_VEC + self._integ[i]
      gripper_a = GRIPPER_CLOSED
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
    else:
      # Pull along the goal direction; re-engage if the grip slipped.
      gripper_a = GRIPPER_CLOSED
      raw_err = gto - TIP_VEC
      if np.linalg.norm(raw_err) > SLIP_DIST:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        return raw_err, _ROT_GRASP, GRIPPER_OPEN
      dist = np.linalg.norm(o2g)
      anchor = raw_err + self._integ[i]
      if dist < GOAL_TOL:
        pos_err = anchor
      else:
        pos_err = anchor + o2g / (dist + 1e-8) * min(dist, PULL_STEP)

    return pos_err, _ROT_GRASP, gripper_a
