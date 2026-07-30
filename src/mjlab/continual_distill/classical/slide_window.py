"""Scripted SlideWindow teacher — lateral FACE-PUSH on the vertical grab bar.

Geometry (window.xml, verified live): a slide joint along world +y, range
[0, 0.25], target 0.22 m (threshold 0.03 m, shortfall-only so overshoot counts).
The handle is a VERTICAL bar (half-extents 0.012 x 0.012 x 0.07) on the pane's
leading edge at body-local (-0.04, -0.10, 0). The mount has no yaw randomisation,
so the slide axis is exactly world +y. Measured: handle site at z = 0.25, mount
x in [0.50, 0.70], y in [-0.1, 0.1].

Strategy — a PUSH, not a grasp. Domain randomisation drives fingertip friction to
0.3, so nothing here may depend on grip. The bar is 14 cm tall, which makes it a
generous vertical target: we bring the closed gripper to the bar's -y face and
drive it +y for the full 0.22 m stroke. The contact is pure face-to-face normal
force along the direction of travel, so friction is irrelevant — the same
principle that makes open_drawer's hook robust, applied side-on.

Approach detail: the standoff point sits on the -y side of the bar, and the
descent to it comes from ABOVE and from -y, so the pad never has to pass through
the pane. Alignment in x is what matters most (the bar is only 2.4 cm thick in x);
z is forgiving over the bar's 14 cm height.

Only relative terms are used: gto = obs[40:43] (handle - gripper), o2g =
obs[43:46] (goal - handle, ~ +0.22 in y). Per-env origin offsets cancel.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# EE z-axis straight down: fingers hang vertically alongside the vertical bar, and
# the closed pad's broad face points laterally — exactly the pushing face we want.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

GRIPPER_CLOSED = -1.0

# Standoff from the bar centre along -y to the pad's pushing face: bar half-width
# 0.012 plus the closed pad's half-thickness, plus a little clearance.
PUSH_STANDOFF = 0.028
# Contact a bit BELOW the site: the site is the bar's mid-height, and biasing down
# keeps the wrist clear of the pane's top edge while staying well inside the 14 cm bar.
CONTACT_Z = -0.015
HOVER_Y = 0.10  # stand off this much further in -y during the approach
HOVER_Z = 0.06
ALIGN_TOL = 0.045  # x-z alignment before closing the last of the -y gap
CONTACT_TOL = 0.035
EMA_ALPHA = 0.4
INTEG_GAIN = 0.22
SETTLE = 2
PUSH_STEP = 0.12  # aggressive +y stride; 150 steps for a 0.22 m stroke
SEAT_OVERSHOOT = 0.04  # keep driving past the goal to seat on the open stop
GOAL_TOL = 0.015
LOST_TOL = 0.13  # pad this far off the bar face => re-seat


class SlideWindowClassicalPolicy(ClassicalPolicyBase):
  """Push the vertical grab bar side-on along +y with a closed gripper."""

  max_dq = 0.15

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

  def _target_error(self, i: int, obs_i: np.ndarray):
    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]
    o2g = obs_i[43:46]

    # Where the gripper site must sit to have its pad on the bar's -y face.
    seat = gto + np.array([0.0, -PUSH_STANDOFF, CONTACT_Z])

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Stand off in -y and above, then align in x and z.
      pos_err = seat + np.array([0.0, -HOVER_Y, HOVER_Z])
      if np.linalg.norm(pos_err[[0, 2]]) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Close the remaining -y gap and drop to contact height. Integral action
      # nulls the DLS steady-state bias so the pad actually reaches the face.
      raw = seat
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
      # Push +y. Hold the seat (so the pad stays on the face) while translating
      # along o2g, which shrinks as the pane slides.
      if np.linalg.norm(seat) > LOST_TOL:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        return seat, _DOWN_AXIS, gripper_a
      anchor = seat + self._integ[i]
      dist = float(np.linalg.norm(o2g))
      if dist < GOAL_TOL:
        pos_err = anchor + np.array([0.0, SEAT_OVERSHOOT, 0.0])
      else:
        pos_err = anchor + o2g / (dist + 1e-8) * min(dist + SEAT_OVERSHOOT, PUSH_STEP)

    return pos_err, _DOWN_AXIS, gripper_a
