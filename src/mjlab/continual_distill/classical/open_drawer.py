"""Scripted OpenDrawer teacher policy — top-down HOOK strategy.

Geometry (drawer.xml): the handle is a horizontal bar 16cm wide (y), 2cm tall
(z in [-0.01, 0.01]), 2cm thick (x), sitting ~1cm in front of the drawer panel
— the gap between the bar's rear face (x=-0.03) and the panel's front face
(x=-0.02) is only 1cm. The slide joint opens along -x; success = drawer pulled
to within 2cm of the -0.25m stop (i.e. >=0.23m), latched over the episode.

Strategy: NOT a side cage-grasp (the old approach, ~36%). Instead, come straight
DOWN from above with the fingers CLOSED, and drop the fingertip into the 1cm slot
behind the bar. The closed fingertip pad (~1.75cm across its closing axis) is
wider than the 1cm slot, so it wedges — a friction-free geometric hook. Pulling
-x then drives the pad's rear face into the bar's rear face (normal force, not
tangential friction), so the domain-randomized low fingertip friction (down to
0.3) is irrelevant. We deliberately bias the descent into the panel face (a large
40x20cm wall, impossible to miss) and let the tip slide down into the corner.

Only relative obs are used (offsets cancel): gto = obs[40:43] (handle - gripper),
o2g = obs[43:46] (goal - handle). Robot starts at NEUTRAL_QPOS (+-10deg reset).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Approach axis: EE z-axis points straight DOWN (-world z), so the fingers hang
# vertically and the closed fingertip drops into the slot behind the bar. Yaw is
# left free (axis-only orientation target), so the solver keeps the wrist near
# its posture default and only the approach axis is constrained.
_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

# The 'gripper' site sits ~0.4cm ABOVE the closed fingertip pads (measured from
# panda.xml with the wrist oriented top-down), so the site is essentially AT the
# tip — no tool offset needed. When the site z-axis points straight down, the
# finger-closing axis is world y (along the 16cm bar), so closed fingers
# straddle along the bar and the tip drops into the 1cm slot in x. Verified.
TIP_DROP = 0.004  # metres, gripper-site -> closed-fingertip along -z

# Lateral/depth standoffs, all in the offset-free gto frame.
HOVER_Z = 0.10  # hover this far above the bar before descending
PANEL_BIAS_X = -0.015  # bias the descent ~1.5cm into the panel -> seats in corner
# xy alignment tol before starting the descent. Loose on purpose: the bar is
# 16cm wide in y and the panel-bias makes x self-seat into the corner, so a ~5cm
# xy offset still drops the tip into the slot. A tight tol (0.03) stranded the
# arm hovering at its ~3.5cm DLS steady-state bias for the whole episode.
ALIGN_TOL = 0.055
SEATED_TOL = 0.035  # tip within this of slot depth => hooked, start pulling
EMA_ALPHA = 0.4  # smooth the noisy gto (obs noise ~+-1.4cm effective)
INTEG_GAIN = 0.25  # integral nulls the DLS steady-state bias on the descent
DESCEND_SETTLE = 2

PULL_STEP = 0.12  # aggressive -x pull; 150-step budget, 0.23m stroke
GOAL_TOL = 0.015
SEAT_OVERSHOOT = 0.04  # keep driving -x this far past goal to seat on the stop
UNHOOK_Z = 0.06  # tip risen this far above slot while pulling => popped out
GRIPPER_CLOSED = -1.0


class OpenDrawerClassicalPolicy(ClassicalPolicyBase):
  """Top-down hook: drop a closed fingertip behind the bar and pull -x."""

  max_dq = 0.15  # 150-step budget: brisk descent + 0.23m pull

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
    gto_raw = obs_i[40:43]  # handle - gripper (bar center)
    o2g = obs_i[43:46]  # goal - handle  (points -x, magnitude = remaining stroke)
    if not self._ema_init[i]:
      self._gto_ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    gto = self._gto_ema[i]

    # Vector from the gripper site to the desired fingertip seat point: the bar
    # center, biased into the panel corner, with the tip dropped TIP_DROP below
    # the site. gto already targets the bar center relative to the site; add the
    # tool offset so the *fingertip* (not the site) lands there.
    tip_offset = np.array([0.0, 0.0, -TIP_DROP])  # site -> closed tip, world frame
    seat = gto + np.array([PANEL_BIAS_X, 0.0, 0.0]) - tip_offset

    gripper_a = GRIPPER_CLOSED  # fingers closed throughout: pads act as one hook

    if self._phase[i] == 0:
      # Hover directly above the seat point, HOVER_Z up. Get xy aligned first so
      # the descent goes down a clean vertical line into the slot.
      pos_err = seat + np.array([0.0, 0.0, HOVER_Z])
      if np.linalg.norm(pos_err[:2]) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Descend into the slot. Integral action nulls the DLS steady-state bias
      # so the tip actually reaches slot depth instead of hovering short.
      raw_err = seat
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw_err, -0.06, 0.06)
      pos_err = raw_err + self._integ[i]
      if np.linalg.norm(raw_err) < SEATED_TOL:
        self._settle[i] += 1
        if self._settle[i] >= DESCEND_SETTLE:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
    else:
      # Pull -x along o2g until the slide seats on the stop. Re-descend if the
      # hook popped out of the slot (tip rose well above slot height).
      tip_err = seat  # residual site->seat error; +z means tip is too high
      if tip_err[2] > UNHOOK_Z:
        self._phase[i] = 1
        self._settle[i] = 0
        self._integ[i] = 0.0
        return seat + self._integ[i], _DOWN_AXIS, gripper_a
      dist = np.linalg.norm(o2g)
      # Hold the seat (keep the tip in the slot) while translating along o2g.
      anchor = seat + self._integ[i]
      if dist < GOAL_TOL:
        # At goal: keep driving a little further -x to seat firmly on the stop.
        pos_err = anchor + o2g / (dist + 1e-8) * SEAT_OVERSHOOT if dist > 1e-6 \
          else anchor + np.array([-SEAT_OVERSHOOT, 0.0, 0.0])
      else:
        pos_err = anchor + o2g / (dist + 1e-8) * min(dist + SEAT_OVERSHOOT, PULL_STEP)

    return pos_err, _DOWN_AXIS, gripper_a
