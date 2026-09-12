"""Scripted PushDisc teacher policy.

Strategy (the shepherding idea from push_cuboid.py, re-derived for the current
placements): ride the CLOSED fingers at disc height, always on the far side of
the disc from the goal, and creep toward the goal so the fingers nudge the disc
along the ground. The tuning constants below are NOT inherited from
push_cuboid.py — that policy is mid-retune against the post-workspace-fix
placements — they were measured against this task.

Geometry, all verified in-sim:
  * The ``gripper`` site's lowest collision geometry (the finger pads) sits only
    0.0135 m below the site. ``workspace.SITE_TO_FINGERTIP = 0.10`` does NOT
    apply to this site.
  * The binding constraint is the ``ee_ground_collision`` termination on the
    whole link7 subtree. Objects sit on the GROUND (no table), so the pushing
    altitude is a few cm above a kill-switch. Measured, holding station over a
    floor object for 250 steps: a DIRECT low approach trips it repeatedly, while
    hovering high, aligning in xy, and only then descending vertically to a site
    height of 0.045 trips it zero times. So the approach is staged, exactly as
    for the lift teacher, and PUSH_SITE_Z never goes below that floor.
  * The disc is a cylinder r=0.02 h=0.04, resting centre at z=0.020. At site
    z=0.05 the pads straddle the disc's upper half and push it edge-on.

The success bar here is TIGHT and it is worth being explicit about why. The goal
sits at a fixed z = 0.03 while the disc's centre rests at z = 0.020, and success
is a 3D distance under 0.02 m. The 0.01 m vertical residual is unavoidable, so
only sqrt(0.02^2 - 0.01^2) = 0.017 m of lateral budget remains. A pusher must
therefore park the disc within ~1.7cm of the goal in xy — several times tighter
than the 5cm ball the lift tasks are scored against.

Only RELATIVE observations are used (offsets cancel):
  obs[40:43] gripper_to_object, obs[43:46] object_to_goal.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0

OBJ_CENTER_Z = 0.020  # disc half-height


class PushDiscClassicalPolicy(ClassicalPolicyBase):
  """Shepherd the disc to the goal with closed fingers riding at disc height."""

  DEFAULT_QPOS = HOME_QPOS  # push_disc env uses get_franka_robot_cfg (home)
  max_dq = 0.05
  orientation_weight = 0.3  # vertical approach, yaw free

  # Site height while hovering / traversing above the disc.
  HOVER_SITE_Z = 0.20
  # Site height while pushing. 0.045 is the measured ground-collision floor;
  # 0.050 keeps a margin while still contacting the disc's upper half.
  PUSH_SITE_Z = 0.050
  # Stand this far behind the disc (opposite the goal) so the pads contact its
  # rim rather than driving through its centre.
  BEHIND = 0.045
  ALIGN_TOL = 0.020
  ALIGN_TIMEOUT = 70
  DESCENT_RATE = 0.02
  # How far past the disc to aim while pushing. Larger = faster but sloppier.
  ADVANCE = 0.05
  # Below this remaining distance, creep instead of shoving, so the disc is not
  # kicked past the 1.7cm lateral budget.
  FINE_DIST = 0.06
  FINE_ADVANCE = 0.012
  GOAL_TOL = 0.012

  EMA_ALPHA = 0.4
  RESET_JUMP = 0.12

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._prev_gto = np.zeros((self.num_envs, 3))
    else:
      self._ema_init[env_ids] = False
      self._settle[env_ids] = 0

  def _rewind(self, i: int) -> None:
    self._phase[i] = 0
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._ema_init[i] = False

  def _site_err_to(self, gto: np.ndarray, site_z: float, xy_off: np.ndarray):
    """Error putting the site at ``xy_off`` from the disc, at absolute height
    ``site_z``. Built entirely from the relative gto, so origins cancel."""
    return np.array(
      [gto[0] + xy_off[0], gto[1] + xy_off[1], gto[2] + (site_z - OBJ_CENTER_Z)]
    )

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]
    # Mid-episode auto-reset (ground collision / out-of-bounds respawns the
    # disc): rewind, or the rest of the episode is spent pushing thin air.
    if self._ema_init[i] and np.linalg.norm(raw_gto - self._prev_gto[i]) > self.RESET_JUMP:
      self._rewind(i)
    self._prev_gto[i] = raw_gto

    if not self._ema_init[i]:
      self._gto_ema[i] = raw_gto
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = (
        self.EMA_ALPHA * raw_gto + (1 - self.EMA_ALPHA) * self._gto_ema[i]
      )
    gto = self._gto_ema[i]
    o2g = obs_i[43:46]  # goal - disc

    # Unit vector from disc toward the goal, in the ground plane.
    d_xy = o2g[:2]
    dist = float(np.linalg.norm(d_xy))
    u = d_xy / (dist + 1e-8)
    behind = -u * self.BEHIND  # stand on the far side from the goal

    phase = self._phase[i]

    if phase == 0:
      # Hover high, behind the disc, and get xy right before descending. Going
      # in low trips the ground-collision termination on the swing-in.
      pos_err = self._site_err_to(gto, self.HOVER_SITE_Z, behind)
      if np.linalg.norm(pos_err[:2]) < self.ALIGN_TOL:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = 1
          self._phase_steps[i] = 0
          self._settle[i] = 0
      else:
        self._settle[i] = max(0, self._settle[i] - 1)
      if self._phase_steps[i] > self.ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
        self._settle[i] = 0
      return pos_err, _DOWN_AXIS, GRIPPER_CLOSED

    if phase == 1:
      # Straight down to push height, still behind the disc.
      pos_err = self._site_err_to(gto, self.PUSH_SITE_Z, behind)
      pos_err[2] = max(pos_err[2], -self.DESCENT_RATE)
      if abs(gto[2] + (self.PUSH_SITE_Z - OBJ_CENTER_Z)) < 0.012:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > 120:
        self._phase[i] = 2
        self._phase_steps[i] = 0
      return pos_err, _DOWN_AXIS, GRIPPER_CLOSED

    # phase 2: push. Aim THROUGH the disc toward the goal; how far through sets
    # the push speed. Back right off near the goal — the lateral budget is only
    # ~1.7cm, so a hard shove overshoots and the disc has to be chased back.
    if dist < self.GOAL_TOL:
      # Arrived: hold station just behind the disc, do not nudge it further.
      pos_err = self._site_err_to(gto, self.PUSH_SITE_Z, behind)
      return pos_err, _DOWN_AXIS, GRIPPER_CLOSED

    adv = self.FINE_ADVANCE if dist < self.FINE_DIST else self.ADVANCE
    adv = min(adv, dist)
    pos_err = self._site_err_to(gto, self.PUSH_SITE_Z, u * adv)

    # If the disc has drifted well off the push line (the pads slid past its
    # rim), lift back to the hover and re-approach from behind rather than
    # scrubbing it sideways.
    # 2D cross product, written out: numpy 2 removed the 2-vector np.cross.
    lateral = abs(float(gto[0] * u[1] - gto[1] * u[0]))
    if lateral > 0.07:
      self._rewind(i)

    return pos_err, _DOWN_AXIS, GRIPPER_CLOSED
