"""Scripted OpenDoor teacher policy.

The door is a hinged panel; its handle is a VERTICAL bar (16cm tall, 2cm
thick) protruding ~4cm toward the robot on the front face, at the edge far
from the hinge. Positive hinge motion swings the panel toward the robot;
success = hinge angle at the 90-degree target.

Strategy: approach the handle horizontally (EE z-axis pointing forward,
finger-closure axis along world y so the fingers straddle the vertical bar),
land the FINGERTIPS on the bar (the gripper site sits ~6cm behind the tips),
close, then pull along the LOCAL TANGENT of the handle's hinge arc while
rotating the grasp with the door. Pulling along the chord to the goal presses
into the hinge constraint and the door barely moves.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import ClassicalPolicyBase

# Grasp orientation: approach horizontally along +x with a slight downward
# tilt, fingers closing along world y (across the vertical bar).
_Z_COL = np.array([0.98, 0.0, -0.2])
_Z_COL = _Z_COL / np.linalg.norm(_Z_COL)
_Y_COL = np.array([0.0, 1.0, 0.0])
_X_COL = np.cross(_Y_COL, _Z_COL)
_X_COL /= np.linalg.norm(_X_COL)
_ROT_GRASP = np.column_stack([_X_COL, _Y_COL, _Z_COL])

# Door geometry (door frame, yaw 0): hinge at (0, -0.3), handle/object_site at
# (-0.04, +0.25) -> handle-to-hinge radius r0 = (-0.04, 0.55, 0).
_R0 = np.array([-0.04, 0.55, 0.0])

STANDOFF_X = 0.10  # pre-grasp standoff in front of the handle (-x side)
# Fingertips sit ~6cm beyond the gripper site along the approach axis. The
# site stops short of the handle by only 2.5cm so the TIPS REACH PAST the bar
# (2cm wide, ~3cm clearance to the panel): the bar must sit BETWEEN the
# fingers, not at their very tips, or the pull slides straight off.
TIP_VEC = 0.04 * _Z_COL
ALIGN_TOL = 0.06
ENGAGE_TOL = 0.04
ENGAGE_SETTLE = 1
INTEG_GAIN = 0.15  # integral action nulls the ~2-3cm DLS steady-state bias
EMA_ALPHA = 0.4  # smooth the noisy gto (obs noise ~±1.4cm)
CLOSE_STEPS = 3
GOAL_TOL = 0.03
PULL_STEP = 0.12
PULL_DTHETA = np.radians(14.0)  # arc-waypoint advance per control step
PULL_MODE = "arc"  # "arc" (waypoint at theta+dtheta) or "tangent"
PULL_ROT = "rotate"  # "rotate" | "fixed" | "free"
SLIP_DIST = 0.07
DRAG_SLIP_DIST = 0.13
FLING_THETA = np.radians(26.0)  # release here and let the door coast
STALL_CHECK = 8  # steps between coast-stall checks
HOOK_THETA = np.radians(6.0)  # door cracked open enough to hook the edge
# Edge-hook geometry: the panel free edge is at radius 0.60 from the hinge;
# the gripper site sits so the fingertips (6cm past the site along z_ee,
# pointing inboard -y_door) reach ~4cm past the edge.
HOOK_R = 0.64
PUSH_DTHETA = np.radians(22.0)  # arc advance of the hooked sweep waypoint
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = -1.0
GRIPPER_CAGE = -0.5  # ~4cm finger gap: loose cage around the 2cm bar


def _rot_z(theta: float) -> np.ndarray:
  c, s = np.cos(theta), np.sin(theta)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hook_rot(rot_th: np.ndarray) -> np.ndarray:
  """Hand orientation for the edge hook: fingers (z_ee) point inboard along
  -y_door so the open finger gap (closure axis = hand y) straddles the panel
  plane (x_door)."""
  z_ee = rot_th @ np.array([0.0, -1.0, 0.0])
  y_ee = rot_th @ np.array([-1.0, 0.0, 0.0])
  x_ee = np.cross(y_ee, z_ee)
  return np.column_stack([x_ee, y_ee, z_ee])


class OpenDoorClassicalPolicy(ClassicalPolicyBase):
  """Grasp the vertical door handle and pull it open along the hinge arc."""

  max_dq = 0.15  # 150-step budget: brisk approach + 0.86m arc pull

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._theta_hist = np.zeros(self.num_envs)
    else:
      self._theta_hist[env_ids] = 0.0
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0

  def _target_error(self, i: int, obs_i: np.ndarray):
    # Per-phase servo speed: brisk approach, careful pinch, aggressive hooked
    # sweep (the geometric cage tolerates speed; the pinch does not).
    ph = self._phase[i]
    self.max_dq = 0.30 if ph < 1 else (0.15 if ph < 3 else 0.30)
    # Sustained command-lead during the drag: pulling the door needs force,
    # and commanding from the lagging actual joints caps it.
    self.cmd_lead_max = 0.35 if ph >= 3 else 0.0
    # The cage-drag needs long strides and only loose orientation: a
    # full-weight 3x3 target binds the IK near wrist limits (hand sags and
    # crawls); the cage only needs approximate finger alignment.
    self.max_pos_err = 0.10 if ph < 3 else 0.20
    self.orientation_weight = 0.3 if ph < 3 else 0.12
    gto_raw = obs_i[40:43]  # handle - gripper
    o2g = obs_i[43:46]  # goal - handle
    if not self._ema_init[i]:
      self._gto_ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    gto = self._gto_ema[i]

    # Door opening angle from the chord to the goal: the goal is the handle
    # position at the 90-degree target, so |o2g| = 2 r sin((pi/2 - theta)/2)
    # with r = |handle - hinge| = 0.551. (object_quat reports the static door
    # base, not the swinging panel, so it cannot be used for theta.)
    r_len = float(np.linalg.norm(_R0))
    chord = np.clip(np.linalg.norm(o2g) / (2.0 * r_len), 0.0, 1.0)
    theta = np.pi / 2.0 - 2.0 * np.arcsin(chord)
    rot_th = _rot_z(theta)
    tip_vec = rot_th @ TIP_VEC
    # Rotate the hand at 0.6x the door angle: full tracking churns the wrist
    # (slowing the drag) and the cage tolerates ~30deg of misalignment.
    grasp_rot = _rot_z(0.6 * theta) @ _ROT_GRASP

    gripper_a = GRIPPER_OPEN
    target_rot = grasp_rot
    if self._phase[i] == 0:
      # Stand off in front of the handle at handle height, fingers open.
      # Far out, constrain only the approach axis (full 3x3 targets make the
      # IK crawl at range); enforce the full grasp orientation once close.
      pos_err = gto + rot_th @ np.array([-STANDOFF_X, 0.0, 0.0])
      if np.linalg.norm(pos_err) > 0.15:
        target_rot = rot_th @ _Z_COL
      if np.linalg.norm(pos_err) < ALIGN_TOL:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Move in until the fingertips straddle the bar; integral action nulls
      # the DLS steady-state bias.
      raw_err = gto - tip_vec
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw_err, -0.03, 0.03)
      # Fine-positioning mode near the bar: full-gain tracking of the noisy
      # EMA overshoots in y (servo limit cycle) and the close gate never
      # latches; halve the commanded error when close.
      pos_err = raw_err + self._integ[i]
      # Gate the close on the closure-critical axes: y must be inside the
      # finger gap (bar 2cm, gap 8cm) or the fingers grip nothing.
      # y is closure-critical (bar 2cm, gap 8cm); z barely matters (bar is
      # 16cm tall); x only needs the tips past the bar.
      aligned = (
        abs(raw_err[1]) < 0.025
        and abs(raw_err[2]) < 0.05
        and abs(raw_err[0]) < ENGAGE_TOL
      )
      if aligned:
        self._settle[i] += 1
        if self._settle[i] >= ENGAGE_SETTLE:
          self._phase[i] = 2
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
    elif self._phase[i] == 2:
      # Narrow the cage around the bar: semi-close to a ~4cm gap (bar is
      # 2cm). NOT a pinch — the bar stays loosely caged between the fingers,
      # which drags the door without needing fingertip friction.
      pos_err = gto - tip_vec + self._integ[i]
      gripper_a = GRIPPER_CAGE
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
    elif self._phase[i] == 3:
      # Cage-drag along the hinge arc: waypoint = where the bar will be at
      # theta + dtheta, hand orientation rotating with the door. The caged
      # bar presses against the trailing finger — geometric containment, so
      # speed does not tear it off the way it tears off a pinch.
      # The drag rate is joint-velocity-limited to ~0.7deg/step, too slow to
      # reach 90deg in-episode — so past FLING_THETA we release and let the
      # door COAST: the panel is heavy (~14kg, I~1.7 kg m^2) and the hinge
      # damping tiny (0.1), so it keeps swinging for tens of degrees.
      gripper_a = GRIPPER_CAGE
      raw_err = gto - tip_vec
      if theta >= FLING_THETA:
        self._phase[i] = 4
        self._phase_steps[i] = 0
        self._theta_hist[i] = theta
        return rot_th @ np.array([-0.22, 0.06, 0.0]), None, GRIPPER_OPEN
      # Slip check on RAW obs (the EMA lags behind during a fast drag and
      # false-triggers), with a drag-sized threshold. Recover via the
      # rotated standoff (phase 0) so the re-approach cannot press the
      # panel closed.
      if np.linalg.norm(gto_raw - tip_vec) > DRAG_SLIP_DIST:
        self._phase[i] = 0
        self._settle[i] = 0
        self._integ[i] = 0.0
        return gto + rot_th @ np.array([-STANDOFF_X, 0.0, 0.0]), grasp_rot, GRIPPER_OPEN
      anchor = raw_err
      ramp = min(1.0, (self._phase_steps[i] + 1) / 6.0)
      if np.linalg.norm(o2g) < GOAL_TOL:
        pos_err = anchor
      else:
        r_now = rot_th @ _R0
        r_next = _rot_z(theta + PULL_DTHETA * ramp) @ _R0
        pos_err = anchor + (r_next - r_now)
    else:
      # Phase 4: released — back the hand away from the swinging panel and
      # let the door coast. If it stalls short of the goal, re-engage.
      gripper_a = GRIPPER_OPEN
      pos_err = rot_th @ np.array([-0.22, 0.06, 0.0])
      target_rot = None
      if self._phase_steps[i] % STALL_CHECK == 0:
        stalled = (theta - self._theta_hist[i]) < np.radians(1.5)
        self._theta_hist[i] = theta
        if stalled and np.linalg.norm(o2g) > GOAL_TOL:
          self._phase[i] = 0
          self._settle[i] = 0
          self._integ[i] = 0.0

    return pos_err, target_rot, gripper_a
