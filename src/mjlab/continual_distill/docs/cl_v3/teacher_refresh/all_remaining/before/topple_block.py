"""Scripted TopBlock teacher policy -- poke an ungraspable standing block over.

TASK
----
Block half-extents 0.05 x 0.08 x 0.105 (block.xml = YCB 003_cracker_box scaled
to 100 x 160 x 210 mm; geom size verified live). EVERY full width (0.10 x,
0.16 y) exceeds the Franka's 0.08 m gripper aperture -- and its ~0.095 m of
actual open-pad gap -- so force closure is impossible by construction: this is
a POKE task, never a grasp (see CONTEXT.md sec 6, "four tasks are ungraspable
by construction"). The block spawns STANDING (local z-axis == world z at spawn,
no roll/pitch), with a uniformly random yaw, CoM at z = 0.105 (half the 0.21 m
height).

block.xml documents the intended mechanism directly: "it preferentially
topples over the +-x edges (0.05 half-extent) -- a rotation about y that
carries the BODY X-AXIS to vertical". Concretely: the two faces whose outward
normal is the block's LOCAL X axis are 0.16 x 0.21 m (the "X-normal faces");
poking one of them, above the centre of mass, rotates the block about a
horizontal axis parallel to local Y until that X-normal face is the new
"floor" -- i.e. the local X axis ends up vertical. Poking a Y-normal face
instead would rotate the WRONG body axis to vertical and never satisfy the
predicate below, regardless of tolerance.

SUCCESS (ReorientObjectCommand, symmetric_axis=True, body_axis=(1,0,0))
-------------------------------------------------------------------------
    acos(|dot(object_body_x, world_z)|) < angle_threshold   (0.35 rad, ~20 deg)
  AND
    ||object_xy - spawn_xy|| < max_drift                     (0.30 m)
(a clean topple moves the centre by half_x + half_z = 0.155 m, so the bound
still separates "fell over" from "was flung across the table".)
Either landing face counts (symmetric_axis handles the sign), so only the
*direction* chosen for the push is a design decision, not a correctness
requirement. The sign is chosen to push the block AWAY from the robot base
(positive world-x component) -- matching block.xml's own framing and the
env cfg's asymmetric spawn padding, which only budgets extra clearance on the
FAR edge of the spawn box for the ~0.09 m of topple travel (env_cfgs.py:
"_poke_far ... fallen block reaches 0.60, inside x_bounds"). Pushing toward
the robot base instead would send the fallen block toward the NEAR edge,
which the placement was never audited for.

STRATEGY
--------
1. Latch the block's local-X world direction from ``object_orientation``
   (obs[34:40] = rows 1,2 of the body rotation matrix; row 0 recovered by
   orthonormality, local X is COLUMN 0 of the reconstructed matrix) at first
   sight, then FREEZE it for the episode. Re-deriving it live once the block
   starts tumbling under the poke is a live risk, not a hypothetical one --
   CONTEXT.md sec 5 names exactly this failure shape twice (Flip-Switch's
   unsigned gate, Rotate-Valve's swept-angle deadlock): a signal that is fine
   at rest becomes actively misleading once the object is disturbed.
2. Hover above-and-behind the near X-normal face with the gripper CLOSED (a
   single rigid pushing pad, exactly like every other non-grasp push teacher
   in this module -- turn_lever, slide_window, open_drawer, open_lid -- all
   use a closed fingertip and a straight-down approach axis regardless of the
   actual push-plane orientation), then close the remaining gap onto a
   contact point 7.8 cm above the block's CoM: still under the 10.5 cm
   half-height (a real lever arm above the tipping point, with 2.7 cm of face
   left above the pad), and clear of the ground-collision guard.
3. PUNCH: once seated, latch a single FIXED target ~0.32 m further along the
   push direction (well inside the block's solid geometry -- unreachable, so
   the DLS solve keeps driving at the servo limit for the whole punch phase)
   and hold it for a fixed step budget. Deliberately NOT re-derived from the
   live object pose once contact starts: the block accelerates through the
   poke and its rot6d observation becomes a poor waypoint source exactly when
   it matters most (same reasoning as the frozen push direction, item 1).
4. Retreat up and back out of the way, then hold for the remainder of the
   episode so a still-settling block is not re-struck or shoved past the
   drift bound.

Observation layout (60 dims, ``topple_block`` task -- see
``topple_block_env_cfg.py`` for the authoritative term list):
  0:9    robot_joint_pos (relative to HOME_QPOS -- this task uses
         get_franka_robot_cfg, the HOME pose, NOT the neutral pose the
         door/drawer/button/flap group uses)
  9:18   robot_joint_vel
  18:21  object_pos      (block centre, world frame, ABSOLUTE -- not used;
         base.py: absolute terms carry a per-env scene offset that only
         cancels in relative vectors)
  21:25  object_quat
  25:28  gripper_pos     (absolute, unused)
  28:34  gripper_orientation (rot6d, unused)
  34:40  object_orientation  (rot6d) -- USED, to recover the local X axis
  40:43  gripper_to_object vector (object centre - gripper site)  -- USED
  43:46  object_to_goal vector (spawn anchor - object centre)     -- unused
  46:52  goal_orientation_diff (rot6d, unused)
  52:60  control_qpos_diff (unused)
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0

# block.xml geom size = "0.05 0.07 0.09" -- exact geometry, not a tuned constant.
# block.xml: YCB 003_cracker_box at 100 x 160 x 210 mm, collision half-extents
# 0.050 x 0.080 x 0.105. The x half-extent is unchanged from the primitive (that is
# why the box was stretched in x at all — see the asset PROVENANCE.md).
FACE_HALF_X = 0.05

# Contact height above the block's CoM (top face is now +0.105, was +0.09).
#
# Scaling the retired 0.055 by half-height alone gives 0.61 * 0.105 = 0.064, and
# that is NOT enough: the CL-V2 block is a real cracker box at 0.550 kg against the
# primitive's unphysical 0.150 kg, so the same tipping angle needs ~3.3x the torque,
# and the punch is force-limited, not geometry-limited. MEASURED at n=32:
#
#   CONTACT_Z  PUNCH_STEPS:RETREAT_STEPS   SR
#     0.064          70:20                0.844   <- pure half-height scaling
#     0.064          90:15                0.906
#     0.078          70:20                0.906   <- shipped
#     0.078          90:15                0.844   <- both together is NOT additive
#
# 0.078 is 74% of the way from the centre to the top face (against 61% before), so
# the lever arm about the tipping edge grows from 0.169 m to 0.183 m while the pad
# still lands squarely on the face (0.027 m of face left above it). Shipped as the
# single-constant change: it is the geometry constant the packaging contract says to
# re-derive, and it leaves the phase budget (align 60 + seat 50 + punch 70 +
# retreat 20 = the whole 200-step episode) untouched.
#
# Still far above the tip-rather-than-slide floor, which is
# half_x / friction = 0.05 / 0.8 = 0.0625 m ABOVE THE GROUND: 0.078 + 0.105 = 0.183 m
# clears it by 2.9x. That criterion is mass independent, so the heavier box still
# tips rather than skids.
CONTACT_Z = 0.078
STANDOFF = 0.08  # extra clearance behind the face during the hover/approach
HOVER_Z = 0.10  # extra height above the contact point during phase 0
PUNCH_DEPTH = 0.32  # fixed target this far beyond the face along push_dir

ALIGN_TOL = 0.05
CONTACT_TOL = 0.035
EMA_ALPHA = 0.4
INTEG_GAIN = 0.2
SETTLE = 2

# Watchdogs: force progress rather than stall forever short of tolerance --
# the recurring deadlock shape named in CONTEXT.md sec 5 (Rotate-Valve: a
# handoff gate keyed on swept angle could never fire once an engagement
# jammed). 200-step episode: budget hover/approach generously, still cap them.
ALIGN_TIMEOUT = 60
SEAT_TIMEOUT = 50
PUNCH_STEPS = 70  # steps spent holding the fixed punch target
RETREAT_STEPS = 20  # steps spent actively retreating before holding station

RETREAT_UP = 0.10
RETREAT_BACK = 0.10

MAX_WAYPOINT = 0.30


def _object_x_axis(obs_i: np.ndarray) -> np.ndarray:
  """Block's local X axis in world frame, from the rot6d observation.

  obs[34:40] is rows 1 and 2 of the body rotation matrix (mirrors
  ``reorient_object._object_axis``). Row 0 is recovered by orthonormality;
  the local X axis is COLUMN 0 of the reconstructed matrix.
  """
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  r1 = r1 / (np.linalg.norm(r1) + 1e-9)
  r2 = r2 / (np.linalg.norm(r2) + 1e-9)
  r0 = np.cross(r1, r2)
  return np.array([r0[0], r1[0], r2[0]])


class ToppleBlockClassicalPolicy(ClassicalPolicyBase):
  """Poke the ungraspable standing block over onto one of its long faces."""

  DEFAULT_QPOS = HOME_QPOS  # topple_block env uses get_franka_robot_cfg (home)
  max_dq = 0.12
  orientation_weight = 0.2

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._push_dir = np.zeros((self.num_envs, 3))
      self._push_dir_ok = np.zeros(self.num_envs, dtype=bool)
      self._anchor = np.zeros((self.num_envs, 3))
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._integ = np.zeros((self.num_envs, 3))
      self._punch_steps = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._push_dir_ok[env_ids] = False
      self._settle[env_ids] = 0
      self._ema_init[env_ids] = False
      self._integ[env_ids] = 0.0
      self._punch_steps[env_ids] = 0

  def _latch_push_dir(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Freeze the push direction (world xy, unit norm) at first sight."""
    if not self._push_dir_ok[i]:
      ax = _object_x_axis(obs_i)
      ax_xy = ax[:2].copy()
      n = float(np.linalg.norm(ax_xy))
      if n < 1e-6:
        ax_xy = np.array([1.0, 0.0])  # degenerate read; arbitrary fallback
      else:
        ax_xy = ax_xy / n
      if ax_xy[0] < 0.0:  # push AWAY from the robot base (+world-x)
        ax_xy = -ax_xy
      self._push_dir[i] = np.array([ax_xy[0], ax_xy[1], 0.0])
      self._push_dir_ok[i] = True
    return self._push_dir[i]

  def _target_error(self, i: int, obs_i: np.ndarray):
    push_dir = self._latch_push_dir(i, obs_i)

    gto_raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._ema[i] = gto_raw
      self._ema_init[i] = True
    else:
      self._ema[i] = EMA_ALPHA * gto_raw + (1 - EMA_ALPHA) * self._ema[i]
    gto = self._ema[i]

    # Point on the near X-normal face, CONTACT_Z above the CoM.
    face = gto - push_dir * FACE_HALF_X + np.array([0.0, 0.0, CONTACT_Z])

    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Hover behind (toward the robot) and above the face.
      pos_err = face - push_dir * STANDOFF + np.array([0.0, 0.0, HOVER_Z])
      perp = pos_err - np.dot(pos_err, push_dir) * push_dir
      if (
        np.linalg.norm(perp) < ALIGN_TOL
        or self._phase_steps[i] > ALIGN_TIMEOUT
      ):
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Close the gap onto the face; integral action nulls the DLS bias.
      raw = face
      self._integ[i] = np.clip(self._integ[i] + INTEG_GAIN * raw, -0.06, 0.06)
      pos_err = raw + self._integ[i]
      seated = np.linalg.norm(raw) < CONTACT_TOL
      if seated:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= SETTLE or self._phase_steps[i] > SEAT_TIMEOUT:
        # Latch the punch target from the CURRENT smoothed estimate, then
        # freeze it -- see module docstring, strategy item 3.
        self._anchor[i] = face + push_dir * PUNCH_DEPTH
        self._phase[i] = 2
        self._phase_steps[i] = 0
    elif self._phase[i] == 2:
      # PUNCH: drive at a fixed point well inside the block's geometry.
      pos_err = self._anchor[i]
      self._punch_steps[i] += 1
      if self._punch_steps[i] >= PUNCH_STEPS:
        self._phase[i] = 3
        self._phase_steps[i] = 0
    else:
      # Retreat up and back for a bounded number of steps, then hold station
      # so a still-settling block is not re-struck.
      if self._phase_steps[i] < RETREAT_STEPS:
        pos_err = -push_dir * RETREAT_BACK + np.array([0.0, 0.0, RETREAT_UP])
      else:
        pos_err = np.zeros(3)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)
    return pos_err, _DOWN_AXIS, gripper_a
