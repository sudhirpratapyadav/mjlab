"""Scripted PivotLift teacher policy -- pivot-against-wall grasp (Lane D, D-2).

TASK
----
board.xml: a scanned wooden chopping board 0.12 x 0.219 x 0.0201 (half-extents
0.06, 0.109, 0.01004), mass 0.17 kg, friction 0.7. It spawns flat (z=0.0101,
near-zero yaw within +-0.15 rad) at x
0.38-0.41, in front of a static mocap brick wall whose inner face sits at x=0.508
(``PivotLiftCommandCfg.wall_spawn_range``, written PER-ENV every resample --
read it from the relative observation terms, never the MJCF). Both in-plane
widths exceed the 0.08 m gripper aperture and the 20 mm thickness lies flush
with the ground, so the board is ungraspable where it starts -- CONTEXT.md sec
6 and the Wave-1 design note ("push the ungraspable flat board INTO the wall to
pivot it up, then pinch and lift"). The goal is airborne (z 0.15-0.30,
LiftingCommand semantics unchanged), so sliding alone can never satisfy it.

SUCCESS (PivotLiftCommand = LiftingCommand, unmodified)
---------------------------------------------------------
    ||board_pos - goal_pos|| < 0.05, latched over the 300-step / 6.0 s episode.

THE PIVOT MECHANISM (measured, not assumed -- see LOGS.md 2026-09-09 D-2)
---------------------------------------------------------------------------
Pushing the board's TOP-REAR face (a point near the board's own top surface,
behind its centre, along +x) drives its leading (+x) edge into the wall. Once
that edge is blocked, continued horizontal force applied ABOVE the ground
creates a torque about the wall/floor contact line (a horizontal axis along
world Y) that lifts the TRAILING (robot-side) edge up and forward -- the same
"push above the pivot line" mechanism topple_block.py already uses to tip its
block, but here directed by a WALL rather than the object's own tipping point,
and the moment arm is much smaller (board top is only 2 cm above the ground,
vs. topple's whole standing block). At full rotation the board's ORIGINAL
length axis (local X, half 0.06) becomes vertical and its ORIGINAL thickness
axis (local Z) becomes roughly horizontal, leaning back against the wall --
i.e. the exposed 20 mm cross-section is now available for a TOP-DOWN pinch
at any height along the board's now-standing length, with the gripper's
default (yaw-free, straight-down) orientation, because the thickness axis
ends up aligned with world X (the push direction), not Y.

Progress is read from ``object_orientation`` (rows 1-2 of the body rotation
matrix). Row 0 is recovered by orthonormality (row0 = row1 x row2, exact for
a proper rotation matrix -- same trick topple_block.py uses in COLUMN space).
The board's local-Z axis in world frame is then COLUMN 2 of the reconstructed
matrix: ``cos_tilt = local_z_world[2]`` starts near +1 (flat) and falls toward
0 (standing on edge) and below (leaning past vertical, against the wall).

STRATEGY -- push/pivot, clear, re-approach, pinch, lift
-----------------------------------------------------------
  0 HOVER    align above-and-behind the board centre (fixed push_dir=+x: the
             wall is always further +x than the board, by construction).
  1 SEAT     close the gap onto the board's top-rear contact point.
  2 PIVOT    fixed anchor far along +x (through the board and wall --
             unreachable, so the DLS solve drives at the servo limit, same
             "punch" mechanism as topple_block), held until ``cos_tilt`` drops
             below ``PIVOT_DONE_COS`` or a step budget expires.
  3 CLEAR    lift straight up and back, open fingers, well clear of the
             leaned board -- reorienting or descending here risks knocking
             a marginally-standing board back down.
  4 RE_HOVER top-down above the board's CURRENT centre (which has both risen
             and moved toward the wall), fingers open.
  5 DESCEND  straight down to a pinch height along the standing board.
  6 CLOSE    squeeze.
  7 LIFT     climb clear, then drive ``object_to_goal`` with the fingers shut
             (mirrors lift_object.py's own phase 3, including its grasp-miss
             recovery).

Observations: 60-D layout, matches lift_object/topple_block.
  40:43 gripper_to_object (board - gripper), 43:46 object_to_goal
  (goal - board), 34:40 object_orientation (rows 1-2 of the body rotation
  matrix).

THE BLOCKING MECHANISM: the push-approach point falls in the arm's documented
"folds back over its own base" dead zone (measured, see LOGS.md 2026-09-09 D-2)
------------------------------------------------------------------------------
Three real mechanism bugs were found and fixed in the HOVER/SEAT approach
(bad contact height missing the board entirely, a too-small standoff landing
the contact target INSIDE the board's own volume, and an align-check that
mixed a z offset into what should have been an xy-only test -- all documented
at their constants above). None of the three fixes got the arm to actually
close the gap. Instrumented per-step (``gripper_to_object`` printed every
control step for env 0, several configurations): x converges partway then
PLATEAUS 10-17cm short of the seat target and never recovers, in HOVER, SEAT
and PIVOT alike, regardless of orientation constraint (MEASURED AND REVERTED:
setting ``orientation_weight = 0.0``, i.e. fully free wrist orientation, to
rule out an orientation-vs-position conflict -- reproduced the same ~10-15cm
plateau, so orientation is not the limiting factor).

The reason: the required seat point is ``board_x - STANDOFF`` with
``STANDOFF = BOARD_HALF_X + PAD_RADIUS + clearance ~= 0.076``. The board spawns
at x 0.32-0.38 (``PivotLiftCommandCfg.object_pose_range``, itself constrained
tightly against the wall band -- see the 2026-09-02 placement audit note in
``CLASS_A_WAVE1.md``), so the seat point's x falls in 0.244-0.304 -- BELOW or
right at ``workspace.GRASP_RADIAL_MIN`` (0.28), which ``workspace.py`` documents
explicitly: "objects closer than this fold the arm back over its own base". An
isolated FK/DLS probe (fresh HOME_QPOS start, no policy/physics, this file's
own IK code) DOES converge on these exact targets to a normal ~3cm residual --
so the kinematics are not impossible in the abstract. But the REAL rollout's
IK re-solves every control step from the ARM'S CURRENT PHYSICAL JOINT STATE
(base.py: ``q_goal = q_abs.copy()``, not a cached solver state), and once the
arm has moved through its own HOVER descent into a posture near this dead
zone, subsequent solves warm-started from that same posture do not escape it
-- the same "posture is a local-minimum trap" phenomenon this package's own
``axial_extract.py`` documents (there, fixed by biasing the posture-
regularization target; not attempted here given the time-box, since a fix
would need to fight a genuine placement conflict, not a bad constant: the
board's spawn band is ITSELF constrained to sit just in front of the wall
band at x=0.485 -- see ``PivotLiftCommandCfg.wall_spawn_range`` -- leaving no
room to move the board spawn further from the base without either invalidating
the wall-clearance placement audit or shrinking push travel to near zero).

Net effect: for roughly the lower two-thirds of the board's spawn x-range, the
"stand behind the board and push" approach point that any single-rigid-pad
pusher needs is not reliably reachable by this DLS/servo stack, so contact
with the board -- the precondition for the push/pivot torque this file's
strategy section describes -- is rarely made at all. This is upstream of, and
independent from, the pivot-torque question itself.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

# board.xml (exact geometry, not tuned): half-extents 0.06 (x) x 0.05 (y) x
# 0.010 (z). Resting centre z=0.0101 (from the env cfg's object_pose_range).
# CL-V2 (W1-c): the board is now a scanned 219 x 120 x 20 mm wooden chopping board,
# yawed so its LONG axis lies across the push direction -- the x half-extent and the
# thickness are unchanged from the primitive, so all three of these are unchanged
# except for the 0.1 mm of half-thickness the scan actually measures.
BOARD_HALF_X = 0.06
BOARD_HALF_Z = 0.01004
BOARD_CENTER_Z = 0.0101

PUSH_DIR = np.array([1.0, 0.0, 0.0])  # wall is always further +x than the board

# -- phase 0/1: hover + seat -------------------------------------------------
# Contact point, as an offset above the board's CENTRE.
#
# MEASURED AND REVERTED: 0.009 (1mm below the board's own top face, chosen to
# maximise the torque arm about the wall/floor pivot line -- see module
# docstring) NEVER MADE CONTACT AT ALL. Instrumented (see LOGS.md 2026-09-09
# D-2): raw object x-displacement stayed within +-1.5cm (pure observation
# noise, Unoise +-0.01 on object_pos) for the full 90-step PIVOT phase, with
# no directional trend whatsoever -- not the "pushes a little then friction-
# stalls" signature seen elsewhere in this package (Drag-Pull, Push-Cuboid),
# but a flat line from step 1. Same mechanism as Drag-Pull's own documented
# RIDE_HEIGHT bug (grazed the box's top edge): the DLS solve's ordinary
# ~1-2cm steady-state bias has zero margin against a target only 1mm below a
# 20mm-thick board's top face, so the pad hovers just clear of it every
# episode. Moved to the board's CENTRE height (0mm offset) -- matching
# push_cuboid's own contact convention -- to guarantee contact first; this
# sacrifices torque-arm length (see PIVOT_DONE_COS's docstring for what that
# costs).
CONTACT_Z_ABOVE_CENTER = 0.0
# MEASURED AND FIXED: this was 0.02 alone, i.e. the "behind the board" contact
# target sat only 2cm behind the board's CENTRE -- INSIDE its own 0.06 half-
# length, not behind its rear (near-robot) face at all. Instrumented (see
# LOGS.md 2026-09-09 D-2): raw object x-displacement stayed within the +-1.5cm
# observation-noise band for the entire 90-step PIVOT phase with zero net
# trend, in BOTH the original top-rear contact height and the corrected
# board-centre height -- ruling out a height bug and pointing at the contact
# geometry itself. Every other push teacher in this package (push_cuboid,
# topple_block, edge_grasp, drag_pull) uses ``half_extent + pad_radius +
# clearance`` for this distance; this file skipped the half-extent term.
PAD_RADIUS = 0.010
STANDOFF_CLEARANCE = 0.006
STANDOFF = BOARD_HALF_X + PAD_RADIUS + STANDOFF_CLEARANCE  # ~0.076
HOVER_Z = 0.14
ALIGN_TOL = 0.05
CONTACT_TOL = 0.03
ALIGN_TIMEOUT = 40
SEAT_TIMEOUT = 30

# -- phase 2: pivot -----------------------------------------------------------
PIVOT_DEPTH = 0.55  # fixed anchor this far along +x -- through board and wall
PIVOT_STEPS_MAX = 90
# cos(tilt) of the board's local-Z axis vs world-Z: 1.0 = flat, 0.0 = standing
# on edge, negative = leaned past vertical (against the wall). Stop pushing
# once clearly past the standing point -- continuing risks shoving the
# now-vertical board over the top of the wall instead of leaving it leaning.
PIVOT_DONE_COS = 0.15

# -- phase 3: clear -----------------------------------------------------------
CLEAR_STEPS = 20
CLEAR_UP = 0.16
CLEAR_BACK = 0.14

# -- phase 4/5: re-approach top-down ------------------------------------------
RE_HOVER_Z = 0.22
RE_ALIGN_TOL = 0.020
RE_ALIGN_TIMEOUT = 60
# Pinch height: a modest distance above the board's (now higher) centre, so
# the fingers close on standing material rather than right at the ground/wall
# corner where the board is hardest to reach cleanly.
PINCH_Z_ABOVE_CENTER = 0.02
DESCENT_RATE = 0.02
SEAT_TOL = 0.015
CLOSE_STEPS = 15
HELD_TOL = 0.07

# -- phase 7: lift ------------------------------------------------------------
CLIMB_STEPS = 25
CLIMB_ERR = 0.10

MAX_WAYPOINT = 0.55
FLOOR_MIN_Z = 0.030
RESET_JUMP = 0.14
EMA_ALPHA = 0.4

P_HOVER = 0
P_SEAT = 1
P_PIVOT = 2
P_CLEAR = 3
P_RE_HOVER = 4
P_DESCEND = 5
P_CLOSE = 6
P_LIFT = 7


def _reconstruct_local_z_world(obs_i: np.ndarray) -> np.ndarray:
  """Board's local-Z axis (thickness axis) in world frame.

  ``object_orientation`` ([34:40]) is rows 1-2 of the body rotation matrix
  (mat[3:9] flattened row-major). Row 0 is recovered by orthonormality
  (row0 = row1 x row2, exact for det=+1). The local-Z axis in world frame is
  COLUMN 2 of the reconstructed matrix, i.e. (row0[2], row1[2], row2[2]).
  """
  r1 = obs_i[34:37]
  r2 = obs_i[37:40]
  n1 = np.linalg.norm(r1) + 1e-9
  n2 = np.linalg.norm(r2) + 1e-9
  r1 = r1 / n1
  r2 = r2 / n2
  r0 = np.cross(r1, r2)
  return np.array([r0[2], r1[2], r2[2]])


class PivotLiftClassicalPolicy(ClassicalPolicyBase):
  """Push the flat board into the wall to pivot it up, then pinch and lift."""

  DEFAULT_QPOS = HOME_QPOS  # pivot_lift env uses get_franka_robot_cfg (home)
  max_dq = 0.12
  orientation_weight = 0.2

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._pivot_steps = np.zeros(self.num_envs, dtype=np.int64)
      self._anchor = np.zeros((self.num_envs, 3))
      self._gto_ema = np.zeros((self.num_envs, 3))
      self._ema_init = np.zeros(self.num_envs, dtype=bool)
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._gto_seen = np.zeros(self.num_envs, dtype=bool)
    else:
      self._settle[env_ids] = 0
      self._pivot_steps[env_ids] = 0
      self._ema_init[env_ids] = False
      self._gto_seen[env_ids] = False

  def _rewind(self, i: int) -> None:
    self._phase[i] = 0
    self._phase_steps[i] = 0
    self._settle[i] = 0
    self._pivot_steps[i] = 0
    self._ema_init[i] = False

  def _gto(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    raw = obs_i[40:43]
    if not self._ema_init[i]:
      self._gto_ema[i] = raw
      self._ema_init[i] = True
    else:
      self._gto_ema[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._gto_ema[i]
    return self._gto_ema[i]

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]
    if self._gto_seen[i]:
      if np.linalg.norm(raw_gto - self._prev_gto[i]) > RESET_JUMP:
        self._rewind(i)
    else:
      self._gto_seen[i] = True
    self._prev_gto[i] = raw_gto

    gto = self._gto(i, obs_i)  # board - gripper (smoothed)
    o2g = obs_i[43:46]  # goal - board
    gripper_a = GRIPPER_CLOSED
    phase = self._phase[i]

    if phase == P_HOVER:
      # MEASURED AND FIXED: the align check used to test ``perp`` (pos_err
      # with its push_dir/x component zeroed), which still carries the FULL
      # +HOVER_Z (0.14) offset in z -- so norm(perp) only dropped under
      # ALIGN_TOL once the arm's natural (still-x-misaligned) descent
      # happened to cross z~=HOVER_Z above the board, a coincidence unrelated
      # to xy readiness. Instrumented (see LOGS.md 2026-09-09 D-2): the real
      # rollout transitioned to SEAT with gto_x still at -0.26 (should be
      # +0.076 at convergence -- a 34cm miss), and having started SEAT from
      # that far-off, still-descending joint configuration, the DLS solve
      # never recovered -- |raw| sat flat at ~0.33 for the entire 30-step SEAT
      # budget with no progress. Fixed to the same xy-then-z convention every
      # other top-down phase 0 in this package uses (lift_object.py, etc).
      gripper_a = GRIPPER_CLOSED
      contact = gto + np.array([0.0, 0.0, CONTACT_Z_ABOVE_CENTER])
      pos_err = contact - PUSH_DIR * STANDOFF + np.array([0.0, 0.0, HOVER_Z])
      if (
        np.linalg.norm(pos_err[:2]) < ALIGN_TOL and abs(pos_err[2]) < 0.05
      ) or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = P_SEAT
        self._phase_steps[i] = 0

    elif phase == P_SEAT:
      gripper_a = GRIPPER_CLOSED
      contact = gto + np.array([0.0, 0.0, CONTACT_Z_ABOVE_CENTER])
      raw = contact - PUSH_DIR * STANDOFF
      pos_err = raw
      seated = np.linalg.norm(raw) < CONTACT_TOL
      if seated:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= 2 or self._phase_steps[i] > SEAT_TIMEOUT:
        contact_now = gto + np.array([0.0, 0.0, CONTACT_Z_ABOVE_CENTER])
        self._anchor[i] = contact_now + PUSH_DIR * PIVOT_DEPTH
        self._phase[i] = P_PIVOT
        self._phase_steps[i] = 0

    elif phase == P_PIVOT:
      gripper_a = GRIPPER_CLOSED
      pos_err = self._anchor[i]
      self._pivot_steps[i] += 1
      cos_tilt = _reconstruct_local_z_world(obs_i)[2]
      if cos_tilt < PIVOT_DONE_COS or self._pivot_steps[i] >= PIVOT_STEPS_MAX:
        self._phase[i] = P_CLEAR
        self._phase_steps[i] = 0

    elif phase == P_CLEAR:
      gripper_a = GRIPPER_OPEN
      if self._phase_steps[i] < CLEAR_STEPS:
        pos_err = -PUSH_DIR * CLEAR_BACK + np.array([0.0, 0.0, CLEAR_UP])
      else:
        self._phase[i] = P_RE_HOVER
        self._phase_steps[i] = 0
        pos_err = np.array([0.0, 0.0, 0.02])

    elif phase == P_RE_HOVER:
      gripper_a = GRIPPER_OPEN
      pos_err = gto + np.array([0.0, 0.0, RE_HOVER_Z])
      if (
        np.linalg.norm(pos_err[:2]) < RE_ALIGN_TOL and abs(pos_err[2]) < 0.05
      ) or self._phase_steps[i] > RE_ALIGN_TIMEOUT:
        self._phase[i] = P_DESCEND
        self._phase_steps[i] = 0

    elif phase == P_DESCEND:
      gripper_a = GRIPPER_OPEN
      pos_err = gto + np.array([0.0, 0.0, PINCH_Z_ABOVE_CENTER])
      z_err = pos_err[2]
      pos_err[2] = max(z_err, -DESCENT_RATE)
      if abs(z_err) < SEAT_TOL and np.linalg.norm(pos_err[:2]) < 0.02:
        self._settle[i] += 1
        if self._settle[i] >= 3:
          self._phase[i] = P_CLOSE
          self._phase_steps[i] = 0
      else:
        self._settle[i] = 0
      if self._phase_steps[i] > 100:
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0

    elif phase == P_CLOSE:
      gripper_a = GRIPPER_CLOSED
      pos_err = gto + np.array([0.0, 0.0, PINCH_Z_ABOVE_CENTER])
      pos_err[:2] *= 0.5
      if self._phase_steps[i] >= CLOSE_STEPS:
        self._phase[i] = P_LIFT
        self._phase_steps[i] = 0

    else:  # P_LIFT
      gripper_a = GRIPPER_CLOSED
      if self._phase_steps[i] < CLIMB_STEPS:
        pos_err = np.array([0.0, 0.0, CLIMB_ERR])
      elif np.linalg.norm(gto) > HELD_TOL:
        # Grasp missed: go back and try again rather than flying to the goal
        # holding nothing.
        self._phase[i] = P_RE_HOVER
        self._phase_steps[i] = 0
        gripper_a = GRIPPER_OPEN
        pos_err = gto + np.array([0.0, 0.0, RE_HOVER_Z])
      else:
        pos_err = o2g.copy()

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)

    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err, _DOWN_AXIS, gripper_a
