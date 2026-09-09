"""Scripted EdgeGrasp teacher policy.

THE TRAP THIS TASK IS BUILT AROUND
-----------------------------------
The plate is UNGRASPABLE where it spawns. plate.xml: a 150 mm ceramic side plate,
19.2 mm tall (half-extents 0.075, 0.075, 0.0096) -- both in-plane widths are nearly
twice the 0.08 m aperture (unspannable), and lying flat on the riser its rim offers
no finger clearance underneath (nothing to slide a pad below). The ONLY graspable
configuration is thickness-wise, at an overhang -- manufacturing that clearance
IS the task. A policy that tries to pinch the plate flat on the ledge has
already lost; see CONTEXT.md sec 6 and the Wave-1 design note for Edge-Grasp
("drag to overhang, pinch the exposed 16mm").

GEOMETRY (ledge.xml / plate.xml, measured)
-------------------------------------------
The ledge is a static MOCAP body, written PER-ENV every resample by
``EdgeGraspCommand._resample_command`` (same trap as the place-in-container
bin: its MJCF pose only ever serves env 0). Box half-extents 0.10 (x) x 0.13
(y) x 0.05 (z, geom centred at local z=0.05), so the ledge spans z in
[0, 0.10] and its TOP surface sits at ``ledge_top_height`` (0.10) above the
body origin. **The robot faces the -x edge**: pushing the plate toward -x is
what creates the overhang (ledge.xml's own comment). Ledge x/y is drawn per-env
from ``EdgeGraspCommandCfg.ledge_spawn_range`` (x in [0.39, 0.43], y in
[-0.06, 0.06]); z is fixed at 0.

The plate spawns resting on the ledge top, NOT yet overhanging by construction:
``plate_rel_x`` in [-0.02, 0.02] relative to the ledge centre, so with the
plate's own half-length (0.075) its near (-x) edge lands at worst 0.005 m inland
of the ledge's near edge (ledge_centre_x - 0.10). Every episode therefore
genuinely needs a push before any grasp is possible.

**FK STALENESS WARNING (see CONTEXT.md sec 6):** the ledge's pose is written
by the ENV, not read back through ``site_pos_w`` here -- this policy never
reads the MJCF or any cached model state for ledge position. Every geometric
quantity below is recovered from the OBSERVATION each step (see next section),
which is always current.

RECOVERING THE LEDGE POSITION FROM RELATIVE OBSERVATIONS ONLY
-----------------------------------------------------------------
The policy observation carries the plate (``gripper_to_object``,
[40:43]) but has NO ledge observation term at all -- ``edge_grasp_env_cfg.py``'s
term list only tracks robot state, plate state, and gripper state. However,
``object_to_goal`` ([43:46]) is ``target_pos - plate_pos``, and
``EdgeGraspCommand._resample_command`` sets
``target_pos = ledge_pos + goal_offset`` with the FIXED, known constant
``goal_offset = (-0.13, 0, 0.18)``. So:

    o2g + gto = (target_pos - plate_pos) + (plate_pos - gripper_pos)
              = target_pos - gripper_pos
              = (ledge_pos + goal_offset) - gripper_pos

    ledge_rel := (o2g + gto) - goal_offset  =  ledge_pos - gripper_pos

``ledge_rel`` is exactly as legal to use as ``gto`` itself: it is a difference
of two world positions, so the per-env scene-origin offset cancels. From it:
    ledge_edge_x_rel (near/robot-facing edge) = ledge_rel[0] - LEDGE_HALF_X (0.10)
And from ``gto`` (plate centre relative to gripper):
    plate_near_edge_x_rel (the plate's own -x edge)  = gto[0] - PLATE_HALF_X (0.075)
    overhang = ledge_edge_x_rel - plate_near_edge_x_rel
             = ledge_rel[0] - gto[0] - 0.025
-- how far the plate's near edge has been pushed PAST the ledge edge. This is
the single quantity that drives the push phase and the two phase transitions
that follow it.

SUCCESS (EdgeGraspCommand)
---------------------------
    lift  = plate_pos.z - (ledge_centre.z + ledge_top_height)   > 0.04
    drift = ||plate_pos.xy - ledge_centre.xy||                  < 0.40
Both, latched over the episode. Note this is NOT a Cartesian goal-reaching
task in the usual sense -- there is no tight target position, only "get the
plate held clearly above the ledge top without flinging it away". Sliding
alone cannot satisfy it: pushed off the edge without being caught, the plate
drops BELOW the ledge top under gravity (this is a free-object task -- gravity
is ON, unlike the mocap-mounted mechanism tasks). The failure mode (dropped
off the edge) and the success mode (lifted clear) are separated by
construction.

STRATEGY -- push an overhang, reorient side-on, pinch the thickness, lift
----------------------------------------------------------------------------
Because the plate never rotates in x/y relative to the ledge (``plate_yaw``
range is (0, 0) and the push is a pure -x slide), the whole geometry reduces
to 1-D bookkeeping along x, which is what makes deriving ``overhang`` above
tractable without ever needing the plate's absolute pose.

1. PUSH: top-down, fingers CLOSED (a single rigid pusher pad, same convention
   as push_cuboid/drag_pull), contact the plate's FAR (+x, trailing) edge and
   creep -x. Direction is FIXED here (always -x, unlike push_cuboid's
   arbitrary-direction contact-point servo) because the task geometry is
   fixed, which simplifies the servo to a 1-D version of the same idea. Stop
   once ``overhang`` reaches ``OVERHANG_TARGET`` -- deliberately well under
   the plate's own half-length (0.075 m): past that point the plate's centre of
   mass crosses the ledge edge and it tips/falls on its own, which is not
   recoverable (this is a free-object gravity-on task; a plate that rotates
   off the ledge cannot be un-dropped).
2. RETREAT: rise and pull back, clear of the ledge entirely, while ALSO
   rotating the wrist to the SIDE orientation (position and orientation are
   solved jointly by the DLS step, so there is no need for a separate
   reorientation-only phase). Skipping this and reorienting in place risks
   sweeping the forearm through the ledge.
3. SIDE_HOVER / SIDE_INSERT: approach the exposed overhang HORIZONTALLY from
   outside the ledge's footprint (x less than the ledge edge), fingers open,
   gripper CLOSING axis vertical (world z) so the jaws straddle the plate's
   19 mm rim thickness top/bottom, APPROACH axis horizontal (world +x) so the
   wrist reaches in from outside. Because the pinch point is always chosen
   with x strictly less than the ledge edge (``PINCH_INSET`` margin), the
   approach never has to enter the ledge's own footprint -- below the
   overhang there is nothing but air (the ledge box only exists for
   x >= ledge_edge), so there is no collision risk from the ledge geometry
   itself on this approach.
4. CLOSE: squeeze on the exposed edge.
5. LIFT: with the SIDE orientation held (re-orienting mid-air while gripping
   a thin, marginally-held plate is not worth the risk), drive
   ``object_to_goal`` directly -- it already points up and further -x, away
   from the ledge, which is both the success direction (height) and the safe
   direction (away from any remaining ledge geometry).

Orientation frame for the side pinch (see ``_side_frame`` below): the
Franka gripper's local x-axis is always the FINGER-CLOSING axis and its local
z-axis is always the APPROACH axis (panda geometry, independent of wrist
orientation -- see stack_object.py's module docstring for the top-down case,
which is the same fact expressed for a different target orientation). Here we
want local x -> world z (closing top/bottom on the plate's thickness) and
local z -> world +x (approaching from outside, reaching forward).

OBSERVATIONS: 60-D layout (see ``edge_grasp_env_cfg.py``). Only relative terms
used: [40:43] gripper_to_object (plate - gripper), [43:46] object_to_goal
(target_pos - plate). Absolute terms (object_pos, gripper_pos, obs[18:21],
obs[25:28]) carry the per-env scene origin and are never used.

Robot starts at HOME_QPOS (this task uses ``get_franka_robot_cfg``, the same
convention as lift/push_cube/cuboid/disc), with NO reset offset
(``reset_robot_joints`` position_range is (0, 0) here) -- so ``obs[0:7]`` reads
exactly zero on the first post-reset observation, which is what makes the
robot-state auto-reset detector below reliable (same trick as
``stack_object.py``'s ``_detect_reset``).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

GRIPPER_OPEN = 1.0
GRIPPER_CLOSED = -1.0

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])

# Fixed constant from EdgeGraspCommandCfg.goal_offset -- used to recover the
# ledge's position from the two relative observation terms (see module
# docstring, "RECOVERING THE LEDGE POSITION").
GOAL_OFFSET = np.array([-0.13, 0.0, 0.18])

# plate.xml / ledge.xml geometry (half-extents, NOT tuned constants).
# RE-DERIVED for the CL-V2 assets (W1-c): the plate is a 150 mm scanned ceramic side
# plate (GSO Room_Essentials_Salad_Plate_Turquoise scaled to a 6 in bread plate) and the
# ledge is a 200 x 260 x 100 mm oak riser (widened from 180 mm so the 150 mm plate spawns
# fully supported). Note the push contact point, ledge_x + plate_rel_x_max + PLATE_HALF_X,
# is UNCHANGED at ledge_x + 0.095 because plate_rel_x shrank by exactly as much as the
# plate grew -- so EdgeGraspCommandCfg.ledge_spawn_range's reach derivation still holds.
PLATE_HALF_X = 0.075
LEDGE_HALF_X = 0.10

# Phases.
P_PUSH_HOVER = 0
P_PUSH_DESCEND = 1
P_PUSH_ADVANCE = 2
P_RETREAT = 3
P_SIDE_HOVER = 4
P_SIDE_INSERT = 5
P_CLOSE = 6
P_LIFT = 7

# -- push phase -------------------------------------------------------------
# Ride the closed pad against the plate's trailing (+x) face at roughly its
# own mid-height. PAD_RADIUS/STANDOFF follow drag_pull.py's derivation (its
# HALF_WIDTH+PAD_RADIUS+STANDOFF formula) with the plate's own half-length in
# place of the cuboid's.
PAD_RADIUS = 0.010
STANDOFF = 0.006
BEHIND = PLATE_HALF_X + PAD_RADIUS + STANDOFF  # ~0.066
PUSH_HOVER_HEIGHT = 0.13
PUSH_ALIGN_TOL = 0.04
PUSH_DESCENT_RATE = 0.03
# MEASURED MECHANISM BUG: the plate is only 16mm thick resting on a WIDE, RIGID
# ledge -- unlike pushing a cuboid on open ground, undershooting the descent
# just parks the closed pad on the ledge's own top surface (a big flat rigid
# plane extending well past the plate) instead of missing cleanly, and there
# is no ee_ground_collision-style signal to catch it. The original 0.025
# tolerance let PUSH_ADVANCE start ~2cm too high with the pad resting on the
# ledge, not the plate's trailing face -- traced live, the plate sat
# completely motionless for the entire advance phase while the (blocked) site
# z crept slowly toward the plate height instead of ever generating a push.
# Tightened to comfortably inside the plate's own half-thickness (0.0096).
PUSH_SEAT_TOL = 0.008
# Target overhang: comfortably under the plate's own half-length (0.075) so the
# centre of mass never crosses the ledge edge and the plate stays flat and
# supported (does not tip on its own -- see module docstring, step 1). Held at the
# same 64% of the half-length the primitive plate used (0.032 / 0.05).
OVERHANG_TARGET = 0.048
PUSH_GAIN = 0.7
PUSH_STEP_MIN = 0.010
PUSH_STEP_MAX = 0.032
# If the pusher is no longer meaningfully behind the plate (drifted level with
# or past its centre -- an overtake), back off and re-seat rather than push
# from the wrong side. Mirrors drag_pull.py's MIN_BEHIND/RESEAT_STEPS.
MIN_BEHIND = 0.014
RESEAT_STEPS = 10
PUSH_TIMEOUT = 90

# -- retreat / reorient -------------------------------------------------------
# MEASURED MECHANISM BUG: a single constant retreat vector with both -x and +z
# active from step 1 keeps the still-closed fingertip in contact with the
# plate's face WHILE it retreats -- height rises too slowly to break contact
# before the -x component has already dragged the plate another 5-10cm past
# OVERHANG_TARGET, well past the point (plate half-length 0.075) where it tips
# off the ledge on its own (traced live: overhang went from ~0.03 at the
# push/retreat transition to >0.15 by the time the plate stopped moving, and
# the plate ended up on the ground, lift ~= -0.09).
#
# TWO compounding mechanism bugs were found here, not one:
#   1. A pure-vertical-only clear vector was tried first and STILL let the
#      plate slide ~0.10m before contact broke -- the closed fingertip stays
#      geometrically overlapped with the plate's edge until the site has
#      risen a surprising amount (traced: the drag did not stop until
#      site z reached ~0.20, ~0.09m above the plate's own 0.108 resting
#      height), so "rise in place" is slow to disengage.
#   2. Commanding the SIDE orientation (_SIDE_ROT) from the very first retreat
#      step -- while the fingertip is still at/near the plate -- sweeps the
#      CLOSED fingertip through a ~90deg arc as the wrist reorients: the
#      fingertip is offset several cm from the ``gripper`` site, so a
#      "vertical-only" SITE position error does nothing to stop the
#      fingertip's own swept path from crossing the plate.
# THIRD bug, and the one actually responsible for the continued drag even
# after both fixes above: the retreat vector's x-component was still -x --
# i.e. the SAME direction as the push itself. "Retreating" at x=-0.16 does
# not disengage the pad from the plate at all; it continues advancing through
# it, just with the fingers open instead of closed. The pusher stands BEHIND
# the plate on its far (+x) side (see BEHIND), so disengaging means moving
# back toward +x -- away from the ledge, not further across it -- while also
# rising clear. Still held at the SAME _DOWN_AXIS orientation the push used;
# reorientation to the side frame happens only once this retreat is complete
# (SIDE_HOVER, which starts genuinely clear of the ledge).
RETREAT_STEPS = 24
RETREAT_VEC = np.array([0.14, 0.0, 0.14])  # back away from the ledge (+x), and up

# -- side approach ------------------------------------------------------------
# Margin the pinch point is kept INSIDE the plate's protruding tip (away from
# the very edge, for a stable pinch) and clear of the ledge edge (see module
# docstring: with overhang >= OVERHANG_TARGET, this leaves
# OVERHANG_TARGET - PINCH_INSET >= 0.018 m of clearance from the ledge).
PINCH_INSET = 0.014
# MEASURED MECHANISM issue: the SIDE orientation's reachability is NOT smooth
# in x -- an isolated FK/DLS probe (this class's own IK, no policy or physics)
# found a clean convergence (4-9cm residual) at x in [0.28, 0.30] but a
# completely different, badly-wrong local minimum (11-21cm residual, latching
# onto a fixed point around x~0.36 regardless of the actual target) at x=0.25
# and worse still at x=0.15. The pinch point itself sits around x in
# [0.27, 0.32] (ledge x in [0.39, 0.43] minus the ledge half-extent and
# overhang), so a standoff of 0.11 (the top-down convention used elsewhere in
# this file) pulls the HOVER waypoint down into that bad x<0.20 region before
# ever committing to the pinch. Kept small so the hover point stays inside the
# region the side-approach IK actually converges in.
SIDE_STANDOFF_X = 0.045  # approach from this far outside before inserting
SIDE_HOVER_TOL = 0.045
SIDE_HOVER_TIMEOUT = 60
SIDE_INSERT_TOL = 0.020
SIDE_INSERT_TIMEOUT = 60
# Rate cap on the SIDE_INSERT approach (per axis, per control step). MEASURED:
# an un-rate-limited insert -- commanding the full remaining ``pinch`` vector
# every step -- repeatedly knocked the plate off the ledge entirely (traced:
# ``plate_lift`` dropping from +0.007 to -0.09 within a handful of steps of
# entering this phase). The side-approach IK also carries a larger residual
# bias here than the top-down approaches elsewhere in this file (see
# SIDE_STANDOFF_X's docstring: 4-9cm even at full convergence), so a fast
# final approach strikes the 16mm-thick plate off-centre rather than
# straddling it cleanly. A slow creep gives contact forces time to settle
# instead of a collision.
SIDE_INSERT_RATE = 0.012
CLOSE_STEPS = 16
# After closing, if the plate is not where the pinch was aimed the grasp
# missed; retry the insert once rather than lifting nothing (mirrors
# lift_object.py's post-climb grasp check).
HELD_TOL = 0.05

EMA_ALPHA = 0.4
RESET_JUMP = 0.12
FLOOR_MIN_Z = 0.030


def _side_frame() -> np.ndarray:
  """Rotation matrix for the horizontal edge pinch.

  local x (finger-closing axis) -> world +z: jaws open/close top-to-bottom,
    straddling the plate's 19mm rim thickness.
  local z (approach axis)       -> world +x: the wrist reaches forward,
    horizontally, in from outside the ledge's footprint.
  """
  ex = np.array([0.0, 0.0, 1.0])
  ez = np.array([1.0, 0.0, 0.0])
  ey = np.cross(ez, ex)
  return np.column_stack([ex, ey, ez])


_SIDE_ROT = _side_frame()


class EdgeGraspClassicalPolicy(ClassicalPolicyBase):
  """Push the plate to an overhang, reorient side-on, pinch the exposed edge, lift."""

  DEFAULT_QPOS = HOME_QPOS  # edge_grasp uses get_franka_robot_cfg (home)
  max_dq = 0.10
  orientation_weight = 0.3

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._ema = np.zeros((self.num_envs, 3))
      self._ema_ok = np.zeros(self.num_envs, dtype=bool)
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._reseat = np.zeros(self.num_envs, dtype=np.int64)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._started = np.zeros(self.num_envs, dtype=bool)
      self._retry = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._ema_ok[env_ids] = False
      self._reseat[env_ids] = 0
      self._settle[env_ids] = 0
      self._started[env_ids] = False
      self._retry[env_ids] = 0

  # -- auto-reset detection --------------------------------------------------
  # reset_robot_joints has position_range (0, 0) here, so obs[0:7] reads
  # exactly zero on the first post-reset observation and never does once the
  # arm is moving (same trick as stack_object.py's _detect_reset). Needed
  # because ee_ground_collision and object_out_of_bounds both auto-reset the
  # env mid-episode without telling the harness.
  reset_qpos_tol = 0.03

  def _detect_reset(self, i: int, obs_i: np.ndarray) -> None:
    at_home = np.max(np.abs(obs_i[0:7])) < self.reset_qpos_tol
    if at_home and self._started[i]:
      self._phase[i] = P_PUSH_HOVER
      self._phase_steps[i] = 0
      self._ema_ok[i] = False
      self._reseat[i] = 0
      self._settle[i] = 0
      self._retry[i] = 0
      self._started[i] = False
    elif not at_home:
      self._started[i] = True

  def _smooth(self, i: int, raw: np.ndarray) -> np.ndarray:
    if not self._ema_ok[i]:
      self._ema[i] = raw
      self._ema_ok[i] = True
    else:
      self._ema[i] = EMA_ALPHA * raw + (1 - EMA_ALPHA) * self._ema[i]
    return self._ema[i]

  def _guard(self, err: np.ndarray, obs_i: np.ndarray) -> np.ndarray:
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + err[2] < FLOOR_MIN_Z:
      err = err.copy()
      err[2] = FLOOR_MIN_Z - z
    return err

  @staticmethod
  def _x_guard(err: np.ndarray, ledge_edge_x: float, margin: float = 0.012) -> np.ndarray:
    """Clamp the commanded x-displacement so the SITE can never be commanded
    past the ledge's near edge, independent of how well the IK converges.

    ``ledge_edge_x`` is already expressed as a displacement FROM THE GRIPPER
    (see the module docstring's derivation), so "target x == ledge edge" is
    exactly ``err[0] == ledge_edge_x``; clamping the command there directly
    prevents ever asking the arm to plant the site inside the ledge's own
    footprint. This is a hard, physics-independent backstop for the side
    approach's known-unreliable IK convergence (see SIDE_STANDOFF_X's
    docstring) -- MEASURED to be necessary: without it, an imprecise
    convergence during SIDE_INSERT drove the site (and hence the fingertip)
    into the ledge/plate, knocking the plate off entirely (traced live:
    ``plate_lift`` dropping from +0.008 to -0.09 -- to the ground -- within a
    handful of steps of entering SIDE_INSERT).
    """
    cap = ledge_edge_x - margin
    if err[0] > cap:
      err = err.copy()
      err[0] = cap
    return err

  def _target_error(self, i: int, obs_i: np.ndarray):
    self._detect_reset(i, obs_i)

    # Higher damping (more regularization, slower convergence) ONLY for the
    # side-approach phases. MEASURED: the full _SIDE_ROT orientation target
    # combined with the default damping produces a genuine numerical
    # OSCILLATION at this reach/height, not just a steady residual -- traced
    # live, the site's y position swung from -0.02 to +0.07 and back past
    # -0.04 over ~30 steps while the pinch target barely moved, a limit cycle
    # rather than a converging approach. This is set/reset per env, per call
    # (safe: base.py's IK step for env ``i`` runs synchronously right after
    # this method returns, before any other env is touched), so it never
    # leaks into the top-down push/retreat phases' tuned dynamics.
    if self._phase[i] >= P_SIDE_HOVER:
      self.damping = 0.45
      self.max_dq = 0.05
    else:
      self.damping = 0.2
      self.max_dq = 0.10

    raw_gto = obs_i[40:43]  # plate - gripper
    o2g = obs_i[43:46]  # target_pos - plate
    if self._ema_ok[i] and np.linalg.norm(raw_gto - self._prev_gto[i]) > RESET_JUMP:
      self._phase[i] = P_PUSH_HOVER
      self._phase_steps[i] = 0
      self._reseat[i] = 0
      self._settle[i] = 0
      self._retry[i] = 0
      self._ema_ok[i] = False
    self._prev_gto[i] = raw_gto

    gto = self._smooth(i, raw_gto)
    # ledge position relative to the gripper (see module docstring); computed
    # fresh every step from the un-smoothed o2g/gto so it tracks the gripper's
    # own motion exactly (it is a difference of two absolute positions, one of
    # which -- the gripper -- moves every step).
    ledge_rel = (o2g + raw_gto) - GOAL_OFFSET
    ledge_edge_x = ledge_rel[0] - LEDGE_HALF_X
    plate_near_edge_x = gto[0] - PLATE_HALF_X
    overhang = ledge_edge_x - plate_near_edge_x

    phase = self._phase[i]

    if phase == P_PUSH_HOVER:
      pos_err = gto + np.array([BEHIND, 0.0, PUSH_HOVER_HEIGHT])
      if np.linalg.norm(pos_err[:2]) < PUSH_ALIGN_TOL:
        self._phase[i] = P_PUSH_DESCEND
        self._phase_steps[i] = 0
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_PUSH_DESCEND:
      raw = gto + np.array([BEHIND, 0.0, 0.0])
      pos_err = raw.copy()
      pos_err[2] = max(pos_err[2], -PUSH_DESCENT_RATE)
      # Gate on Z ALONE, not the full 3D norm: xy was already brought inside
      # the (loose, 0.04) PUSH_ALIGN_TOL by the hover phase, and re-demanding
      # that same xy precision here just to read a combined norm would make
      # the now-tight PUSH_SEAT_TOL practically unreachable. Height is the
      # quantity that actually matters (see PUSH_SEAT_TOL's docstring).
      if abs(raw[2]) < PUSH_SEAT_TOL:
        self._phase[i] = P_PUSH_ADVANCE
        self._phase_steps[i] = 0
      elif self._phase_steps[i] > 100:
        self._phase[i] = P_PUSH_ADVANCE  # stalled: push anyway
        self._phase_steps[i] = 0
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_PUSH_ADVANCE:
      if overhang >= OVERHANG_TARGET or self._phase_steps[i] > PUSH_TIMEOUT:
        self._phase[i] = P_RETREAT
        self._phase_steps[i] = 0
        return self._guard(RETREAT_VEC.copy(), obs_i), _DOWN_AXIS, GRIPPER_OPEN

      along = -gto[0]  # + while the pusher is correctly behind (on +x side)
      if self._reseat[i] > 0:
        self._reseat[i] -= 1
        pos_err = gto + np.array([1.6 * BEHIND, 0.0, 0.0])
      elif along < MIN_BEHIND:
        self._reseat[i] = RESEAT_STEPS
        pos_err = gto + np.array([1.6 * BEHIND, 0.0, 0.0])
      else:
        remaining = max(OVERHANG_TARGET - overhang, 0.0)
        step = float(np.clip(PUSH_GAIN * remaining, PUSH_STEP_MIN, PUSH_STEP_MAX))
        pos_err = gto + np.array([BEHIND - step, 0.0, 0.0])
      pos_err[2] = max(pos_err[2], -PUSH_DESCENT_RATE)
      return self._guard(pos_err, obs_i), _DOWN_AXIS, GRIPPER_CLOSED

    if phase == P_RETREAT:
      # Held at _DOWN_AXIS throughout -- see RETREAT_VEC's docstring for why
      # reorienting here (before the arm is actually clear of the plate) is
      # the mechanism bug this phase exists to avoid. Reorientation to the
      # side frame happens in SIDE_HOVER, once this retreat has finished.
      if self._phase_steps[i] >= RETREAT_STEPS:
        self._phase[i] = P_SIDE_HOVER
        self._phase_steps[i] = 0
      return self._guard(RETREAT_VEC.copy(), obs_i), _DOWN_AXIS, GRIPPER_OPEN

    # From here on, the pinch target is derived fresh every step: a point
    # PINCH_INSET inside the plate's protruding tip, at the plate's own
    # (unchanged) y and z -- see module docstring for why this stays clear of
    # the ledge's own footprint by construction.
    pinch = np.array(
      [plate_near_edge_x + PINCH_INSET, gto[1], gto[2]]
    )

    if phase == P_SIDE_HOVER:
      pos_err = pinch - np.array([SIDE_STANDOFF_X, 0.0, 0.0])
      if (
        np.linalg.norm(pos_err[1:]) < SIDE_HOVER_TOL
        or self._phase_steps[i] > SIDE_HOVER_TIMEOUT
      ):
        self._phase[i] = P_SIDE_INSERT
        self._phase_steps[i] = 0
      pos_err = self._x_guard(pos_err, ledge_edge_x)
      return self._guard(pos_err, obs_i), _SIDE_ROT, GRIPPER_OPEN

    if phase == P_SIDE_INSERT:
      pos_err = pinch.copy()
      if (
        np.linalg.norm(pos_err) < SIDE_INSERT_TOL
        or self._phase_steps[i] > SIDE_INSERT_TIMEOUT
      ):
        self._phase[i] = P_CLOSE
        self._phase_steps[i] = 0
      pos_err = np.clip(pos_err, -SIDE_INSERT_RATE, SIDE_INSERT_RATE)
      pos_err = self._x_guard(pos_err, ledge_edge_x)
      return self._guard(pos_err, obs_i), _SIDE_ROT, GRIPPER_OPEN

    if phase == P_CLOSE:
      pos_err = pinch * 0.4
      if self._phase_steps[i] >= CLOSE_STEPS:
        if np.linalg.norm(gto) > HELD_TOL and self._retry[i] < 1:
          # Missed: try the insert once more before giving up and lifting air.
          self._retry[i] += 1
          self._phase[i] = P_SIDE_INSERT
          self._phase_steps[i] = 0
        else:
          self._phase[i] = P_LIFT
          self._phase_steps[i] = 0
      pos_err = self._x_guard(pos_err, ledge_edge_x)
      return self._guard(pos_err, obs_i), _SIDE_ROT, GRIPPER_CLOSED

    # P_LIFT: drive object_to_goal directly. target_pos is up and further -x
    # from the ledge (GOAL_OFFSET), i.e. simultaneously the success direction
    # (height above the ledge top) and the safe direction (away from the
    # ledge's own geometry) -- see module docstring, step 5.
    pos_err = o2g.copy()
    return self._guard(pos_err, obs_i), _SIDE_ROT, GRIPPER_CLOSED
