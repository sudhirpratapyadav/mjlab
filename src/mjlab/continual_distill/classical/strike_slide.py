"""Scripted StrikeSlide teacher policy -- calibrated impulse strike (Lane D, D-1).

TASK
----
puck.xml: CL-V2 (W1-b) replaced the primitive with a **regulation ice-hockey puck** --
cylinder r=0.0381 h=0.0254 (half-height 0.0127), **mass 0.16 kg** (was 0.03), sliding
friction 0.4 unchanged. The 5.3x heavier puck is the dominant dynamics change on this
task: the arm has to deliver 5.3x the impulse for the same release speed, and the
STRIKE phase's lever is contact DURATION at a fixed rate cap, so the reachable slide
distance falls. The puck spawns in the reachable grasp box (x 0.30-0.46,
y -0.15..0.15) but the goal band (x 0.88-1.05, y -0.18..0.18) sits BEYOND the
arm's absolute ~0.85 m stretch -- see CONTEXT.md sec 6 and
``strike_slide_env_cfg.py``. The goal can never be reached quasi-statically: the
puck must be STRUCK so it slides there ballistically after the arm has already
let go. success_threshold = 0.08 m (latched over the 200-step / 4.0 s episode),
looser than a push task's tolerance but still a real window against slide
distances of 0.42-0.9 m.

SUCCESS (PushingCommand, reused unmodified)
--------------------------------------------
    ||puck_pos - goal_pos|| < 0.08, latched (torch.maximum) over the episode.

STRATEGY -- approach / seat / windup / strike / retreat
---------------------------------------------------------
This is the same "punch a fixed target well beyond the object, hold for a
step budget, let the DLS solve drive at the servo limit" mechanism
``topple_block.py`` uses to topple the block -- but topple only needs the
punch to EXCEED a force threshold (any strong-enough punch tips the block, a
binary event with no distance requirement). Strike-slide needs the puck to
land inside an 8 cm window after travelling anywhere from 0.42 to 0.9 m, i.e.
release SPEED must be calibrated continuously, not just exceed a minimum. That
is the whole difference this file exists to measure.

  0 HOVER    align above-and-behind the puck (opposite the goal direction),
             frozen push_dir computed once from the initial object_to_goal.
  1 SEAT     close the gap onto the puck's near face at ride height.
  2 WINDUP   retreat further behind along -push_dir, at ride height, to give
             the arm room (and TIME -- see below) to accelerate before contact.
             Windup distance scales with the required slide distance.
  3 STRIKE   command a fixed anchor far beyond the puck along push_dir (well
             inside/through its geometry, unreachable) at a BOOSTED max_dq, for
             a fixed step budget. Anchor depth and duration are both
             calibrated from the required distance -- see CALIBRATION.
  4 RETREAT  pull up and back immediately, then hold station so the puck is
             never re-struck while it is still sliding.

CALIBRATION (measured, see LOGS.md 2026-09-09 D-1 entry for the instrumented
numbers this produced)
--------------------------------------------------------------------------
The arm is a POSITION servo, not a force/impulse source: nothing in the
teacher contract exposes "apply impulse J". What actually sets the puck's
release speed is how much the physical joint-space PD actuators manage to
accelerate the gripper site during WINDUP + the first few STRIKE steps before
contact -- i.e. release speed is a function of (windup distance, strike
max_dq, elapsed contact-free steps), not a single dial. This file exposes a
single scalar knob, ``STRIKE_GAIN``, and derives windup distance + strike
max_dq from it, scaled by the required slide distance read each episode from
``object_to_goal``. See the module-level constants for the measured mapping
and its scatter.

Observations: 60-D layout, matches push_cuboid/topple_block.
  40:43 gripper_to_object (puck - gripper), 43:46 object_to_goal (goal - puck).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.base import HOME_QPOS, ClassicalPolicyBase

_DOWN_AXIS = np.array([0.0, 0.0, -1.0])
GRIPPER_CLOSED = -1.0

# puck.xml: cylinder r=0.0381 h=0.0254 (half-height 0.0127). Object centre at
# z=0.0127. Measured (see LOGS.md): pads reach ~1.2-1.4cm below the gripper
# site in a top-down pose, same figure push_cuboid/lift_object measured for
# this gripper -- re-derived here, not assumed, per the Drag-Pull lesson.
PUCK_HALF_HEIGHT = 0.0127
PUCK_RADIUS = 0.0381
# Absolute site height ABOVE THE PUCK CENTRE (added directly, same convention as
# push_cuboid's RIDE_HEIGHT -- NOT netted against the object's own half-height,
# which was this file's first bug: `RIDE_HEIGHT - PUCK_HALF_HEIGHT` put the
# absolute site target at z=0.020, BELOW the 0.030 ee_ground_collision floor
# guard every other teacher in this package uses, so the "seat" phase drove the
# hand into the floor almost every episode -- instrumented via a raw object_pos
# probe (see LOGS.md 2026-09-09 D-1): gto_z oscillated back to its ~-0.37
# hover-start value mid-episode, i.e. env auto-resets from ee_ground_collision,
# which the state machine did not detect and so ran on with stale phase/anchor
# state. Puck top sits at z=0.0254 (half-height 0.0127); pads reach ~1.3cm
# below the site (measured figure this package uses throughout), so
# RIDE_HEIGHT=0.020 puts the pad bottom at ~0.0327-0.013=0.0197 -- inside the
# puck's own 0-0.0254 vertical span -- while the site itself (0.0327 absolute)
# clears the 0.030 floor.
#
# RE-DERIVED, CL-V2 (W1-b), AND THIS CORRECTS A REAL BUG. The value shipped
# here was 0.026 while the comment above derives 0.020, and 0.026 was already
# marginal on the old 24 mm puck (site 0.038 -> pad bottom 0.025, i.e. 1 mm
# ABOVE the puck's own top face). On the 25.4 mm regulation puck it puts the pad
# bottom at 0.0257 against a top face at 0.0254: the fingers skate OVER the puck
# instead of striking its rim. Set to the value the derivation actually gives --
# the largest RIDE_HEIGHT that keeps the pads inside the puck's vertical span
# while still leaving 2.7 mm of margin above the FLOOR_MIN_Z guard.
RIDE_HEIGHT = 0.020

# Behind-the-puck standoff during hover/seat: puck radius + a pad half-width (0.025).
STANDOFF = 0.063
HOVER_Z = 0.14
ALIGN_TOL = 0.05
CONTACT_TOL = 0.03
SEAT_INTEG_GAIN = 0.2

ALIGN_TIMEOUT = 40
SEAT_TIMEOUT = 30

# WINDUP: retreat this far behind the seated contact point before striking.
# Scales with the required slide distance -- see CALIBRATION. Bounds keep the
# windup inside the episode's step budget (200 steps @ 0.02s = 4.0s) and away
# from folding the arm back over its own base (GRASP_RADIAL_MIN = 0.28).
WINDUP_MIN = 0.05
WINDUP_MAX = 0.18
WINDUP_STEPS = 20

# STRIKE: fixed anchor this far beyond the puck along push_dir (through its
# geometry -- unreachable, so the DLS solve drives at the servo limit for the
# whole phase, same mechanism as topple_block's PUNCH_DEPTH).
STRIKE_DEPTH = 0.45
# Per-episode strike duration and max_dq, both derived from the required
# slide distance (0.42-0.9m over the sampled goal/puck pair). Longer WINDUP
# and more STRIKE steps give the physical joint PD servos more time to reach
# the boosted rate cap before contact -- this is the only lever available; a
# fixed value under- or over-shoots most episodes (see LOGS.md).

# MEASURED AND REVERTED: STRIKE_MAX_DQ up to 0.55 rad/step (vs. 0.15 elsewhere
# in this package) reliably drove ee_ground_collision terminations DURING the
# strike itself -- instrumented (see LOGS.md 2026-09-09 D-1): 8/8 envs at
# n=8, almost all firing in phase 3, none in phase 0/1/2. The internal FK/IK
# model that computes q_goal respects the floor guard by construction, but at
# that rate cap the joint-space step is ~14x push_cuboid's, and the physical
# PD-servo'd arm does not track a trajectory that aggressive cleanly -- it
# overshoots into the floor even though every COMMANDED waypoint stayed legal.
# The lever this file uses instead is STRIKE_STEPS (contact duration) at a
# max_dq already proven safe elsewhere in this package.
STRIKE_MAX_DQ_MIN = 0.15
STRIKE_MAX_DQ_MAX = 0.15
STRIKE_STEPS_MIN = 5
STRIKE_STEPS_MAX = 26

RETREAT_STEPS = 20
RETREAT_UP = 0.12
RETREAT_BACK = 0.10

MAX_WAYPOINT = 0.45

# Absolute lower bound on the commanded gripper-site height (robot base frame).
# Same figure push_cuboid/lift_object/topple_block use: over the geoms that
# actually collide, the hand capsule's bounding volume reaches ~3.1cm below the
# site. Site height comes from FK on the arm's own joint angles (robot-base
# frame, free of the per-env scene-origin offset), not from an absolute obs.
FLOOR_MIN_Z = 0.030
# If the observed puck jumps by more than this in one control step, the env
# auto-reset underneath us (ee_ground_collision or out-of-bounds ends the
# episode and respawns the puck) -- the per-env state machine must be rewound
# or it keeps windup/striking at a puck that is no longer there (see
# lift_object.py's RESET_JUMP for the same pattern).
RESET_JUMP = 0.12

# Distance band the puck must cover, purely for scaling the calibration law
# (NOT used to change the task -- see strike_slide_env_cfg.py: puck spawns
# 0.30-0.46, goal 0.88-1.05, so realised slide distances span roughly this).
DIST_LO = 0.40
DIST_HI = 0.95


def _lerp(lo: float, hi: float, t: float) -> float:
  t = float(np.clip(t, 0.0, 1.0))
  return lo + t * (hi - lo)


class StrikeSlideClassicalPolicy(ClassicalPolicyBase):
  """Approach, seat, windup, and strike the puck with a calibrated punch."""

  DEFAULT_QPOS = HOME_QPOS  # strike_slide env uses get_franka_robot_cfg (home)
  max_dq = 0.12
  orientation_weight = 0.2

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._push_dir = np.zeros((self.num_envs, 3))
      self._push_dir_ok = np.zeros(self.num_envs, dtype=bool)
      self._settle = np.zeros(self.num_envs, dtype=np.int64)
      self._integ = np.zeros((self.num_envs, 3))
      self._windup_steps = np.zeros(self.num_envs, dtype=np.int64)
      self._strike_steps = np.zeros(self.num_envs, dtype=np.int64)
      self._req_dist = np.zeros(self.num_envs)
      self._anchor = np.zeros((self.num_envs, 3))
      self._strike_max_dq = np.full(self.num_envs, STRIKE_MAX_DQ_MIN)
      self._strike_step_budget = np.full(self.num_envs, STRIKE_STEPS_MIN, dtype=np.int64)
      self._prev_gto = np.zeros((self.num_envs, 3))
      self._gto_init = np.zeros(self.num_envs, dtype=bool)
    else:
      self._push_dir_ok[env_ids] = False
      self._settle[env_ids] = 0
      self._integ[env_ids] = 0.0
      self._windup_steps[env_ids] = 0
      self._strike_steps[env_ids] = 0
      self._gto_init[env_ids] = False
      self.max_dq = 0.12

  def _rewind(self, i: int) -> None:
    """Rewind the state machine after a detected mid-episode auto-reset (a
    voided strike from ee_ground_collision or object_out_of_bounds). Without
    this the phase/anchor/push_dir state keeps running against a puck that has
    been respawned underneath it -- see RIDE_HEIGHT's docstring."""
    self._phase[i] = 0
    self._phase_steps[i] = 0
    self._push_dir_ok[i] = False
    self._settle[i] = 0
    self._integ[i] = 0.0
    self._windup_steps[i] = 0
    self._strike_steps[i] = 0

  def _latch_push_dir(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """Freeze the push direction (world xy, unit norm) and required distance
    from the initial object_to_goal -- the goal never moves, and re-deriving
    this once the puck is airborne/sliding would be actively misleading (same
    reasoning as topple_block's frozen push_dir)."""
    if not self._push_dir_ok[i]:
      o2g = obs_i[43:46]
      d = float(np.linalg.norm(o2g[:2]))
      self._req_dist[i] = d
      t = (d - DIST_LO) / (DIST_HI - DIST_LO)
      self._strike_max_dq[i] = _lerp(STRIKE_MAX_DQ_MIN, STRIKE_MAX_DQ_MAX, t)
      self._strike_step_budget[i] = int(
        round(_lerp(STRIKE_STEPS_MIN, STRIKE_STEPS_MAX, t))
      )
      dirxy = o2g[:2] / (d + 1e-8)
      self._push_dir[i] = np.array([dirxy[0], dirxy[1], 0.0])
      self._push_dir_ok[i] = True
    return self._push_dir[i]

  def _target_error(self, i: int, obs_i: np.ndarray):
    raw_gto = obs_i[40:43]
    if self._gto_init[i]:
      if np.linalg.norm(raw_gto - self._prev_gto[i]) > RESET_JUMP:
        self._rewind(i)
    else:
      self._gto_init[i] = True
    self._prev_gto[i] = raw_gto

    push_dir = self._latch_push_dir(i, obs_i)
    gto = obs_i[40:43]  # puck - gripper

    contact = gto + np.array([0.0, 0.0, RIDE_HEIGHT])
    gripper_a = GRIPPER_CLOSED

    if self._phase[i] == 0:
      # Hover behind (toward the robot, i.e. -push_dir) and above the puck.
      pos_err = contact - push_dir * STANDOFF + np.array([0.0, 0.0, HOVER_Z])
      perp = pos_err - np.dot(pos_err, push_dir) * push_dir
      if np.linalg.norm(perp) < ALIGN_TOL or self._phase_steps[i] > ALIGN_TIMEOUT:
        self._phase[i] = 1
        self._phase_steps[i] = 0
    elif self._phase[i] == 1:
      # Seat just behind the puck's near face at ride height.
      raw = contact - push_dir * STANDOFF
      # MEASURED AND REVERTED: an integral term here (clip +-0.05, matching
      # topple_block's seat integrator) drove repeated ee_ground_collision
      # terminations DURING phase 1 itself -- the floor margin at this
      # object's height is only ~8mm (see RIDE_HEIGHT docstring), an order of
      # magnitude smaller than the +-5cm the integrator is allowed to wind up,
      # so any windup toward -z blew straight through it. The floor guard at
      # the end of this function clips the COMMANDED target, but the puck sits
      # close enough to the true collision limit that the transient path still
      # crosses it. Disabled; the plain proportional term plus the endgame
      # floor guard is the safer (if slower-converging) choice for this object.
      self._integ[i] = 0.0
      pos_err = raw + self._integ[i]
      seated = np.linalg.norm(raw) < CONTACT_TOL
      if seated:
        self._settle[i] += 1
      else:
        self._settle[i] = 0
      if self._settle[i] >= 2 or self._phase_steps[i] > SEAT_TIMEOUT:
        self._phase[i] = 2
        self._phase_steps[i] = 0
    elif self._phase[i] == 2:
      # WINDUP: retreat further behind, scaled by the required slide distance.
      d = self._req_dist[i]
      t = (d - DIST_LO) / (DIST_HI - DIST_LO)
      windup = _lerp(WINDUP_MIN, WINDUP_MAX, t)
      pos_err = contact - push_dir * (STANDOFF + windup)
      self._windup_steps[i] += 1
      if self._windup_steps[i] >= WINDUP_STEPS:
        self._anchor[i] = contact + push_dir * STRIKE_DEPTH
        self._phase[i] = 3
        self._phase_steps[i] = 0
    elif self._phase[i] == 3:
      # STRIKE: fixed anchor far beyond the puck, boosted max_dq, calibrated
      # duration. The base class reads ``self.max_dq`` once per act_single
      # call, so mutate it per-env here (single-env call, safe).
      self.max_dq = float(self._strike_max_dq[i])
      pos_err = self._anchor[i]
      self._strike_steps[i] += 1
      if self._strike_steps[i] >= self._strike_step_budget[i]:
        self._phase[i] = 4
        self._phase_steps[i] = 0
        self.max_dq = 0.12
    else:
      # RETREAT: up and back, then hold so a still-sliding puck is not
      # re-struck by a lingering hand.
      self.max_dq = 0.12
      if self._phase_steps[i] < RETREAT_STEPS:
        pos_err = -push_dir * RETREAT_BACK + np.array([0.0, 0.0, RETREAT_UP])
      else:
        pos_err = np.zeros(3)

    n = float(np.linalg.norm(pos_err))
    if n > MAX_WAYPOINT:
      pos_err = pos_err * (MAX_WAYPOINT / n)

    # ABSOLUTE FLOOR GUARD (robot base frame, via FK -- immune to the per-env
    # scene-origin offset that makes the absolute obs terms unusable here).
    q_abs = self.default_qpos + obs_i[0:9]
    z = float(self._fk(q_abs)[0][2])
    if z + pos_err[2] < FLOOR_MIN_Z:
      pos_err[2] = FLOOR_MIN_Z - z
    return pos_err, _DOWN_AXIS, gripper_a
