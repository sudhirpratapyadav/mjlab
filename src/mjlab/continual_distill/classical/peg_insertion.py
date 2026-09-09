"""Scripted PegInsertion teacher policy.

GEOMETRY (peg.xml / hole_board.xml, CL-V2 shape-sorter assets, measured)
------------------------------------------------------------------------
* The peg is a 2.5cm-square, 10cm-tall wooden shape-sorter post (2 mm lead chamfer
  on the insert end) that spawns UPRIGHT with its centre at z=0.05, i.e. standing on
  the ground. Its ``object_site`` is at the peg TIP (body-local ``0 0 -0.05``), so
  ``gripper_to_object`` points at the BOTTOM of the peg, 5cm below its centre and
  10cm below its top. Grasping where that vector points would drive the fingers into
  the ground.
  -> We grasp at ``tip + GRASP_UP`` (7.5cm up the peg, 2.5cm below the top). That
     leaves the whole lower half of the peg free to enter the hole.
* The board is a 12cm-square, 3cm-thick plywood sorter lid (centre z=0.015, top
  z=0.03) with a 3cm square through-bore and a 45 deg x 6 mm lead-in chamfer, so the
  MOUTH is 4.2cm across at z=0.03 and narrows to 3.0cm at z=0.024. Clearance on the
  bore is 2.5 mm per side. The chamfer is real collision geometry (four rotated
  boxes), so a tip landing up to ~6 mm off-centre is funnelled in rather than
  stopping on the rim -- this is the capture range the whole task hangs on.
  An inserted peg stands on the ground with its centre back at z=0.05.
* The board's ``object_site`` is the hole opening (body +0.015). ``object_to_goal``
  is (board_root + 0.035) - peg_tip, i.e. it points from the peg tip to the height an
  inserted peg's CENTRE reaches.

SUCCESS (StackingCommand with tightened tolerances)
---------------------------------------------------
Measured on the peg's BODY ORIGIN (its centre), not on the tip site the observation
uses -- a 5cm offset that must be accounted for:
    xy error < 0.015   (6x the 2.5 mm per-side bore clearance: in the bore => success)
    |z error| < 0.015  against goal z = board_root_z + 0.035 = 0.050
A peg standing on the ground through the hole has centre z=0.05, so a fully seated
peg has ZERO height error; resting on the board top is 0.030 and jammed on the
chamfer is 0.021, both outside. The task therefore reduces to the xy alignment, and
1.5cm is tighter than the ~1cm effective observation noise plus the DLS steady-state
bias -- which is what makes this the hardest task in the set.

STRATEGY
--------
Grasp the peg high with the finger-closing axis at a fixed yaw, lift, translate so
the peg TIP is directly over the hole opening (the observation's ``object_to_goal``
already expresses exactly that, since it starts at the tip), then descend slowly and
release. A final settle phase lets the peg drop the last centimetre on its own so
that the fingers do not fight the hole walls.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.stack_object import (
  GRIPPER_CLOSED,
  P_HOVER,
  P_PLACE,
  P_RELEASE,
  GraspTransportPolicy,
)

# How far ABOVE the tip site to close the fingers. The peg is 10cm; 0.075 puts the
# pads 2.5cm below the top face (a solid grip) and keeps 7.5cm of peg below them, so
# the fingers stay clear of the board while the lower half enters the hole.
GRASP_UP = 0.075

# Extra descent past the point where the tip reaches the observed goal. RE-DERIVED
# for the CL-V2 geometry: the goal now sits at board_root_z + 0.035 = 0.050 (the
# height an inserted peg's CENTRE reaches). ``_place`` servos ``d[2] - INSERT_DEPTH``
# to zero, i.e. it drives tip_z -> goal_z - INSERT_DEPTH; the hole is a through-bore,
# so a fully inserted peg stands on the GROUND with its tip at z=0, which needs
# INSERT_DEPTH = goal_z = 0.050. (It was 0.025 when the goal was at 0.025.)
INSERT_DEPTH = 0.050

# Unconditional CARRY-phase escape. THE MECHANISM (instrumented 2026-09-09, W3-b):
# every other phase in the shared spine has an unconditional timeout fallback
# (DESCEND: phase_steps>60, CLOSE: phase_steps>=close_steps, LIFT: phase_steps>=
# lift_steps, PLACE: phase_steps>70) -- CARRY does not. Its only exit is
# ``norm(d[:2]) < carry_tol``, which is unreachable if the peg was never actually
# grasped (fingers close on nothing, or a hold established in CLOSE/LIFT is lost
# before/during CARRY): with nothing gripped the peg sits wherever it was dropped
# and ``object_to_goal`` reports the SAME large, roughly-constant offset forever, so
# the condition can never fire.
#
# Per-step traces (true `robot.data.site_pos_w` / `object.data.root_link_pos_w`,
# not the noisy obs) over 16 envs x 1000 steps show this is not rare: the
# gripper-empty signature (finger aperture collapses to ~0.0000, i.e. fully closed
# with NOTHING between the pads -- the peg's true 2.4cm width would hold it near
# 0.024-0.025) appears in 14/16 traced envs, and once it appears the env's own
# `xy_err`/`peg_z` FREEZE at a fixed value for the rest of the episode -- one
# instrumented env sat frozen in CARRY for 580+ consecutive steps out of a
# 1000-step budget on a SINGLE failed grasp, with zero further attempts, because
# nothing in this phase's own logic can ever break out of it. Two distinct root
# causes were seen feeding this same terminal state: the fingers closing on empty
# air during CLOSE (aperture goes to 0 without ever showing a true-width plateau),
# and a hold that WAS established (aperture briefly ~0.025, peg genuinely rising
# during LIFT/CARRY) being lost partway through the carry (aperture collapses from
# ~0.025 to ~0 mid-phase, peg_z falling back to ~0.012 -- the peg's lying-flat
# half-width, i.e. it topples over once dropped rather than merely sliding).
# Whichever the cause, only an EXTERNAL auto-reset (``ee_ground_collision`` /
# ``object_out_of_bounds``, caught by ``_detect_reset``) was ever observed to free
# a deadlocked env -- never the policy's own logic. The ONE success in a 16-env
# instrumented episode got there by having three complete grasp attempts inside the
# first 400 of 1000 steps (via auto-reset escapes each time an early attempt
# failed); envs that instead deadlocked in CARRY got only ONE attempt for the
# whole episode and could not succeed no matter how good the CARRY servo itself is
# (a noise-free kinematic replay of this exact CARRY/PLACE law, fed realistic
# +-1cm noise, converges to within ``carry_tol``/the success tolerance in ~95-100%
# of trials in under 10 of the 70 available PLACE steps when it actually gets a
# grasp to work with -- the servo is not the bottleneck, the lack of a retry is).
#
# Fix: give CARRY the same unconditional escape every other phase already has.
# Falls through to P_HOVER (not a task change -- this only touches how the
# TEACHER recovers from its own missed grasp), which is a state ``_detect_reset``
# already proves safe to land in (fingers command OPEN there, so a phantom "hold"
# is released rather than dragged around). The integrator and EMA are cleared so
# the retry does not inherit a bias/estimate from the failed attempt.
CARRY_TIMEOUT = 100


class PegInsertionClassicalPolicy(GraspTransportPolicy):
  """Grasp the standing peg near its top, align the tip over the hole, insert."""

  gto_slice = slice(37, 40)  # 51-D two-object layout
  o2g_slice = slice(40, 43)

  # Yaw is CONSTRAINED here (unlike stack): the peg is square and the hole is square,
  # and a 45deg-rotated 2.4cm square has a 3.4cm diagonal -- wider than the 3cm hole.
  # Both spawn at yaw 0, so holding the fingers on the world axes keeps the peg's
  # faces parallel to the hole's walls.
  grasp_yaw = 0.0

  # RE-DERIVED with INSERT_DEPTH: ``_carry`` holds the tip at goal_z + hover_height,
  # and goal_z moved from 0.025 to 0.050, so the carry altitude drops by the same
  # 0.025 to keep the peg tip at the SAME physical 0.155 m (well clear of the 0.03
  # board top, and low enough that the grasp point at tip+0.075 stays in reach).
  hover_height = 0.105
  align_tol = 0.018
  descend_tol = 0.012
  lift_height = 0.12  # only needs to clear the 3cm board
  lift_steps = 20
  carry_tol = 0.012  # tight: the whole task is this xy alignment
  place_tol = 0.010
  release_steps = 16
  max_dq = 0.08  # slower than stack: precision over speed

  # KEEP THE OLD, LOWER FLOOR GUARD. The shared spine raised ``floor_min_z`` to
  # 0.030 because the true end-effector collision clearance is ~1.4cm (pads) and
  # the hand capsule reaches ~3.1cm below the site, so 0.022 was tripping
  # ``ee_ground_collision`` on transient dips. That is right for tasks working
  # over the bare floor -- but this task deliberately drives the peg DOWN THROUGH
  # a hole to the ground (``_place`` targets ``d[2] - INSERT_DEPTH``), so a 0.030
  # guard clamps the insertion itself and the peg never seats: measured 0.000
  # across 32 episodes with the raised guard, against a 0.03-0.06 baseline.
  # The peg is inside the board's hole here, not over open floor, so the extra
  # margin buys nothing and costs the whole task.
  floor_min_z = 0.022

  def _gripper_to_grasp(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    # obs points at the TIP; the grasp point is GRASP_UP above it.
    return self._gto(i, obs_i) + np.array([0.0, 0.0, GRASP_UP])

  def _carry(self, i, obs_i, rot):
    """Hold the peg high and centre its TIP over the hole opening.

    ``object_to_goal`` starts at the tip, so translating the gripper by its xy
    component puts the tip over the hole -- no offset bookkeeping needed. We stay at
    carry altitude until xy is inside ``carry_tol``, because descending while still
    laterally off lands the peg on the board's rim instead of in the gap.

    INTEGRAL ACTION IS MANDATORY HERE. With yaw fully constrained the DLS solve
    leaves a persistent 2-3.6cm lateral steady-state offset (measured over the peg's
    spawn box: 0.016-0.036m, with the orientation term converging perfectly the whole
    time -- the posture and orientation weights are simply fighting the position
    term). That bias alone is larger than the 1.5cm success tolerance, so a purely
    proportional carry can never satisfy ``carry_tol`` and would just time out at
    whatever offset it stalled at. The integrator drives it to zero.
    """
    d = obs_i[self.o2g_slice]
    self._integ[i] = np.clip(
      self._integ[i] + 0.18 * np.array([d[0], d[1], 0.0]), -0.06, 0.06
    )
    err = np.array([d[0], d[1], 0.0]) + self._integ[i]
    err[2] = self.hover_height + d[2]
    if np.linalg.norm(d[:2]) < self.carry_tol and self._phase_steps[i] > 10:
      self._phase[i] = P_PLACE
      self._phase_steps[i] = 0
      # Keep the integrator: it is holding out the same steady-state bias during the
      # descent, and zeroing it here would let the peg drift straight back off-centre.
    elif self._phase_steps[i] > CARRY_TIMEOUT:
      # UNCONDITIONAL ESCAPE -- see CARRY_TIMEOUT's module-level note. Reaching here
      # means carry_tol was never satisfied in 100 steps, which the noise-free probe
      # says should take under 25; the only way that happens is an empty or lost
      # grip. Give up on this attempt and retry from HOVER rather than burn the rest
      # of the episode's step budget on a peg that is not moving.
      self._phase[i] = P_HOVER
      self._phase_steps[i] = 0
      self._integ[i] = 0.0
      self._ema_ok[i] = False
    return err, rot, GRIPPER_CLOSED

  def _place(self, i, obs_i, rot):
    """Lower the tip through the opening, correcting xy the whole way down.

    The vertical target is the observed goal MINUS ``INSERT_DEPTH`` so the peg is
    driven down to the ground through the hole rather than stopping 1cm above the
    board. xy is servoed continuously: the hole's chamfer-free square walls give no
    self-centring, so any lateral error at contact wedges the peg on the rim.
    """
    d = obs_i[self.o2g_slice]
    target = np.array([d[0], d[1], d[2] - INSERT_DEPTH])
    self._integ[i] = np.clip(
      self._integ[i] + 0.18 * np.array([d[0], d[1], 0.0]), -0.06, 0.06
    )
    if abs(target[2]) < self.place_tol or self._phase_steps[i] > 70:
      self._phase[i] = P_RELEASE
      self._phase_steps[i] = 0
    # Descend at a bounded rate so contact with the rim is gentle enough to slide.
    target[2] = np.clip(target[2], -0.03, 0.03)
    return target + self._integ[i], rot, GRIPPER_CLOSED
