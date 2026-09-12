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
* The board's ``object_site`` is the hole opening (body +0.015).
  The public ``object_to_goal`` now points from the peg TIP to its fully inserted
  TIP at floor height. Rewards and the goal replica use that same reference.
  `_act_single` converts this vector to the legacy internal center target so the
  existing grasp, carry and release waypoints retain their physical heights.

SUCCESS (StackingCommand insertion mode, corrected 2026-09-11)
------------------------------------------------------------
The square cross section must fit the straight bore, including at both ends when
slightly tilted; the peg must be within 3 mm of seated center height, within 5 degrees
of upright, moving slowly, and released (no robot contact). A hovering peg or a peg
lying across the board is not a successful insertion. Historical teacher rates use
a different predicate and must be remeasured.

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
  P_CARRY,
  P_DONE,
  P_HOVER,
  P_LIFT,
  P_PLACE,
  P_RELEASE,
  GraspTransportPolicy,
)

# How far ABOVE the tip site to close the fingers. The peg is 10cm; 0.075 puts the
# pads 2.5cm below the top face (a solid grip) and keeps 7.5cm of peg below them, so
# the fingers stay clear of the board while the lower half enters the hole.
# CL-V3 (W1-G, 2026-09-09) -- 0.055 IS GEOMETRICALLY IMPOSSIBLE. MEASURED on the private
# panda model (``hand_capsule``: capsule r 0.040, half-length 0.060, centre 0.070 behind
# the gripper site, axis along the site's y = the finger-closing axis): in a top-down
# pose the capsule's lowest point is exactly ``site_z + 0.030`` and it sits directly over
# the site's xy -- i.e. over the peg. The peg is 10 cm tall standing on the ground, so
# its top is at 0.100 and the site cannot go below 0.070 without the hand BODY landing on
# the peg's top face. Grasping at 0.055 drives the capsule 1.5 cm into the peg: the peg is
# shoved/toppled before the pads ever close, which is what the 12/31 ``grasp_lost`` and
# the exhausted-retry DONE envs in the n = 32 reads actually were. 0.078 leaves 8 mm of
# capsule clearance over the peg's top and still puts the pads on the peg's top 1.0-3.4 cm
# (the pads span site + 0.004 +- 0.008 below the site). The 2.8 cm pendulum below the
# pinch is the price and is paid for with a slow carry (``carry_max_dq`` 0.04).
GRASP_UP = 0.078  # 8 mm of hand-capsule clearance over the standing peg's top

# Extra descent past the point where the tip reaches the observed goal. RE-DERIVED
# for the CL-V2 geometry: the goal now sits at board_root_z + 0.035 = 0.050 (the
# height an inserted peg's CENTRE reaches). ``_place`` servos ``d[2] - INSERT_DEPTH``
# to zero, i.e. it drives tip_z -> goal_z - INSERT_DEPTH; the hole is a through-bore,
# so a fully inserted peg stands on the GROUND with its tip at z=0, which needs
# INSERT_DEPTH = goal_z = 0.050 to DRIVE it there. (It was 0.025 when the goal was at
# 0.025.)
#
# CL-V3 (W1-G, 2026-09-09) -- 0.050 IS THE WRONG NUMBER AND IT COST THE TASK.
# Driving the tip all the way to z = 0 puts the gripper SITE at tip + GRASP_UP =
# 0.055, i.e. the fingertip pads (1.4 cm below the site) at 0.041 and the whole hand
# inside the ground-collision envelope the moment the peg catches on the rim and the
# servo keeps pressing. MEASURED (diagnose, n = 32): 16 of 31 failures are
# ``ee_ground_collision`` inside PLACE, with the site logged down at z 0.004-0.017 --
# far past the 0.022 floor guard, because the guard clamps the TARGET while the
# command lead (0.12 rad ~ 6 cm) keeps the command below the actual joints.
# The peg does not have to be PUSHED to the ground: the bore is a through-hole with
# 2.5 mm clearance per side, so once the tip is a couple of centimetres inside it the
# peg is captured laterally and falls the rest of the way on its own when the fingers
# open. 0.032 drives the tip to z = 0.018 (1.2 cm below the board top, inside the
# straight bore, past the chamfer that ends at 0.024) and releases there; the peg
# drops 1.8 cm and stands with its centre at the goal's 0.050. The site at release is
# then 0.073 and the pads 0.059 -- nowhere near the floor.
# CL-V3 (W1-G2, 2026-09-09) -- DO NOT PRESS THE PEG IN. DROP IT IN.
# MEASURED (diagnose, n = 32, trace `object_quat`): in essentially every failure the peg
# is TILTED or flat by the end -- the body z-axis' vertical component is 0.60/0.45/0.41 at
# the best-aligned moment (53-66 deg off vertical) and 0.00 +-0.5 at the end, with the peg
# centre at 0.042 = a 25 mm post LYING FLAT on the 30 mm-thick board. 13 of 30 failures are
# `ee_ground_collision` in PLACE with the aperture at 0.000 (the peg already gone).
# The mechanism is a lever: the pads can only grip the peg's top 1-3.4 cm (the hand capsule
# blocks anything lower, see GRASP_UP), so the pinch is 2.8 cm ABOVE the peg's centre of
# mass and ~9 cm above its tip. The instant the tip touches the board off-centre, the
# arm's downward command pivots the peg about that contact and tips it out of the pads --
# and the compliance spiral, which drags the pressed tip across the board, does it faster.
# The bore is a THROUGH-hole with a 45 deg x 6 mm chamfer whose mouth is 4.2 cm across on
# a 2.5 cm peg, i.e. an 8.5 mm capture radius. So the peg does not need to be pushed
# anywhere: hold it VERTICAL with its tip 1.5 cm above the board, centre it, and OPEN.
# It falls into the mouth, the chamfer funnels it, and it stands on the ground with its
# centre at the goal's 0.050. Same lesson as Place-In-Container: drop, do not lower.
# ``_place`` servos ``d[2] - INSERT_DEPTH`` to zero, i.e. tip_z -> 0.050 - INSERT_DEPTH.
INSERT_DEPTH = 0.005  # release the tip at z 0.045, 1.5 cm above the board top

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


# CL-V3 (W1-G2, 2026-09-10) -- PITCH THE HAND AND GRIP THE PEG AT ITS CENTRE OF MASS.
#
# The problem the pitch solves, measured end to end:
#  * ``hand_capsule`` (r 0.040, half-length 0.060, centre 0.070 behind the gripper site,
#    axis along the finger-closing axis) reaches to exactly ``site_z + 0.030`` in a
#    TOP-DOWN pose, directly over the site's xy. A 10 cm peg standing on the ground
#    therefore blocks the site above 0.070, so a top-down grasp can only take the peg's
#    top 1-3.4 cm -- 2.8 cm ABOVE its centre of mass and ~9 cm above its tip.
#  * That pinch is a pendulum AND, whenever the closing axis is a few degrees off the
#    peg's faces, a HINGE. MEASURED (diagnose, n = 32): the aperture the pads actually
#    stall at runs 0.0249-0.0409 on a 25 mm peg whose diagonal is 0.0354 (median 0.0289 =
#    ~13 deg of yaw error, i.e. two EDGE contacts, not two faces), and the peg first tilts
#    past 32 deg during LIFT in 13 envs, CARRY in 9 and PLACE in 10 -- with the grip still
#    closed at 0.025-0.029 every time. A hinged, top-heavy grip cannot deliver a vertical
#    peg to a 30 mm bore however good the servo is.
#  * Pitching the approach axis by GRASP_PITCH about the (horizontal) closing axis swings
#    the capsule sideways: its nearest surface moves to 0.070*sin(beta) - 0.040 from the
#    site axis, which clears the peg's 12.5 mm half-width for beta >= 60 deg, and its
#    lowest point drops to 0.070*cos(beta) - 0.040 (BELOW the site at 65 deg). The site
#    can then descend the peg to its centre of mass, where there is no pendulum at all.
#  * Reachability probed on the CPU with this repo's own DLS over the peg's spawn box at
#    both the grasp height (0.050) and the insert height (0.095), 50 poses: position
#    residual median 3 mm / max 5 mm and orientation residual under 0.5 deg at every pitch
#    from 0 to 75 deg (with posture_weight 0.0005 and ik_joint_limit_factor 0.9). The
#    pitched pose is no harder to reach than the top-down one.
# The pads stay on two vertical faces because the closing axis stays horizontal, and the
# descent stays vertical: with the jaws open the pads straddle the peg all the way down
# and only the capsule needed to get out of the way.
GRASP_PITCH = np.radians(65.0)


def pitched_frame(theta: float, r_hat: np.ndarray, pitch: float = GRASP_PITCH) -> np.ndarray:
  """EE rotation: finger-closing axis (site y) horizontal at heading ``theta``; approach
  axis (site z) pitched ``pitch`` from straight down, leaning so the hand BODY ends up on
  the robot side of the peg (``r_hat`` = outward radial direction to the object)."""
  ey = np.array([np.cos(theta), np.sin(theta), 0.0])
  down = np.array([0.0, 0.0, -1.0])
  ex0 = np.cross(ey, down)  # horizontal, perpendicular to the closing axis
  if float(np.dot(ex0[:2], r_hat[:2])) < 0.0:
    ex0 = -ex0  # keep the capsule (which sits opposite the approach axis) near the base
  ez = np.cos(pitch) * down + np.sin(pitch) * ex0
  ex = np.cross(ey, ez)
  return np.column_stack([ex, ey, ez])


class PegInsertionClassicalPolicy(GraspTransportPolicy):
  """Grasp the standing peg near its top, align the tip over the hole, insert."""

  gto_slice = slice(37, 40)  # 51-D two-object layout
  o2g_slice = slice(40, 43)

  # Yaw is CONSTRAINED here (unlike stack): the peg is square and the hole is square,
  # and a 45deg-rotated 2.4cm square has a 3.4cm diagonal -- wider than the 3cm hole.
  #
  # CL-V3 (W1-G): the peg now spawns with +-30 deg of yaw (the board stays at 0, W0-a's
  # narrowing: the hole yaw is unobservable). So the grasp aligns the finger-closing
  # axis with the PEG's faces (``yaw_align`` from ``object_quat``, 4-fold symmetry),
  # and during CARRY/PLACE the latched heading is ramped to the nearest multiple of
  # 90 deg, which rotates the held peg's faces parallel to the hole's walls.
  grasp_yaw = None
  yaw_align = True
  quat_slice = slice(21, 25)
  yaw_symmetry = np.pi / 2
  hole_yaw = 0.0  # the board's yaw on the frozen (narrowed) spec
  yaw_ramp = 0.02  # rad per control step while rotating the held peg (0.05 swung it)

  # CL-V3 (W1-G): Lift-Cube's grasp phases (see stack_object.GraspTransportPolicy):
  # the grasp point is 7.5 cm up the standing peg, i.e. an ABSOLUTE site height of
  # 0.075 (the tip site rests on the ground); hover at 0.22. Jump-based reset
  # detection (the "home" test is dead under +-10 deg joint noise); retries from DONE.
  grasp_style = "lift"
  lift_grasp_site_z = GRASP_UP
  lift_hover_site_z = 0.22
  lift_max_dq = 0.05
  lift_integ_gain = 0.05  # the yaw-constrained frame carries a 2-3.6 cm DLS bias; 0.15/0.04 with the 0.12 lead ORBITS (Tool-Pull CPU trace)
  lift_integ_clip = 0.02
  lift_align_tol = 0.012  # a 25 mm peg: closing 2-3 cm off-centre tilts it before the pinch
  lift_seat_xy_tol = 0.012
  # A 10 cm standing post is toppled both by the pad edge arriving at the finger servo's
  # step speed and by the lift-style CLOSE's lateral creep, so the close is RAMPED and
  # the xy creep is off; the descent is halved for the same reason (0.02/step covers the
  # 14 cm from the hover in 7 steps, and a pad that clips the peg at that speed knocks it
  # over). Most of the exhausted-retry DONE envs are ``_placed_ok`` giving up on a peg
  # that is no longer upright.
  # NEGATIVE RESULT (W1-G2, n=32): ramping the close (12 steps), freezing the lateral
  # creep during it and halving the descent rate -- all three aimed at "the pad edge
  # topples the standing post" -- measured 0.281 -> 0.156 TOGETHER. Reverted to the
  # spine defaults; the hooks stay in ``stack_object.py`` (default-off) for whoever
  # tests them one at a time.
  # Do not close until the pads are actually facing the peg's faces (see
  # ``close_orient_tol``): a 13 deg yaw error already widens the stalled aperture from
  # 25 mm to 29 mm, which means edge contacts, which means a hinge.
  close_orient_tol = 0.10
  # NEGATIVE RESULT (W1-G2, n = 64), kept in the code because the mechanism it fixes is
  # real and the next person should not have to re-derive it. ``grasp_pitch = GRASP_PITCH``
  # (65 deg) with GRASP_UP = 0.050 (the peg's CoM), posture_weight 0.0005 and
  # ik_joint_limit_factor 0.9 DOES fix the grip: the aperture the pads stall at goes from
  # a 0.0249-0.0409 spread (median 0.0289 = ~13 deg of yaw error = two EDGE contacts on a
  # 25 mm peg whose diagonal is 0.0354) to 0.0244-0.026 in 21 of 29 envs -- proper face
  # grips at the centre of mass, no pendulum. It did NOT pay: 0.281 -> 0.172, because it
  # trades one tilt mechanism for another. With the hand pitched, the CARRY's gravity sag
  # (site logged sinking 0.19 -> 0.12 against a hold-at-0.205 command) makes the DLS trade
  # ORIENTATION against the growing position error, so the hand tilts and the peg, held in
  # a firm grip (aperture pinned at 0.025 the whole time), tilts with it: 22/32 envs first
  # exceed 32 deg of tilt inside CARRY. Referencing the carry to the previous command
  # (W1-D's ratchet cure) was tried on top and measured 0.156. Set grasp_pitch to
  # GRASP_PITCH to re-enable.
  grasp_pitch = 0.0
  lift_climb_to = 0.20  # tip clears the 3 cm board top; a fixed 30-step climb went to 0.37
  carry_max_dq = 0.04  # the pinch holds the peg against swing only by pad friction
  reset_detect = "jump"
  retry_from_done = True
  max_attempts = 5
  lead = 0.12
  # PLACE: exit on xy AND z jointly (phase-1 flagged the z-only exit), descend slowly
  # near the mouth, and on a stall do a small spiral in xy while keeping z pressure.
  place_xy_tol = 0.006  # the chamfer's capture radius is 8.5 mm; leave margin for the drop
  # Absolute site floor DURING THE INSERT. The grasp itself has to reach site z 0.055
  # (GRASP_UP above the tip resting on the ground), so ``floor_min_z`` cannot be
  # raised globally; this clamp applies only while pressing the peg down, where the
  # site should never need to go below tip_target + GRASP_UP = 0.018 + 0.055 = 0.073.
  place_floor_z = 0.085
  # Command lead while pressing (see ``_lead_for``): the full 0.12 rad lets the
  # command sit ~6 cm below the actual joints and sinks the wrist into the floor when
  # the peg catches on the rim.
  place_lead = 0.04
  place_slow_rate = 0.010
  place_stall_steps = 12
  wiggle_radius = 0.003  # legacy; the search now uses the expanding spiral below
  wiggle_period = 16
  spiral_growth = 0.0006  # legacy: the compliance search is gone (it tipped the peg over)
  spiral_max = 0.014
  place_timeout = 120
  place_settle = 4  # consecutive steps inside the release gate before opening

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

  # CL-V3 (W1-G): back to the spine's 0.030 now that ``INSERT_DEPTH`` releases the peg
  # inside the bore instead of driving it to the ground -- the deepest the site ever
  # needs to go is 0.055 for the GRASP, and 0.030 does not clamp that. The note below
  # is the CL-V2 reasoning for the old 0.022 and is kept because it explains why the
  # guard cannot simply be raised to the insert height.
  #
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
  floor_min_z = 0.030
  # Guard the true lowest hand point (see GraspTransportPolicy.guard_mode). This task
  # works at 1-2 cm of ground clearance during the insert and 13/30 first-terminations
  # were floor contacts.
  # NEGATIVE RESULT (W1-G2, n=32): 0.188 -> 0.062. This task's targets are all well clear
  # of the floor once the insert releases early (site 0.123 at the drop), so the hand
  # guard only LOOSENS the site floor from 0.030 to ~0.020 here and buys nothing.
  guard_mode = "site"

  def _lead_for(self, i: int) -> float:
    ph = self._phase[i]
    return self.place_lead if ph in (P_PLACE, P_RELEASE) else self.lead

  # Disabled with the pitch (see grasp_pitch): on the top-down tree the carry sag does not
  # tilt the peg, and command-referencing it measured no better.
  carry_cmd_ref = False

  def _act_single(self, i, obs_i):
    """Reference the joint command to the PREVIOUS COMMAND while carrying.

    W1-D's descent-ratchet defect: with ``cmd_ref = "actual"`` the IK target is
    FK(actual) + err, so the servo's own sag is re-added to the target every step and the
    position loop's fixed point is "error == sag" -- about 0.11 rad of shoulder droop.
    On this task that is not a cosmetic offset: the CARRY is supposed to hold the peg's
    tip at 0.155 and the trace shows the site sinking 0.19 -> 0.12 instead, and because
    the hand is PITCHED the DLS trades orientation against that growing position error --
    so the hand tilts, and the peg, held in a firm face grip (aperture pinned at 0.025
    throughout), tilts with it. 22 of 32 envs first tilt past 32 deg inside CARRY.
    There is no floor to protect at carry altitude, so referencing every axis to the
    command is safe here; the grasp and insert phases keep "actual" so ``_guard`` and
    ``place_floor_z`` still bind.
    """
    # The public observation now points tip-to-inserted-tip. Keep the internal
    # transport controller's center-height reference without changing its physical
    # waypoints (the peg is 100 mm long, desired orientation upright).
    obs_i = obs_i.copy()
    obs_i[self.o2g_slice][2] += .05
    if self._phase[i] == P_RELEASE:
      self.cmd_ref = "command"
      self.cmd_ref_axes = None
    elif self.carry_cmd_ref and self._phase[i] in (P_LIFT, P_CARRY):
      self.cmd_ref = "command"
      self.cmd_ref_axes = None
    else:
      self.cmd_ref = "actual"
    return super()._act_single(i, obs_i)

  def _approach_rot(self, i: int, obs_i: np.ndarray):
    """Pitched closing frame (see GRASP_PITCH). Falls back to the spine's top-down frame
    when ``grasp_pitch`` is 0, so the old behaviour is one constant away."""
    if self.grasp_pitch <= 0.0:
      return super()._approach_rot(i, obs_i)
    q_abs = self.default_qpos + obs_i[0:9]
    site = self._fk(q_abs)[0]
    obj_xy = site[:2] + self._gto(i, obs_i)[:2]  # peg tip xy in the robot base frame
    n = float(np.linalg.norm(obj_xy))
    r_hat = obj_xy / n if n > 1e-6 else np.array([1.0, 0.0])
    return pitched_frame(self._closing_heading(i, obs_i), r_hat, self.grasp_pitch)

  def _gripper_to_grasp(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    # obs points at the TIP; the grasp point is GRASP_UP above it.
    return self._gto(i, obs_i) + np.array([0.0, 0.0, GRASP_UP])

  # Running-mean object estimator (see GraspTransportPolicy.gto_estimator): this task's
  # whole margin is the chamfer's 8.5 mm capture radius against a 10-13 mm/axis
  # observation noise, so the estimator is the binding term, not the servo.
  gto_estimator = "mean"

  def _o2g_s(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    """SMOOTHED object_to_goal. The spine EMAs ``gripper_to_object`` but hands the RAW
    ``object_to_goal`` to ``_carry``/``_place`` -- fine for a 3 cm stacking tolerance,
    fatal here: this task's exits are ``|d[:2]| < 0.012`` (carry) and ``< 0.008``
    (place) evaluated on an observation carrying +-1 cm of uniform noise PER AXIS, so
    they fire at random while the true tip is a centimetre off the 30 mm bore. The
    chamfer only captures +-6 mm. Same alpha as the spine's gto EMA."""
    raw = obs_i[self.o2g_slice]
    if self.gto_estimator == "mean":
      # The GOAL (the board's hole) never moves, so average its position in the robot
      # base frame: goal = FK(q_obs) + gripper_to_object + object_to_goal. Combined with
      # the hand-frame mean of ``gripper_to_object`` after the grasp, both halves of this
      # vector are then averages over tens of samples instead of 3-sample EMAs.
      site = self._fk(self.default_qpos + obs_i[0:9])[0]
      self._goal_sum[i] += site + obs_i[self.gto_slice] + raw
      self._goal_n[i] += 1
      goal = self._goal_sum[i] / self._goal_n[i]
      return goal - site - self._gto(i, obs_i)
    if not self._o2g_ok[i]:
      self._o2g_ema[i] = raw
      self._o2g_ok[i] = True
    else:
      self._o2g_ema[i] = self.ema_alpha * raw + (1 - self.ema_alpha) * self._o2g_ema[i]
    return self._o2g_ema[i]

  def reset(self, env_ids=None) -> None:
    super().reset(env_ids)
    if env_ids is None:
      self._release_hold = np.full((self.num_envs, 3), np.nan)
      self._best_dz = np.full(self.num_envs, -np.inf)
      self._stall = np.zeros(self.num_envs, dtype=np.int64)
      self._wiggle_t = np.zeros(self.num_envs, dtype=np.int64)
      self._o2g_ema = np.zeros((self.num_envs, 3))
      self._o2g_ok = np.zeros(self.num_envs, dtype=bool)
      self._goal_sum = np.zeros((self.num_envs, 3))
      self._goal_n = np.zeros(self.num_envs, dtype=np.int64)
    else:
      self._release_hold[env_ids] = np.nan
      self._best_dz[env_ids] = -np.inf
      self._stall[env_ids] = 0
      self._wiggle_t[env_ids] = 0
      self._o2g_ok[env_ids] = False
      self._goal_sum[env_ids] = 0.0
      self._goal_n[env_ids] = 0

  def _plan(self, i, obs_i):
    releasing = self._phase[i] == P_RELEASE
    if releasing:
      position, _ = self._fk(self.default_qpos + obs_i[:9])
      if self._phase_steps[i] <= 1 or np.isnan(self._release_hold[i]).any():
        self._release_hold[i] = position
    error, rotation, grip = super()._plan(i, obs_i)
    if releasing:
      # Zero relative error follows the falling arm and lets it knock over the
      # released peg. Keep this pose fixed while the fingers open, then retreat.
      error = self._release_hold[i] - position
    return error, rotation, grip

  def _closing_heading(self, i: int, obs_i: np.ndarray) -> float:
    """Faces of the peg while grasping; ramp to the hole's yaw once it is held."""
    if self._phase[i] >= P_CARRY and not np.isnan(self._yaw_target[i]):
      sym = self.yaw_symmetry
      target = self.hole_yaw + np.round((self._yaw_target[i] - self.hole_yaw) / sym) * sym
      d = float(target - self._yaw_target[i])
      self._yaw_target[i] += float(np.clip(d, -self.yaw_ramp, self.yaw_ramp))
      return float(self._yaw_target[i])
    return super()._closing_heading(i, obs_i)

  def _peg_upright(self, obs_i: np.ndarray) -> bool:
    w, x, y, z = obs_i[self.quat_slice]
    # z component of the body z-axis: R[2,2] = 1 - 2(x^2 + y^2)
    return bool(1.0 - 2.0 * (x * x + y * y) > 0.7)

  def _placed_ok(self, i: int, obs_i: np.ndarray) -> bool:
    """Seated: tip on the ground through the bore. object_to_goal = (board_root +
    0.035) - tip, so a seated peg reads d_z ~ 0.05 (on the board top 0.02, on the
    chamfer 0.029) with xy inside the 0.015 predicate."""
    d = self._o2g_s(i, obs_i)
    if not self._peg_upright(obs_i):
      # Fallen over: this teacher cannot re-grasp a lying peg. Stop retrying.
      self._attempts[i] = self.max_attempts
      return True
    return bool(np.linalg.norm(d[:2]) < 0.012 and d[2] > 0.04)

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
    d = self._o2g_s(i, obs_i)
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
    """Hold the peg VERTICAL over the hole, centre it, and let it go.

    No pressing, no spiral: see the INSERT_DEPTH note. The only two things that matter
    here are (a) the tip is above the chamfer's 8.5 mm capture radius and (b) the peg is
    still vertical, which it is exactly as long as nothing pushes it against anything.
    """
    d = self._o2g_s(i, obs_i)
    self._integ[i] = np.clip(
      self._integ[i] + 0.18 * np.array([d[0], d[1], 0.0]), -0.06, 0.06
    )
    if self._phase_steps[i] == 0:
      self._settle[i] = 0
    lat = float(np.linalg.norm(d[:2]))
    z_err = d[2] - INSERT_DEPTH
    err = np.array([d[0], d[1], float(np.clip(z_err, -self.place_slow_rate, 0.03))])
    err = err + self._integ[i]
    # Hard site floor: the drop height needs site = tip + GRASP_UP = 0.045 + 0.078, so
    # anything below ``place_floor_z`` means something has gone wrong; never command it.
    z_site = self._site_z(obs_i)
    err[2] = max(err[2], self.place_floor_z - z_site)
    # Release gate held over consecutive steps (obs noise is +-1 cm per axis even EMA'd).
    if abs(z_err) < 0.008 and lat < self.place_xy_tol:
      self._settle[i] += 1
    else:
      self._settle[i] = 0
    if self._settle[i] >= self.place_settle:
      self._phase[i] = P_RELEASE
      self._phase_steps[i] = 0
      self._settle[i] = 0
    elif self._phase_steps[i] > self.place_timeout:
      if lat < 0.012:
        self._phase[i] = P_RELEASE  # inside the chamfer's capture: let it go
      else:
        self._phase[i] = P_CARRY  # re-centre from carry altitude (unconditional exit)
        self._attempts[i] += 1
      self._phase_steps[i] = 0
      self._settle[i] = 0
    return err, rot, GRIPPER_CLOSED
