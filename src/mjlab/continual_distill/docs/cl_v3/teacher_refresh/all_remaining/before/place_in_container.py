"""Scripted PlaceInContainer teacher policy.

GEOMETRY (container.xml, measured -- CL-V2, W1-b)
-------------------------------------------------
The bin is a STATIC mocap body, and is now a real moulded plastic basket (GSO
`Spritz_Easter_Basket_Plastic_Teal`, 191 x 183 x 129 mm at scanned scale) rather than
five boxes. Measured cavity: a 4mm floor, an inner clear span of 16.3cm at the floor
flaring to 18.6 x 17.8cm, and a scalloped rim whose LOWEST point stands 11.3cm above
the underside (peaks 12.9cm). Its ``object_site`` -- the reference every tolerance is
measured against -- sits 2cm above the body origin, i.e. 1.6cm above the inner floor
surface and 9.3cm below the rim. The carried object is the 4.6cm mini cube.

SUCCESS (PlaceInContainerCommand) -- CONTAINMENT + RELEASE + SETTLING
---------------------------------------------------------------------
    lateral(cube - site) < 0.0585    inside the footprint (0.0815 inner half-span
                                     minus the cube's 0.0230 half-width, i.e. exactly
                                     "a face touching the wall")
    (cube - site).z      < 0.093     below the rim
    (cube - site).z      > -0.020    not tunnelled through the floor
    |cube linear velocity| < 0.12    released and settled, NOT still carried
The velocity term is the whole point: holding the cube perfectly inside the bin
scores ZERO, because a carried object inherits the arm's motion. The teacher must
open the fingers and wait.

WHY WE DROP RATHER THAN LOWER
-----------------------------
A cube resting on the bin floor sits 0.7cm above the site, well inside the band, so
the cube does not have to be placed gently -- it only has to end up inside.
Meanwhile the walls make a lowering approach expensive: the fingertips would have to
go INSIDE the basket while holding the cube, and the rim is now 11.3cm tall. Every
lowering variant clipped a wall and either popped the cube out or jammed the arm.
So: centre the cube over the bin at a safe altitude ABOVE the rim, hold still until
the swing has damped, open, and let it fall. The real basket makes this trade STRICTLY
better than it was on the primitive: the fall grew from 9.2cm to 14.3cm, but the
containment radius grew from an effective 3.4cm (0.054 inner half-span minus the old
2cm cube half-width) to 5.85cm -- a 3x larger landing area.

OBSERVATIONS: 60-D layout. [40:43] gripper_to_object (cube - gripper),
[43:46] object_to_goal (bin interior site - cube).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.stack_object import (
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  P_CARRY,
  P_DONE,
  P_PLACE,
  P_RELEASE,
  GraspTransportPolicy,
)

# Height ABOVE the bin's interior site at which the cube is released.
# RE-DERIVED for the real basket (W1-b). The binding constraint is the LATERAL CARRY,
# not the release: the cube travels in over the wall, so its underside must clear the
# rim's highest point. Scallop peaks are 0.129 above the bin's underside and the site
# is at 0.020, i.e. 0.109 above the site; plus the cube's 0.0226 half-height that is
# 0.132, and 0.15 leaves 18mm of servo margin. The fingers sit ~10mm above the cube's
# centre, so they clear the rim by ~4cm and never enter the basket.
DROP_HEIGHT = 0.15
# CL-V3 (W1-G): after arriving at DROP_HEIGHT above the rim, LOWER the cube to this height
# above the bin site before opening (cube centre 0.11 above the ground: hand capsule bottom
# 1.3 cm above the rim peaks, pads inside the opening). MEASURED (diagnose, 32 envs): 5/6
# failures were releases from 0.15 that landed outside the 5.85 cm footprint.
# CL-V3 (W1-G2, 2026-09-09): 0.15 -> 0.09 -> 0.060. The limit on going deeper is NOT the
# hand capsule (measured on the panda model: in a top-down pose the capsule's lowest point
# is site + 0.030, i.e. ABOVE the site -- it can never enter the basket before the fingers
# do) but the FINGER BODIES, whose outermost geom sits 0.066 from the site axis. At a cube
# centre 0.060 above the bin's interior site the gripper site is at 0.070 and the fingers
# span that height, where the basket's inner half-span is ~0.084 -- 18 mm of clearance if
# the hand is centred, which the release gate already guarantees. Halving the fall from
# 7 cm to 3.5 cm halves the drop scatter inside a 5.85 cm containment radius.
RELEASE_HEIGHT = 0.060
LOWER_RATE = 0.012
# Do not START lowering into the opening until the cube is roughly centred: the fingers
# only clear the wall by 18 mm at the release height, and lowering at 5 cm lateral puts
# them on the rim.
LOWER_GATE_LAT = 0.030
# Lateral gate for the release (must hold 3 consecutive steps). The containment radius
# is 0.0585; the cube then falls ~7 cm and bounces, so the drop's own scatter needs the
# margin. 0.030 was the old single-step gate.
RELEASE_LAT_TOL = 0.028

# Steps to hold the cube motionless over the bin before opening. The arm arrives with
# residual swing; releasing into that swing throws the cube sideways past the wall.
DAMP_STEPS = 25

# Steps to hold the (open) hand still after release so the cube can fall and settle
# below ``settle_speed``. Success latches during this window.
SETTLE_STEPS = 60


class PlaceInContainerClassicalPolicy(GraspTransportPolicy):
  """Grasp the cube, carry it above the bin's rim, damp out, open, let it drop."""

  gto_slice = slice(40, 43)  # 60-D layout
  o2g_slice = slice(43, 46)

  hover_height = 0.13
  align_tol = 0.022
  descend_tol = 0.012
  # Must clear the bin on the way across. RE-DERIVED (W1-b): the real basket's rim
  # peaks 0.129 above the ground where the primitive's stood at 0.082, so the required
  # RISE of the cube from its grasp height went from 0.082 to 0.129. This is a
  # commanded displacement, not an altitude, and the DLS solve does not fully converge
  # inside ``lift_steps``; keeping the same ~2x headroom over the requirement.
  lift_height = 0.26
  lift_steps = 26
  carry_tol = 0.020
  max_dq = 0.09
  # Losing the cube during the LIFT, not missing the bin, is this task's dominant
  # failure -- and it is worth recording because it contradicts the obvious guess.
  # Instrumented over 32 episode-instances, 19 runs dropped the cube and 12 of
  # those dropped it inside P_LIFT; only 13 runs ever kept hold of it at all. So
  # dropping from lower, centring harder or damping longer all tune a phase that
  # most failures never reach.
  #
  # Deepening the grasp to 0.004 (straddling the cube's centre of mass) and
  # ramping the lift were both tried against that diagnosis and measured WORSE
  # over 96 episode-instances: 0.302 baseline -> 0.188. This task has the tallest
  # carry in the file (lift_height 0.26, to clear a 12.9cm basket), so the lower wrist
  # pose costs more clearance than the firmer grip buys back. Kept at 1cm.
  #
  # W2-c (2026-09-09): the "LIFT is where it's lost" diagnosis above named the right
  # symptom (P_LIFT is where the loss first becomes DETECTABLE) but not the true
  # cause -- see the instrumented finding in ``stack_object.GraspTransportPolicy``'s
  # P_LIFT docstring, confirmed to reproduce here too (traced 16 envs of this task:
  # 14/17 established grasps dropped, 9 inside P_LIFT, and the CLOSE_ENTRY moment was
  # well-centred in essentially every case -- xy offset 1-2cm, descend converging in
  # 10-30 of the 60-step budget). The object is usually ejected sideways DURING
  # P_CLOSE itself (aperture collapses monotonically from the cube's true ~4cm
  # contact width to fully-closed while the gripper-cube xy offset grows in lockstep,
  # over P_CLOSE's own fixed hold window, before P_LIFT's first command is ever
  # issued). Two tested fixes (a partial ``grip_close_action`` instead of
  # fully-closed, -0.4 and -0.85) were measured WORSE on Stack-Cube (0.000/32 and
  # 0.188/32 vs baseline ~0.33) and NOT re-tried here since the mechanism -- and its
  # failure -- is shared; see stack_object.py and LOGS.md for the full record.
  grasp_z_offset = 0.010

  # CL-V3 (W1-G, 2026-09-09): same finding as Stack -- the old spine's fast,
  # unsettled descent lands the hand on the cube and shoves it before the close.
  # Lift-Cube's grasp phases reused (grasp_style="lift"), jump-based reset detector,
  # unconditional CARRY escape, retries from DONE while the budget lasts.
  grasp_style = "lift"
  reset_detect = "jump"
  carry_timeout = 90
  retry_from_done = True
  lead = 0.12
  # The carry has to clear the 12.9 cm rim: climb to site z 0.21 (cube bottom 0.155),
  # then carry slowly. MEASURED (diagnose, 32 envs): a fixed 36-step climb sent the
  # site to ~0.37 m and 7/8 failures lost the cube in the fast 0.18 m descent that the
  # carry then had to make.
  lift_climb_steps = 40
  lift_climb_to = 0.21
  carry_max_dq = 0.06

  def _placed_ok(self, i: int, obs_i: np.ndarray) -> bool:
    """Containment from object_to_goal (= bin site - cube): inside the 0.0585
    footprint, and the cube between the floor and the rim."""
    d = obs_i[self.o2g_slice]
    return bool(np.linalg.norm(d[:2]) < 0.055 and -0.09 < d[2] < 0.02)

  def _carry(self, i, obs_i, rot):
    """Centre the cube DROP_HEIGHT above the bin's interior site.

    ``object_to_goal`` points from the cube to that site, so the desired gripper
    displacement is that vector raised by DROP_HEIGHT. Both terms are relative, so
    the per-env origin offsets cancel.

    Integral action on the lateral axes, for the same reason as Stack: the DLS solve
    parks 2-3cm off laterally, and with a 5.85cm containment radius that bias plus the
    drop's own scatter is most of the budget.
    """
    d = obs_i[self.o2g_slice]
    self._integ[i] = np.clip(
      self._integ[i] + self.carry_integ_gain * np.array([d[0], d[1], 0.0]),
      -self.carry_integ_clip,
      self.carry_integ_clip,
    )
    err = d + self._integ[i] + np.array([0.0, 0.0, DROP_HEIGHT])
    if err[2] > 0.04:
      # Still well below the carry altitude: climb first, translate after, so the
      # cube's underside never sweeps into the basket's 12.9 cm rim.
      err[:2] = 0.0
      self._integ[i] = 0.0
    if np.linalg.norm(d[:2]) < self.carry_tol and self._phase_steps[i] > 10:
      self._phase[i] = P_PLACE
      self._phase_steps[i] = 0
    return err, rot, GRIPPER_CLOSED

  def _place(self, i, obs_i, rot):
    """Hold station over the bin until the residual swing has damped, then release.

    Keeping the servo on the same waypoint (rather than freezing the command) is what
    actually removes the swing: the DLS solve pulls the remaining error to zero while
    the arm decelerates.
    """
    d = obs_i[self.o2g_slice]
    # CL-V3 (W1-G, 2026-09-09) -- THE STALE-INTEGRATOR BUG. The integrator update used
    # to sit BELOW the damp branch, so for the whole DAMP_STEPS window ``_integ`` held
    # the value it had wound up to during the carry (clipped at +-0.06). MEASURED on the
    # diagnose trace (env 22, steps 115-150): the carry handed over at lat 0.013, the
    # frozen integrator then made the commanded error zero at a TRUE lateral offset of
    # 0.060-0.075 -- exactly ``-integ`` -- and the arm sat at that offset for the entire
    # damp window; every one of the five drop failures released between 0.056 and 0.098
    # lateral against a 0.0585 containment radius. The integrator now runs every step,
    # so the damp window converges instead of parking on the stored bias.
    self._integ[i] = np.clip(
      self._integ[i] + self.carry_integ_gain * np.array([d[0], d[1], 0.0]),
      -self.carry_integ_clip,
      self.carry_integ_clip,
    )
    lat = float(np.linalg.norm(d[:2]))
    if self._phase_steps[i] < DAMP_STEPS:
      err = d + self._integ[i] + np.array([0.0, 0.0, DROP_HEIGHT])
      err[:2] *= 0.6  # gentle terminal xy correction; avoid re-exciting the swing
      return err, rot, GRIPPER_CLOSED
    # Damped: lower into the opening at a bounded rate WHILE servoing xy at full gain
    # with the integrator (lowering reconfigures the arm and the xy error regrows:
    # diagnose #3 released at 5-10 cm lateral, every failure landed on the rim).
    z_err = d[2] + RELEASE_HEIGHT
    dz = max(z_err, -LOWER_RATE) if lat < LOWER_GATE_LAT else max(z_err, 0.0)
    err = np.array([d[0], d[1], dz]) + self._integ[i]
    # The release gate is checked on THREE CONSECUTIVE steps (``_settle``): obs noise is
    # +-1 cm per axis, so a single-step read of ``lat`` fires ~1 cm optimistic and the
    # cube leaves the hand at the edge of the footprint.
    if abs(z_err) < 0.015 and lat < RELEASE_LAT_TOL:
      self._settle[i] += 1
    else:
      self._settle[i] = 0
    if self._settle[i] >= 3:
      self._phase[i] = P_RELEASE
      self._phase_steps[i] = 0
      self._settle[i] = 0
    elif self._phase_steps[i] >= DAMP_STEPS + 80:
      if lat < 0.040:
        self._phase[i] = P_RELEASE  # inside the 5.85 cm footprint: let it go
      else:
        self._phase[i] = P_CARRY  # re-centre from carry altitude (unconditional exit)
        self._attempts[i] += 1
      self._phase_steps[i] = 0
      self._settle[i] = 0
    return err, rot, GRIPPER_CLOSED

  def _plan(self, i: int, obs_i: np.ndarray):
    """Override the tail of the spine: after opening we must STAY PUT, not retreat.

    Retreating immediately drags the still-closing fingers across the cube and can
    flick it out of the bin; it also risks clipping the rim on the way up. Holding
    the open hand still for SETTLE_STEPS lets the cube fall, bounce and settle, which
    is exactly the window in which ``at_goal`` (contained AND slow) can latch.
    """
    self._detect_reset(i, obs_i)  # the branches below may not reach super()
    if self._phase[i] == P_RELEASE:
      if self._phase_steps[i] >= SETTLE_STEPS:
        self._phase[i] = P_DONE
        self._phase_steps[i] = 0
      rot = self._approach_rot(i, obs_i)
      return np.zeros(3), rot, GRIPPER_OPEN
    if self._phase[i] == P_DONE:
      if (
        self.retry_from_done
        and self._phase_steps[i] > 30
        and self._attempts[i] < self.max_attempts
        and not self._placed_ok(i, obs_i)
      ):
        self._rewind(i)
        return super()._plan(i, obs_i)
      # Rise straight up, away from the rim, and stay open.
      return np.array([0.0, 0.0, 0.10]), self._approach_rot(i, obs_i), GRIPPER_OPEN
    return super()._plan(i, obs_i)
