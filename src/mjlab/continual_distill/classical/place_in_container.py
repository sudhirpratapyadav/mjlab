"""Scripted PlaceInContainer teacher policy.

GEOMETRY (container.xml, measured)
----------------------------------
The bin is a STATIC mocap body: an 8mm-thick floor with four walls whose inner clear
span is ~10.8cm and whose rim stands 7cm above the floor's centre. Its
``object_site`` -- the reference every tolerance is measured against -- sits 2cm
above the body origin, i.e. 1.2cm above the inner floor surface and 5cm below the
rim. The carried object is the 4cm cube.

SUCCESS (PlaceInContainerCommand) -- CONTAINMENT + RELEASE + SETTLING
---------------------------------------------------------------------
    lateral(cube - site) < 0.055     inside the footprint
    (cube - site).z      < 0.05      below the rim
    (cube - site).z      > -0.04     not tunnelled through the floor
    |cube linear velocity| < 0.12    released and settled, NOT still carried
The velocity term is the whole point: holding the cube perfectly inside the bin
scores ZERO, because a carried object inherits the arm's motion. The teacher must
open the fingers and wait.

WHY WE DROP RATHER THAN LOWER
-----------------------------
A cube resting on the bin floor sits 2.8cm above the site, well inside the +-5cm
band, so the cube does not have to be placed gently -- it only has to end up inside.
Meanwhile the walls make a lowering approach expensive: to put the cube 1cm below the
rim the fingertips must go INSIDE a 10.8cm span while holding a 4cm cube, leaving
~3cm of clearance per side for a gripper that is wider than that. Every lowering
variant clipped a wall and either popped the cube out or jammed the arm.
So: centre the cube over the bin at a safe altitude ABOVE the rim, hold still until
the swing has damped, open, and let it fall the last ~10cm. The fall is short enough
that the lateral scatter stays far inside the 5.5cm footprint, and the cube settles
under 0.12 m/s within ~20 steps.

OBSERVATIONS: 60-D layout. [40:43] gripper_to_object (cube - gripper),
[43:46] object_to_goal (bin interior site - cube).
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.stack_object import (
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  P_DONE,
  P_PLACE,
  P_RELEASE,
  GraspTransportPolicy,
)

# Height ABOVE the bin's interior site at which the cube is released. The rim is 5cm
# above the site, so this clears it by 5cm -- enough that the fingers (which hang
# level with the cube's centre) never enter the bin.
DROP_HEIGHT = 0.10

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
  lift_height = 0.22  # must clear the 9cm-tall bin on the way across
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
  # carry in the file (lift_height 0.22, to clear a 9cm bin), so the lower wrist
  # pose costs more clearance than the firmer grip buys back. Kept at 1cm.
  grasp_z_offset = 0.010

  def _carry(self, i, obs_i, rot):
    """Centre the cube DROP_HEIGHT above the bin's interior site.

    ``object_to_goal`` points from the cube to that site, so the desired gripper
    displacement is that vector raised by DROP_HEIGHT. Both terms are relative, so
    the per-env origin offsets cancel.

    Integral action on the lateral axes, for the same reason as Stack: the DLS solve
    parks 2-3cm off laterally, and with a 5.5cm containment radius that bias plus the
    drop's own scatter is most of the budget.
    """
    d = obs_i[self.o2g_slice]
    self._integ[i] = np.clip(
      self._integ[i] + self.carry_integ_gain * np.array([d[0], d[1], 0.0]),
      -self.carry_integ_clip,
      self.carry_integ_clip,
    )
    err = d + self._integ[i] + np.array([0.0, 0.0, DROP_HEIGHT])
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
    err = d + self._integ[i] + np.array([0.0, 0.0, DROP_HEIGHT])
    err[:2] *= 0.6  # gentle terminal xy correction; avoid re-exciting the swing
    if self._phase_steps[i] >= DAMP_STEPS:
      self._phase[i] = P_RELEASE
      self._phase_steps[i] = 0
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
      # Rise straight up, away from the rim, and stay open.
      return np.array([0.0, 0.0, 0.10]), self._approach_rot(i, obs_i), GRIPPER_OPEN
    return super()._plan(i, obs_i)
