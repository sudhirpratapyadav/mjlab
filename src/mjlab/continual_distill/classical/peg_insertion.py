"""Scripted PegInsertion teacher policy.

GEOMETRY (peg.xml / hole_board.xml, measured)
---------------------------------------------
* The peg is a 2.4cm-square, 10cm-tall box that spawns UPRIGHT with its centre at
  z=0.05, i.e. standing on the ground. Its ``object_site`` is at the peg TIP
  (body-local ``0 0 -0.05``), so ``gripper_to_object`` points at the BOTTOM of the
  peg, 5cm below its centre and 10cm below its top. Grasping where that vector
  points would drive the fingers into the ground.
  -> We grasp at ``tip + GRASP_UP`` (7.5cm up the peg, 2.5cm below the top). That
     leaves the whole lower half of the peg free to enter the hole.
* The board is a square frame with a 3cm x 3cm central gap and a 3cm-thick body
  (centre z=0.015, top z=0.03). The gap goes right through, so an inserted peg
  stands on the ground with its centre back at z=0.05.
* The board's ``object_site`` is the hole opening (body +0.015). ``object_to_goal``
  is (board_root + 0.01) - peg_tip, i.e. it points from the peg tip to a point 1cm
  above the board's centre.

SUCCESS (StackingCommand with tightened tolerances)
---------------------------------------------------
Measured on the peg's BODY ORIGIN (its centre), not on the tip site the observation
uses -- a 5cm offset that must be accounted for:
    xy error < 0.015   (half the hole's clearance: the peg is 2.4cm in a 3cm hole)
    |z error| < 0.03   against goal z = board_root_z + 0.01 = 0.025
A peg standing on the ground through the hole has centre z=0.05, so the height test
passes with 5mm to spare and the task reduces to the xy alignment. 1.5cm is tighter
than the ~1cm effective observation noise plus the DLS steady-state bias, which is
what makes this the hardest task in the set.

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
  P_PLACE,
  P_RELEASE,
  GraspTransportPolicy,
)

# How far ABOVE the tip site to close the fingers. The peg is 10cm; 0.075 puts the
# pads 2.5cm below the top face (a solid grip) and keeps 7.5cm of peg below them, so
# the fingers stay clear of the board while the lower half enters the hole.
GRASP_UP = 0.075

# Extra descent past the point where the tip reaches the observed goal. The goal sits
# at board_root_z + 0.01 = 0.025, i.e. 1cm below the board's top face; the hole is a
# through-gap, so a fully inserted peg stands on the GROUND with its tip at z=0. This
# offset therefore targets tip_z = 0 exactly (0.025 - 0.025), which is also where the
# peg physically bottoms out.
INSERT_DEPTH = 0.025


class PegInsertionClassicalPolicy(GraspTransportPolicy):
  """Grasp the standing peg near its top, align the tip over the hole, insert."""

  gto_slice = slice(37, 40)  # 51-D two-object layout
  o2g_slice = slice(40, 43)

  # Yaw is CONSTRAINED here (unlike stack): the peg is square and the hole is square,
  # and a 45deg-rotated 2.4cm square has a 3.4cm diagonal -- wider than the 3cm hole.
  # Both spawn at yaw 0, so holding the fingers on the world axes keeps the peg's
  # faces parallel to the hole's walls.
  grasp_yaw = 0.0

  hover_height = 0.13
  align_tol = 0.018
  descend_tol = 0.012
  lift_height = 0.12  # only needs to clear the 3cm board
  lift_steps = 20
  carry_tol = 0.012  # tight: the whole task is this xy alignment
  place_tol = 0.010
  release_steps = 16
  max_dq = 0.08  # slower than stack: precision over speed

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
