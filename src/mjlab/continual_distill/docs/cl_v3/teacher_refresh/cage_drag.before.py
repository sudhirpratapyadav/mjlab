"""Scripted Cage-Drag teacher (CL-V3, W1-P).

*** THE TRAP (unchanged from phase 1) ***
``CageDragCommand`` latches the MINIMUM of ``finger_joint1 + finger_joint2`` over the
episode and success requires ``min_aperture > 0.055`` as well as goal error < 0.03. The
gripper action is therefore a hard constant (``GRIPPER_OPEN``) on every code path and
``__call__`` asserts it — a closing action anywhere would void the episode silently.

Geometry (MEASURED with panda.xml FK, see ``push_cuboid.py`` / ``logs/W1-P.md``): the
finger-closing axis is the site **y** axis. The phase-1 teacher believed it was x and
yawed the hand so that site x lay along object→goal: the open fingers then straddled the
goal line LEFT/RIGHT of the cube with nothing behind it — its "trailing pad" sat 4.5 cm
off to the side and transport happened by side friction only (0.141 → 0.016).

Strategy: ``PlanarPushPolicy`` in cage mode — the finger axis (site y) is put ALONG the
goal line so the two open pads (inner faces at y = ±q_finger ≈ ±0.040, an 8.0 cm gap)
cage the 46 mm cube fore-and-aft (support ≤ 0.065 at 45° yaw). The cube ENTERS the cage
from the side at ride height (never from above, never at speed: momentum on a pad is
what forced the aperture below 0.055 in phase 1), the trailing pad's inner face is the
paddle (offset −q_finger from the site, read from the finger observation), the lead /
steering / stop-and-hold law is the shared one, and the leading pad makes the stop
precise. The hand starts flush with the cube's nearest axis (widest entry clearance) and
turns toward the goal line while pushing; the cube follows the flat pad.
"""

from __future__ import annotations

import numpy as np

from mjlab.continual_distill.classical.push_cuboid import (
  PAD_HALF_WIDTH_OPEN,
  PlanarPushPolicy,
)

# Never anything else. See the module docstring.
GRIPPER_OPEN = 1.0


class CageDragClassicalPolicy(PlanarPushPolicy):
  """Straddle the cube with fingers pinned open and drag it to the goal."""

  OBJ_HALF = np.array([0.0230, 0.0226, 0.0226])
  SUCCESS_TOL = 0.03
  CAGE = True
  GRIPPER_ACTION = GRIPPER_OPEN
  PAD_HALF_W = PAD_HALF_WIDTH_OPEN  # one open pad's inner face, across the push
  # Cube 0..0.0452. A mid-height contact would want the site at 0.0265, but with the
  # fingers OPEN the pads reach 5.5 cm out to the side, so a few degrees of wrist roll
  # drops a pad corner ~5 mm below the site's own clearance -- at 0.0265 that is the
  # floor, and 94 % of episodes terminated on `ee_ground_collision` (measured, n = 32).
  # At 0.032 the pad face still spans 0.0201-0.0366 against a 0.0452 cube, which is a
  # solid contact, and the extra tipping moment does not matter: the success predicate
  # is the cube's CENTRE position and a 46 mm cube tipping onto another face keeps its
  # centre at the same height.
  RIDE_Z = 0.032
  FLOOR_MIN_Z = 0.024
  # The push-height SAW-TOOTH is a Push-Cuboid/Drag-Pull device: it exists because the
  # 3.0 cm carton gives the pad only ~1 cm of contact band, so the height reference has
  # to be recycled to keep the pad slipping upward. The 4.5 cm cube has twice that band
  # and the caged cube is held fore-and-aft by both pads, so recycling only re-loads it.
  # MEASURED (n = 128): saw-tooth 0.383, climb-and-hold 0.461.
  PUSH_CYCLE = False
  PUSH_CREEP = 0.0009
  PUSH_CYCLE_AMP = 0.006
  # the cage is a precise stop by construction; a slightly larger lead moves the cube
  # briskly and the leading pad bounds any overshoot
  LEAD_MAX = 0.045

  def __call__(self, obs: np.ndarray, *, active_env_ids=None) -> np.ndarray:
    actions = super().__call__(obs, active_env_ids=active_env_ids)
    assert np.all(actions[:, 7] >= GRIPPER_OPEN - 1e-6), (
      "cage_drag policy attempted to close the gripper — this would silently "
      "void the episode via CageDragCommand's min-aperture latch"
    )
    actions[:, 7] = GRIPPER_OPEN
    return actions
