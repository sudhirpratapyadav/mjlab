"""Open-gripper Cage-Drag teacher using the shared planar transport controller.

The corrected task requires a physical aperture above approximately 69.5 mm,
actual geometric enclosure, and at least 5 mm of progress while caged. Every
policy action keeps the fingers open; external contacts can still force them
closed, invalidating the episode. Success is not mere cube/goal proximity.

The finger-closing axis is site y. The controller centers the open cage from
above, then uses its trailing inner pad to transport the cube toward the goal.
The September 11 centered-entry alternatives scored 5/32 against this teacher's
8/32 under the same CPU protocol and were rejected. See teacher_refresh/WORK_LOG.md.
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
  # solid contact, and the extra tipping moment does not matter: a 46 mm cube tipping onto another face keeps its
  # centre at the same height, although the corrected predicate also requires enclosure.
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
