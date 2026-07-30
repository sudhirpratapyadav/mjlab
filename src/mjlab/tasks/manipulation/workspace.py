"""Class A (Franka arm + 2-finger gripper) workspace definition.

**Single source of truth for where objects may be placed in Class A tasks.**

Every Class A task shares one embodiment, so it shares one reachable workspace. Spawn
ranges were previously chosen per task by eye, which put several tasks in the sparse
tail of the arm's reachable set (see the numbers below) — objects at the very edge of
reach, mechanisms the gripper can only touch in near-singular configurations. This
module replaces that guesswork with a measured envelope plus per-task placement rules.

## How these numbers were obtained

Forward kinematics over 600k uniform joint samples drawn from **90% of each joint's
range** (staying off the hard stops, which ``joint_pos_limits`` penalises at -10.0),
recording the ``gripper`` site pose. "Top-down" means the EE z-axis is within ~32 deg
of straight down, the approach every Class A grasp uses.

Reproduce with ``python -m mjlab.scripts.audit_workspace``.

Measured radial reach (sqrt(x^2+y^2) from the base) for TOP-DOWN poses:

    site height z        p5      p50     p90     p95
    0.06-0.14 (grasp)    0.276   0.414   0.726   0.783
    0.14-0.25 (carry)    0.319   0.483   0.751   0.810
    0.25-0.40 (carry)    0.294   0.574   0.782   0.818
    0.40-0.60 (mech)     0.260   0.574   0.727   0.764

The p90/p95 figures are NOT usable workspace: they are near-singular, fully-extended
configurations. The *density* of comfortable top-down poses peaks at x~0.30-0.35.

## Why the old placements were wrong

Fraction of comfortable top-down grasp poses (at object-grasp height) that land inside
each spawn box — i.e. how much freedom the policy has to approach an object there:

    lift / stack     x 0.60-0.80          0.9%   <- outer edge of reach
    tool-pull puck   x 0.78-0.88          0.2%   <- essentially unreachable
    place-in-container x 0.45-0.60        0.9%
    reorient         x 0.45-0.60          1.7%
    PROPOSED         x 0.30-0.50          9.9%   <- 10x more approach freedom

A box in the tail is not merely "harder": the arm can only enter it near singularities,
so grasp orientation is heavily constrained and the reward's reach term is fighting the
joint-limit penalty. That is a task-design bug, not difficulty.

## The envelope

``GRASP_*`` is where free objects may rest on the ground for a top-down grasp.
``MECHANISM_*`` is where articulated mechanisms may be mounted (they are approached
from the side, so they use the any-orientation envelope and sit higher).
"""

from __future__ import annotations

from dataclasses import dataclass

# --- Free objects resting on the ground plane, grasped top-down -------------------

GRASP_X_RANGE: tuple[float, float] = (0.30, 0.52)
"""Comfortable fore/aft band for a graspable object. Centred on the density peak
(x~0.33) and stopping well short of the p90 reach tail."""

GRASP_Y_RANGE: tuple[float, float] = (-0.25, 0.25)
"""Lateral band. The arm is symmetric in y; +-0.25 stays inside the comfortable cone
at these x values."""

GRASP_RADIAL_MAX: float = 0.55
"""Hard ceiling on sqrt(x^2+y^2) for anything that must be GRASPED. Beyond this the
top-down approach is only reachable near singularities."""

GRASP_RADIAL_MIN: float = 0.28
"""Objects closer than this fold the arm back over its own base."""

# --- Goal / target positions (need only be REACHED, not grasped) ------------------

GOAL_X_RANGE: tuple[float, float] = (0.30, 0.55)
GOAL_Y_RANGE: tuple[float, float] = (-0.28, 0.28)
GOAL_Z_RANGE: tuple[float, float] = (0.10, 0.35)
"""Lift/place goals hang in free space; a slightly wider band than GRASP is fine
because no grasp has to be executed there."""

# --- Articulated mechanisms (mounted, approached from the side) -------------------

MECHANISM_X_RANGE: tuple[float, float] = (0.42, 0.56)
"""Mechanisms present a handle toward the robot, so their MOUNT x sits further out
than a graspable object; the handle itself protrudes back toward the arm."""

MECHANISM_Y_RANGE: tuple[float, float] = (-0.12, 0.12)
MECHANISM_Z_RANGE: tuple[float, float] = (0.35, 0.50)
"""Mount height. The any-orientation envelope at z=0.45-0.60 has p50 radial 0.563, so
a handle at radial ~0.45 is comfortably mid-range."""

MECHANISM_HANDLE_RADIAL_MAX: float = 0.58
"""The HANDLE (not the mount) must stay within this radius — that is the point the
gripper actually has to reach."""

# --- Shared object geometry -------------------------------------------------------

GROUND_Z: float = 0.0
"""Class A scenes have no table; free objects rest on the ground plane."""

SITE_TO_FINGERTIP: float = 0.10
"""The ``gripper`` site sits 10cm behind the fingertips along the approach axis. A
grasp of an object whose centre is at z=h puts the SITE near z=h+0.10 — the reason the
grasp-height analysis above uses site z~0.10 for a floor-resting object."""


@dataclass(frozen=True)
class PlacementRange:
  """An (x, y, z) spawn box, in env-local coordinates."""

  x: tuple[float, float]
  y: tuple[float, float]
  z: tuple[float, float]

  def as_pose_range(self) -> dict[str, tuple[float, float]]:
    """Format expected by ``reset_root_state_uniform`` events."""
    return {"x": self.x, "y": self.y, "z": self.z}

  def center(self) -> tuple[float, float, float]:
    return (
      (self.x[0] + self.x[1]) / 2,
      (self.y[0] + self.y[1]) / 2,
      (self.z[0] + self.z[1]) / 2,
    )

  def max_radial(self) -> float:
    """Worst-case sqrt(x^2+y^2) over the box corners."""
    xs = (abs(self.x[0]), abs(self.x[1]))
    ys = (abs(self.y[0]), abs(self.y[1]))
    return (max(xs) ** 2 + max(ys) ** 2) ** 0.5


def validate_grasp_placement(rng: PlacementRange) -> list[str]:
  """Return human-readable problems with a graspable-object spawn box (empty = ok)."""
  problems = []
  if rng.max_radial() > GRASP_RADIAL_MAX:
    problems.append(
      f"max radial {rng.max_radial():.3f} > GRASP_RADIAL_MAX {GRASP_RADIAL_MAX} "
      "(corner only reachable near a singularity)"
    )
  near = (min(abs(rng.x[0]), abs(rng.x[1])) ** 2) ** 0.5
  if near < GRASP_RADIAL_MIN and abs(rng.y[0]) < 0.05 and abs(rng.y[1]) < 0.05:
    problems.append(
      f"min radial {near:.3f} < GRASP_RADIAL_MIN {GRASP_RADIAL_MIN} "
      "(arm folds back over its own base)"
    )
  if rng.x[0] < GRASP_X_RANGE[0] - 1e-6 or rng.x[1] > GRASP_X_RANGE[1] + 1e-6:
    problems.append(f"x {rng.x} outside GRASP_X_RANGE {GRASP_X_RANGE}")
  if rng.y[0] < GRASP_Y_RANGE[0] - 1e-6 or rng.y[1] > GRASP_Y_RANGE[1] + 1e-6:
    problems.append(f"y {rng.y} outside GRASP_Y_RANGE {GRASP_Y_RANGE}")
  return problems
