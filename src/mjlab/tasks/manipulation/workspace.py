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
(x~0.33) and stopping well short of the p90 reach tail.

CAUTION: this and ``GRASP_Y_RANGE`` bound each axis INDEPENDENTLY, so the box corner
escapes the radial ceiling: hypot(0.52, 0.25) = 0.577 > GRASP_RADIAL_MAX (0.55). A task
that uses both ranges verbatim will occasionally sample an out-of-envelope corner. Use
``grasp_box(y_max=...)`` instead of pairing the raw constants — it clamps x so the whole
box, corners included, respects the ceiling."""

GRASP_Y_RANGE: tuple[float, float] = (-0.25, 0.25)
"""Lateral band. The arm is symmetric in y; +-0.25 stays inside the comfortable cone
at these x values. See the corner caveat on ``GRASP_X_RANGE``."""

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

GOAL_RADIAL_MAX: float = 0.60
"""Hard ceiling on sqrt(x^2+y^2) for a GOAL point.

Looser than ``GRASP_RADIAL_MAX`` because a goal only has to be REACHED (often while
already holding the object), not entered with a free top-down approach. Note that
``GOAL_X_RANGE`` x ``GOAL_Y_RANGE`` has the same corner-escape problem as the grasp
box — hypot(0.55, 0.28) = 0.617 — so a task pairing the raw ranges will occasionally
sample past this ceiling. ``audit_workspace`` checks the SAMPLED goals, not the
declared ranges, so it catches that.

Tasks whose goal is deliberately unreachable (strike-slide, throw-to-bin) are
exempted by name in ``audit_workspace._GOAL_EXEMPT``, never by loosening this.

This is the CARRY ceiling — it applies to a goal the robot must bring an object to.
Mechanism goals use ``MECHANISM_GOAL_RADIAL_MAX`` instead."""

MECHANISM_GOAL_RADIAL_MAX: float = 0.70
"""Hard ceiling on sqrt(x^2+y^2) for a MECHANISM's goal point.

A mechanism goal is the far end of the member's own swept arc — the door handle at
70 degrees open, the flap panel at the end of its hinge travel. Three things make it
a looser bound than the carry ceiling:

  - no object is carried there, so the arm is not fighting a grasp constraint;
  - it is approached with ANY orientation, not top-down, and the any-orientation p90
    at mechanism height is 0.727;
  - the arm arrives there by FOLLOWING the arc, already extended, rather than by
    reaching into the point cold.

0.70 sits just under that p90. Above it the arm would have to finish the arc in a
near-singular configuration, which is the same task-design bug the grasp ceiling
exists to catch."""

# --- Articulated mechanisms (mounted, approached from the side) -------------------

MECHANISM_X_RANGE: tuple[float, float] = (0.42, 0.56)
"""Mechanisms present a handle toward the robot, so their MOUNT x sits further out
than a graspable object; the handle itself protrudes back toward the arm."""

MECHANISM_Y_RANGE: tuple[float, float] = (-0.12, 0.12)

MECHANISM_Z_RANGE: tuple[float, float] = (0.35, 0.50)
"""Mount height that would be ideal for REACH alone (the any-orientation envelope at
z=0.45-0.60 has p50 radial 0.563).

SUPERSEDED for actual placement: mount height is dictated by FLOOR CLEARANCE, not
reach. Mechanisms hang downward from their mount and Class A scenes have no table or
wall to hang them from, so a mount at 0.35-0.50 buries the taller assets in the ground
plane. Use ``min_mechanism_mount_z(asset)``.

The consequence is that mechanisms sit lower than a human-scale workstation would put
them (the drawer handle ends up near 0.33 rather than 0.5). That is correct for THIS
scene — a floor-standing cabinet, not a wall-mounted one — and the arm reaches it
comfortably. Kept here as the reference for what a table-based Class A variant should
use, if one is ever added."""

MECHANISM_HANDLE_RADIAL_MAX: float = 0.58
"""The HANDLE (not the mount) must stay within this radius — that is the point the
gripper actually has to reach."""

MECHANISM_DROP_BELOW_MOUNT: dict[str, float] = {
  "lever": 0.150,
  "valve": 0.150,
  "switch": 0.060,
  "window": 0.220,
  "lid": 0.070,
  "door": 0.318,
  "drawer": 0.315,
  "button": 0.030,
  # Wave-1 mechanisms (CATALOG_100_TASKS.md). Both swept over the joint range:
  # the flap rotates about a VERTICAL axis so its below-mount extent never changes
  # (post half-height 0.16 governs); the plug slides UP only, so the socket walls'
  # 0.05 below-mount extent is already the swept value.
  "flap": 0.160,
  "plug": 0.050,
}
"""How far each mechanism's geometry extends BELOW its mount body, in metres.

This is the SWEPT extent over the joint's full range of motion, not the extent at the
closed/rest pose. The distinction is not academic: the lid's flap swings DOWN as it
opens, dropping 0.157m at full travel versus 0.070m closed. Mounting for the closed
pose left the open lid's lip below the floor, which made the task literally
unsolvable — the success angle required driving the lip underground, so no policy,
scripted or learned, could ever pass it. Always sweep the joint range.

Mechanisms hang downward from a mocap mount, and Class A scenes have no table or wall
for them to hang from — so a mount placed at a height chosen purely for reachability
buries the mechanism in the ground plane. The door used to be the extreme case, at
0.80m below its mount; CL-V2 replaced that 1.2m slab with a real 300mm cabinet door
(carcass 636mm tall) and the drop fell to 0.318m.

Mount z must therefore be at least this value (plus clearance). Use
``min_mechanism_mount_z()`` rather than reading this directly. Regenerate with
``python -m mjlab.scripts.audit_workspace --measure-drops`` if an asset changes.
"""

MECHANISM_FLOOR_CLEARANCE: float = 0.03
"""Gap left between a mechanism's lowest geom and the ground plane."""


def min_mechanism_mount_z(asset: str) -> float:
  """Lowest mount height that keeps ``asset`` clear of the ground plane."""
  try:
    drop = MECHANISM_DROP_BELOW_MOUNT[asset]
  except KeyError:
    raise KeyError(
      f"unknown mechanism {asset!r}; add its drop to MECHANISM_DROP_BELOW_MOUNT "
      "(measure with audit_workspace --measure-drops)"
    ) from None
  return drop + MECHANISM_FLOOR_CLEARANCE

# --- Shared object geometry -------------------------------------------------------

GROUND_Z: float = 0.0
"""Class A scenes have no table; free objects rest on the ground plane."""

SITE_TO_FINGERTIP: float = 0.019
"""Distance from the ``gripper`` site to the lowest fingertip geometry, along the
approach axis.

MEASURED (fingers open, home pose): the site is 0.0189m above ``right_finger_pad``.

An earlier value of 0.10 here was WRONG and actively harmful: a teacher that offsets
its target by 10cm commands the gripper a full 10cm too high and never touches the
object. ``gripper_to_object`` in the observations is already effectively a
pad-to-object vector, so most teachers need no offset at all. Verify against the model
before applying any offset:

    site z 0.3819, lowest hand/finger geom 0.3630 -> 0.0189

The grasp-height analysis in this module's docstring uses site z~0.10 for a
floor-resting object, which remains correct for a different reason: an object centre at
z~0.02 plus clearance puts the SITE near 0.10 during the approach, not at contact."""


def grasp_box(
  y_max: float | None = None,
  x_min: float | None = None,
  margin: float = 0.01,
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Return an (x_range, y_range) grasp box whose CORNERS respect the radial ceiling.

  ``GRASP_X_RANGE`` and ``GRASP_Y_RANGE`` bound each axis independently, so pairing
  them directly yields a box whose far corner (hypot(0.52, 0.25) = 0.577) exceeds
  ``GRASP_RADIAL_MAX`` (0.55). A task sampling uniformly in that box occasionally
  spawns an object outside the comfortable envelope — intermittently, so it may pass
  one audit and fail the next depending on the sample.

  This keeps the full lateral spread (the y variation is what gives lift/push their
  character) and pulls x back so the corner lands on the ceiling:

      x_max = sqrt(GRASP_RADIAL_MAX^2 - y_max^2) - margin

  Args:
    y_max: half-width of the lateral band. Defaults to the full ``GRASP_Y_RANGE``.
    x_min: near edge. Defaults to ``GRASP_X_RANGE[0]``.
    margin: safety gap below the ceiling, in metres.
  """
  y = GRASP_Y_RANGE[1] if y_max is None else y_max
  x0 = GRASP_X_RANGE[0] if x_min is None else x_min
  if abs(y) >= GRASP_RADIAL_MAX:
    raise ValueError(
      f"grasp_box: y_max={y} is beyond GRASP_RADIAL_MAX ({GRASP_RADIAL_MAX}); "
      "no x band can satisfy the ceiling."
    )
  x1 = min(GRASP_X_RANGE[1], (GRASP_RADIAL_MAX**2 - y**2) ** 0.5 - margin)
  if x1 <= x0:
    raise ValueError(
      f"grasp_box: y_max={y} leaves no valid x band (x_min={x0}, x_max={x1:.3f}). "
      "Reduce y_max."
    )
  return (x0, x1), (-y, y)


def goal_box(
  y_max: float | None = None,
  x_min: float | None = None,
  margin: float = 0.01,
) -> tuple[tuple[float, float], tuple[float, float]]:
  """Return an (x_range, y_range) GOAL box whose CORNERS respect ``GOAL_RADIAL_MAX``.

  The goal-side twin of ``grasp_box``. ``GOAL_X_RANGE`` x ``GOAL_Y_RANGE`` bound each
  axis independently, so their far corner escapes the ceiling exactly as the grasp
  box's does: hypot(0.55, 0.28) = 0.617 > GOAL_RADIAL_MAX (0.60).

  The lift tasks paired the raw ranges and sampled goals out to radial 0.610 — an
  intermittent fault (only the corner of the box offends), which is why it survived
  the entity-only audits: the CUBE was always placed correctly, and nothing looked at
  where the cube was being asked to go.
  """
  y = GOAL_Y_RANGE[1] if y_max is None else y_max
  x0 = GOAL_X_RANGE[0] if x_min is None else x_min
  if abs(y) >= GOAL_RADIAL_MAX:
    raise ValueError(
      f"goal_box: y_max={y} is beyond GOAL_RADIAL_MAX ({GOAL_RADIAL_MAX}); "
      "no x band can satisfy the ceiling."
    )
  x1 = min(GOAL_X_RANGE[1], (GOAL_RADIAL_MAX**2 - y**2) ** 0.5 - margin)
  if x1 <= x0:
    raise ValueError(
      f"goal_box: y_max={y} leaves no valid x band (x_min={x0}, x_max={x1:.3f})."
    )
  return (x0, x1), (-y, y)


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
