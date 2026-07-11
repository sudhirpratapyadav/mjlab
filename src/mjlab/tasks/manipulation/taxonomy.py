"""Taxonomy for the manipulation-diversity continual-learning benchmark.

The benchmark is organized along three orthogonal axes so that tasks can be queried,
grouped, and sequenced for continual-learning studies:

- ``Embodiment``: which robot/hand the task uses. The benchmark deliberately spans
  three classes (arm + parallel gripper, arm + dexterous hand, floating dexterous
  hand) — a gap in existing MuJoCo manipulation benchmarks.
- ``SkillFamily``: the *manipulation skill* exercised (push, grasp, insert, ...). The
  benchmark maximizes skill diversity (RLBench-style), NOT scene/spatial diversity
  (anti-LIBERO). This is the primary diversity axis.
- ``Fragility``: how all-or-nothing the task is. Forgiving/planar tasks are trivially
  retained by regularization-based CL; precision/dexterous tasks are catastrophically
  forgotten. Sampling all fragility tiers is what makes CL comparisons discriminating
  (see continual_distill/docs/FINDINGS.md, Finding 8).

See ``continual_distill/docs/benchmark/PLAN.md`` for the full design.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class Embodiment(enum.Enum):
  """Embodiment class of a task."""

  ARM_GRIPPER = "arm_gripper"
  """Arm + 2-finger parallel gripper (e.g. Franka Panda + hand). Class A."""

  ARM_HAND = "arm_hand"
  """Arm + 5-finger dexterous hand (e.g. Franka + Allegro/Shadow). Class B."""

  FLOATING_HAND = "floating_hand"
  """Floating 5-finger hand, no arm (e.g. Adroit / Shadow). Class C."""


class SkillFamily(enum.Enum):
  """Manipulation skill exercised by a task (the primary diversity axis)."""

  REACH = "reach"
  """Move end-effector / hand to a target position. No object contact required."""

  PLANAR_PUSH = "planar_push"
  """Push / slide an object along a surface toward a goal (push, sweep, slide)."""

  ARTICULATION = "articulation"
  """Actuate an articulated mechanism (open door/drawer/cabinet, turn faucet, flip
  switch, press button)."""

  PICK_PLACE = "pick_place"
  """Grasp, lift, and place/stack a free object (pick, lift, place, stack, bin-pick)."""

  INSERTION = "insertion"
  """Precise fitting of one part into another (peg-in-hole, nut-on-bolt, plug, snapfit,
  gear-mesh, assembly)."""

  TOOL_USE = "tool_use"
  """Use a grasped object as a tool (hammer, sweep-with-tool, scoop, reach-with-stick)."""

  IN_HAND = "in_hand"
  """Reorient / regrasp an object within the hand (rotate block/pen/egg, in-hand
  repositioning). Dexterous-hand specific."""

  DEFORMABLE = "deformable"
  """Manipulate a deformable object (rope, cloth). Expensive in MJX; deferred."""


class Fragility(enum.IntEnum):
  """How all-or-nothing a task is (the CL-difficulty axis).

  Ordered from most forgiving (easily retained by regularization CL) to most fragile
  (catastrophically forgotten). Using ``IntEnum`` so orderings can sort/grade on it.
  """

  PLANAR = 1
  """Forgiving: the object keeps sliding toward the goal after imperfect actions.
  Regularization CL retains these well (they finish near 1.0 and stay there)."""

  MILD_CONTACT = 2
  """Some contact dependence; small errors recoverable (e.g. push-cuboid)."""

  PRECISION_GRASP = 3
  """All-or-nothing at the grasp instant: a small error means no contact, no recovery
  (e.g. lift-cube). Catastrophically forgotten under regularization-only CL."""

  DEXTEROUS = 4
  """Multi-contact / dexterous coordination (insertion, in-hand reorientation). The
  hardest to learn and to retain."""


@dataclass(frozen=True)
class TaskTaxonomy:
  """Benchmark taxonomy tags attached to a registered task.

  Attached to a task at registration time via ``register_mjlab_task(..., taxonomy=)``.
  Optional so non-benchmark tasks (velocity, tracking) need not supply it.

  Args:
    embodiment: Which embodiment class the task belongs to.
    skill: The primary manipulation skill family.
    fragility: The CL-difficulty tier.
    source: Where the task design/assets came from (e.g. "native", "metaworld",
      "maniskill", "adroit"). Used for the asset-provenance / license ledger.
    contact_rich: Whether the task is contact-rich (grasp/insert/tool). Convenience
      flag; derivable from skill but explicit for querying.
    notes: Free-form notes (e.g. solvability evidence, known caveats).
  """

  embodiment: Embodiment
  skill: SkillFamily
  fragility: Fragility
  source: str = "native"
  contact_rich: bool = False
  notes: str = ""

  def as_dict(self) -> dict:
    """Serialize for manifests / metadata export."""
    return {
      "embodiment": self.embodiment.value,
      "skill": self.skill.value,
      "fragility": int(self.fragility),
      "fragility_name": self.fragility.name.lower(),
      "source": self.source,
      "contact_rich": self.contact_rich,
      "notes": self.notes,
    }
