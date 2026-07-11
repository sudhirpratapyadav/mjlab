"""Tests for the manipulation-benchmark taxonomy + query/ordering API."""

import mjlab  # noqa: F401  (import side-effect: auto-registers task packages)
from mjlab.tasks.manipulation import benchmark
from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import list_taxonomy, load_taxonomy


def test_existing_franka_tasks_are_tagged() -> None:
  """The 6 native Franka tasks carry taxonomy tags."""
  tagged = list_taxonomy()
  for task_id in (
    "Mjlab-Lift-Cube-Franka",
    "Mjlab-Push-Cuboid-Franka",
    "Mjlab-Push-Disc-Franka",
    "Mjlab-Push-Button-Franka",
    "Mjlab-Open-Door-Franka",
    "Mjlab-Open-Drawer-Franka",
  ):
    assert task_id in tagged, f"{task_id} missing taxonomy"
    assert isinstance(tagged[task_id], TaskTaxonomy)


def test_untagged_tasks_return_none() -> None:
  """Non-benchmark tasks (velocity/tracking) stay untagged (backward-compatible)."""
  # These are registered but must not require taxonomy.
  assert load_taxonomy("Mjlab-Velocity-Flat-Unitree-Go1") is None


def test_lift_cube_is_the_fragile_grasp() -> None:
  tax = load_taxonomy("Mjlab-Lift-Cube-Franka")
  assert tax is not None
  assert tax.embodiment is Embodiment.ARM_GRIPPER
  assert tax.skill is SkillFamily.PICK_PLACE
  assert tax.fragility is Fragility.PRECISION_GRASP
  assert tax.contact_rich is True


def test_filter_contact_rich() -> None:
  contact = set(benchmark.filter_tasks(contact_rich=True))
  assert "Mjlab-Lift-Cube-Franka" in contact
  assert "Mjlab-Push-Button-Franka" not in contact  # button press is not contact-rich


def test_filter_by_embodiment() -> None:
  arm = benchmark.filter_tasks(embodiment=Embodiment.ARM_GRIPPER)
  assert len(arm) >= 6
  assert benchmark.filter_tasks(embodiment=Embodiment.FLOATING_HAND) == []  # none yet


def test_fragility_graded_ordering_puts_fragile_first() -> None:
  order = benchmark.fragility_graded_ordering(fragile_first=True)
  assert order[0] == "Mjlab-Lift-Cube-Franka"  # most fragile leads
  # easy-first puts a planar/mild task first, grasp last
  easy = benchmark.fragility_graded_ordering(fragile_first=False)
  assert easy[-1] == "Mjlab-Lift-Cube-Franka"


def test_skill_diverse_ordering_covers_all_tasks() -> None:
  order = benchmark.skill_diverse_ordering(embodiment=Embodiment.ARM_GRIPPER)
  assert set(order) == set(benchmark.filter_tasks(embodiment=Embodiment.ARM_GRIPPER))
  # adjacent tasks should differ in skill where possible (first two differ)
  taxo = benchmark.all_benchmark_tasks()
  assert taxo[order[0]].skill is not taxo[order[1]].skill


def test_manifest_shape() -> None:
  m = benchmark.export_manifest()
  assert m["num_tasks"] >= 6
  assert m["num_distinct_skills"] >= 3
  assert "arm_gripper" in m["counts_by_embodiment"]
  assert set(m["tasks"]) == set(benchmark.all_benchmark_tasks())
