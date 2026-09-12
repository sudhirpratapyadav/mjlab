"""Tests for the Class A motion-profile expansion (8 -> 16 distinct profiles).

See ``src/mjlab/continual_distill/docs/benchmark/CLASS_A_EXPANSION.md``.

The important tests here are the SUCCESS-PREDICATE ones. ``benchmark-smoke`` proves an
env builds, resets and steps; it does NOT prove the success predicate can ever fire, and
a predicate that never fires makes a task silently unsolvable rather than loudly broken.
These drive each mechanism/object into a success state and assert the predicate latches,
plus a near-miss negative control so we know it discriminates instead of always firing.

That is not a hypothetical concern: it is exactly how the Tool-Pull goal was caught
sitting 0.3m above the puck's resting height, which made the task unsatisfiable.
"""

import pytest
import torch

import mjlab  # noqa: F401  (import side-effect: auto-registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.taxonomy import Embodiment, SkillFamily
from mjlab.tasks.registry import list_taxonomy, load_env_cfg

NEW_TASKS = (
  "Mjlab-Turn-Lever-Franka",
  "Mjlab-Rotate-Valve-Franka",
  "Mjlab-Flip-Switch-Franka",
  "Mjlab-Slide-Window-Franka",
  "Mjlab-Open-Lid-Franka",
  "Mjlab-Place-In-Container-Franka",
  "Mjlab-Reorient-Object-Franka",
  "Mjlab-Tool-Pull-Franka",
)

# (task_id, command_name, asset_name, joint target that must count as success)
ARTICULATION_CASES = (
  ("Mjlab-Turn-Lever-Franka", "turn_lever", "lever", -1.5707963),
  ("Mjlab-Rotate-Valve-Franka", "rotate_valve", "valve", 4.712389),
  ("Mjlab-Flip-Switch-Franka", "flip_switch", "switch", 0.5235988),
  ("Mjlab-Slide-Window-Franka", "slide_window", "window", 0.22),
  ("Mjlab-Open-Lid-Franka", "open_lid", "lid", -1.308997),
)


def test_all_new_tasks_registered_and_tagged() -> None:
  tagged = list_taxonomy()
  for task_id in NEW_TASKS:
    assert task_id in tagged, f"{task_id} missing taxonomy"
    assert tagged[task_id].embodiment is Embodiment.ARM_GRIPPER


def test_tool_pull_is_the_first_tool_use_task() -> None:
  """Tool-Pull introduces the TOOL_USE skill family to the benchmark."""
  tagged = list_taxonomy()
  assert tagged["Mjlab-Tool-Pull-Franka"].skill is SkillFamily.TOOL_USE


def test_new_tasks_keep_the_uniform_class_a_interface() -> None:
  """Uniform interfaces are a PLAN.md commitment: joint-space, 8-D action."""
  for task_id in NEW_TASKS:
    cfg = load_env_cfg(task_id, test=True)
    action = cfg.actions["robot_joint_pos"]
    assert action.scale == pytest.approx(0.04), task_id


def _make_env(task_id: str, num_envs: int = 2) -> ManagerBasedRlEnv:
  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = num_envs
  return ManagerBasedRlEnv(cfg, device="cpu")


@pytest.mark.parametrize(
  "task_id,command_name,asset_name,target", ARTICULATION_CASES
)
def test_articulation_success_predicate_fires(
  task_id: str, command_name: str, asset_name: str, target: float
) -> None:
  """Driving the joint to its target latches success; reset does not false-positive."""
  env = _make_env(task_id)
  try:
    env.reset()
    command = env.command_manager.get_term(command_name)
    entity = env.scene[asset_name]
    env_ids = torch.arange(env.num_envs, device=env.device)

    assert not bool(
      (command.metrics["episode_success"] > 0).any()
    ), f"{task_id} reports success at reset"

    entity.write_joint_position_to_sim(
      torch.full((env.num_envs, 1), float(target), device=env.device),
      env_ids=env_ids,
    )
    env.sim.forward()
    command._update_metrics()

    assert bool(
      (command.metrics["episode_success"] > 0).all()
    ), f"{task_id} success never fires at its own target"
  finally:
    env.close()


def _place(env, entity_name: str, pos: torch.Tensor, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
  entity = env.scene[entity_name]
  env_ids = torch.arange(env.num_envs, device=env.device)
  quat_t = torch.tensor(quat, device=env.device, dtype=torch.float32).expand(
    env.num_envs, 4
  )
  entity.write_root_link_pose_to_sim(torch.cat([pos, quat_t], dim=-1), env_ids=env_ids)
  entity.write_root_link_velocity_to_sim(
    torch.zeros(env.num_envs, 6, device=env.device), env_ids=env_ids
  )


def _at_goal(env, command) -> bool:
  env.sim.forward()
  command._update_metrics()
  return bool((command.metrics["at_goal"] > 0).all())


def test_place_in_container_requires_actual_containment() -> None:
  env = _make_env("Mjlab-Place-In-Container-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("place_in_container")
    goal = command.target_pos.clone()
    goal[:,2] -= .0005  # establish a finite solver contact with the bin floor

    _place(env, "cube", goal)
    assert _at_goal(env, command), "cube inside the bin should count"

    _place(env, "cube", goal + torch.tensor([0.0, 0.30, 0.0]))
    assert not _at_goal(env, command), "cube outside the bin must not count"

    # Held above the rim — the case a plain distance threshold would wrongly accept.
    _place(env, "cube", goal + torch.tensor([0.0, 0.0, 0.25]))
    assert not _at_goal(env, command), "cube above the rim must not count"
  finally:
    env.close()


def test_reorient_scores_orientation_not_position() -> None:
  env = _make_env("Mjlab-Reorient-Object-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("reorient_object")
    pos = command.target_pos

    _place(env, "cylinder", pos, quat=(1.0, 0.0, 0.0, 0.0))
    assert _at_goal(env, command), "upright cylinder should count"

    # Same position, wrong orientation — must fail, or the task is positional.
    _place(env, "cylinder", pos, quat=(0.7071, 0.7071, 0.0, 0.0))
    assert not _at_goal(env, command), "cylinder lying down must not count"
  finally:
    env.close()


def test_every_manipulated_entity_is_observable() -> None:
  """A task may not require manipulating an object the policy cannot see.

  Regression: Tool-Pull bound EVERY observation term to the puck, so the stick — the
  tool the task exists to make you pick up — appeared nowhere in the 60-D observation.
  It spawns uniformly over a ~20x19cm box, so no function of the observation could
  locate it: the task was unsolvable, not merely hard, for scripted and learned policies
  alike. Found by a classical teacher that could not do tool use because it could not
  see the tool.

  Checked structurally (which entities are referenced by observation terms) rather than
  by rollout, so it stays fast and catches the omission at authoring time.
  """
  for task_id in NEW_TASKS:
    cfg = load_env_cfg(task_id, test=True)
    manipulable = {
      name
      for name in cfg.scene.entities
      if name not in ("robot", "mocap_goal")
    }
    referenced: set[str] = set()
    for term in cfg.observations["policy"].terms.values():
      for key in ("object_asset_name", "container_asset_name", "tool_asset_name"):
        value = term.params.get(key)
        if isinstance(value, str):
          referenced.add(value)

    # A static receptacle whose pose the command owns (the container) is exempt: the
    # goal vector already encodes it.
    exempt = {"container"}
    missing = manipulable - referenced - exempt
    assert not missing, (
      f"{task_id}: entities {sorted(missing)} must be manipulated but appear in no "
      "observation term — the policy cannot locate them."
    )


def test_flip_switch_detent_is_actually_bistable() -> None:
  """The switch must SNAP to whichever side of centre it is released on.

  Regression: the detent was first modelled with a joint spring
  (``stiffness``/``springref``). A single linear spring has exactly ONE rest pose, so
  the toggle always fell back to OFF even when released past centre — monostable, which
  silently reduces the task to "hold against a spring" and destroys the ballistic-commit
  motion profile the task exists to test. It is now an over-centre weighted lever, whose
  centre-of-mass must stay directly ABOVE the pivot (any x-offset adds a constant
  gravity torque that also destroys bistability).
  """
  env = _make_env("Mjlab-Flip-Switch-Franka")
  try:
    env.reset()
    switch = env.scene["switch"]
    command = env.command_manager.get_term("flip_switch")
    env_ids = torch.arange(env.num_envs, device=env.device)

    def settle_from(angle: float) -> float:
      switch.write_joint_position_to_sim(
        torch.full((env.num_envs, 1), angle, device=env.device), env_ids=env_ids
      )
      switch.write_joint_velocity_to_sim(
        torch.zeros(env.num_envs, 1, device=env.device), env_ids=env_ids
      )
      for _ in range(150):
        env.sim.step()
      return float(switch.data.joint_pos[0, 0])

    assert settle_from(-0.05) < -0.5, "released before centre it must fall back to OFF"
    assert settle_from(0.05) > 0.5, "released past centre it must snap to ON"

    command._update_metrics()
    assert bool(
      (command.metrics["episode_success"] > 0).all()
    ), "reaching the ON state must count as success"
  finally:
    env.close()


def test_tool_pull_goal_is_physically_reachable() -> None:
  """Regression: the goal must sit at the puck's RESTING height.

  The goal_offset z was originally 0.31 while the puck rests at z=0.012, so no amount
  of correct dragging could satisfy the 0.07m threshold — an unsolvable task that every
  structural check still passed.
  """
  env = _make_env("Mjlab-Tool-Pull-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("tool_pull")
    goal = command.target_pos

    _place(env, "puck", goal)
    assert not _at_goal(env, command), "puck at goal without tool interaction must not count"

    _place(env, "puck", goal + torch.tensor([0.35, 0.0, 0.0]))
    assert not _at_goal(env, command), "puck still far away must not count"

    # The goal must be at table height, not floating above it.
    puck_rest_z = 0.012
    assert abs(float(command.cfg.goal_offset[2]) - puck_rest_z) < 0.01
  finally:
    env.close()
