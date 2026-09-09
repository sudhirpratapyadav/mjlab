"""Tests for the Class A Wave-1 expansion (16 -> 25 distinct profiles).

See ``src/mjlab/continual_distill/docs/benchmark/CATALOG_100_TASKS.md`` (T17-T25).

Same philosophy as ``test_class_a_expansion.py``: benchmark-smoke proves an env
builds and steps, but only these tests prove each SUCCESS PREDICATE can fire — and,
via the near-miss negative controls, that it discriminates instead of always firing.
Wave 1 adds two predicate shapes that did not exist before and get dedicated
regression coverage here: an episode-long NEGATIVE constraint (cage-drag's
min-aperture latch) and goals placed DELIBERATELY outside the reach envelope
(strike-to-slide, throw-to-bin), which the workspace audit would otherwise flag as
bugs.
"""

import math

import pytest
import torch

import mjlab  # noqa: F401  (import side-effect: auto-registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.taxonomy import Embodiment, SkillFamily
from mjlab.tasks.registry import list_taxonomy, load_env_cfg

NEW_TASKS = (
  "Mjlab-Drag-Pull-Franka",
  "Mjlab-Strike-Slide-Franka",
  "Mjlab-Cage-Drag-Franka",
  "Mjlab-Topple-Block-Franka",
  "Mjlab-Push-Flap-Franka",
  "Mjlab-Axial-Extract-Franka",
  "Mjlab-Edge-Grasp-Franka",
  "Mjlab-Pivot-Lift-Franka",
  "Mjlab-Throw-To-Bin-Franka",
)

# (task_id, command_name, asset_name, joint target that must count as success)
ARTICULATION_CASES = (
  ("Mjlab-Push-Flap-Franka", "push_flap", "flap", -1.2217305),
  ("Mjlab-Axial-Extract-Franka", "axial_extract", "plug", 0.10),
)


def test_all_new_tasks_registered_and_tagged() -> None:
  tagged = list_taxonomy()
  for task_id in NEW_TASKS:
    assert task_id in tagged, f"{task_id} missing taxonomy"
    assert tagged[task_id].embodiment is Embodiment.ARM_GRIPPER


def test_wave1_introduces_the_non_prehensile_family() -> None:
  tagged = list_taxonomy()
  assert tagged["Mjlab-Cage-Drag-Franka"].skill is SkillFamily.NON_PREHENSILE
  assert tagged["Mjlab-Topple-Block-Franka"].skill is SkillFamily.NON_PREHENSILE


def test_axial_extract_is_tagged_by_skill_not_by_machinery() -> None:
  """The plug is a joint, so the command term subclasses the articulation base — but
  the SKILL is the inverse of peg-insertion, and the family tag names the skill.
  Mis-tagging it ARTICULATION left INSERTION a family of one and put the task in the
  wrong bucket of every skill-diverse CL ordering."""
  assert list_taxonomy()["Mjlab-Axial-Extract-Franka"].skill is SkillFamily.INSERTION


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


def test_drag_pull_goal_is_nearer_than_the_object() -> None:
  """The pull inversion must actually hold, or this is push with a new name."""
  env = _make_env("Mjlab-Drag-Pull-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("drag_pull")
    origins = env.scene.env_origins

    obj_x = env.scene["cuboid"].data.root_link_pos_w[:, 0] - origins[:, 0]
    goal_x = command.target_pos[:, 0] - origins[:, 0]
    assert bool((goal_x < obj_x).all()), "goal must spawn NEARER the base than the cuboid"

    _place(env, "cuboid", command.target_pos)
    assert _at_goal(env, command), "cuboid at the goal should count"

    _place(env, "cuboid", command.target_pos + torch.tensor([0.15, 0.0, 0.0]))
    assert not _at_goal(env, command), "cuboid still far away must not count"
  finally:
    env.close()


def test_strike_slide_goal_is_beyond_the_reach_envelope() -> None:
  """The goal band past ~0.85 m is the WHOLE task; a reachable goal is a push."""
  env = _make_env("Mjlab-Strike-Slide-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("strike_slide")
    origins = env.scene.env_origins

    radial = torch.norm((command.target_pos - origins)[:, :2], dim=-1)
    assert bool((radial > 0.85).all()), (
      "strike goal inside the arm's stretch: the task degenerates to quasi-static push"
    )

    _place(env, "puck", command.target_pos)
    assert _at_goal(env, command), "puck at the goal should count"

    _place(env, "puck", command.target_pos - torch.tensor([0.4, 0.0, 0.0]))
    assert not _at_goal(env, command), "puck mid-corridor must not count"
  finally:
    env.close()


def test_cage_drag_pinching_voids_the_episode() -> None:
  """The min-aperture latch is the task: cube-at-goal only counts if never pinched."""
  env = _make_env("Mjlab-Cage-Drag-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("cage_drag")
    robot = env.scene["robot"]
    env_ids = torch.arange(env.num_envs, device=env.device)

    # Fingers start OPEN (0.04 + 0.04): cube at goal must count.
    _place(env, "cube", command.target_pos)
    assert _at_goal(env, command), "caged (open-finger) transport should count"

    # Now pinch ONCE: close the fingers, tick metrics, reopen.
    f1 = robot.joint_names.index("finger_joint1")
    f2 = robot.joint_names.index("finger_joint2")
    pinched = robot.data.joint_pos.clone()
    pinched[:, f1] = 0.0
    pinched[:, f2] = 0.0
    robot.write_joint_position_to_sim(pinched, env_ids=env_ids)
    env.sim.forward()
    command._update_metrics()

    reopened = pinched.clone()
    reopened[:, f1] = 0.04
    reopened[:, f2] = 0.04
    robot.write_joint_position_to_sim(reopened, env_ids=env_ids)

    # Cube is still at the goal and the fingers are open again — but the episode
    # pinched once, so at_goal must now be False. This is the negative-constraint
    # latch that distinguishes caging from pinch-and-drag.
    assert not _at_goal(env, command), (
      "a pinch mid-episode must permanently void cage-drag success"
    )
    assert float(command.metrics["min_aperture"].max()) < 0.055

    # A resample must clear the violation (fresh episode, fresh latch).
    command._resample_command(env_ids)
    _place(env, "cube", command.target_pos)
    assert _at_goal(env, command), "resample must reset the min-aperture latch"
  finally:
    env.close()


def test_topple_scores_either_landing_face() -> None:
  """Body x-axis vertical (either sign) counts; still standing does not."""
  env = _make_env("Mjlab-Topple-Block-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("topple_block")
    pos = command.target_pos.clone()
    pos[:, 2] = 0.05  # Lying on a side face: the 0.05 half-extent is now vertical

    # Rotation of +-90 deg about y sends body x to -+world z.
    _place(env, "block", pos, quat=(0.7071068, 0.0, 0.7071068, 0.0))
    assert _at_goal(env, command), "toppled (+90 about y) must count"

    _place(env, "block", pos, quat=(0.7071068, 0.0, -0.7071068, 0.0))
    assert _at_goal(env, command), "toppled the other way must count (symmetric axis)"

    standing = command.target_pos.clone()
    standing[:, 2] = 0.09
    _place(env, "block", standing, quat=(1.0, 0.0, 0.0, 0.0))
    assert not _at_goal(env, command), "still standing must not count"
  finally:
    env.close()


def test_edge_grasp_requires_lifting_above_the_ledge_top() -> None:
  """Lifted above the top counts; resting on it or fallen to the ground does not."""
  env = _make_env("Mjlab-Edge-Grasp-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("edge_grasp")
    ledge_top = command.ledge_center.clone()
    ledge_top[:, 2] += command.cfg.ledge_top_height

    _place(env, "plate", ledge_top + torch.tensor([0.0, 0.0, 0.06]))
    assert _at_goal(env, command), "plate lifted above the ledge top should count"

    _place(env, "plate", ledge_top + torch.tensor([0.0, 0.0, 0.010]))
    assert not _at_goal(env, command), "plate resting ON the ledge must not count"

    # Pushed off the edge: lands on the ground BELOW the top — the failure mode
    # sliding alone produces, separated from success by construction.
    fallen = command.ledge_center.clone()
    fallen[:, 0] -= 0.16
    fallen[:, 2] = 0.008
    _place(env, "plate", fallen)
    assert not _at_goal(env, command), "plate fallen to the ground must not count"
  finally:
    env.close()


def test_pivot_lift_goal_is_airborne_and_wall_is_placed_per_env() -> None:
  env = _make_env("Mjlab-Pivot-Lift-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("pivot_lift")
    origins = env.scene.env_origins

    assert bool(((command.target_pos - origins)[:, 2] >= 0.15).all()), (
      "pivot-lift goal must be airborne, or sliding could satisfy it"
    )

    # The wall must stand in EVERY env, not just env 0 (the static-mocap trap).
    # Read the expected x from the cfg rather than hardcoding it: the wall band is
    # geometry that gets retuned (it moved 0.57 -> 0.50 when the board was found to
    # spawn inside it), and a literal here fails the next time for the wrong reason.
    env.sim.forward()
    wall_x = command.cfg.wall_spawn_range.x[0]
    wall_pos = env.scene["wall"].data.root_link_pos_w - origins
    assert bool((torch.abs(wall_pos[:, 0] - wall_x) < 1e-4).all()), (
      "wall not written per-env: only env 0 would have its pivot fixture"
    )

    # The board must spawn CLEAR of the wall, with travel left to push it. The
    # original 0.57 wall paired with an x 0.42-0.50 board band put boards up to
    # 13.4 mm INSIDE the wall — the episode began in contact with nowhere to push,
    # and no test noticed because nothing compared the two placements.
    board_half_x = 0.06 * math.cos(0.3) + 0.05 * math.sin(0.3)  # yaw-swept, yaw +-0.3
    wall_inner_face = wall_x - 0.015  # wall half-thickness (wall.xml)
    board_x = (env.scene["board"].data.root_link_pos_w - origins)[:, 0]
    gap = wall_inner_face - (board_x + board_half_x)
    assert bool((gap > 0.0).all()), (
      f"board spawns into the wall: min gap {float(gap.min()):+.4f} m"
    )

    _place(env, "board", command.target_pos)
    assert _at_goal(env, command), "board at the airborne goal should count"

    grounded = command.target_pos.clone()
    grounded[:, 2] = 0.011
    _place(env, "board", grounded)
    assert not _at_goal(env, command), "board flat on the ground must not count"
  finally:
    env.close()


def test_throw_bin_is_beyond_reach_and_requires_containment() -> None:
  env = _make_env("Mjlab-Throw-To-Bin-Franka")
  try:
    env.reset()
    command = env.command_manager.get_term("place_in_container")
    origins = env.scene.env_origins

    radial = torch.norm((command.target_pos - origins)[:, :2], dim=-1)
    assert bool((radial > 0.7).all()), (
      "bin inside the reach envelope: the task degenerates to place-in-container"
    )

    _place(env, "cube", command.target_pos)
    assert _at_goal(env, command), "cube settled in the bin should count"

    _place(env, "cube", command.target_pos + torch.tensor([0.0, 0.0, 0.25]))
    assert not _at_goal(env, command), "cube in flight above the bin must not count"

    _place(env, "cube", command.target_pos + torch.tensor([0.0, 0.30, 0.0]))
    assert not _at_goal(env, command), "cube landed beside the bin must not count"
  finally:
    env.close()


def test_every_manipulated_entity_is_observable() -> None:
  """A task may not require manipulating an object the policy cannot see.

  Same structural check as the first expansion (the Tool-Pull invisible-stick
  regression). Command-owned static fixtures are exempt: the goal vector already
  encodes the bin/ledge/wall, which the policy never has to localize independently.
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

    exempt = {"container", "ledge", "wall"}
    missing = manipulable - referenced - exempt
    assert not missing, (
      f"{task_id}: entities {sorted(missing)} must be manipulated but appear in no "
      "observation term — the policy cannot locate them."
    )
