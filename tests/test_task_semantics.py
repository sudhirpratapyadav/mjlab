"""Task contracts checked against actual compiled states, contacts and rewards."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.task_state_probe import (
  cage_at_goal,
  held_at_goal,
  place,
  tick_metrics,
)
from mjlab.tasks.manipulation.mdp.rewards import (
  articulation_task_reward,
  cage_transport_reward,
  object_at_goal_reward,
  orientation_task_reward,
)
from mjlab.tasks.registry import load_env_cfg

TASKS = [
  r["task_id"]
  for r in json.loads(
    Path(
      "src/mjlab/continual_distill/docs/cl_v3/physics_refresh/results.json"
    ).read_text()
  )
]


@pytest.fixture
def make_env():
  created = []

  def build(task):
    cfg = load_env_cfg(f"Mjlab-{task}-Franka", test=True)
    cfg.scene.num_envs = 2
    env = ManagerBasedRlEnv(cfg, device="cpu")
    created.append(env)
    env.reset()
    return env, next(iter(env.command_manager._terms.values()))

  yield build
  for env in created:
    env.close()


@pytest.mark.parametrize("task", TASKS)
def test_evaluation_preserves_task_rules(task):
  train, test = load_env_cfg(task), load_env_cfg(task, test=True)
  assert train.episode_length_s == test.episode_length_s
  assert train.terminations.keys() == test.terminations.keys()
  assert train.commands == test.commands
  assert train.rewards == test.rewards


def test_peg_reward_marker_and_success_share_inserted_pose(make_env):
  env, c = make_env("Peg-Insertion")
  place(env, "object", c.target_pos.clone())
  tick_metrics(env, c)
  assert c.compute_success().all()
  torch.testing.assert_close(
    object_at_goal_reward(env, "stack_object", max_dist=0.35), torch.ones(2)
  )
  torch.testing.assert_close(
    c.target_site_pos, env.scene["object"].data.site_pos_w[:, 0]
  )
  torch.testing.assert_close(
    env.scene["mocap_goal"].data.root_link_pos_w, c.target_site_pos
  )
  # Compare source/replica visual geometry, not just marker origins.
  a = env.scene["object"].data.geom_pos_w[:, 0]
  b = env.scene["mocap_goal"].data.geom_pos_w[:, 0]
  torch.testing.assert_close(a, b, atol=1e-5, rtol=0)
  for offset in (0.014, 0.05):
    p = c.target_pos.clone()
    p[:, 2] += offset
    place(env, "object", p)
    tick_metrics(env, c)
    assert not c.compute_success().any()
    assert (object_at_goal_reward(env, "stack_object", max_dist=0.35) < 1).all()
  place(
    env, "object", c.target_pos.clone(), (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8))
  )
  tick_metrics(env, c)
  assert not c.compute_success().any(), "45-degree square peg cannot fit a 30 mm bore"


def test_stack_rejects_hover_and_motion(make_env):
  env, c = make_env("Stack-Cube")
  p = c.target_pos.clone()
  p[:, 2] -= 0.0005  # finite solver contact tolerance
  place(env, "object", p)
  tick_metrics(env, c)
  assert c.compute_success().all()
  p = c.target_pos.clone()
  p[:, 2] += 0.01
  place(env, "object", p)
  tick_metrics(env, c)
  assert not c.compute_success().any(), "nearby airborne cube is not stacked"


@pytest.mark.parametrize("task", ["Place-In-Container", "Throw-To-Bin"])
def test_container_requires_support_full_extent_and_release(make_env, task):
  from mjlab.utils.lab_api.math import quat_apply, quat_mul

  env, c = make_env(task)
  root = c.target_pos.clone()
  root[:, 2] -= 0.0005
  place(env, "cube", root)
  tick_metrics(env, c)
  assert c.compute_success().all()
  hover = root.clone()
  hover[:, 2] += 0.015
  place(env, "cube", hover)
  tick_metrics(env, c)
  assert not c.compute_success().any(), (
    "zero velocity in midair does not establish settling"
  )
  # At this offset a 45-degree cube crosses the inner wall although its center
  # would pass the former 58.5 mm radial threshold.
  q = c.container.data.root_link_quat_w
  offset = quat_apply(q, torch.tensor([0.055, 0.0, 0.0]).expand(2, 3))
  rotated = quat_mul(
    q,
    torch.tensor(
      [np.cos(np.pi / 8), 0.0, 0.0, np.sin(np.pi / 8)], dtype=torch.float32
    ).expand(2, 4),
  )
  place(env, "cube", root + offset, rotated)
  tick_metrics(env, c)
  assert not c.compute_success().any()


def test_cage_requires_enclosure_transport_and_revokes_reward(make_env):
  env, c = make_env("Cage-Drag")
  place(env, "cube", c.target_pos.clone())
  tick_metrics(env, c)
  assert not c.compute_success().any()
  cage_at_goal(env, c)
  assert c.compute_success().all()
  q = env.scene["robot"].data.joint_pos.clone()
  q[:, list(c.finger_idx)] = 0
  env.scene["robot"].write_joint_position_to_sim(q)
  tick_metrics(env, c)
  q[:, list(c.finger_idx)] = 0.04
  env.scene["robot"].write_joint_position_to_sim(q)
  tick_metrics(env, c)
  assert not c.episode_success.any()
  assert not cage_transport_reward(env, "cage_drag").any()


def test_cage_rejects_a_diagonal_cube_pinch_above_old_threshold(make_env):
  env, c = make_env("Cage-Drag")
  cage_at_goal(env, c)
  q = env.scene["robot"].data.joint_pos.clone()
  q[:, list(c.finger_idx)] = 0.032
  env.scene["robot"].write_joint_position_to_sim(q)
  place(env, "cube", c.target_pos.clone(), (np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)))
  tick_metrics(env, c)
  assert (c._aperture() > 0.055).all()
  assert not c.compute_success().any()
  assert not c.episode_success.any()


@pytest.mark.parametrize(
  "task,asset,good,bad",
  [
    ("Reorient-Object", "cylinder", (1, 0, 0, 0), (2**-0.5, 0, 2**-0.5, 0)),
    ("Topple-Block", "block", (2**-0.5, 0, 2**-0.5, 0), (1, 0, 0, 0)),
  ],
)
def test_orientation_reward_prefers_success_at_identical_position(
  make_env, task, asset, good, bad
):
  env, c = make_env(task)
  name = next(iter(env.command_manager._terms))
  pos = c.target_pos.clone()
  place(env, asset, pos, good)
  tick_metrics(env, c)
  score = orientation_task_reward(env, name, asset)
  assert c.compute_success().all()
  place(env, asset, pos, bad)
  tick_metrics(env, c)
  assert not c.compute_success().any()
  assert (score > orientation_task_reward(env, name, asset)).all()


def test_valve_reward_is_monotonic_through_270_degrees(make_env):
  env, c = make_env("Rotate-Valve")
  scores = []
  for fraction in (0, 0.25, 0.5, 0.75, 1):
    c.asset.write_joint_position_to_sim((c.target_value * fraction)[:, None])
    tick_metrics(env, c)
    scores.append(
      articulation_task_reward(
        env, "rotate_valve", "valve", robot_asset_cfg=c.robot_cfg
      )
    )
  # Isolate bringing by holding the approach term fixed: the precise completion
  # reward itself must never reward a shortcut in the wrong angular direction.
  exact = []
  for fraction in (0, 0.25, 0.5, 0.75, 1):
    c.asset.write_joint_position_to_sim((c.target_value * fraction)[:, None])
    tick_metrics(env, c)
    exact.append(articulation_task_reward(env, "rotate_valve", "valve"))
  assert torch.stack(exact).diff(dim=0).ge(0).all()
  assert torch.stack(exact)[-1].eq(1).all()


@pytest.mark.parametrize("task", ["Lift-Cube", "Edge-Grasp"])
def test_lift_success_requires_actual_bilateral_contact(make_env, task):
  env, c = make_env(task)
  place(env, c.cfg.asset_name, c.target_pos.clone())
  tick_metrics(env, c)
  assert not c.compute_success().any()
  held_at_goal(env, c)
  assert c.compute_success().all()


def test_pivot_goal_witness_clears_the_wall(make_env):
  from mjlab.scripts.task_state_probe import goal_oracle

  env, c = make_env("Pivot-Lift")
  assert goal_oracle(env, c).all()
  assert c.pivoted.all()
