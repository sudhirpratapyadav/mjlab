"""Every Class A task must have a registered classical teacher, and it must run.

These are cheap structural checks, not accuracy measurements. Measured success rates
live in `continual_distill/docs/benchmark/CLASSICAL_TEACHERS.md` and are produced by
`continual_distill.classical.test_classical`, which is far too slow for CI (a single
20s-episode task takes minutes on CPU).

What is worth pinning here is the wiring: that the registry covers the suite, and that
each teacher actually produces finite, correctly-shaped actions from a real observation.
A teacher that crashes or emits NaN is a different class of failure from one that merely
scores badly, and only the first should fail a test.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import mjlab  # noqa: F401  (import side-effect: registers task packages)
from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation import benchmark
from mjlab.tasks.manipulation.taxonomy import Embodiment
from mjlab.tasks.registry import load_env_cfg


def _class_a_tasks() -> list[str]:
  return sorted(
    t
    for t, tax in benchmark.all_benchmark_tasks().items()
    if tax.embodiment is Embodiment.ARM_GRIPPER
  )


def test_registry_covers_every_class_a_task() -> None:
  tasks = set(_class_a_tasks())
  registered = set(CLASSICAL_POLICIES)
  assert not tasks - registered, f"no classical teacher for: {sorted(tasks - registered)}"
  assert not registered - tasks, f"teacher registered for unknown task: {sorted(registered - tasks)}"


@pytest.mark.parametrize("task_id", _class_a_tasks())
def test_teacher_produces_valid_actions(task_id: str) -> None:
  """Build the env, step the teacher a few times, check the actions are usable."""
  num_envs = 2
  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    obs, _ = env.reset()
    policy = CLASSICAL_POLICIES[task_id](num_envs=num_envs)
    policy.reset()
    action_dim = int(env.action_manager.total_action_dim)

    for _ in range(5):
      actions = policy(obs["policy"].detach().cpu().numpy())
      assert actions.shape == (num_envs, action_dim), (
        f"{task_id}: teacher returned {actions.shape}, expected {(num_envs, action_dim)}"
      )
      assert np.isfinite(actions).all(), f"{task_id}: teacher emitted non-finite actions"
      obs, *_ = env.step(torch.from_numpy(actions).to("cpu"))
  finally:
    env.close()


def test_teachers_use_relative_observations_only() -> None:
  """Teachers must not read absolute position terms.

  Observations are world-frame and carry each env's scene-origin offset, so a teacher
  built on absolute terms (``object_pos``, ``gripper_pos``) works in env 0 and is wrong
  everywhere else — the single most common way a teacher here silently breaks. The
  relative terms (``gripper_to_object``, ``object_to_goal``) cancel the offset.

  Checked behaviourally: run the same teacher on two envs whose objects are at the same
  RELATIVE pose but different origins, and require the same action. A teacher reading
  absolute terms produces different actions for identical relative geometry.
  """
  task_id = "Mjlab-Lift-Cube-Franka"
  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = 4
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    obs, _ = env.reset()
    origins = env.scene.env_origins
    # Different origins is the precondition that makes this test meaningful.
    assert float(torch.norm(origins[1] - origins[0])) > 0.1

    policy = CLASSICAL_POLICIES[task_id](num_envs=4)
    policy.reset()
    raw = obs["policy"].detach().cpu().numpy()
    # Copy env 0's observation into env 1, then shift ONLY the absolute terms by the
    # origin delta. Relative terms stay identical, so a correct teacher is unaffected.
    raw[1] = raw[0]
    delta = (origins[1] - origins[0]).cpu().numpy()
    raw[1, 18:21] += delta  # object_pos (absolute)
    raw[1, 25:28] += delta  # gripper_pos (absolute)

    actions = policy(raw)
    assert np.allclose(actions[0], actions[1], atol=1e-6), (
      f"{task_id}: teacher's action changed when only ABSOLUTE observation terms moved "
      "— it is reading world-frame positions and will be wrong in every env but one."
    )
  finally:
    env.close()
