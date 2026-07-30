"""Every Class A object must spawn inside the measured workspace.

Placement bugs are invisible to `benchmark-smoke` (the env still builds and steps) and
to the success-predicate tests (teleporting an object into a success state does not
care where it spawned). They only show up as a policy that cannot learn, or as an
object at the very edge of reach in a rendered video.

These tests pin the envelope so placements cannot silently drift back out. The
authority for the numbers is `tasks/manipulation/workspace.py`, derived by FK sampling;
see `scripts/audit_workspace.py --measure` to reproduce them.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import mjlab  # noqa: F401  (import side-effect: registers task packages)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation import benchmark, workspace
from mjlab.tasks.manipulation.taxonomy import Embodiment
from mjlab.tasks.registry import load_env_cfg

NUM_ENVS = 16

# Entities deliberately outside the direct-grasp envelope, with the bound that does
# apply. Keep in sync with `scripts/audit_workspace.py::_REACH_EXEMPT`.
REACH_EXEMPT = {
  ("Mjlab-Tool-Pull-Franka", "puck"): (0.58, 0.75),
}


def _class_a_tasks() -> list[str]:
  return sorted(
    t
    for t, tax in benchmark.all_benchmark_tasks().items()
    if tax.embodiment is Embodiment.ARM_GRIPPER
  )


def _spawn_positions(task_id: str) -> dict[str, np.ndarray]:
  """Env-local position of every non-robot entity after a reset."""
  cfg = load_env_cfg(task_id, test=True)
  cfg.scene.num_envs = NUM_ENVS
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    origins = env.scene.env_origins
    out = {}
    for name in cfg.scene.entities:
      if name in ("robot", "mocap_goal"):
        continue
      entity = env.scene[name]
      if "object_site" in entity.site_names:
        idx = entity.site_names.index("object_site")
        pos = entity.data.site_pos_w[:, idx]
      else:
        pos = entity.data.root_link_pos_w
      out[name] = (pos - origins).cpu().numpy()
    return out
  finally:
    env.close()


@pytest.mark.parametrize("task_id", _class_a_tasks())
def test_objects_spawn_within_reach(task_id: str) -> None:
  """No Class A object may spawn beyond the arm's comfortable reach."""
  for name, pos in _spawn_positions(task_id).items():
    radial = float(np.linalg.norm(pos[:, :2], axis=1).max())
    exempt = REACH_EXEMPT.get((task_id, name))
    if exempt is not None:
      lo, hi = exempt
      assert lo <= radial <= hi, (
        f"{task_id}/{name} is reach-exempt but radial {radial:.3f} is outside "
        f"[{lo}, {hi}] — it must stay draggable by the tool"
      )
      continue
    # Mechanisms are approached from the side and mounted higher than free objects.
    is_mechanism = float(pos[:, 2].mean()) > 0.25
    ceiling = (
      workspace.MECHANISM_HANDLE_RADIAL_MAX
      if is_mechanism
      else workspace.GRASP_RADIAL_MAX
    )
    assert radial <= ceiling, (
      f"{task_id}/{name} spawns at radial {radial:.3f} > {ceiling} — only reachable "
      "near a singularity. See workspace.py."
    )


@pytest.mark.parametrize("task_id", _class_a_tasks())
def test_objects_spawn_per_env(task_id: str) -> None:
  """Each env must place its objects relative to its OWN origin.

  Regression: the place-in-container bin is a static mocap body that was never written
  per-env, so every env shared the world-frame pose from the MJCF — the bin sat at the
  world origin while each robot sat at its own env origin, and only env 0 was correct.
  A per-env spread far larger than any configured randomization is the signature.
  """
  for name, pos in _spawn_positions(task_id).items():
    spread = float(np.ptp(pos, axis=0).max())
    assert spread < 1.0, (
      f"{task_id}/{name} varies by {spread:.2f}m across envs in env-local coords — "
      "it is almost certainly being placed in world frame, not per-env"
    )


def test_grasp_box_corners_respect_the_radial_ceiling() -> None:
  """`grasp_box()` must be corner-safe, unlike pairing the raw ranges.

  GRASP_X_RANGE x GRASP_Y_RANGE bounds each axis independently, so its far corner
  (hypot(0.52, 0.25) = 0.577) exceeds GRASP_RADIAL_MAX (0.55). Sampling that rectangle
  spawns out-of-envelope objects intermittently — passing one audit and failing the
  next depending on the draw.
  """
  raw = float(np.hypot(workspace.GRASP_X_RANGE[1], workspace.GRASP_Y_RANGE[1]))
  assert raw > workspace.GRASP_RADIAL_MAX, (
    "raw ranges are now corner-safe; grasp_box() and this test's rationale need review"
  )
  for y_max in (None, 0.25, 0.20, 0.12):
    (x0, x1), (_, y1) = workspace.grasp_box(y_max=y_max)
    assert x0 < x1
    corner = float(np.hypot(x1, y1))
    assert corner <= workspace.GRASP_RADIAL_MAX + 1e-9, (
      f"grasp_box(y_max={y_max}) corner {corner:.4f} exceeds the ceiling"
    )


def test_grasp_box_rejects_impossible_lateral_band() -> None:
  with pytest.raises(ValueError, match="beyond GRASP_RADIAL_MAX"):
    workspace.grasp_box(y_max=workspace.GRASP_RADIAL_MAX + 0.05)
