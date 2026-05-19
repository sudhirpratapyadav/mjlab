"""Glue between mjviser's PerturbationHandler and mjlab's batched scene.

mjviser's ``PerturbationHandler`` was designed for a single-CPU ``MjData``
viewer: it expects one mesh handle per body, ``instance_index`` cycling
through that body's instances, and forces written via ``mj_applyFT`` into
``MjData.qfrc_applied``.

mjlab's viser scene is GPU-batched: each ``BatchedGlbHandle`` packs one
batch instance per (env, body) pair, and forces live on a ``(num_envs,
nbody, 6)`` warp tensor at ``sim.data.xfrc_applied``. This module bridges
the two: it filters drag events so only the currently-followed env reacts,
and writes the handler's spring force/torque directly to ``xfrc_applied``
(which is already in body-origin Cartesian frame).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
  import mujoco
  import viser
  from mjviser.interaction import PerturbationHandler


def attach_active_env_filter(
  handler: "PerturbationHandler",
  mesh_handle: "viser.BatchedGlbHandle",
  body_ids: np.ndarray,
  env_ids: np.ndarray | None,
  active_env_idx_fn,
) -> None:
  """Register drag/click handlers that only fire for the active env.

  Two batched-handle layouts are supported:

  * Upstream ``_MeshGroup`` (no per-world DR): the handle holds
    ``num_envs * len(body_ids)`` instances tiled as
    ``[env0_body0, ..., env0_bodyN-1, env1_body0, ...]``.  Pass
    ``env_ids=None``; we recover ``(env_idx, body_id)`` from
    ``divmod(instance_index, len(body_ids))``.

  * mjlab ``_PerWorldMeshGroup`` (per-world DR active): the handle holds
    one instance per active (env, body) pair, with parallel arrays
    ``env_ids[i]`` / ``body_ids[i]``.  Pass both arrays.
  """
  import mujoco
  import time
  import viser

  n_bodies = len(body_ids)
  if n_bodies == 0:
    return
  # Skip world-body-only groups (matches upstream's no-op for body 0).
  if not bool(np.any(np.asarray(body_ids) != 0)):
    return

  per_world = env_ids is not None
  n_instances = len(env_ids) if per_world else None  # parallel-array length

  def _resolve(event) -> tuple[int, int] | None:
    """Return (env_idx, body_id) if event is on the active env, else None."""
    idx = event.instance_index
    if idx is None:
      return None
    if per_world:
      idx = idx % n_instances
      env_idx = int(env_ids[idx])
      bid = int(body_ids[idx])
    else:
      env_idx, body_idx = divmod(int(idx), n_bodies)
      bid = int(body_ids[body_idx])
    if env_idx != active_env_idx_fn():
      return None
    if bid == 0:
      return None
    return env_idx, bid

  @mesh_handle.on_click
  def _(event: viser.SceneNodePointerEvent) -> None:  # type: ignore[type-arg]
    resolved = _resolve(event)
    if resolved is None:
      return
    _, bid = resolved
    with handler._lock:
      handler.selected_body_id = bid
    name = mujoco.mj_id2name(handler._model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if handler._info_text is not None:
      handler._info_text.value = name or f"body_{bid}"

  def _make_drag(mode: str):
    async def _handler(event: viser.SceneNodeDragEvent) -> None:  # type: ignore[type-arg]
      if event.phase == "start":
        resolved = _resolve(event)
        if resolved is None:
          return
        _, bid = resolved
        target_viser = np.array(event.start_position)
        grab_world_mj = target_viser - handler._scene_offset
        grab_local = handler._world_to_body_local(bid, grab_world_mj)
        with handler._lock:
          handler._drag_body_id = bid
          handler._drag_grab_local = grab_local
          handler._drag_target = target_viser
          handler._drag_mode = mode
          handler.selected_body_id = bid
          handler._prev_grab_world = target_viser.copy()
          handler._prev_time = time.perf_counter()
        name = mujoco.mj_id2name(handler._model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if handler._info_text is not None:
          handler._info_text.value = name or f"body_{bid}"
      elif event.phase == "update":
        with handler._lock:
          if handler._drag_body_id is not None:
            handler._drag_target = np.array(event.end_position)
      else:  # "end"
        with handler._lock:
          handler._drag_body_id = None
          handler._drag_grab_local = None
          handler._drag_target = None
          handler._drag_mode = None
          handler._prev_grab_world = None
          handler._prev_time = None

    return _handler

  mesh_handle.on_drag("left", modifier="cmd/ctrl")(_make_drag("translate"))
  mesh_handle.on_drag("left", modifier="cmd/ctrl+shift")(_make_drag("rotate"))


def write_perturbation_to_xfrc(
  pert,
  xfrc_applied: torch.Tensor,
  env_idx: int,
) -> tuple[int, int] | None:
  """Write the handler's force/torque to ``sim.data.xfrc_applied``.

  ``PerturbationHandler.get_perturbation`` returns a wrench whose torque
  already includes the moment arm contribution (cross(point - body_pos,
  force)), so it can be applied at the body origin directly -- which is
  exactly the slot ``xfrc_applied`` provides. Returns the
  (env_idx, body_id) that was written, so the caller can zero it later.
  """
  if pert is None:
    return None
  bid = int(pert.body_id)
  if bid <= 0 or bid >= xfrc_applied.shape[1]:
    return None
  force = torch.as_tensor(pert.force, dtype=xfrc_applied.dtype, device=xfrc_applied.device)
  torque = torch.as_tensor(pert.torque, dtype=xfrc_applied.dtype, device=xfrc_applied.device)
  xfrc_applied[env_idx, bid, 0:3] = force
  xfrc_applied[env_idx, bid, 3:6] = torque
  return env_idx, bid
