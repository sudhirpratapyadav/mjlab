from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab.continual_distill.classical.record_outcomes import encode
from mjlab.viewer.offscreen_renderer import OffscreenRenderer


@pytest.mark.parametrize("fps", [25, 30, 50, 60])
def test_stream_encoder_duration_is_simulation_duration(monkeypatch, tmp_path, fps):
  from mjlab.continual_distill.classical import record_outcomes as module

  written = []

  def writer(*args, **kwargs):
    while True:
      frame = yield
      written.append(frame)

  monkeypatch.setattr(module.imageio_ffmpeg, "write_frames", writer)
  monkeypatch.setattr(module, "restore", lambda env, state: None)
  env = SimpleNamespace(
    step_dt=0.02, render=lambda: np.zeros((1080, 1920, 3), dtype=np.uint8)
  )
  _, timing = encode(env, [{}] * 100, tmp_path / "test.mp4", fps, 1920, 1080)
  assert timing["simulation_seconds"] == 2
  assert timing["video_seconds"] == 2
  assert len(written) == 2 * fps


def test_selected_environment_is_not_added_twice(monkeypatch):
  from mjlab.viewer import offscreen_renderer as module

  added = []
  monkeypatch.setattr(module.mujoco, "mj_forward", lambda *a: None)
  monkeypatch.setattr(
    module.mujoco, "mjv_addGeoms", lambda *a: added.append(a[1].qpos.copy())
  )
  renderer = object.__new__(OffscreenRenderer)
  renderer._cfg = SimpleNamespace(env_idx=1)
  renderer._model = SimpleNamespace(nmocap=0)
  renderer._data = SimpleNamespace(qpos=np.zeros(1), qvel=np.zeros(1))
  renderer._cam = None
  renderer._opt = None
  renderer._pert = None
  renderer._catmask = SimpleNamespace(value=1)
  renderer._renderer = SimpleNamespace(update_scene=lambda *a, **k: None, scene=None)
  data = SimpleNamespace(
    qpos=torch.tensor([[10.0], [20.0], [30.0]]), qvel=torch.zeros((3, 1))
  )
  renderer.update(data)
  assert [x.item() for x in added] == [10.0, 30.0]


def test_batch_state_normalization_preserves_robot_and_object_sites():
  import mujoco

  from mjlab.continual_distill.classical.record_measured_batch import normalize_world
  from mjlab.continual_distill.classical.record_outcomes import restore, snapshot
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  task = "Mjlab-Axial-Extract-Franka"
  cfg = load_env_cfg(task, test=True)
  cfg.scene.num_envs = 2
  batch = ManagerBasedRlEnv(cfg, device="cpu")
  cfg1 = load_env_cfg(task, test=True)
  cfg1.scene.num_envs = 1
  single = ManagerBasedRlEnv(cfg1, device="cpu")
  try:
    batch.reset()
    single.reset()
    batch.sim.forward()
    state = {k: v[1:2] for k, v in snapshot(batch).items()}
    origin = batch.scene.env_origins[1]
    local_ids = [
      i
      for i in range(single.sim.mj_model.nsite)
      if single.sim.mj_model.site_bodyid[i] != 0
    ]
    site_ids = [
      mujoco.mj_name2id(
        batch.sim.mj_model,
        mujoco.mjtObj.mjOBJ_SITE,
        mujoco.mj_id2name(single.sim.mj_model, mujoco.mjtObj.mjOBJ_SITE, i),
      )
      for i in local_ids
    ]
    expected = batch.sim.data.site_xpos[1, site_ids].detach() - origin
    normalized = normalize_world(state, origin, batch.sim.mj_model)
    restore(single, normalized)
    torch.testing.assert_close(
      single.sim.data.site_xpos[0, local_ids].detach(), expected, atol=2e-5, rtol=0
    )
  finally:
    batch.close()
    single.close()
