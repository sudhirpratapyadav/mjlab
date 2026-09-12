"""Capture both outcomes directly from a reproducible batched evaluation."""

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture
from mjlab.continual_distill.classical.record_outcomes import (
  STATE_FIELDS,
  encode,
  restore,
  snapshot,
)
from mjlab.continual_distill.classical.render_rollout import (
  _build_env,
  _success_term,
  source_fingerprint,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.record_task_videos import RecordConfig, _autoframe_camera
from mjlab.utils.torch import configure_torch_backends


def normalize_world(state, origin, model):
  result = {key: value.clone() for key, value in state.items()}
  for address, kind in zip(model.jnt_qposadr, model.jnt_type, strict=True):
    if kind == mujoco.mjtJoint.mjJNT_FREE:
      result["qpos"][:, address : address + 3] -= origin
  result["mocap_pos"] -= origin
  return result


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--task", required=True)
  p.add_argument("--out", type=Path, required=True)
  p.add_argument("--stats-from", type=Path, required=True)
  a = p.parse_args()
  configure_torch_backends()
  old = json.loads(a.stats_from.read_text())
  seed = old["stats_seed"]
  n = old["n"]
  device = "cuda:0"
  ManagerBasedRlEnv.seed(seed)
  env = _build_env(a.task, n, device, False, 0, 0)
  histories = [[] for _ in range(n)]
  selected = {}
  count = 0
  try:
    policy = CLASSICAL_POLICIES[a.task](num_envs=n)
    obs, _ = env.reset()
    policy.reset()
    cmd = _success_term(env)
    alive = np.ones(n, dtype=bool)
    successes = np.zeros(n, dtype=bool)
    terminal = {}

    def before_reset(ids):
      cmd._update_command()
      state = snapshot(env)
      for i in ids.tolist():
        terminal[i] = {k: v[i : i + 1] for k, v in state.items()}

    with TerminalSuccessCapture(env, cmd, before_reset) as capture:
      for step in range(int(env.max_episode_length)):
        active = np.flatnonzero(alive)
        action = policy(obs["policy"].detach().cpu().numpy(), active_env_ids=active)
        obs, _, terminated, truncated, _ = capture.step(
          torch.from_numpy(action).to(device)
        )
        done = (terminated | truncated).detach().cpu().numpy()
        success = (
          torch.where(
            terminated | truncated, capture.terminal_success, cmd.episode_success > 0
          )
          .detach()
          .cpu()
          .numpy()
        )
        current = snapshot(env)
        for i in active:
          state = (
            terminal[i] if done[i] else {k: v[i : i + 1] for k, v in current.items()}
          )
          histories[i].append(state)
          if done[i] or success[i]:
            successes[i] = success[i]
            alive[i] = False
            label = "success" if success[i] else "failure"
            if label not in selected:
              origin = env.scene.env_origins[i].cpu()
              selected[label] = (
                int(i),
                [normalize_world(s, origin, env.sim.mj_model) for s in histories[i]],
              )
            histories[i] = []
        terminal.clear()
        if step % 50 == 0:
          print(
            "STEP",
            step,
            "alive",
            int(alive.sum()),
            "successes",
            int(successes.sum()),
            flush=True,
          )
        if not alive.any():
          break
    count = int(successes.sum())
    print("MEASURED", count, n, "previous", old["num_success"], flush=True)
    if count != old["num_success"]:
      raise RuntimeError("Measured-batch replay did not reproduce the published rate")
    if not {"success", "failure"}.issubset(selected):
      raise RuntimeError("Both outcomes were not present in batch")
  finally:
    env.close()
  env = _build_env(a.task, 1, device, True, 1920, 1080)
  try:
    env.reset()
    for term in env.command_manager._terms.values():
      term.cfg.debug_vis = False
    a.out.mkdir(parents=True, exist_ok=True)
    r = dict(old)
    r.update(
      source_sha256=source_fingerprint(),
      stats_source_sha256=old["source_sha256"],
      stats_reused_visual_only=True,
      visualization_revision="axis-goals-hd-20260910",
      has_teacher_clip=True,
      has_failure_clip=True,
      render_episodes_used=n,
      rendered_episode_count=2,
      search_method="exact_measured_batch",
      search_successes=count,
      search_failures=n - count,
      physics_dt=0.005,
      control_dt=0.02,
      physics_hz=200.0,
      control_hz=50.0,
      decimation=4,
      width=1920,
      height=1080,
      fps=50,
      debug_overlays=False,
      missing_success_reason=None,
      missing_failure_reason=None,
      clip_end="first termination/timeout or latched success",
      clips={},
    )
    for label, (i, states) in selected.items():
      restore(env, states[0])
      _autoframe_camera(env, RecordConfig())
      np.savez_compressed(
        a.out / f"{label}_states.npz",
        **{k: torch.stack([s[k] for s in states]).numpy() for k in STATE_FIELDS},
      )
      filename = "teacher.mp4" if label == "success" else "failure.mp4"
      thumb, timing = encode(env, states, a.out / filename, 50, 1920, 1080)
      r["clips"][label] = dict(timing, batch_env_index=i, seed=seed)
      if label == "success":
        imageio.imwrite(a.out / "thumb.jpg", thumb)
    (a.out / "result.json").write_text(json.dumps(r, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
