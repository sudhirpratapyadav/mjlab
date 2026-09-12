"""Find real outcomes cheaply, then render their recorded physics states in HD.

No action replay, re-simulation, fabricated failures, or wall-clock timing enters
an encoded clip. Only the visual forward pass is recomputed from recorded states.
"""

from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture

STATE_FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat")


def snapshot(env):
  return {
    name: getattr(env.sim.data, name).detach().cpu().clone() for name in STATE_FIELDS
  }


def restore(env, state):
  for name, value in state.items():
    getattr(env.sim.data, name).copy_(value.to(env.device))
  env.sim.forward()


def encode(env, states, output, fps, width, height, debug_dir=None, debug_every=5):
  """Stream frames; HD memory use does not grow with episode duration."""
  count = max(1, round(len(states) * float(env.step_dt) * fps))
  indices = np.minimum(
    (np.arange(count) / (fps * float(env.step_dt))).astype(int), len(states) - 1
  )
  writer = imageio_ffmpeg.write_frames(
    str(output),
    (width, height),
    fps=fps,
    codec="libx264",
    pix_fmt_out="yuv420p",
    macro_block_size=1,
    output_params=["-crf", "18", "-preset", "fast", "-threads", "2"],
    ffmpeg_log_level="error",
  )
  writer.send(None)
  thumb = None
  try:
    for frame_index, state_index in enumerate(indices):
      restore(env, states[state_index])
      frame = np.asarray(env.render())
      if frame.ndim == 4:
        frame = frame[0]
      if frame.dtype != np.uint8:
        frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
      assert frame.shape == (height, width, 3), frame.shape
      writer.send(np.ascontiguousarray(frame))
      if debug_dir is not None and frame_index % debug_every == 0:
        import imageio.v2 as imageio

        debug_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(debug_dir / f"{frame_index:05d}.png", frame)
      if frame_index == len(indices) // 2:
        thumb = frame.copy()
  finally:
    writer.close()
  return thumb, {
    "control_steps": len(states),
    "simulation_seconds": len(states) * float(env.step_dt),
    "encoded_frames": count,
    "video_seconds": count / fps,
  }


def record_outcomes(
  task_id,
  device,
  out_dir,
  max_episodes,
  fps,
  width,
  height,
  stats,
  seed,
  debug=False,
  debug_every=5,
):
  import imageio.v2 as imageio

  from mjlab.continual_distill.classical.render_rollout import _build_env, _success_term
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.scripts.record_task_videos import RecordConfig, _autoframe_camera

  ManagerBasedRlEnv.seed(seed)
  env = _build_env(task_id, 1, device, True, width, height)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  selected = {}
  result = {
    "has_teacher_clip": False,
    "has_failure_clip": False,
    "render_episodes_used": 0,
    "search_successes": 0,
    "search_failures": 0,
    "clips": {},
  }
  try:
    env.reset()
    _autoframe_camera(env, RecordConfig())
    cmd = _success_term(env)
    # The physical ghost is the authoritative visualization. Do not overlay
    # debug spheres computed from the last search episode during state playback.
    for term in env.command_manager._terms.values():
      term.cfg.debug_vis = False
    policy = CLASSICAL_POLICIES[task_id](num_envs=1)
    for ep in range(max_episodes):
      need_success = "success" not in selected
      need_failure = "failure" not in selected
      # Never infer certainty from 98%. Zero measured successes cannot supply a
      # success clip; a perfect measured teacher need not supply a failure clip.
      if stats is not None:
        need_success &= stats.num_success > 0
        need_failure &= stats.num_success < stats.n
      if not need_success and not need_failure:
        break
      # Startup randomization (notably friction) must also vary between search
      # attempts, as it does across the independently randomized stats worlds.
      ManagerBasedRlEnv.seed(seed + ep)
      if "startup" in env.event_manager.available_modes:
        env.event_manager.apply(mode="startup")
      obs, _ = env.reset()
      policy.reset()
      states = []
      terminal_state = [None]

      def before_reset(_ids, terminal_state=terminal_state):
        cmd._update_command()
        terminal_state[0] = snapshot(env)

      succeeded = False
      first_success_step = None
      with TerminalSuccessCapture(env, cmd, before_reset) as capture:
        for step in range(int(env.max_episode_length)):
          action = policy(obs["policy"].detach().cpu().numpy())
          obs, _, terminated, truncated, _ = capture.step(
            torch.from_numpy(action).to(device)
          )
          done = bool((terminated | truncated).any())
          success = capture.terminal_success if done else cmd.episode_success > 0
          succeeded |= bool(success[0])
          states.append(terminal_state[0] if done else snapshot(env))
          if succeeded and first_success_step is None:
            first_success_step = step
          if done or (
            first_success_step is not None
            and step - first_success_step >= round(0.5 / env.step_dt)
          ):
            break
      label = "success" if succeeded else "failure"
      result["search_successes" if succeeded else "search_failures"] += 1
      result["render_episodes_used"] += 1
      print(
        f"[search] {task_id} attempt={ep + 1} {label} steps={len(states)}", flush=True
      )
      if label not in selected:
        selected[label] = (ep, states)
    result.update(
      physics_dt=float(env.physics_dt),
      control_dt=float(env.step_dt),
      physics_hz=1 / float(env.physics_dt),
      control_hz=1 / float(env.step_dt),
      decimation=env.cfg.decimation,
      debug_overlays=False,
      search_randomizes_startup=True,
      width=width,
      height=height,
      fps=fps,
    )
    for label, (ep, states) in selected.items():
      filename = "teacher.mp4" if label == "success" else "failure.mp4"
      np.savez_compressed(
        out_dir / f"{label}_states.npz",
        **{
          name: torch.stack([state[name] for state in states]).numpy()
          for name in STATE_FIELDS
        },
      )
      thumb, timing = encode(
        env,
        states,
        out_dir / filename,
        fps,
        width,
        height,
        out_dir / "debug_frames" / label if debug else None,
        debug_every,
      )
      result["has_teacher_clip" if label == "success" else "has_failure_clip"] = True
      result["clips"][label] = dict(timing, seed=seed + ep, search_attempt=ep + 1)
      if label == "success" or not (out_dir / "thumb.jpg").exists():
        imageio.imwrite(out_dir / "thumb.jpg", thumb)
    result["rendered_episode_count"] = len(selected)
    result["missing_success_reason"] = (
      None
      if "success" in selected
      else (
        "No successes in the measured trials"
        if stats and stats.num_success == 0
        else f"Not found in {max_episodes} search attempts"
      )
    )
    result["missing_failure_reason"] = (
      None
      if "failure" in selected
      else (
        "All measured trials succeeded"
        if stats and stats.num_success == stats.n
        else f"Not found in {max_episodes} search attempts"
      )
    )
  finally:
    env.close()
  return result
