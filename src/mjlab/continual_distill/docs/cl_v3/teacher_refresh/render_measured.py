"""Render one success and failure from the exact measured batch, with provenance."""

import argparse
import json
from datetime import date
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

from mjlab.continual_distill.classical.render_rollout import _build_env, _git_head

FIELDS = ("qpos", "qvel", "mocap_pos", "mocap_quat")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--success-env", type=int)
  parser.add_argument("--failure-env", type=int)
  args = parser.parse_args()
  result = json.loads((args.run / "result.json").read_text())
  traces = np.load(args.run / "trace.npz")
  finals = np.load(args.run / "final_states.npz")
  args.out.mkdir(parents=True, exist_ok=True)
  env = _build_env(result["task_id"], 1, "cpu", False, 0, 0)
  try:
    env.reset()
    model = env.sim.mj_model
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 1080, 1920)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0.43, 0, 0.21]
    camera.distance = 1.30
    camera.azimuth = 60
    camera.elevation = -28
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    options.sitegroup[:] = 0
    clips = {}
    for success, label, filename in (
      (True, "success", "teacher.mp4"),
      (False, "failure", "failure.mp4"),
    ):
      candidates = [x for x in result["outcomes"] if x["success"] == success]
      selected = args.success_env if success else args.failure_env
      if selected is not None:
        candidates = [x for x in candidates if x["env"] == selected]
        assert candidates, f"Environment {selected} has no {label} outcome"
      if not candidates:
        continue
      outcome = candidates[0]
      i, steps = outcome["env"], outcome["steps"]
      origin = finals["origins"][i]
      states = {
        key: np.concatenate([traces[key][:steps, i], finals[key][i : i + 1]])
        for key in FIELDS
      }
      for address, kind in zip(model.jnt_qposadr, model.jnt_type, strict=True):
        if kind == mujoco.mjtJoint.mjJNT_FREE:
          states["qpos"][:, address : address + 3] -= origin
      states["mocap_pos"] -= origin
      # Keep the exact final state as the final video frame, with one frame per
      # control interval. Raw state files additionally contain the reset frame.
      np.savez_compressed(args.out / f"{label}_states.npz", **states)
      writer = imageio_ffmpeg.write_frames(
        str(args.out / filename),
        (1920, 1080),
        fps=50,
        macro_block_size=1,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "20", "-preset", "fast", "-threads", "2"],
        ffmpeg_log_level="error",
      )
      writer.send(None)
      panels = []
      try:
        for index in range(1, steps + 1):
          for key in FIELDS:
            getattr(data, key)[:] = states[key][index]
          mujoco.mj_forward(model, data)
          renderer.update_scene(data, camera=camera, scene_option=options)
          frame = renderer.render()
          writer.send(np.ascontiguousarray(frame))
          if index in {1, max(1, steps // 2), steps}:
            panels.append(Image.fromarray(frame.copy()))
          if index == max(1, steps // 2) and (
            success or not (args.out / "thumb.jpg").exists()
          ):
            Image.fromarray(frame).save(args.out / "thumb.jpg")
      finally:
        writer.close()
      strip = Image.new("RGB", (1920, 360))
      for panel_index, panel in enumerate(panels):
        strip.paste(panel.resize((640, 360)), (panel_index * 640, 0))
      strip.save(args.out / f"{label}_review.jpg")
      clips[label] = dict(
        batch_env_index=i,
        seed=result["stats_seed"],
        control_steps=steps,
        simulation_seconds=steps * env.step_dt,
        encoded_frames=steps,
        video_seconds=steps / 50,
        outcome=outcome,
      )
      print("RENDERED", label, i, steps, flush=True)
    renderer.close()
    result.update(
      bar=0.90,
      pass_bar=result["sr"] >= 0.90,
      head=_git_head(),
      date=date.today().isoformat(),
      geometry_revision="2026-09-11-corrected-semantics",
      visualization_revision="measured-batch-20260911",
      has_teacher_clip="success" in clips,
      has_failure_clip="failure" in clips,
      render_episodes_used=result["n"],
      rendered_episode_count=len(clips),
      search_method="exact_measured_batch",
      fps=50,
      width=1920,
      height=1080,
      control_dt=float(env.step_dt),
      physics_dt=0.005,
      physics_hz=200,
      control_hz=50,
      decimation=4,
      clip_end="first success, termination or timeout",
      clips=clips,
      missing_success_reason=None
      if "success" in clips
      else "No successes in the measured batch.",
      missing_failure_reason=None
      if "failure" in clips
      else "No failures in the measured batch.",
    )
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
