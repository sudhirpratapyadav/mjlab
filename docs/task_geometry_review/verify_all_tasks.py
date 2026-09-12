"""Sequential per-task reset, MuJoCo-state, reward, rollout and visual verification."""

import argparse
import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
import mujoco
import numpy as np
import torch
from PIL import Image, ImageDraw

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture
from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.task_state_probe import goal_oracle, native_state
from mjlab.tasks.registry import load_env_cfg


def capture_state(env):
  return {
    k: getattr(env.sim.data, k)[0].cpu().numpy().copy()
    for k in ("qpos", "qvel", "mocap_pos", "mocap_quat")
  }


def contact_summary(model, data):
  return [
    dict(
      a=model.geom(c.geom[0]).name, b=model.geom(c.geom[1]).name, distance=float(c.dist)
    )
    for c in data.contact
    if c.dist < -0.003
  ]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--tasks", nargs="*")
  parser.add_argument(
    "--out", type=Path, default=Path("docs/task_geometry_review/verified_tasks")
  )
  parser.add_argument("--resets", type=int, default=8)
  args = parser.parse_args()
  tasks = args.tasks or [
    r["task_id"]
    for r in json.loads(
      Path(
        "src/mjlab/continual_distill/docs/cl_v3/physics_refresh/results.json"
      ).read_text()
    )
  ]
  source = hashlib.sha256()
  for folder in (
    "src/mjlab/tasks/manipulation",
    "src/mjlab/asset_zoo",
    "src/mjlab/continual_distill/classical",
  ):
    for path in sorted(Path(folder).rglob("*")):
      if path.suffix in (".py", ".xml"):
        source.update(str(path).encode())
        source.update(path.read_bytes())
  args.out.mkdir(parents=True, exist_ok=True)
  for task in tasks:
    folder = args.out / task
    folder.mkdir(exist_ok=True)
    cfg = load_env_cfg(task, test=True)
    cfg.scene.num_envs = 1
    ManagerBasedRlEnv.seed(20260911)
    env = ManagerBasedRlEnv(cfg, device="cpu")
    try:
      cmd = next(iter(env.command_manager._terms.values()))
      result = dict(
        task=task,
        source_sha256=source.hexdigest(),
        protocol="one strict diagnostic episode; not a success-rate estimate",
        command=type(cmd).__name__,
        episode_seconds=cfg.episode_length_s,
        terminations=list(cfg.terminations),
        resets=[],
        rewards={
          k: dict(function=v.func.__name__, weight=v.weight)
          for k, v in cfg.rewards.items()
        },
      )
      for _reset in range(args.resets):
        env.reset()
        env.sim.forward()
        cmd._update_metrics()
        m, d = native_state(env)
        result["resets"].append(
          dict(
            success=bool(cmd.compute_success()[0]),
            finite=bool(np.isfinite(d.qpos).all()),
            deep_contacts=contact_summary(m, d),
          )
        )
      env.reset()
      env.sim.forward()
      rejected = capture_state(env)
      try:
        accepted = bool(goal_oracle(env, cmd)[0])
        result["oracle"] = dict(
          accepted=accepted,
          description="constructed MuJoCo goal/contact state; not a rollout",
        )
      except RuntimeError as exc:
        result["oracle"] = dict(accepted=False, error=str(exc))
      oracle_state = capture_state(env)
      om, od = native_state(env)
      result["oracle"]["deep_contacts"] = contact_summary(om, od)
      np.savez_compressed(folder / "goal_state.npz", **oracle_state)
      obs, _ = env.reset()
      policy = CLASSICAL_POLICIES[task](num_envs=1)
      states = [capture_state(env)]
      rewards = []
      metrics = []
      terminal = [None]
      terminal_metrics = [None]

      def before_reset(
        ids, cmd=cmd, env=env, terminal=terminal, terminal_metrics=terminal_metrics
      ):
        terminal_metrics[0] = {k: float(v[0]) for k, v in cmd.metrics.items()}
        cmd._update_command()
        env.sim.forward()
        terminal[0] = capture_state(env)

      with TerminalSuccessCapture(env, cmd, before_reset) as capture:
        for step in range(int(env.max_episode_length)):
          action = policy(obs["policy"].cpu().numpy())
          obs, reward, terminated, truncated, _ = capture.step(torch.from_numpy(action))
          done = bool((terminated | truncated)[0])
          success = bool(
            (capture.terminal_success if done else cmd.episode_success > 0)[0]
          )
          states.append(terminal[0] if done else capture_state(env))
          rewards.append(float(reward[0]))
          metrics.append(
            terminal_metrics[0]
            if done
            else {k: float(v[0]) for k, v in cmd.metrics.items()}
          )
          assert all(np.isfinite(v).all() for v in states[-1].values()), task
          assert np.isfinite(rewards[-1]), task
          if done or success:
            result["rollout"] = dict(
              steps=step + 1,
              success=success,
              terminated=bool(terminated[0]),
              timeout=bool(truncated[0]),
              simulation_seconds=(step + 1) * env.step_dt,
            )
            break
      np.savez_compressed(
        folder / "trajectory.npz",
        **{k: np.stack([s[k] for s in states]) for k in states[0]},
        reward=np.array(rewards),
      )
      result["final_metrics"] = metrics[-1]
      m, d = native_state(env)
      m.vis.global_.offwidth = 640
      m.vis.global_.offheight = 480
      renderer = mujoco.Renderer(m, 480, 640)
      camera = mujoco.MjvCamera()
      camera.lookat[:] = [0.48, 0, 0.18]
      camera.distance = 1.4
      camera.azimuth = 60
      camera.elevation = -30
      options = mujoco.MjvOption()
      options.geomgroup[3] = 0
      options.sitegroup[:] = 0

      def render(
        state,
        label,
        d=d,
        m=m,
        renderer=renderer,
        camera=camera,
        options=options,
        task=task,
      ):
        for k, v in state.items():
          getattr(d, k)[:] = v
        mujoco.mj_forward(m, d)
        renderer.update_scene(d, camera=camera, scene_option=options)
        im = Image.fromarray(renderer.render())
        ImageDraw.Draw(im).text((10, 10), f"{task}\n{label}", fill="white")
        return im

      panels = []
      render(oracle_state, "constructed goal-state oracle").save(
        folder / "goal_oracle.png"
      )
      # Inspect the geometry at the task goal at a useful scale, in addition to
      # the scene-wide context frames and real rollout video.
      saved_lookat = camera.lookat.copy()
      saved_distance = camera.distance
      goal_body = int(env.scene["mocap_goal"].data.indexing.root_body_id)
      render(oracle_state, "constructed goal-state oracle")
      camera.lookat[:] = d.xpos[goal_body] + np.array([0.0, 0.0, 0.04])
      camera.distance = 0.65
      saved_elevation = camera.elevation
      saved_azimuth = camera.azimuth
      if "Container" in task or "Throw-To-Bin" in task:
        camera.elevation = -65
        camera.azimuth = 240
      render(oracle_state, "goal detail").save(folder / "goal_detail.png")
      camera.lookat[:] = saved_lookat
      camera.distance = saved_distance
      camera.elevation = saved_elevation
      camera.azimuth = saved_azimuth
      render(rejected, "reset negative control").save(folder / "negative_control.png")
      for index, label in [
        (0, "reset"),
        (len(states) // 2, "mid-rollout"),
        (len(states) - 1, "final"),
      ]:
        im = render(states[index], label)
        im.save(folder / f"{label}.png")
        panels.append(im)
      strip = Image.new("RGB", (1920, 480))
      for i, im in enumerate(panels):
        strip.paste(im, (640 * i, 0))
      strip.save(folder / "review.jpg")
      fps = 10
      writer = imageio_ffmpeg.write_frames(
        str(folder / "rollout.mp4"),
        (640, 480),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast", "-threads", "2"],
        ffmpeg_log_level="error",
      )
      writer.send(None)
      count = max(1, round((len(states) - 1) * env.step_dt * fps))
      try:
        for i in range(count):
          index = min(round(i / (fps * env.step_dt)), len(states) - 1)
          writer.send(
            np.asarray(render(states[index], f"t={index * env.step_dt:.2f}s"))
          )
      finally:
        writer.close()
      renderer.close()
      result["video"] = dict(fps=fps, frames=count)
      (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
      print("VERIFIED", task, result["rollout"], flush=True)
    finally:
      env.close()
  rows = []
  for folder in sorted(args.out.iterdir()):
    if (folder / "result.json").exists():
      rows.append(json.loads((folder / "result.json").read_text()))
  (args.out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
  sheet = Image.new("RGB", (5 * 384, ((len(rows) + 4) // 5) * 288), "#202020")
  for i, row in enumerate(rows):
    im = Image.open(args.out / row["task"] / "final.png").resize((384, 288))
    sheet.paste(im, ((i % 5) * 384, (i // 5) * 288))
  sheet.save(args.out / "all_tasks.jpg")


if __name__ == "__main__":
  main()
