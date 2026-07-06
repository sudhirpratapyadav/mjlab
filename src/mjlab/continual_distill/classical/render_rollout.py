#!/usr/bin/env python3
"""Render a single-env classical-policy rollout to PNG frames for inspection."""

import argparse
from pathlib import Path

import numpy as np
import torch

from mjlab.continual_distill.classical import (
  OpenDoorClassicalPolicy,
  OpenDrawerClassicalPolicy,
  PushButtonClassicalPolicy,
  PushCuboidClassicalPolicy,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

POLICIES = {
  "Mjlab-Push-Button-Franka": PushButtonClassicalPolicy,
  "Mjlab-Push-Cuboid-Franka": PushCuboidClassicalPolicy,
  "Mjlab-Open-Door-Franka": OpenDoorClassicalPolicy,
  "Mjlab-Open-Drawer-Franka": OpenDrawerClassicalPolicy,
}


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--task", default="Mjlab-Open-Door-Franka")
  parser.add_argument("--out", default="/tmp/door_frames")
  parser.add_argument("--every", type=int, default=5)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()

  import imageio.v2 as imageio

  configure_torch_backends()
  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.distance = 1.4
  env_cfg.viewer.azimuth = 160.0
  env_cfg.viewer.elevation = -25.0
  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device, render_mode="rgb_array")
  policy = POLICIES[args.task](num_envs=1)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  obs, _ = env.reset()
  policy.reset()
  episode_length = int(env.max_episode_length)
  door = env.scene["door"]
  robot = env.scene["robot"]
  hinge_idx = door.joint_names.index("door_hinge")
  osite = door.site_names.index("object_site")
  gsite = robot.site_names.index("gripper")
  for t in range(episode_length):
    obs_np = obs["policy"].detach().cpu().numpy()
    actions = policy(obs_np)
    obs, *_ = env.step(torch.from_numpy(actions).to(args.device))
    if t % args.every == 0:
      frame = env.render()
      if frame is not None:
        ph = int(policy._phase[0])
        imageio.imwrite(out / f"t{t:03d}_ph{ph}.png", frame)
      ang = float(door.data.joint_pos[0, hinge_idx])
      hs = door.data.site_pos_w[0, osite].cpu().numpy()
      gs = robot.data.site_pos_w[0, gsite].cpu().numpy()
      print(
        f"t={t:3d} ph={int(policy._phase[0])} hinge={np.degrees(ang):6.1f}deg "
        f"handle_w=[{hs[0]:+.3f} {hs[1]:+.3f} {hs[2]:+.3f}] "
        f"grip_w=[{gs[0]:+.3f} {gs[1]:+.3f} {gs[2]:+.3f}]",
        flush=True,
      )
  env.close()
  print(f"frames in {out}")


if __name__ == "__main__":
  main()
