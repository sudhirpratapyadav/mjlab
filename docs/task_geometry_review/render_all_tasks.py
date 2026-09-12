from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.benchmark import all_benchmark_tasks
from mjlab.tasks.manipulation.taxonomy import Embodiment
from mjlab.tasks.registry import load_env_cfg

out = Path("docs/task_geometry_review")
thumbs = []
for task, tax in all_benchmark_tasks().items():
  if tax.embodiment != Embodiment.ARM_GRIPPER:
    continue
  cfg = load_env_cfg(task, test=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  env.reset()
  env.sim.forward()
  m = env.sim.mj_model
  d = mujoco.MjData(m)
  d.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  d.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
  d.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(m, d)
  goal = env.scene["mocap_goal"].data.root_link_pos_w[0].cpu().numpy()
  cam = mujoco.MjvCamera()
  cam.lookat[:] = [0.42, 0, 0.18]
  cam.distance = 1.25
  cam.azimuth = 135
  cam.elevation = -30
  m.vis.global_.offwidth = 640
  m.vis.global_.offheight = 480
  r = mujoco.Renderer(m, 480, 640)
  opt = mujoco.MjvOption()
  opt.geomgroup[3] = 0
  opt.sitegroup[:] = 0
  r.update_scene(d, camera=cam, scene_option=opt)
  im = Image.fromarray(r.render())
  ImageDraw.Draw(im).text((10, 10), task, fill="white")
  im.save(out / (task + ".png"))
  thumbs.append(im.resize((320, 240)))
  r.close()
  env.close()
  print("RENDERED", task, flush=True)
sheet = Image.new("RGB", (5 * 320, 6 * 240), "#202020")
for i, im in enumerate(thumbs):
  sheet.paste(im, ((i % 5) * 320, (i // 5) * 240))
sheet.save(out / "all_tasks.jpg")
