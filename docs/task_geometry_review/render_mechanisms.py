import importlib
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw

out = Path("docs/task_geometry_review")
shots = []
for name in (
  "switch",
  "lever",
  "valve",
  "lid",
  "flap",
  "plug",
  "window",
  "door",
  "drawer",
  "button",
):
  mod = importlib.import_module(
    f"mjlab.asset_zoo.objects.articulated.{name}.{name}_constants"
  )
  spec = getattr(mod, f"get_{name}_spec")()
  m = spec.compile()
  d = mujoco.MjData(m)
  m.vis.global_.offwidth = 600
  m.vis.global_.offheight = 400
  m.vis.headlight.ambient[:] = 0.5
  m.vis.headlight.diffuse[:] = 0.8
  lo, hi = m.jnt_range[0]
  angles = [lo, 4.712389 if name == "valve" else (lo + hi) / 2, hi]
  for k, q in enumerate(angles):
    d.qpos[0] = q
    mujoco.mj_forward(m, d)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = d.xpos[1]
    cam.distance = 0.45 if name in ("switch", "plug", "button") else 0.75
    if name in ("door", "drawer", "window"):
      cam.distance = 1.2
      cam.lookat[1] += 0.12
    cam.azimuth = 45
    cam.elevation = -25
    r = mujoco.Renderer(m, 400, 600)
    opt = mujoco.MjvOption()
    opt.geomgroup[3] = 0
    opt.sitegroup[:] = 0
    r.update_scene(d, camera=cam, scene_option=opt)
    im = Image.fromarray(r.render())
    r.close()
    ImageDraw.Draw(im).text((10, 10), f"{name}: q={q:.3f}", fill="white")
    im.save(out / f"{name}_sweep_{k}.png")
    shots.append(im.resize((300, 200)))
sheet = Image.new("RGB", (900, 2000))
for i, im in enumerate(shots):
  sheet.paste(im, ((i % 3) * 300, (i // 3) * 200))
sheet.save(out / "mechanism_sweeps.jpg")
