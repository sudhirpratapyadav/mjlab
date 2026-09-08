"""Render an object XML on its own: a still PNG and a turntable MP4 (G6 evidence).

Loads the object MJCF exactly as the env does (``mujoco.MjSpec.from_file``), drops it
into a lit studio scene (textured floor, three lights, skybox) and orbits a camera
around it.  Articulated objects get their joints swept over the full range during the
turn so the video also shows the mechanism moving.

Needs a GPU node with ``MUJOCO_GL=egl``::

    MUJOCO_GL=egl PYTHONPATH=src .venv/bin/python -m mjlab.scripts.render_asset \
        src/mjlab/asset_zoo/objects/free/cube/xmls/cube.xml \
        --out src/mjlab/continual_distill/docs/cl_v2/renders/Lift-Cube \
        --joint-sweep

Outputs ``<out>/still.png``, ``<out>/turntable.mp4``, ``<out>/colliders.png``
(collision geoms only, for the G2 review).  The scene, camera and lights are the
same rig ``mjlab.scene`` uses for task videos, so the still is representative.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np

STUDIO_ASSETS = """
  <asset>
    <texture name="studio_sky" type="skybox" builtin="gradient" rgb1="0.86 0.88 0.92"
             rgb2="0.55 0.60 0.68" width="256" height="256"/>
    <texture name="studio_floor_tex" type="2d" builtin="checker" rgb1="0.62 0.60 0.56"
             rgb2="0.56 0.54 0.50" width="512" height="512" mark="edge" markrgb="0.5 0.48 0.45"/>
    <material name="studio_floor" texture="studio_floor_tex" texrepeat="12 12"
              texuniform="true" reflectance="0.05" specular="0.1" shininess="0.1"/>
  </asset>
"""


def _studio(spec: mujoco.MjSpec, floor_z: float, sky: bool = True) -> None:
  """Add floor, lights and camera to a spec (idempotent names)."""
  spec.visual.global_.offwidth = 1280
  spec.visual.global_.offheight = 720
  spec.visual.quality.shadowsize = 4096
  spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
  spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
  spec.visual.headlight.specular = [0.1, 0.1, 0.1]
  spec.visual.map.znear = 0.005
  if sky:
    tex = spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[0.86, 0.88, 0.92], rgb2=[0.55, 0.60, 0.68],
                           width=256, height=256)
  ftex = spec.add_texture(name="studio_floor_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                          builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                          rgb1=[0.62, 0.60, 0.56], rgb2=[0.56, 0.54, 0.50],
                          width=512, height=512, mark=mujoco.mjtMark.mjMARK_EDGE,
                          markrgb=[0.5, 0.48, 0.45])
  mat = spec.add_material(name="studio_floor", texrepeat=[12, 12], texuniform=True,
                          reflectance=0.05, specular=0.1, shininess=0.1)
  mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "studio_floor_tex"
  spec.worldbody.add_geom(name="studio_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                          size=[3, 3, 0.05], pos=[0, 0, floor_z], material="studio_floor",
                          contype=1, conaffinity=1)
  spec.worldbody.add_light(name="key", pos=[1.2, -1.0, 1.8], dir=[-0.55, 0.45, -0.7],
                           diffuse=[0.75, 0.72, 0.68], specular=[0.3, 0.3, 0.3],
                           castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.worldbody.add_light(name="fill", pos=[-1.5, -0.8, 1.2], dir=[0.7, 0.4, -0.5],
                           diffuse=[0.35, 0.36, 0.40], specular=[0.05, 0.05, 0.05],
                           castshadow=False, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.worldbody.add_light(name="rim", pos=[0.2, 1.6, 1.4], dir=[-0.1, -0.8, -0.6],
                           diffuse=[0.30, 0.30, 0.32], specular=[0.15, 0.15, 0.15],
                           castshadow=False, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
  spec.worldbody.add_camera(name="orbit", pos=[1, 0, 0.5], fovy=35)


def _object_bounds(model: mujoco.MjModel, data: mujoco.MjData, exclude_prefix="studio"):
  lo = np.full(3, np.inf)
  hi = np.full(3, -np.inf)
  for g in range(model.ngeom):
    if model.geom(g).name.startswith(exclude_prefix):
      continue
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
      continue
    r = model.geom_rbound[g]
    c = data.geom_xpos[g]
    lo = np.minimum(lo, c - r)
    hi = np.maximum(hi, c + r)
  if not np.all(np.isfinite(lo)):
    return np.array([-0.1, -0.1, 0]), np.array([0.1, 0.1, 0.2])
  return lo, hi


def render(xml: Path, out: Path, frames: int = 120, fps: int = 30, joint_sweep: bool = True,
           width: int = 1280, height: int = 720, elev_deg: float = 28.0,
           settle_steps: int = 0, floor: str = "auto") -> dict:
  import imageio.v2 as imageio

  out.mkdir(parents=True, exist_ok=True)
  spec = mujoco.MjSpec.from_file(str(xml))
  # Freeze mocap bases in place; drop free joints so the object sits still while we orbit.
  free_bodies = []
  for j in spec.joints:
    if j.type == mujoco.mjtJoint.mjJNT_FREE:
      free_bodies.append(j)
  m0 = spec.compile()
  d0 = mujoco.MjData(m0)
  mujoco.mj_forward(m0, d0)
  lo, hi = _object_bounds(m0, d0)
  floor_z = lo[2] if floor == "auto" else float(floor)
  _studio(spec, floor_z)
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  for _ in range(settle_steps):
    mujoco.mj_step(model, data)
  lo, hi = _object_bounds(model, data)
  center = (lo + hi) / 2
  size = float(np.max(hi - lo))
  # rbound spheres over-estimate flat objects; frame on ~0.75 of the sphere extent.
  dist = max(0.25, 0.75 * size / math.tan(math.radians(35) / 2))
  cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "orbit")

  # Articulation: joints we can sweep (hinge/slide with a range), ignore free.
  sweep = []
  for j in range(model.njnt):
    jt = model.jnt_type[j]
    if jt in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and model.jnt_limited[j]:
      sweep.append((model.jnt_qposadr[j], float(model.jnt_range[j][0]), float(model.jnt_range[j][1]),
                    model.joint(j).name))

  renderer = mujoco.Renderer(model, height=height, width=width)
  opt = mujoco.MjvOption()
  opt.geomgroup[:] = 0
  opt.geomgroup[0] = 1
  opt.geomgroup[1] = 1
  opt.geomgroup[2] = 1
  opt_col = mujoco.MjvOption()
  opt_col.geomgroup[:] = 0
  opt_col.geomgroup[3] = 1
  opt_col.geomgroup[0] = 1
  opt_col.flags[mujoco.mjtVisFlag.mjVIS_CONVEXHULL] = 1
  cam = mujoco.MjvCamera()

  def set_cam(az_deg: float):
    az = math.radians(az_deg)
    el = math.radians(elev_deg)
    pos = center + dist * np.array([math.cos(az) * math.cos(el), math.sin(az) * math.cos(el), math.sin(el)])
    model.cam_pos[cam_id] = pos
    fwd = center - pos
    fwd /= np.linalg.norm(fwd)
    up = np.array([0, 0, 1.0])
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    up2 = np.cross(right, fwd)
    # MuJoCo camera looks along -z of its frame; x=right, y=up.
    R = np.stack([right, up2, -fwd], axis=1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.flatten())
    model.cam_quat[cam_id] = q

  clips = []
  qpos0 = data.qpos.copy()
  for f in range(frames):
    t = f / max(1, frames - 1)
    set_cam(-135 + 360 * t)
    if joint_sweep and sweep:
      s = 0.5 - 0.5 * math.cos(2 * math.pi * t)  # 0 -> 1 -> 0
      data.qpos[:] = qpos0
      for adr, lo_r, hi_r, _ in sweep:
        data.qpos[adr] = lo_r + (hi_r - lo_r) * s if qpos0[adr] <= (lo_r + hi_r) / 2 else hi_r - (hi_r - lo_r) * s
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera="orbit", scene_option=opt)
    clips.append(renderer.render().copy())
    if f == int(frames * 0.12):
      imageio.imwrite(out / "still.png", clips[-1])
      renderer.update_scene(data, camera="orbit", scene_option=opt_col)
      imageio.imwrite(out / "colliders.png", renderer.render())
  imageio.mimwrite(out / "turntable.mp4", clips, fps=fps, codec="libx264", quality=8,
                   macro_block_size=None, ffmpeg_params=["-pix_fmt", "yuv420p"])
  renderer.close()
  info = {"xml": str(xml), "frames": frames, "object_extent_m": (hi - lo).round(4).tolist(),
          "swept_joints": [s[3] for s in sweep], "nmesh": int(model.nmesh),
          "ntex": int(model.ntex), "ngeom": int(model.ngeom)}
  (out / "render.json").write_text(__import__("json").dumps(info, indent=1))
  print(info)
  return info


def _main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("xml")
  ap.add_argument("--out", required=True)
  ap.add_argument("--frames", type=int, default=120)
  ap.add_argument("--fps", type=int, default=30)
  ap.add_argument("--joint-sweep", action="store_true")
  ap.add_argument("--elev", type=float, default=28.0)
  ap.add_argument("--settle-steps", type=int, default=0)
  ap.add_argument("--floor", default="auto", help="'auto' = lowest geom, or a z value")
  a = ap.parse_args()
  render(Path(a.xml), Path(a.out), frames=a.frames, fps=a.fps, joint_sweep=a.joint_sweep,
         elev_deg=a.elev, settle_steps=a.settle_steps, floor=a.floor)


if __name__ == "__main__":
  _main()
