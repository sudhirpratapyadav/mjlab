"""Studio lighting / floor for the Class A (Franka) manipulation scenes.

The manipulation scenes had no ``<light>`` and no skybox: everything was lit by the
headlight only and the floor was a blue procedural checker (see ``CODE_MAP.md`` §5).
Realistic textured assets look flat under that rig, so CL-V2 adds one shared studio
setup, applied through ``SceneCfg.spec_fn`` to every registered Franka task
(``config/franka/__init__.py``):

  * three directional lights (key with shadows, fill, rim) plus a dimmer headlight;
  * a gradient skybox;
  * the terrain ``groundplane`` material re-textured with a CC0 wood floor
    (``asset_zoo/scenes/studio/floor_wood.png``, provenance in the same folder).

Physics, observations and rewards are untouched: lights, textures and materials are
render-only in MuJoCo. Set ``MJLAB_NO_STUDIO=1`` to disable (e.g. to reproduce the
old look).
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH

FLOOR_TEXTURE: Path = MJLAB_SRC_PATH / "asset_zoo" / "scenes" / "studio" / "floor_wood.png"


def studio_spec_fn(spec: mujoco.MjSpec) -> None:
  """``SceneCfg.spec_fn`` hook: add the studio rig to an assembled scene spec."""
  if os.environ.get("MJLAB_NO_STUDIO"):
    return
  names = {l.name for l in spec.lights}
  if "studio_key" in names:
    return  # idempotent

  # Headlight down: it flattens everything when it is the only light.
  spec.visual.headlight.ambient = [0.30, 0.30, 0.30]
  spec.visual.headlight.diffuse = [0.30, 0.30, 0.30]
  spec.visual.headlight.specular = [0.05, 0.05, 0.05]
  spec.visual.rgba.haze = [0.9, 0.92, 0.95, 1.0]

  d = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
  spec.worldbody.add_light(name="studio_key", pos=[1.5, -1.5, 2.5], dir=[-0.45, 0.45, -0.75],
                           diffuse=[0.70, 0.68, 0.64], specular=[0.25, 0.25, 0.25],
                           castshadow=True, type=d)
  spec.worldbody.add_light(name="studio_fill", pos=[-1.5, -1.0, 1.5], dir=[0.6, 0.4, -0.55],
                           diffuse=[0.32, 0.33, 0.38], specular=[0.03, 0.03, 0.03],
                           castshadow=False, type=d)
  spec.worldbody.add_light(name="studio_rim", pos=[0.5, 2.0, 1.8], dir=[-0.15, -0.8, -0.55],
                           diffuse=[0.28, 0.28, 0.30], specular=[0.12, 0.12, 0.12],
                           castshadow=False, type=d)

  if "studio_sky" not in {t.name for t in spec.textures}:
    spec.add_texture(name="studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.88, 0.90, 0.94], rgb2=[0.60, 0.65, 0.72], width=256, height=256)

  # Re-texture the terrain plane if it is the default procedural checker.
  if FLOOR_TEXTURE.exists():
    for tex in spec.textures:
      if tex.name == "groundplane":
        tex.builtin = mujoco.mjtBuiltin.mjBUILTIN_NONE
        tex.type = mujoco.mjtTexture.mjTEXTURE_2D
        tex.file = str(FLOOR_TEXTURE)
        tex.width = 0
        tex.height = 0
    for mat in spec.materials:
      if mat.name == "groundplane":
        mat.texrepeat = [1.0, 1.0]
        mat.texuniform = True
        mat.reflectance = 0.08
        mat.specular = 0.15
        mat.shininess = 0.2
        mat.rgba = [1, 1, 1, 1]


def apply_studio(cfg) -> None:
  """Attach the studio hook to an env cfg (keeps any pre-existing spec_fn)."""
  prev = cfg.scene.spec_fn
  if prev is None or prev is studio_spec_fn:
    cfg.scene.spec_fn = studio_spec_fn
    return

  def _chain(spec: mujoco.MjSpec, _prev=prev) -> None:
    _prev(spec)
    studio_spec_fn(spec)

  cfg.scene.spec_fn = _chain
