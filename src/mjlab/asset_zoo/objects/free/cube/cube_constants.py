"""Cube constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

CUBE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "cube" / "xmls" / "cube.xml"
)
assert CUBE_XML.exists(), f"XML not found: {CUBE_XML}"

CUBE_ASSETS_DIR: Path = CUBE_XML.parent / "assets"

# Collision-box half-extents of the packaged mesh (assets/cube_package.json:
# extent_m = [0.046, 0.0453, 0.0451], body frame at the bbox centre). Every
# geometry constant that used to read 0.02 off the primitive box comes from here.
CUBE_HALF_EXTENTS: tuple[float, float, float] = (0.0230, 0.0226, 0.0226)
CUBE_HALF_HEIGHT: float = CUBE_HALF_EXTENTS[2]
"""Resting height of the cube centre above the ground plane."""
CUBE_HALF_WIDTH: float = CUBE_HALF_EXTENTS[0]
"""Widest horizontal half-extent: what a pinch/cage/push must clear."""


##
# Spec functions.
##

def get_cube_spec() -> mujoco.MjSpec:
    """Load Cube MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(CUBE_XML))
    # Franka pattern (franka_constants.py:21-37): ship the meshes/textures in the
    # spec so Scene.to_zip / Entity.to_zip work away from the source tree.
    assets: dict = {}
    update_assets(assets, CUBE_ASSETS_DIR, spec.meshdir)
    spec.assets = assets
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_cube_spec())


##
# Entity configs.
##

def get_cube_cfg() -> EntityCfg:
    """Get a fresh cube configuration instance."""
    return EntityCfg(
        spec_fn=get_cube_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
