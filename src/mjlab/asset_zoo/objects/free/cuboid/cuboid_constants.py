"""Cuboid constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

CUBOID_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "cuboid" / "xmls" / "cuboid.xml"
)
assert CUBOID_XML.exists(), f"XML not found: {CUBOID_XML}"

CUBOID_ASSETS_DIR: Path = CUBOID_XML.parent / "assets"

# Collision-box half-extents of the packaged mesh (assets/cuboid_package.json:
# extent_m = [0.0729, 0.0892, 0.0301], body frame at the bbox centre).
CUBOID_HALF_EXTENTS: tuple[float, float, float] = (0.0365, 0.0446, 0.0150)
CUBOID_HALF_HEIGHT: float = CUBOID_HALF_EXTENTS[2]
"""Resting height of the cuboid centre above the ground plane (unchanged from the
primitive it replaces, which is why no spawn-z or goal-z constant moved)."""


##
# Spec functions.
##

def get_cuboid_spec() -> mujoco.MjSpec:
    """Load Cuboid MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(CUBOID_XML))
    # Franka pattern (franka_constants.py:21-37).
    assets: dict = {}
    update_assets(assets, CUBOID_ASSETS_DIR, spec.meshdir)
    spec.assets = assets
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_cuboid_spec())


##
# Entity configs.
##

def get_cuboid_cfg() -> EntityCfg:
    """Get a fresh cuboid configuration instance."""
    return EntityCfg(
        spec_fn=get_cuboid_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
