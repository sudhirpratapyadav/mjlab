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
    """Create mocap goal (orange box matching cuboid) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=list(CUBOID_HALF_EXTENTS),  # Matches the cuboid collision box
        rgba=[1, 0.5, 0, 0.1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


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
