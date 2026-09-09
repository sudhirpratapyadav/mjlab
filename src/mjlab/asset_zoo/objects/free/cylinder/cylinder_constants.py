"""Cylinder constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

CYLINDER_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "cylinder" / "xmls" / "cylinder.xml"
)
assert CYLINDER_XML.exists(), f"XML not found: {CYLINDER_XML}"

CYLINDER_ASSETS_DIR: Path = CYLINDER_XML.parent / "assets"


def get_assets() -> dict:
  """Mesh + texture blobs keyed as the MJCF's ``meshdir``/``texturedir`` expect."""
  assets: dict = {}
  update_assets(assets, CYLINDER_ASSETS_DIR, "assets")
  return assets


##
# Spec functions.
##

def get_cylinder_spec() -> mujoco.MjSpec:
    """Load Cylinder MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(CYLINDER_XML))
    spec.assets = get_assets()
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create mocap goal (orange cylinder matching object) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        # Matches the bottle's collision extent (radius 0.0150, half-length 0.0266).
        size=[0.015, 0.0266, 0.0],
        rgba=[1, 0.5, 0, 1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_cylinder_cfg() -> EntityCfg:
    """Get a fresh cylinder configuration instance."""
    return EntityCfg(
        spec_fn=get_cylinder_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
