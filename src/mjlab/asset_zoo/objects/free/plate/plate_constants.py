"""Thin-plate constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T23). Follows the
free-object convention (freejoint + object_site) so the shared manipulation MDP terms
work unmodified.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

PLATE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "plate" / "xmls" / "plate.xml"
)
assert PLATE_XML.exists(), f"XML not found: {PLATE_XML}"


##
# Spec functions.
##

def get_plate_spec() -> mujoco.MjSpec:
    """Load Plate MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(PLATE_XML))
    assets: dict = {}
    update_assets(assets, PLATE_XML.parent / "assets", "assets")
    spec.assets = assets
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create the orange mocap goal marker (visual only, no collision)."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[0.075, 0.0096, 0.0],  # 150 mm side plate, 19.2 mm tall
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_plate_cfg() -> EntityCfg:
    """Get a fresh plate configuration instance."""
    return EntityCfg(
        spec_fn=get_plate_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
