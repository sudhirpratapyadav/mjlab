"""Cube constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

CUBE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "cube" / "xmls" / "cube.xml"
)
assert CUBE_XML.exists(), f"XML not found: {CUBE_XML}"


##
# Spec functions.
##

def get_cube_spec() -> mujoco.MjSpec:
    """Load Cube MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(CUBE_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create mocap goal (orange box matching cube) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],  # Matches cube size
        rgba=[1, 0.5, 0, 1],  # Orange (same as other mocap goals)
        contype=0,
        conaffinity=0,
    )
    return spec


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
