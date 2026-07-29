"""Puck constants and configuration.

Part of the Class A motion-profile expansion (see CLASS_A_EXPANSION.md). Follows the
free-object convention (freejoint + object_site) so the shared manipulation MDP terms
work unmodified.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

PUCK_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "puck" / "xmls" / "puck.xml"
)
assert PUCK_XML.exists(), f"XML not found: {PUCK_XML}"


##
# Spec functions.
##

def get_puck_spec() -> mujoco.MjSpec:
    """Load Puck MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(PUCK_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Create the orange mocap goal marker (visual only, no collision)."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.035, 0.035, 0.012],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Entity configs.
##

def get_puck_cfg() -> EntityCfg:
    """Get a fresh puck configuration instance."""
    return EntityCfg(
        spec_fn=get_puck_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
