"""Disc constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

##
# MJCF paths.
##

DISC_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "disc" / "xmls" / "disc.xml"
)
assert DISC_XML.exists(), f"XML not found: {DISC_XML}"


##
# Spec functions.
##

def get_disc_spec() -> mujoco.MjSpec:
    """Load Disc MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(DISC_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_disc_spec())


##
# Entity configs.
##

def get_disc_cfg() -> EntityCfg:
    """Get a fresh disc configuration instance."""
    return EntityCfg(
        spec_fn=get_disc_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
