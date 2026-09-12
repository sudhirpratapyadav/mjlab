"""Ellipsoid object constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

ELLIPSOID_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "ellipsoid" / "xmls" / "ellipsoid.xml"
)
assert ELLIPSOID_XML.exists(), f"XML not found: {ELLIPSOID_XML}"


def get_ellipsoid_spec() -> mujoco.MjSpec:
    """Load Ellipsoid MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(ELLIPSOID_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_ellipsoid_spec())


def get_ellipsoid_cfg() -> EntityCfg:
    """Get a fresh ellipsoid configuration instance."""
    return EntityCfg(spec_fn=get_ellipsoid_spec)


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(spec_fn=get_mocap_goal_spec)
