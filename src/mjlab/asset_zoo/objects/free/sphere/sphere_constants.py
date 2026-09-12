"""Sphere object constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

SPHERE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "sphere" / "xmls" / "sphere.xml"
)
assert SPHERE_XML.exists(), f"XML not found: {SPHERE_XML}"


def get_sphere_spec() -> mujoco.MjSpec:
    """Load Sphere MjSpec from XML."""
    return mujoco.MjSpec.from_file(str(SPHERE_XML))


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_sphere_spec())


def get_sphere_cfg() -> EntityCfg:
    """Get a fresh sphere configuration instance."""
    return EntityCfg(spec_fn=get_sphere_spec)


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(spec_fn=get_mocap_goal_spec)
