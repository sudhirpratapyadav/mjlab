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
    """Create mocap goal (orange sphere) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.022, 0, 0],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


def get_sphere_cfg() -> EntityCfg:
    """Get a fresh sphere configuration instance."""
    return EntityCfg(spec_fn=get_sphere_spec)


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(spec_fn=get_mocap_goal_spec)
