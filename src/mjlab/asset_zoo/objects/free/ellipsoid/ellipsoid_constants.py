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
    """Create mocap goal (orange ellipsoid) for visualization."""
    spec = mujoco.MjSpec()
    mocap_goal = spec.worldbody.add_body(name="mocap_goal")
    mocap_goal.mocap = True
    mocap_goal.pos = [0, 0, 0]
    mocap_goal.add_geom(
        name="mocap_goal_geom",
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[0.035, 0.018, 0.018],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


def get_ellipsoid_cfg() -> EntityCfg:
    """Get a fresh ellipsoid configuration instance."""
    return EntityCfg(spec_fn=get_ellipsoid_spec)


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(spec_fn=get_mocap_goal_spec)
