"""Plug (socket + friction-fit plug) constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T22). Follows the
articulated-object convention (object_site / base_site / handle body naming) so the
shared manipulation MDP terms work unmodified.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

PLUG_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "plug" / "xmls" / "plug.xml"
)
assert PLUG_XML.exists(), f"XML not found: {PLUG_XML}"


##
# Spec functions.
##

def get_plug_spec() -> mujoco.MjSpec:
    """Load Plug MjSpec from XML (textured socket box + Schuko plug moulding)."""
    spec = mujoco.MjSpec.from_file(str(PLUG_XML))
    assets: dict = {}
    update_assets(assets, PLUG_XML.parent / "assets", "assets")
    spec.assets = assets
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Create the orange mocap goal marker (visual only, no collision)."""
    spec = mujoco.MjSpec()
    mocap_target = spec.worldbody.add_body(name="mocap_target")
    mocap_target.mocap = True
    mocap_target.pos = [0, 0, 0]
    mocap_target.add_geom(
        name="mocap_target_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.012],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Initial state.
##

PLUG_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"plug_slide": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_plug_cfg() -> EntityCfg:
    """Get a fresh plug configuration instance."""
    return EntityCfg(
        spec_fn=get_plug_spec,
        init_state=PLUG_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
