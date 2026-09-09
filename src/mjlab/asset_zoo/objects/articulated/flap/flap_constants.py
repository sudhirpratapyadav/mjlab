"""Flap constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T21). Follows the
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

FLAP_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "flap" / "xmls" / "flap.xml"
)
assert FLAP_XML.exists(), f"XML not found: {FLAP_XML}"


##
# Spec functions.
##

def get_flap_assets(meshdir: str) -> dict[str, bytes]:
    """Load the Blender-built visual meshes and their textures (CL-V2 W2-a).

    Mirrors ``franka_constants.get_assets``: ``meshdir`` and ``texturedir`` are both
    "assets", so one sweep of that directory covers the OBJ meshes, their .mtl files
    and the PNG albedos. Required for ``Entity.to_zip`` / ``Scene.to_zip`` and for any
    path that ships the model away from this source tree.
    """
    assets: dict[str, bytes] = {}
    update_assets(assets, FLAP_XML.parent / "assets", meshdir)
    return assets


def get_flap_spec() -> mujoco.MjSpec:
    """Load Flap MjSpec from XML, with its mesh/texture assets."""
    spec = mujoco.MjSpec.from_file(str(FLAP_XML))
    spec.assets = get_flap_assets(spec.meshdir)
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
        size=[0.012, 0.02, 0.05],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Initial state.
##

FLAP_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"flap_hinge": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_flap_cfg() -> EntityCfg:
    """Get a fresh flap configuration instance."""
    return EntityCfg(
        spec_fn=get_flap_spec,
        init_state=FLAP_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
