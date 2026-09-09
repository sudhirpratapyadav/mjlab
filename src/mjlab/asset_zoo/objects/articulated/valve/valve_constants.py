"""Valve constants and configuration.

Auto-generated from the drawer/button template (see CLASS_A_EXPANSION.md); follows the
same object_site / base_site / handle naming convention so the shared manipulation MDP
terms (object_position, gripper_to_object_vector, staged_manipulation_reward, ...)
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

VALVE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "valve" / "xmls" / "valve.xml"
)
assert VALVE_XML.exists(), f"XML not found: {VALVE_XML}"


##
# Spec functions.
##

def get_valve_assets(meshdir: str) -> dict[str, bytes]:
    """Load the Blender-built visual meshes and their textures (CL-V2 W2-a).

    Mirrors ``franka_constants.get_assets``: ``meshdir`` and ``texturedir`` are both
    "assets", so one sweep of that directory covers the OBJ meshes, their .mtl files
    and the PNG albedos. Required for ``Entity.to_zip`` / ``Scene.to_zip`` and for any
    path that ships the model away from this source tree.
    """
    assets: dict[str, bytes] = {}
    update_assets(assets, VALVE_XML.parent / "assets", meshdir)
    return assets


def get_valve_spec() -> mujoco.MjSpec:
    """Load Valve MjSpec from XML, with its mesh/texture assets."""
    spec = mujoco.MjSpec.from_file(str(VALVE_XML))
    spec.assets = get_valve_assets(spec.meshdir)
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
        size=[0.012, 0.055, 0.012],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Initial state.
##

VALVE_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"valve_hinge": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_valve_cfg() -> EntityCfg:
    """Get a fresh valve configuration instance."""
    return EntityCfg(
        spec_fn=get_valve_spec,
        init_state=VALVE_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
