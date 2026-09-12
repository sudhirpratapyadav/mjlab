"""Lever constants and configuration.

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

LEVER_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "lever" / "xmls" / "lever.xml"
)
assert LEVER_XML.exists(), f"XML not found: {LEVER_XML}"


##
# Spec functions.
##

def get_lever_assets(meshdir: str) -> dict[str, bytes]:
    """Load the Blender-built visual meshes and their textures (CL-V2 W2-a).

    Mirrors ``franka_constants.get_assets``: ``meshdir`` and ``texturedir`` are both
    "assets", so one sweep of that directory covers the OBJ meshes, their .mtl files
    and the PNG albedos. Required for ``Entity.to_zip`` / ``Scene.to_zip`` and for any
    path that ships the model away from this source tree.
    """
    assets: dict[str, bytes] = {}
    update_assets(assets, LEVER_XML.parent / "assets", meshdir)
    return assets


def get_lever_spec() -> mujoco.MjSpec:
    """Load Lever MjSpec from XML, with its mesh/texture assets."""
    spec = mujoco.MjSpec.from_file(str(LEVER_XML))
    spec.assets = get_lever_assets(spec.meshdir)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_lever_spec())


##
# Initial state.
##

LEVER_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"lever_hinge": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_lever_cfg() -> EntityCfg:
    """Get a fresh lever configuration instance."""
    return EntityCfg(
        spec_fn=get_lever_spec,
        init_state=LEVER_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
