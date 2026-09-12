"""Lid constants and configuration.

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

LID_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "lid" / "xmls" / "lid.xml"
)
assert LID_XML.exists(), f"XML not found: {LID_XML}"


##
# Spec functions.
##

def get_lid_assets(meshdir: str) -> dict[str, bytes]:
    """Load the mesh/texture assets next to the XML (Franka pattern, CODE_MAP §1)."""
    assets: dict[str, bytes] = {}
    update_assets(assets, LID_XML.parent / "assets", meshdir)
    return assets


def get_lid_spec() -> mujoco.MjSpec:
    """Load Lid MjSpec from XML, with its meshes and textures."""
    spec = mujoco.MjSpec.from_file(str(LID_XML))
    spec.assets = get_lid_assets(spec.meshdir)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_lid_spec())


##
# Initial state.
##

LID_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"lid_hinge": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_lid_cfg() -> EntityCfg:
    """Get a fresh lid configuration instance."""
    return EntityCfg(
        spec_fn=get_lid_spec,
        init_state=LID_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
