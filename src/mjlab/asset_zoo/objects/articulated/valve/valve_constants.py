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
    from mjlab.asset_zoo.objects.collision import add_collision_shell

    base = spec.body("valve_base")
    # The old bonnet had no collision at all and overlapped the wheel spokes.
    # Keep its shaft bore open and put its front behind the rotating spokes.
    spec.geom("vis_bonnet").pos = [0.014, 0, 0]
    for name, lo, hi, radius in (
        ("bonnet_flange", 0.020, 0.034, 0.056),
        ("bonnet_neck", -0.006, 0.020, 0.0412),
        ("bonnet_gland", -0.022, -0.006, 0.027),
    ):
        add_collision_shell(spec, base, name, axis=0, lo=lo, hi=hi,
                            inner=0.021, outer=radius)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_valve_spec())


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
