"""Window constants and configuration.

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

WINDOW_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "window" / "xmls" / "window.xml"
)
assert WINDOW_XML.exists(), f"XML not found: {WINDOW_XML}"


##
# Spec functions.
##

def get_window_assets(meshdir: str) -> dict[str, bytes]:
    """Load the mesh/texture assets next to the XML (Franka pattern, CODE_MAP §1)."""
    assets: dict[str, bytes] = {}
    update_assets(assets, WINDOW_XML.parent / "assets", meshdir)
    return assets


def get_window_spec() -> mujoco.MjSpec:
    """Load Window MjSpec from XML, with its meshes and textures."""
    spec = mujoco.MjSpec.from_file(str(WINDOW_XML))
    spec.assets = get_window_assets(spec.meshdir)
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
        size=[0.012, 0.012, 0.07],
        rgba=[1, 0.5, 0, 1],
        contype=0,
        conaffinity=0,
    )
    return spec


##
# Initial state.
##

WINDOW_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"window_slide": 0.0},
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_window_cfg() -> EntityCfg:
    """Get a fresh window configuration instance."""
    return EntityCfg(
        spec_fn=get_window_spec,
        init_state=WINDOW_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )
