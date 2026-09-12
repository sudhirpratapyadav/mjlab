"""Door constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

DOOR_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "door" / "xmls" / "door.xml"
)
assert DOOR_XML.exists(), f"XML not found: {DOOR_XML}"

MOCAP_TARGET_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "door" / "xmls" / "mocap_target.xml"
)


##
# Spec functions.
##

def get_door_assets(meshdir: str) -> dict[str, bytes]:
    """Load the mesh/texture assets next to the XML (Franka pattern, CODE_MAP §1)."""
    assets: dict[str, bytes] = {}
    update_assets(assets, DOOR_XML.parent / "assets", meshdir)
    return assets


def get_door_spec() -> mujoco.MjSpec:
    """Load Door MjSpec from XML, with its meshes and textures."""
    spec = mujoco.MjSpec.from_file(str(DOOR_XML))
    spec.assets = get_door_assets(spec.meshdir)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_door_spec())


##
# Initial state.
##

# Door starts at 45 degrees (0.785398 rad) to prevent drift to 0
DOOR_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"door_hinge": 0.785398},  # 45 degrees
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_door_cfg() -> EntityCfg:
    """Get a fresh door configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_door_spec,
        init_state=DOOR_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Test door entity
    door = Entity(get_door_cfg())
    print(f"Door entity created with {len(door.joint_names)} joints: {door.joint_names}")

    viewer.launch(door.spec.compile())
