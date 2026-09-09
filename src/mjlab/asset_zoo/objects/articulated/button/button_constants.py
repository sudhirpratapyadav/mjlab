"""Button constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

BUTTON_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "button" / "xmls" / "button.xml"
)
assert BUTTON_XML.exists(), f"XML not found: {BUTTON_XML}"

MOCAP_TARGET_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "button" / "xmls" / "mocap_target.xml"
)


##
# Spec functions.
##

def get_button_assets(meshdir: str) -> dict[str, bytes]:
    """Load the Blender-built visual meshes and their textures (CL-V2 W2-a).

    Mirrors ``franka_constants.get_assets``: ``meshdir`` and ``texturedir`` are both
    "assets", so one sweep of that directory covers the OBJ meshes, their .mtl files
    and the PNG albedos. Required for ``Entity.to_zip`` / ``Scene.to_zip`` and for any
    path that ships the model away from this source tree.
    """
    assets: dict[str, bytes] = {}
    update_assets(assets, BUTTON_XML.parent / "assets", meshdir)
    return assets


def get_button_spec() -> mujoco.MjSpec:
    """Load Button MjSpec from XML, with its mesh/texture assets."""
    spec = mujoco.MjSpec.from_file(str(BUTTON_XML))
    spec.assets = get_button_assets(spec.meshdir)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Load Mocap Target MjSpec from XML."""
    if not MOCAP_TARGET_XML.exists():
        # Fallback: create mocap target programmatically
        spec = mujoco.MjSpec()
        mocap_target = spec.worldbody.add_body(name="mocap_target")
        mocap_target.mocap = True
        mocap_target.pos = [0, 0, 0]
        mocap_target.add_geom(
            name="mocap_target_geom",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.0225, 0.009, 0.0],  # Matches the 45 mm mushroom cap collider
            rgba=[1, 0.5, 0, 1],
            contype=0,
            conaffinity=0,
        )
        return spec
    return mujoco.MjSpec.from_file(str(MOCAP_TARGET_XML))


##
# Initial state.
##

# Button starts at 0.0 (fully extended/unpressed)
# Joint range is -0.05 to 0.0, where 0.0 is unpressed and -0.05 is fully pressed
BUTTON_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"button_slide": 0.0},  # Fully extended (unpressed)
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_button_cfg() -> EntityCfg:
    """Get a fresh button configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_button_spec,
        init_state=BUTTON_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Test button entity
    button = Entity(get_button_cfg())
    print(f"Button entity created with {len(button.joint_names)} joints: {button.joint_names}")

    viewer.launch(button.spec.compile())
