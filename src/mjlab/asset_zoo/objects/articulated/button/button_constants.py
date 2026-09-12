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
    from mjlab.asset_zoo.objects.collision import add_collision_shell

    base = spec.body("button_base")
    # Housing and collar used to be visual-only. Leave a real bore for the rod.
    for name, lo, hi, inner, outer in (
        ("housing", -0.01, 0.031, 0.0115, 0.025),
        ("housing_flange", -0.01, -0.002, 0.0115, 0.0306),
        ("collar", 0.031, 0.036, 0.0165, 0.0303),
    ):
        add_collision_shell(spec, base, name, axis=2, lo=lo, hi=hi,
                            inner=inner, outer=outer)
    # Split the back plate around the shaft instead of letting the rod pass
    # through a solid plate at full depression.
    plate = spec.geom("button_body")
    plate.pos = [0.05575, 0, -0.02]
    plate.size = [0.04425, 0.1, 0.01]
    for name, pos, size in (
        ("panel_left", [-0.05575, 0, -0.02], [0.04425, 0.1, 0.01]),
        ("panel_front", [0, -0.05575, -0.02], [0.0115, 0.04425, 0.01]),
        ("panel_back", [0, 0.05575, -0.02], [0.0115, 0.04425, 0.01]),
    ):
        base.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=pos,
                      size=size, group=3, contype=2, conaffinity=1, mass=0)
    moving = spec.body("handle")
    for name, pos, size in (
        ("rod", [0, 0, 0.061], [0.0105, 0.031, 0]),
        ("rod_stop", [0, 0, 0.0865], [0.0152, 0.0035, 0]),
    ):
        moving.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                        pos=pos, size=size, group=3, contype=2, conaffinity=1, mass=0)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_button_spec())


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
