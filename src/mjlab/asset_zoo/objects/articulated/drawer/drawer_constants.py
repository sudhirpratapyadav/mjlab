"""Drawer constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

DRAWER_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "drawer" / "xmls" / "drawer.xml"
)
assert DRAWER_XML.exists(), f"XML not found: {DRAWER_XML}"

MOCAP_TARGET_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "articulated" / "drawer" / "xmls" / "mocap_target.xml"
)


##
# Spec functions.
##

def get_drawer_assets(meshdir: str) -> dict[str, bytes]:
    """Load the mesh/texture assets next to the XML (Franka pattern, CODE_MAP §1)."""
    assets: dict[str, bytes] = {}
    update_assets(assets, DRAWER_XML.parent / "assets", meshdir)
    return assets


def get_drawer_spec() -> mujoco.MjSpec:
    """Load Drawer MjSpec from XML, with its meshes and textures."""
    spec = mujoco.MjSpec.from_file(str(DRAWER_XML))
    spec.assets = get_drawer_assets(spec.meshdir)
    return spec


def get_mocap_target_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_drawer_spec())


##
# Initial state.
##

# Drawer starts at 0.0 (closed position)
DRAWER_INIT_STATE = EntityCfg.InitialStateCfg(
    joint_pos={"drawer_slide": 0.0},  # Closed
    joint_vel={".*": 0.0},
)


##
# Entity configs.
##

def get_drawer_cfg() -> EntityCfg:
    """Get a fresh drawer configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues.
    """
    return EntityCfg(
        spec_fn=get_drawer_spec,
        init_state=DRAWER_INIT_STATE,
    )


def get_mocap_target_cfg() -> EntityCfg:
    """Get a fresh mocap target configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_target_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    # Test drawer entity
    drawer = Entity(get_drawer_cfg())
    print(f"Drawer entity created with {len(drawer.joint_names)} joints: {drawer.joint_names}")

    viewer.launch(drawer.spec.compile())
