"""Topple-block constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T20). Follows the
free-object convention (freejoint + object_site) so the shared manipulation MDP terms
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

BLOCK_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "block" / "xmls" / "block.xml"
)
assert BLOCK_XML.exists(), f"XML not found: {BLOCK_XML}"

BLOCK_ASSETS_DIR: Path = BLOCK_XML.parent / "assets"

# Collision-box half-extents of the packaged mesh (assets/block_package.json:
# extent_m = [0.10, 0.16, 0.21], body frame at the bbox centre). The x half-extent is
# the tipping lever arm and the z half-extent is both the standing spawn height and
# the topple travel; env_cfgs and the teacher read them from here.
BLOCK_HALF_EXTENTS: tuple[float, float, float] = (0.0500, 0.0800, 0.1050)
BLOCK_HALF_HEIGHT: float = BLOCK_HALF_EXTENTS[2]
"""Height of the standing block's centre above the ground plane."""


##
# Spec functions.
##

def get_block_spec() -> mujoco.MjSpec:
    """Load Block MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(BLOCK_XML))
    # Franka pattern (franka_constants.py:21-37).
    assets: dict = {}
    update_assets(assets, BLOCK_ASSETS_DIR, spec.meshdir)
    spec.assets = assets
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_block_spec())


##
# Entity configs.
##

def get_block_cfg() -> EntityCfg:
    """Get a fresh block configuration instance."""
    return EntityCfg(
        spec_fn=get_block_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
