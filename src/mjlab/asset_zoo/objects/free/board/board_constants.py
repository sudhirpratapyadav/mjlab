"""Board constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T24). Follows the
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

BOARD_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "board" / "xmls" / "board.xml"
)
assert BOARD_XML.exists(), f"XML not found: {BOARD_XML}"


##
# Spec functions.
##

def get_board_spec() -> mujoco.MjSpec:
    """Load Board MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(BOARD_XML))
    assets: dict = {}
    update_assets(assets, BOARD_XML.parent / "assets", "assets")
    spec.assets = assets
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_board_spec())


##
# Entity configs.
##

def get_board_cfg() -> EntityCfg:
    """Get a fresh board configuration instance."""
    return EntityCfg(
        spec_fn=get_board_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
