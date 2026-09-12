"""Puck constants and configuration.

Part of the Class A motion-profile expansion (see CLASS_A_EXPANSION.md). Follows the
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

PUCK_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "puck" / "xmls" / "puck.xml"
)
assert PUCK_XML.exists(), f"XML not found: {PUCK_XML}"

PUCK_ASSETS_DIR: Path = PUCK_XML.parent / "assets"


def get_assets() -> dict:
  """Mesh + texture blobs keyed as the MJCF's ``meshdir``/``texturedir`` expect."""
  assets: dict = {}
  update_assets(assets, PUCK_ASSETS_DIR, "assets")
  return assets


##
# Spec functions.
##

def get_puck_spec() -> mujoco.MjSpec:
    """Load Puck MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(PUCK_XML))
    spec.assets = get_assets()
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_puck_spec())


##
# Entity configs.
##

def get_puck_cfg() -> EntityCfg:
    """Get a fresh puck configuration instance."""
    return EntityCfg(
        spec_fn=get_puck_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
