"""Stick constants and configuration.

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

STICK_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "stick" / "xmls" / "stick.xml"
)
assert STICK_XML.exists(), f"XML not found: {STICK_XML}"

STICK_ASSETS_DIR: Path = STICK_XML.parent / "assets"


def get_assets() -> dict:
  """Mesh + texture blobs keyed as the MJCF's ``meshdir``/``texturedir`` expect."""
  assets: dict = {}
  update_assets(assets, STICK_ASSETS_DIR, "assets")
  return assets


##
# Spec functions.
##

def get_stick_spec() -> mujoco.MjSpec:
    """Load Stick MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(STICK_XML))
    spec.assets = get_assets()
    return spec


def get_mocap_goal_spec() -> mujoco.MjSpec:
    """Translucent replica of the manipulated part at its target pose."""
    from mjlab.asset_zoo.objects.goal import make_goal_spec

    return make_goal_spec(get_stick_spec())


##
# Entity configs.
##

def get_stick_cfg() -> EntityCfg:
    """Get a fresh stick configuration instance."""
    return EntityCfg(
        spec_fn=get_stick_spec,
    )


def get_mocap_goal_cfg() -> EntityCfg:
    """Get a fresh mocap goal configuration instance."""
    return EntityCfg(
        spec_fn=get_mocap_goal_spec,
    )
