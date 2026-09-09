"""Ledge (static raised platform) constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T23). Static mocap
receptacle like the container: EdgeGraspCommand writes its pose per-env every
resample.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

LEDGE_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "ledge" / "xmls" / "ledge.xml"
)
assert LEDGE_XML.exists(), f"XML not found: {LEDGE_XML}"

LEDGE_TOP_HEIGHT: float = 0.10
"""Height of the top surface above the ledge body origin. EdgeGraspCommandCfg's
``ledge_top_height`` must match the geometry here."""


##
# Spec functions.
##

def get_ledge_spec() -> mujoco.MjSpec:
    """Load Ledge MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(LEDGE_XML))
    assets: dict = {}
    update_assets(assets, LEDGE_XML.parent / "assets", "assets")
    spec.assets = assets
    return spec


##
# Entity configs.
##

def get_ledge_cfg() -> EntityCfg:
    """Get a fresh ledge configuration instance."""
    return EntityCfg(
        spec_fn=get_ledge_spec,
    )
