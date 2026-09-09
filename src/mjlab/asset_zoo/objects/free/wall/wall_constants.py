"""Wall (static pivot fixture) constants and configuration.

Part of the Class A Wave-1 expansion (see CATALOG_100_TASKS.md, T24). Static mocap
fixture like the container/ledge: PivotLiftCommand writes its pose per-env every
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

WALL_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "wall" / "xmls" / "wall.xml"
)
assert WALL_XML.exists(), f"XML not found: {WALL_XML}"

WALL_CENTER_Z: float = 0.075
"""Body-origin height that puts the wall flush on the ground plane (half its
height). PivotLiftCommandCfg's wall_spawn_range z must match."""


##
# Spec functions.
##

def get_wall_spec() -> mujoco.MjSpec:
    """Load Wall MjSpec from XML."""
    spec = mujoco.MjSpec.from_file(str(WALL_XML))
    assets: dict = {}
    update_assets(assets, WALL_XML.parent / "assets", "assets")
    spec.assets = assets
    return spec


##
# Entity configs.
##

def get_wall_cfg() -> EntityCfg:
    """Get a fresh wall configuration instance."""
    return EntityCfg(
        spec_fn=get_wall_spec,
    )
