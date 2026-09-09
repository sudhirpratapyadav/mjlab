"""Peg and hole-board constants and configuration (insertion task assets)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF paths.
##

_BASE = MJLAB_SRC_PATH / "asset_zoo" / "objects" / "free" / "peg_in_hole" / "xmls"
PEG_XML: Path = _BASE / "peg.xml"
HOLE_BOARD_XML: Path = _BASE / "hole_board.xml"
assert PEG_XML.exists(), f"XML not found: {PEG_XML}"
assert HOLE_BOARD_XML.exists(), f"XML not found: {HOLE_BOARD_XML}"


##
# Spec functions.
##

def _assets() -> dict:
    """Mesh/texture bytes keyed the way MuJoCo resolves them (the Franka pattern)."""
    assets: dict = {}
    update_assets(assets, _BASE / "assets", "assets")
    return assets


def get_peg_spec() -> mujoco.MjSpec:
    """Load the peg MjSpec from XML (a 25 mm square shape-sorter peg, chamfered)."""
    spec = mujoco.MjSpec.from_file(str(PEG_XML))
    spec.assets = _assets()
    return spec


def get_hole_board_spec() -> mujoco.MjSpec:
    """Load the hole-board MjSpec from XML (chamfered 30 mm square through-hole)."""
    spec = mujoco.MjSpec.from_file(str(HOLE_BOARD_XML))
    spec.assets = _assets()
    return spec


##
# Entity configs.
##

def get_peg_cfg() -> EntityCfg:
    """Get a fresh peg configuration instance."""
    return EntityCfg(spec_fn=get_peg_spec)


def get_hole_board_cfg() -> EntityCfg:
    """Get a fresh hole-board configuration instance."""
    return EntityCfg(spec_fn=get_hole_board_spec)
