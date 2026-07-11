"""Peg and hole-board constants and configuration (insertion task assets)."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityCfg

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

def get_peg_spec() -> mujoco.MjSpec:
    """Load the peg MjSpec from XML (a slender graspable square peg)."""
    return mujoco.MjSpec.from_file(str(PEG_XML))


def get_hole_board_spec() -> mujoco.MjSpec:
    """Load the hole-board MjSpec from XML (a four-wall frame with a square hole)."""
    return mujoco.MjSpec.from_file(str(HOLE_BOARD_XML))


##
# Entity configs.
##

def get_peg_cfg() -> EntityCfg:
    """Get a fresh peg configuration instance."""
    return EntityCfg(spec_fn=get_peg_spec)


def get_hole_board_cfg() -> EntityCfg:
    """Get a fresh hole-board configuration instance."""
    return EntityCfg(spec_fn=get_hole_board_spec)
