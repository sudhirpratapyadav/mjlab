"""LEAP hand constants and configuration (floating dexterous hand, Class C).

The LEAP hand is a 16-DoF anthropomorphic hand (index/middle/ring fingers + thumb,
4 joints each). Here it is set up as a FLOATING hand (a freejoint on the palm, no arm)
for the benchmark's Class-C embodiment. Only the 16 finger joints are actuated; the
palm freejoint is an unactuated floating base (in-hand tasks move the object, not the
hand). Full joint-target action space (16-D), per the benchmark's dexterous-hand
convention.

Model from MuJoCo Menagerie (leap_hand), Apache-2.0; see LICENSE in this directory.
"""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

LEAP_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "leap_hand" / "xmls" / "leap_right_hand.xml"
)
assert LEAP_XML.exists(), f"XML not found: {LEAP_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load LEAP hand mesh assets."""
    assets: dict[str, bytes] = {}
    update_assets(assets, LEAP_XML.parent / "assets", meshdir)
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load the floating LEAP hand MjSpec with assets."""
    spec = mujoco.MjSpec.from_file(str(LEAP_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Joint names.
##

# 16 actuated finger joints (index/middle/ring x mcp/rot/pip/dip + thumb x 4).
FINGER_JOINTS = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

##
# Initial state.
##

# Palm floating above the table, fingers slightly curled (a neutral pre-grasp).
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "palm_freejoint": 0.0,  # freejoint qpos handled via root pose; fingers below
        **{j: 0.2 for j in FINGER_JOINTS},
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##

# Only the finger joints are actuated (XML defines position actuators <joint>_act).
LEAP_ACTUATORS = XmlPositionActuatorCfg(
    joint_names_expr=tuple(FINGER_JOINTS),
)

LEAP_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(LEAP_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_leap_hand_cfg() -> EntityCfg:
    """Get a fresh floating LEAP hand configuration instance."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # Use collisions from XML.
        spec_fn=get_spec,
        articulation=LEAP_ARTICULATION,
    )


# Action scale for finger position deltas.
LEAP_ACTION_SCALE = 0.1
