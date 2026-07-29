"""LEAP hand constants and configuration (floating dexterous hand, Class C).

The LEAP hand is a 16-DoF anthropomorphic hand (index/middle/ring fingers + thumb,
4 joints each). Here it is set up as a FLOATING hand with an ACTUATED 6-DoF base (3
slide + 3 hinge joints on the palm under position control), so the hand is a general
manipulator: it can translate/rotate to reach, pick, and place, and Class-C tasks reuse
the same position-based MDP as the arm tasks (uniform interfaces). Full joint-target
action space: 22-D = 6 base + 16 finger.

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
    """Load the floating LEAP hand MjSpec with assets.

    Mesh collision geoms are disabled (contype/conaffinity = 0): mujoco-warp's
    narrowphase segfaults on this hand's mesh-mesh collision on some GPUs (A100).
    Contact is preserved via the per-phalanx primitive (box) colliders, which remain
    active. Fingertip-mesh collision fidelity is a stage-two TODO (replace with
    primitive fingertip colliders).
    """
    spec = mujoco.MjSpec.from_file(str(LEAP_XML))
    spec.assets = get_assets(spec.meshdir)
    for geom in spec.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0
    return spec


##
# Joint names.
##

# 6-DoF actuated base (3 translate + 3 rotate).
BASE_JOINTS = ["base_tx", "base_ty", "base_tz", "base_rx", "base_ry", "base_rz"]

# 16 actuated finger joints (index/middle/ring x mcp/rot/pip/dip + thumb x 4).
FINGER_JOINTS = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

ALL_JOINTS = BASE_JOINTS + FINGER_JOINTS

##
# Initial state.
##

# Palm positioned above the table (base translate joints), fingers slightly curled.
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "base_tx": 0.0, "base_ty": 0.0, "base_tz": 0.0,
        "base_rx": 0.0, "base_ry": 0.0, "base_rz": 0.0,
        **{j: 0.2 for j in FINGER_JOINTS},
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##

# All 22 joints are actuated (XML defines position actuators <joint>_act).
LEAP_ACTUATORS = XmlPositionActuatorCfg(
    joint_names_expr=(".*",),
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
