"""Franka arm + LEAP hand constants and configuration (Class B: arm + dexterous hand).

Combines the 7-DoF Franka Panda arm with the 16-DoF LEAP hand (the 2-finger gripper is
removed and the fixed-base LEAP hand is attached at link7). Result: a 23-DoF arm+hand
(7 arm + 16 finger), full joint-target action (23-D). Reuses the same position-based
task bases as the other embodiments (uniform interfaces).

Built by composing the two Menagerie/asset_zoo MJCF models via MjSpec.attach.
"""

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import XmlPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

_FRANKA_XML = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "franka_emika_panda" / "xmls" / "panda.xml"
)
_HAND_XML = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "leap_hand" / "xmls" / "leap_right_hand_fixed.xml"
)
assert _FRANKA_XML.exists() and _HAND_XML.exists()

# The LEAP hand joints get this prefix when attached (MjSpec.attach namespacing).
_HAND_PREFIX = "leap_"

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
FINGER_JOINTS = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]
HAND_JOINTS = [_HAND_PREFIX + j for j in FINGER_JOINTS]


def get_spec() -> mujoco.MjSpec:
    """Compose the Franka arm + LEAP hand into one MjSpec.

    Loads the Franka arm (with its meshes), removes the 2-finger gripper body, and
    attaches the fixed-base LEAP hand at link7 (same mount pose as the Franka hand).
    """
    arm = mujoco.MjSpec.from_file(str(_FRANKA_XML))
    # Franka meshes.
    arm_assets: dict[str, bytes] = {}
    update_assets(arm_assets, _FRANKA_XML.parent / "assets", arm.meshdir)
    hand_xml_path = _FRANKA_XML.parent / "hand.xml"
    if hand_xml_path.exists():
        arm_assets["hand.xml"] = hand_xml_path.read_bytes()
    arm.assets = arm_assets

    hand = mujoco.MjSpec.from_file(str(_HAND_XML))
    hand_assets: dict[str, bytes] = {}
    update_assets(hand_assets, _HAND_XML.parent / "assets", hand.meshdir)
    # Disable hand mesh collision (warp narrowphase segfaults on it, some GPUs);
    # per-phalanx box colliders preserve contact. See leap_constants.get_spec.
    for geom in hand.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype = 0
            geom.conaffinity = 0

    # Remove the Franka 2-finger gripper (the "hand" body and its finger children).
    arm.delete(arm.body("hand"))
    # Attach the LEAP hand palm at link7, matching the original gripper mount pose.
    link7 = arm.body("link7")
    frame = link7.add_frame(
        pos=[0, 0, 0.107], quat=[0.9238795, 0, 0, -0.3826834]
    )
    frame.attach_body(hand.body("palm"), _HAND_PREFIX, "")
    for k, v in hand_assets.items():
        arm.assets[k] = v
    return arm


##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint1": 0.0, "joint2": 0.3, "joint3": 0.0, "joint4": -1.57079,
        "joint5": 0.0, "joint6": 2.0, "joint7": -0.7853,
        **{j: 0.2 for j in HAND_JOINTS},
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##

# All actuators are XML position actuators (7 arm + 16 hand); match everything.
FRANKA_LEAP_ACTUATORS = XmlPositionActuatorCfg(joint_names_expr=(".*",))

FRANKA_LEAP_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(FRANKA_LEAP_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_franka_leap_robot_cfg() -> EntityCfg:
    """Get a fresh Franka+LEAP (arm + dexterous hand) robot configuration."""
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),
        spec_fn=get_spec,
        articulation=FRANKA_LEAP_ARTICULATION,
    )


FRANKA_LEAP_ACTION_SCALE = 0.04
