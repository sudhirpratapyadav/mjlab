"""Franka Emika Panda constants and configuration."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

FRANKA_XML: Path = (
    MJLAB_SRC_PATH / "asset_zoo" / "robots" / "franka_emika_panda" / "xmls" / "panda.xml"
)
assert FRANKA_XML.exists(), f"XML not found: {FRANKA_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    """Load Franka mesh assets."""
    assets: dict[str, bytes] = {}
    # Load assets from the xmls/assets directory
    assets_path = FRANKA_XML.parent / "assets"
    update_assets(assets, assets_path, meshdir)
    # Also load hand.xml which is in the xmls directory
    hand_xml_path = FRANKA_XML.parent / "hand.xml"
    if hand_xml_path.exists():
        assets["hand.xml"] = hand_xml_path.read_bytes()
    return assets


def get_spec() -> mujoco.MjSpec:
    """Load Franka MjSpec with assets."""
    spec = mujoco.MjSpec.from_file(str(FRANKA_XML))
    spec.assets = get_assets(spec.meshdir)
    return spec


##
# Joint names.
##

ARM_JOINTS = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

##
# Initial state / Keyframe.
##

# Home pose from mujoco_playground
# qpos: 7 arm joints + 2 finger joints = 9
# Based on "home" keyframe: [0, 0.3, 0, -1.57079, 0, 2.0, -0.7853, 0.04, 0.04]
INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),  # Robot base on ground
    joint_pos={
        "joint1": 0.0,
        "joint2": 0.3,
        "joint3": 0.0,
        "joint4": -1.57079,
        "joint5": 0.0,
        "joint6": 2.0,
        "joint7": -0.7853,
        "finger_joint1": 0.04,  # Open gripper
        "finger_joint2": 0.04,
    },
    joint_vel={".*": 0.0},
)

# Neutral pose - all arm joints at zero
# qpos: 7 arm joints + 2 finger joints = 9
NEUTRAL_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),  # Robot base on ground
    joint_pos={
        "joint1": 0.0,
        "joint2": -1.0,
        "joint3": 0.0,
        "joint4": -1.57079,
        "joint5": 0.0,
        "joint6": 2.0,
        "joint7": 0.785,
        "finger_joint1": 0.04,  # Open gripper
        "finger_joint2": 0.04,
    },
    joint_vel={".*": 0.0},
)

##
# Articulation config.
##
# The panda.xml defines position actuators actuator1-actuator8
# Use XmlPositionActuatorCfg to automatically use actuators defined in XML

from mjlab.actuator import XmlPositionActuatorCfg

# XmlPositionActuatorCfg automatically finds and uses actuators defined in the XML
# for the joints specified by joint_names_expr
# Use regex to match all joints (7 arm + 2 fingers)
FRANKA_ACTUATORS = XmlPositionActuatorCfg(
    joint_names_expr=(".*",),  # Match all joints
)

FRANKA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(FRANKA_ACTUATORS,),
    soft_joint_pos_limit_factor=0.9,
)


def get_franka_robot_cfg() -> EntityCfg:
    """Get a fresh Franka robot configuration instance.

    Returns a new EntityCfg instance each time to avoid mutation issues when
    the config is shared across multiple places.
    """
    return EntityCfg(
        init_state=INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec,
        articulation=FRANKA_ARTICULATION,
    )


def get_franka_robot_cfg_neutral() -> EntityCfg:
    """Get a fresh Franka robot configuration with neutral pose.

    Uses neutral pose (all arm joints at 0) instead of home pose.
    """
    return EntityCfg(
        init_state=NEUTRAL_INIT_STATE,
        collisions=(),  # Use collisions from XML
        spec_fn=get_spec,
        articulation=FRANKA_ARTICULATION,
    )


# Action scale for delta control
# Based on mujoco_playground config: action_scale=0.04
FRANKA_ACTION_SCALE = 0.04


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_franka_robot_cfg())

    viewer.launch(robot.spec.compile())
