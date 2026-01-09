import mujoco
from pathlib import Path

from mjlab.asset_zoo.objects.articulated.door import (
  get_door_cfg,
  get_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.drawer import (
  get_drawer_cfg,
  get_mocap_target_cfg as get_drawer_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.button import (
  get_button_cfg,
  get_mocap_target_cfg as get_button_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.free.cube import (
  get_cube_cfg,
  get_mocap_goal_cfg,
)
from mjlab.asset_zoo.robots import (
  FRANKA_ACTION_SCALE,
  get_franka_robot_cfg,
  get_franka_robot_cfg_neutral,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointDeltaPositionActionCfg, JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_object_env_cfg import make_lift_object_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg, OpenDoorCommandCfg, OpenDrawerCommandCfg, PushButtonCommandCfg
from mjlab.tasks.manipulation.open_door_env_cfg import make_open_door_env_cfg
from mjlab.tasks.manipulation.open_drawer_env_cfg import make_open_drawer_env_cfg
from mjlab.tasks.manipulation.push_button_env_cfg import make_push_button_env_cfg


def franka_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cube": get_cube_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)

  # Override object and target ranges for Franka
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
    z=(0.02, 0.05),
    yaw=(-3.14, 3.14),
  )
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
    z=(0.2, 0.4),
  )

  # Franka uses "gripper" site for end-effector
  # Update all observation terms that use site_names
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  # Franka fingertip geoms for friction randomization
  # Based on Franka hand structure: left_finger_pad, right_finger_pad
  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern - Franka end-effector is link7
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"

  # Set environment spacing
  cfg.scene.env_spacing = 1.5

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def franka_open_door_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka-specific door opening configuration."""
  cfg = make_open_door_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "door": get_door_cfg(),
    "mocap_goal": get_mocap_target_cfg(),
  }

  # Door joint position is set by OpenDoorCommand during _resample_command
  # init_state sets default_joint_pos to prevent drift back to 0

  # Set action scale
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  # Override door pose range for Franka workspace
  assert cfg.commands is not None
  door_command = cfg.commands["open_door"]
  assert isinstance(door_command, OpenDoorCommandCfg)
  door_command.door_pose_range = OpenDoorCommandCfg.DoorPoseRangeCfg(
    x=(1.5, 1.5),  # Fixed at x=1.5m
    y=(0.0, 0.0),  # Fixed at y=0.0
    z=(0.61, 0.61),  # Fixed height
    yaw=(0.0, 0.0),  # No yaw randomization
  )
  # Set robot asset config for gripper position in metrics
  door_command.robot_asset_cfg.site_names = ("gripper",)

  # Franka uses "gripper" site for end-effector
  # Update all observation terms that use site_names
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )

  # Update reward terms that use site_names
  for reward_name in ["reach_object"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["robot_asset_cfg"].site_names = ("gripper",)

  # Franka fingertip geoms for friction randomization
  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensors - Franka end-effector is link7
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"
    elif sensor.name == "ee_door_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"

  # Set environment spacing (wider for door)
  cfg.scene.env_spacing = 2.0

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  # Apply test mode overrides (no corruption, but train episode length).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)

  return cfg


def franka_open_drawer_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka-specific drawer opening configuration."""
  cfg = make_open_drawer_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "drawer": get_drawer_cfg(),
    "mocap_goal": get_drawer_mocap_target_cfg(),
  }

  # Drawer joint position is set by OpenDrawerCommand during _resample_command
  # init_state sets default_joint_pos to prevent drift back to 0

  # Set action scale
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  # Override drawer pose range for Franka workspace
  assert cfg.commands is not None
  drawer_command = cfg.commands["open_drawer"]
  assert isinstance(drawer_command, OpenDrawerCommandCfg)
  drawer_command.drawer_pose_range = OpenDrawerCommandCfg.DrawerPoseRangeCfg(
    x=(0.65, 0.65),  # Fixed at x=0.65m
    y=(-0.1, 0.1),   # 0.0 ± 0.1m (10cm randomization)
    z=(0.5, 0.5),    # Fixed height
    yaw=(0.0, 0.0),  # No yaw randomization
  )
  # Set robot asset config for gripper position in metrics
  drawer_command.robot_asset_cfg.site_names = ("gripper",)

  # Franka uses "gripper" site for end-effector
  # Update all observation terms that use site_names
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )

  # Update reward terms that use site_names
  for reward_name in ["reach_object"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["robot_asset_cfg"].site_names = ("gripper",)

  # Franka fingertip geoms for friction randomization
  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensors - Franka end-effector is link7
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"
    elif sensor.name == "ee_drawer_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"

  # Set environment spacing (wider for drawer)
  cfg.scene.env_spacing = 2.0

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = 4.0  # Reset every 4 seconds for debugging
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  # Apply test mode overrides (no corruption, but keep train episode length).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)

  return cfg


def franka_push_button_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka-specific button pushing configuration."""
  cfg = make_push_button_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "button": get_button_cfg(),
    "mocap_goal": get_button_mocap_target_cfg(),
  }

  # Button joint position is set by PushButtonCommand during _resample_command
  # init_state sets default_joint_pos to prevent drift

  # Set action scale
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  # Override button pose range for Franka workspace
  assert cfg.commands is not None
  button_command = cfg.commands["push_button"]
  assert isinstance(button_command, PushButtonCommandCfg)
  button_command.button_pose_range = PushButtonCommandCfg.ButtonPoseRangeCfg(
    x=(0.65, 0.65),  # Fixed at x=0.65m
    y=(-0.1, 0.1),   # 0.0 ± 0.1m (10cm randomization)
    z=(0.5, 0.5),    # Fixed height
    yaw=(0.0, 0.0),  # No yaw randomization
  )
  # Set robot asset config for gripper position in metrics
  button_command.robot_asset_cfg.site_names = ("gripper",)

  # Franka uses "gripper" site for end-effector
  # Update all observation terms that use site_names
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )

  # Update reward terms that use site_names
  for reward_name in ["reach_object"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["robot_asset_cfg"].site_names = ("gripper",)

  # Franka fingertip geoms for friction randomization
  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensors - Franka end-effector is link7
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"
    elif sensor.name == "ee_button_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"

  # Set environment spacing
  cfg.scene.env_spacing = 2.0

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  # Apply test mode overrides (no corruption, but keep train episode length).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)

  return cfg
