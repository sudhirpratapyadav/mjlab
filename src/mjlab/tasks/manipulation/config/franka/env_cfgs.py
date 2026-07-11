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
from mjlab.asset_zoo.objects.free.cuboid import (
  get_cuboid_cfg,
  get_mocap_goal_cfg as get_cuboid_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.disc import (
  get_disc_cfg,
  get_mocap_goal_cfg as get_disc_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.cylinder import (
  get_cylinder_cfg,
  get_mocap_goal_cfg as get_cylinder_mocap_goal_cfg,
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
from mjlab.tasks.manipulation.reach_target_env_cfg import make_reach_target_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg, OpenDoorCommandCfg, OpenDrawerCommandCfg, PushButtonCommandCfg, PushingCommandCfg, ReachingCommandCfg
from mjlab.tasks.manipulation.open_door_env_cfg import make_open_door_env_cfg
from mjlab.tasks.manipulation.open_drawer_env_cfg import make_open_drawer_env_cfg
from mjlab.tasks.manipulation.push_button_env_cfg import make_push_button_env_cfg
from mjlab.tasks.manipulation.push_cuboid_env_cfg import make_push_cuboid_env_cfg
from mjlab.tasks.manipulation.push_disc_env_cfg import make_push_disc_env_cfg


def franka_lift_cube_env_cfg(
  play: bool = False,
  test: bool = False,
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

  # Apply test mode overrides (no corruption, 150 steps episode length (same as other tasks)).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("object_out_of_bounds", None)
    # Set episode length to 150 steps (150 * 0.02 control_dt = 3.0s)
    cfg.episode_length_s = 5.0

  return cfg


def franka_lift_cylinder_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka grasp-and-lift of a cylinder.

  Mirrors franka_lift_cube_env_cfg but swaps the free object for a cylinder. A cylinder
  is a distinct grasp: it can roll and has no flat top faces, so grasp alignment is
  less forgiving than the cube — a genuinely different pick-place instance for the
  benchmark (grasp-geometry generalization along the same precision-grasp fragility).
  """
  cfg = make_lift_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cylinder": get_cylinder_cfg(),
    "mocap_goal": get_cylinder_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = "cylinder"

  # Same spawn/target ranges as the cube lift.
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

  # Point every object-referencing term at the cylinder.
  for term_name in (
    "object_pos",
    "object_quat",
    "object_orientation",
    "gripper_to_object",
    "object_to_goal",
    "goal_orientation_diff",
  ):
    if term_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[term_name].params["object_asset_name"] = "cylinder"
  cfg.rewards["reach_object"].params["object_asset_name"] = "cylinder"
  cfg.rewards["move_object_to_goal"].params["object_asset_name"] = "cylinder"
  cfg.terminations["object_out_of_bounds"].params["object_name"] = "cylinder"

  # Franka uses "gripper" site for end-effector.
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  # Franka fingertip geoms for friction randomization.
  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Franka end-effector is link7.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 1.5

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("object_out_of_bounds", None)
    cfg.episode_length_s = 5.0

  return cfg


def franka_reach_target_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka end-effector reaching to a sampled 3D target (no object).

  The simplest benchmark skill: dense monotonic distance reward, no contact. Anchors
  the low-fragility (planar) end of the axis and adds the ``reach`` skill family.
  """
  cfg = make_reach_target_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  # Franka uses the "gripper" site for the end-effector.
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_target"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["reach_target"].params["robot_asset_cfg"].site_names = ("gripper",)
  assert cfg.commands is not None
  reach_command = cfg.commands["reach_target"]
  assert isinstance(reach_command, ReachingCommandCfg)
  reach_command.robot_asset_cfg.site_names = ("gripper",)

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 1.5

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.episode_length_s = 5.0

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

  # Apply test mode overrides (no corruption, but keep train episode length).
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


def franka_push_cuboid_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka-specific cuboid pushing configuration."""
  cfg = make_push_cuboid_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cuboid": get_cuboid_cfg(),
    "mocap_goal": get_cuboid_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  push_command = cfg.commands["push_cuboid"]
  assert isinstance(push_command, PushingCommandCfg)

  # Override object and target ranges for Franka
  push_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
    z=(0.015, 0.015),  # Cuboid half-height is 0.015 - spawn at ground level
    yaw=(0.0, 0.0),  # No rotation - keep upright
  )
  push_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
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

  # Apply test mode overrides (no corruption, 150 steps episode length (same as other tasks)).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("object_out_of_bounds", None)

  return cfg


def franka_push_disc_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka-specific disc pushing configuration."""
  cfg = make_push_disc_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "disc": get_disc_cfg(),
    "mocap_goal": get_disc_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  push_command = cfg.commands["push_disc"]
  assert isinstance(push_command, PushingCommandCfg)

  # Override object and target ranges for Franka
  push_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
    z=(0.05, 0.05),  # TEST: Fixed height
    yaw=(0.0, 0.0),  # TEST: NO rotation - keep upright
  )
  push_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(0.6, 0.8),
    y=(-0.15, 0.15),
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

  # Apply test mode overrides (no corruption, 150 steps episode length (same as other tasks)).
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # Disable early termination - only terminate on timeout
    cfg.terminations.pop("ee_ground_collision", None)
    cfg.terminations.pop("object_out_of_bounds", None)
    # Set episode length to 150 steps (150 * 0.02 control_dt = 3.0s)
    cfg.episode_length_s = 5.0

  return cfg
