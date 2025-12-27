import mujoco

from mjlab.asset_zoo.objects.free.cube import (
  get_cube_cfg,
  get_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.cylinder import (
  get_cylinder_cfg,
  get_mocap_goal_cfg as get_cylinder_mocap_goal_cfg,
)
from mjlab.asset_zoo.robots import (
  YAM_ACTION_SCALE,
  get_yam_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointDeltaPositionActionCfg, JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.lift_object_env_cfg import make_lift_object_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg


def yam_lift_cube_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cube": get_cube_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, JointDeltaPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)

  cfg.observations["policy"].terms["gripper_to_object"].params["robot_asset_cfg"].site_names = (
    "grasp_site",
  )
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("grasp_site",)

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg


def yam_lift_cylinder_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_yam_robot_cfg(),
    "cylinder": get_cylinder_cfg(),
    "mocap_goal": get_cylinder_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, JointDeltaPositionActionCfg)
  joint_pos_action.scale = YAM_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = "cylinder"

  cfg.observations["policy"].terms["gripper_to_object"].params["object_asset_name"] = "cylinder"
  cfg.observations["policy"].terms["gripper_to_object"].params["robot_asset_cfg"].site_names = (
    "grasp_site",
  )

  cfg.observations["policy"].terms["object_to_goal"].params["object_asset_name"] = "cylinder"

  cfg.rewards["reach_object"].params["object_asset_name"] = "cylinder"
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("grasp_site",)
  cfg.rewards["move_object_to_goal"].params["object_asset_name"] = "cylinder"

  cfg.terminations["object_out_of_bounds"].params["object_name"] = "cylinder"

  fingertip_geoms = r"[lr]f_down(6|7|8|9|10|11)_collision"
  cfg.events["fingertip_friction_slide"].params[
    "asset_cfg"
  ].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  # Configure collision sensor pattern.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link_6"

  cfg.viewer.body_name = "arm"

  # Apply play mode overrides.
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  return cfg
