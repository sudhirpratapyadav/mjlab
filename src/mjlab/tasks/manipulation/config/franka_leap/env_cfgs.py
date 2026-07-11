"""Class-B (Franka arm + LEAP hand) task configs.

The arm+hand is a 23-DoF manipulator (7 arm + 16 finger), so it reuses the SAME
position-based task bases as the other embodiments — only the robot entity, the
end-effector site (``leap_grasp_site``), and the action scale differ. Uniform
interfaces across all three embodiment classes.
"""

from mjlab.asset_zoo.objects.free.cube import get_cube_cfg, get_mocap_goal_cfg
from mjlab.asset_zoo.objects.free.cuboid import get_cuboid_cfg
from mjlab.asset_zoo.robots.franka_leap import (
  FRANKA_LEAP_ACTION_SCALE,
  get_franka_leap_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointDeltaPositionActionCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg, ReachingCommandCfg, StackingCommandCfg
from mjlab.tasks.manipulation.lift_object_env_cfg import make_lift_object_env_cfg
from mjlab.tasks.manipulation.reach_target_env_cfg import make_reach_target_env_cfg
from mjlab.tasks.manipulation.stack_object_env_cfg import make_stack_object_env_cfg

_EE_SITE = "leap_grasp_site"


def _apply_common(cfg: ManagerBasedRlEnvCfg) -> None:
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_LEAP_ACTION_SCALE

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object", "gripper_to_target"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  # Drop Franka-gripper-specific events/sensors (the LEAP fingers differ).
  for ev in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
    cfg.events.pop(ev, None)
  if cfg.scene.sensors is not None:
    cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name != "ee_ground_collision")
  cfg.terminations.pop("ee_ground_collision", None)

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 1.5


def _finish(cfg: ManagerBasedRlEnvCfg, play: bool, test: bool, drop_oob: bool) -> ManagerBasedRlEnvCfg:
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    if drop_oob:
      cfg.terminations.pop("object_out_of_bounds", None)
    cfg.episode_length_s = 5.0
  return cfg


def franka_leap_reach_target_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Franka+LEAP reaching the hand to a sampled 3D target (Class-B reach)."""
  cfg = make_reach_target_env_cfg()
  cfg.scene.entities = {"robot": get_franka_leap_robot_cfg(), "mocap_goal": get_mocap_goal_cfg()}
  cfg.rewards["reach_target"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  assert cfg.commands is not None
  reach_command = cfg.commands["reach_target"]
  assert isinstance(reach_command, ReachingCommandCfg)
  reach_command.robot_asset_cfg.site_names = (_EE_SITE,)
  _apply_common(cfg)
  return _finish(cfg, play, test, drop_oob=False)


def franka_leap_lift_cube_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Franka+LEAP grasping and lifting a cube (Class-B pick_place)."""
  cfg = make_lift_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_franka_leap_robot_cfg(), "cube": get_cube_cfg(), "mocap_goal": get_mocap_goal_cfg(),
  }
  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = "cube"
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  _apply_common(cfg)
  return _finish(cfg, play, test, drop_oob=True)


def franka_leap_stack_cube_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Franka+LEAP stacking a cube on a cuboid base (Class-B pick_place)."""
  cfg = make_stack_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_franka_leap_robot_cfg(), "object": get_cube_cfg(), "base": get_cuboid_cfg(),
  }
  assert cfg.commands is not None
  stack_command = cfg.commands["stack_object"]
  assert isinstance(stack_command, StackingCommandCfg)
  stack_command.robot_asset_cfg.site_names = (_EE_SITE,)
  stack_command.stack_height = 0.035
  cfg.rewards["stack"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  _apply_common(cfg)
  return _finish(cfg, play, test, drop_oob=True)
