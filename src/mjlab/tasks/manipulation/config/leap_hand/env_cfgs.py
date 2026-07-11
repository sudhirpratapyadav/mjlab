"""Class-C (floating LEAP hand) task configs.

The floating LEAP hand has an actuated 6-DoF base + 16 finger joints (22-D joint-space
action), so it reuses the SAME position-based task bases as the Franka arm — only the
robot entity, the end-effector site (``grasp_site``), and the action scale differ. This
keeps obs/action/reward/success uniform across embodiments.
"""

from mjlab.asset_zoo.objects.free.cube import get_cube_cfg, get_mocap_goal_cfg
from mjlab.asset_zoo.objects.free.cuboid import get_cuboid_cfg
from mjlab.asset_zoo.objects.free.peg_in_hole import get_peg_cfg, get_hole_board_cfg
from mjlab.asset_zoo.robots.leap_hand import LEAP_ACTION_SCALE, get_leap_hand_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointDeltaPositionActionCfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg, ReachingCommandCfg, StackingCommandCfg
from mjlab.tasks.manipulation.lift_object_env_cfg import make_lift_object_env_cfg
from mjlab.tasks.manipulation.reach_target_env_cfg import make_reach_target_env_cfg
from mjlab.tasks.manipulation.stack_object_env_cfg import make_stack_object_env_cfg

_EE_SITE = "grasp_site"


def _apply_leap_common(cfg: ManagerBasedRlEnvCfg) -> None:
  """Wire the LEAP hand's action scale, EE site, and viewer into a base cfg."""
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = LEAP_ACTION_SCALE

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object", "gripper_to_target"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        _EE_SITE,
      )
  # No fingertip-friction / ee-ground-collision wiring: the LEAP model geoms differ
  # from the Franka; drop those Franka-specific events/sensors if present.
  for ev in ("fingertip_friction_slide", "fingertip_friction_spin", "fingertip_friction_roll"):
    cfg.events.pop(ev, None)
  if cfg.scene.sensors is not None:
    cfg.scene.sensors = tuple(
      s for s in cfg.scene.sensors if s.name != "ee_ground_collision"
    )
  cfg.terminations.pop("ee_ground_collision", None)

  cfg.viewer.body_name = "palm"
  cfg.scene.env_spacing = 1.0


def leap_reach_target_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand reaching its palm to a sampled 3D target (Class-C reach)."""
  cfg = make_reach_target_env_cfg()
  cfg.scene.entities = {
    "robot": get_leap_hand_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }
  cfg.rewards["reach_target"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  assert cfg.commands is not None
  reach_command = cfg.commands["reach_target"]
  assert isinstance(reach_command, ReachingCommandCfg)
  reach_command.robot_asset_cfg.site_names = (_EE_SITE,)
  _apply_leap_common(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.episode_length_s = 5.0
  return cfg


def leap_lift_cube_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand grasping and lifting a cube (Class-C pick_place)."""
  cfg = make_lift_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_leap_hand_cfg(),
    "cube": get_cube_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }
  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = "cube"
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  _apply_leap_common(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("object_out_of_bounds", None)
    cfg.episode_length_s = 5.0
  return cfg


def _leap_stack_like_env_cfg(object_cfg_fn, base_cfg_fn, stack_height, xy_thresh,
                             play=False, test=False) -> ManagerBasedRlEnvCfg:
  """Shared builder for LEAP stack/insertion (move 'object' onto/into 'base')."""
  cfg = make_stack_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_leap_hand_cfg(),
    "object": object_cfg_fn(),
    "base": base_cfg_fn(),
  }
  assert cfg.commands is not None
  stack_command = cfg.commands["stack_object"]
  assert isinstance(stack_command, StackingCommandCfg)
  stack_command.robot_asset_cfg.site_names = (_EE_SITE,)
  stack_command.stack_height = stack_height
  stack_command.success_threshold = xy_thresh
  cfg.rewards["stack"].params["robot_asset_cfg"].site_names = (_EE_SITE,)
  _apply_leap_common(cfg)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("object_out_of_bounds", None)
    cfg.episode_length_s = 5.0
  return cfg


def leap_stack_cube_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand stacking a cube on a cuboid base (Class-C pick_place)."""
  return _leap_stack_like_env_cfg(get_cube_cfg, get_cuboid_cfg, 0.035, 0.03, play=play, test=test)


def leap_peg_insertion_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand inserting a peg into a hole board (Class-C insertion)."""
  return _leap_stack_like_env_cfg(get_peg_cfg, get_hole_board_cfg, 0.01, 0.015, play=play, test=test)
