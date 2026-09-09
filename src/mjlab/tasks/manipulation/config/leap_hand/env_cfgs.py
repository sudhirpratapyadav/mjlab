"""Class-C (floating LEAP hand) task configs.

The floating LEAP hand has an actuated 6-DoF base + 16 finger joints (22-D joint-space
action), so it reuses the SAME position-based task bases as the Franka arm — only the
robot entity, the end-effector site (``grasp_site``), and the action scale differ. This
keeps obs/action/reward/success uniform across embodiments.
"""

from mjlab.asset_zoo.objects.free.cube import get_cube_cfg, get_mocap_goal_cfg
from mjlab.asset_zoo.objects.free.cuboid import get_cuboid_cfg
from mjlab.asset_zoo.objects.free.sphere import get_sphere_cfg
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

  # Dexterous hand has many finger geoms -> many contacts. The arm bases' nconmax=200
  # overflows the warp narrowphase buffer (segfault at CUDA-graph capture on some GPUs).
  # Bump the contact/constraint buffers for the hand.
  cfg.sim.nconmax = 800
  cfg.sim.njmax = 3000

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


def _leap_lift_object_env_cfg(
  object_name, object_cfg_fn, play=False, test=False
) -> ManagerBasedRlEnvCfg:
  """Shared builder: floating LEAP hand lifting an arbitrary free object."""
  cfg = make_lift_object_env_cfg()
  cfg.scene.entities = {
    "robot": get_leap_hand_cfg(),
    object_name: object_cfg_fn(),
    "mocap_goal": get_mocap_goal_cfg(),
  }
  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = object_name
  for term_name in (
    "object_pos", "object_quat", "object_orientation",
    "gripper_to_object", "object_to_goal", "goal_orientation_diff",
  ):
    if term_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[term_name].params["object_asset_name"] = object_name
  cfg.rewards["reach_object"].params["object_asset_name"] = object_name
  cfg.rewards["move_object_to_goal"].params["object_asset_name"] = object_name
  cfg.terminations["object_out_of_bounds"].params["object_name"] = object_name
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


def leap_lift_cube_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand grasping and lifting a cube (Class-C pick_place)."""
  return _leap_lift_object_env_cfg("cube", get_cube_cfg, play=play, test=test)


def leap_lift_sphere_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Floating LEAP hand grasping and lifting a sphere."""
  return _leap_lift_object_env_cfg("sphere", get_sphere_cfg, play=play, test=test)


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
  # stack_height 0.035 (not the old 0.01) for the SAME reason as the Franka variant:
  # StackingCommand compares the peg's ROOT to base_root + stack_height, and an inserted
  # peg's root sits at ground + half-length (0.050) while the board root is at 0.015.
  # With 0.01 the goal was 25 mm below the peg's own resting height, which is more than
  # the default 0.02 height_threshold -- i.e. the LEAP task would have become
  # unsatisfiable when the peg/board assets were upgraded. (W1-c, CL-V2)
  cfg = _leap_stack_like_env_cfg(get_peg_cfg, get_hole_board_cfg, 0.035, 0.015, play=play, test=test)
  # The stack base's default spawn z (0.02) fits the cube's half-height, but the peg
  # is 0.10 tall (half-height 0.05): at z=0.02 it spawns 30 mm INSIDE the floor and
  # pops out on the first solver step. Rest the peg on the ground instead.
  command = cfg.commands["stack_object"]
  assert isinstance(command, StackingCommandCfg)
  peg_range = StackingCommandCfg.ObjectPoseRangeCfg()
  peg_range.z = (0.05, 0.05)
  command.object_pose_range = peg_range
  return cfg
