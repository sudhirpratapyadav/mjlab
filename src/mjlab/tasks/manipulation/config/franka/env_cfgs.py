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
  CUBE_HALF_HEIGHT,
  get_cube_cfg,
  get_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.cuboid import (
  CUBOID_HALF_EXTENTS,
  CUBOID_HALF_HEIGHT,
  get_cuboid_cfg,
  get_mocap_goal_cfg as get_cuboid_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.disc import (
  get_disc_cfg,
  get_mocap_goal_cfg as get_disc_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.peg_in_hole import (
  get_peg_cfg,
  get_hole_board_cfg,
)
from mjlab.asset_zoo.objects.free.sphere import (
  get_sphere_cfg,
  get_mocap_goal_cfg as get_sphere_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.ellipsoid import (
  get_ellipsoid_cfg,
  get_mocap_goal_cfg as get_ellipsoid_mocap_goal_cfg,
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
from mjlab.tasks.manipulation import workspace
from mjlab.tasks.manipulation.lift_object_env_cfg import make_lift_object_env_cfg
from mjlab.tasks.manipulation.reach_target_env_cfg import make_reach_target_env_cfg
from mjlab.tasks.manipulation.stack_object_env_cfg import make_stack_object_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg, OpenDoorCommandCfg, OpenDrawerCommandCfg, PushButtonCommandCfg, PushingCommandCfg, ReachingCommandCfg, StackingCommandCfg
from mjlab.tasks.manipulation.mdp.commands import (
  _ObjectSpawnRangeCfg,
  CageDragCommandCfg,
  EdgeGraspCommandCfg,
  PivotLiftCommandCfg,
  PlaceInContainerCommandCfg,
  ReorientObjectCommandCfg,
  ToolPullCommandCfg,
)
from mjlab.asset_zoo.objects.articulated.flap import (
  get_flap_cfg,
  get_mocap_target_cfg as get_flap_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.plug import (
  get_plug_cfg,
  get_mocap_target_cfg as get_plug_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.free.block import (
  BLOCK_HALF_EXTENTS,
  BLOCK_HALF_HEIGHT,
  get_block_cfg,
  get_mocap_goal_cfg as get_block_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.plate import (
  get_plate_cfg,
  get_mocap_goal_cfg as get_plate_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.ledge import get_ledge_cfg
from mjlab.asset_zoo.objects.free.board import (
  get_board_cfg,
  get_mocap_goal_cfg as get_board_mocap_goal_cfg,
)
from mjlab.asset_zoo.objects.free.wall import get_wall_cfg
from mjlab.tasks.manipulation.drag_pull_env_cfg import make_drag_pull_env_cfg
from mjlab.tasks.manipulation.strike_slide_env_cfg import make_strike_slide_env_cfg
from mjlab.tasks.manipulation.cage_drag_env_cfg import make_cage_drag_env_cfg
from mjlab.tasks.manipulation.topple_block_env_cfg import make_topple_block_env_cfg
from mjlab.tasks.manipulation.push_flap_env_cfg import make_push_flap_env_cfg
from mjlab.tasks.manipulation.axial_extract_env_cfg import make_axial_extract_env_cfg
from mjlab.tasks.manipulation.edge_grasp_env_cfg import make_edge_grasp_env_cfg
from mjlab.tasks.manipulation.pivot_lift_env_cfg import make_pivot_lift_env_cfg
from mjlab.asset_zoo.objects.articulated.lever import (
  get_lever_cfg,
  get_mocap_target_cfg as get_lever_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.valve import (
  get_valve_cfg,
  get_mocap_target_cfg as get_valve_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.switch import (
  get_switch_cfg,
  get_mocap_target_cfg as get_switch_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.window import (
  get_window_cfg,
  get_mocap_target_cfg as get_window_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.articulated.lid import (
  get_lid_cfg,
  get_mocap_target_cfg as get_lid_mocap_target_cfg,
)
from mjlab.asset_zoo.objects.free.container import get_container_cfg
from mjlab.asset_zoo.objects.free.stick import get_stick_cfg
from mjlab.asset_zoo.objects.free.puck import (
  get_puck_cfg,
  get_mocap_goal_cfg as get_puck_mocap_goal_cfg,
)
from mjlab.tasks.manipulation.turn_lever_env_cfg import make_turn_lever_env_cfg
from mjlab.tasks.manipulation.rotate_valve_env_cfg import make_rotate_valve_env_cfg
from mjlab.tasks.manipulation.flip_switch_env_cfg import make_flip_switch_env_cfg
from mjlab.tasks.manipulation.slide_window_env_cfg import make_slide_window_env_cfg
from mjlab.tasks.manipulation.open_lid_env_cfg import make_open_lid_env_cfg
from mjlab.tasks.manipulation.place_in_container_env_cfg import (
  make_place_in_container_env_cfg,
)
from mjlab.tasks.manipulation.reorient_object_env_cfg import (
  make_reorient_object_env_cfg,
)
from mjlab.tasks.manipulation.tool_pull_env_cfg import make_tool_pull_env_cfg
from mjlab.tasks.manipulation.open_door_env_cfg import make_open_door_env_cfg
from mjlab.tasks.manipulation.open_drawer_env_cfg import make_open_drawer_env_cfg
from mjlab.tasks.manipulation.push_button_env_cfg import make_push_button_env_cfg
from mjlab.tasks.manipulation.push_cuboid_env_cfg import make_push_cuboid_env_cfg
from mjlab.tasks.manipulation.push_disc_env_cfg import make_push_disc_env_cfg



def _mech_z(asset: str) -> tuple[float, float]:
  """Mount-height range for an articulated mechanism, clear of the ground plane.

  Mechanisms hang DOWNWARD from their mocap mount and Class A scenes have no table or
  wall to hang them from, so mount height is dictated by the asset's own downward
  extent (door: 0.80m) rather than by what is comfortable to reach. Picking a mount z
  for reachability alone buries the mechanism in the floor — invisible to
  benchmark-smoke and to the radial reach audit, but obvious in a rendered frame.
  """
  z = workspace.min_mechanism_mount_z(asset)
  return (z, z)


def _grasp_box_corner_safe() -> tuple[tuple[float, float], tuple[float, float]]:
  """The Class A grasp envelope, with x trimmed so its CORNERS obey GRASP_RADIAL_MAX.

  Thin alias for ``workspace.grasp_box()`` — see there for why the raw
  ``GRASP_X_RANGE`` x ``GRASP_Y_RANGE`` rectangle is not itself corner-safe. Kept as a
  named local so the lift/push call sites read clearly.
  """
  return workspace.grasp_box()


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

  # Object spawn / lift goal live in the measured Class A grasp envelope
  # (mjlab.tasks.manipulation.workspace). z is a property of the cube, not of the
  # workspace: the low end is the collision box's half-height (cube_constants.
  # CUBE_HALF_HEIGHT = 0.0226 for the 46 mm mini 3x3), the high end is unchanged so
  # the cube sometimes spawns a couple of cm up and drops.
  _obj_x, _obj_y = _grasp_box_corner_safe()
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=_obj_x,
    y=_obj_y,
    z=(CUBE_HALF_HEIGHT, 0.05),
    yaw=(-3.14, 3.14),
  )
  # goal_box(), not the raw GOAL_* ranges: their far corner escapes the ceiling
  # (hypot(0.55, 0.28) = 0.617 > GOAL_RADIAL_MAX). See workspace.goal_box.
  _goal_x, _goal_y = workspace.goal_box()
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=_goal_x,
    y=_goal_y,
    z=workspace.GOAL_Z_RANGE,
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

  # Same spawn/target ranges as the cube lift: the shared Class A grasp envelope.
  _obj_x, _obj_y = _grasp_box_corner_safe()
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=_obj_x,
    y=_obj_y,
    # The bottle stands upright here; its resting height is its half-length 0.0266
    # (W1-b, cl_v2: the primitive 0.02-half-height cylinder became a real bottle).
    z=(0.028, 0.05),
    yaw=(-3.14, 3.14),
  )
  # goal_box(), not the raw GOAL_* ranges: their far corner escapes the ceiling
  # (hypot(0.55, 0.28) = 0.617 > GOAL_RADIAL_MAX). See workspace.goal_box.
  _goal_x, _goal_y = workspace.goal_box()
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=_goal_x,
    y=_goal_y,
    z=workspace.GOAL_Z_RANGE,
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


def _franka_lift_object_env_cfg(
  object_name: str,
  object_cfg_fn,
  mocap_goal_cfg_fn,
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Shared builder for a Franka grasp-and-lift task over an arbitrary free object.

  Captures the lift recipe (see franka_lift_cube/cylinder) so new graspable-object
  variants are a one-liner. Keeps obs/action/reward/success identical across all lift
  tasks (uniform interfaces); only the object entity + its name differ.
  """
  cfg = make_lift_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    object_name: object_cfg_fn(),
    "mocap_goal": mocap_goal_cfg_fn(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  lift_command = cfg.commands["lift_object"]
  assert isinstance(lift_command, LiftingCommandCfg)
  lift_command.asset_name = object_name
  # Shared Class A grasp envelope; z (resting height) stays object-specific.
  _obj_x, _obj_y = _grasp_box_corner_safe()
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=_obj_x, y=_obj_y,
    z=(0.02, 0.05), yaw=(-3.14, 3.14),
  )
  # goal_box(): the raw GOAL_* corner exceeds GOAL_RADIAL_MAX. See workspace.goal_box.
  _goal_x, _goal_y = workspace.goal_box()
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=_goal_x, y=_goal_y, z=workspace.GOAL_Z_RANGE,
  )

  for term_name in (
    "object_pos", "object_quat", "object_orientation",
    "gripper_to_object", "object_to_goal", "goal_orientation_diff",
  ):
    if term_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[term_name].params["object_asset_name"] = object_name
  cfg.rewards["reach_object"].params["object_asset_name"] = object_name
  cfg.rewards["move_object_to_goal"].params["object_asset_name"] = object_name
  cfg.terminations["object_out_of_bounds"].params["object_name"] = object_name

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

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


def franka_lift_sphere_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Franka grasp-and-lift of a sphere (rolls, no flat faces — hardest grasp geometry)."""
  return _franka_lift_object_env_cfg(
    "sphere", get_sphere_cfg, get_sphere_mocap_goal_cfg, play=play, test=test
  )


def franka_lift_ellipsoid_env_cfg(play: bool = False, test: bool = False) -> ManagerBasedRlEnvCfg:
  """Franka grasp-and-lift of an elongated ellipsoid (orientation-sensitive grasp)."""
  return _franka_lift_object_env_cfg(
    "ellipsoid", get_ellipsoid_cfg, get_ellipsoid_mocap_goal_cfg, play=play, test=test
  )


def franka_stack_cube_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka stacking a cube on top of a cuboid base.

  A higher-fragility pick_place instance: precise placement + release on top of the
  base, so small errors miss or topple. Goal is dynamic (tracks the base object).
  """
  cfg = make_stack_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "object": get_cube_cfg(),
    "base": get_cuboid_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  # cube half-height 0.0226 + cuboid half-height 0.0150 => stack offset 0.0376.
  # Both are read off the packaged collision boxes (cube_constants /
  # cuboid_constants), not off the retired primitives.
  assert cfg.commands is not None
  stack_command = cfg.commands["stack_object"]
  assert isinstance(stack_command, StackingCommandCfg)
  stack_command.robot_asset_cfg.site_names = ("gripper",)
  stack_command.stack_height = CUBE_HALF_HEIGHT + CUBOID_HALF_HEIGHT

  # Two objects that must not overlap: split the shared GRASP_* box laterally, one
  # either side of y=0. Each half keeps the FULL |y| extent of GRASP_Y_RANGE (that is
  # what makes the two objects' relative bearing vary), so x is squeezed instead: it
  # runs from GRASP_RADIAL_MIN out to 0.47, short of GRASP_X_RANGE's 0.52, putting the
  # far corner at sqrt(0.47^2 + 0.25^2) = 0.532 — inside GRASP_RADIAL_MAX (0.55).
  _STACK_X = (workspace.GRASP_RADIAL_MIN, 0.47)
  stack_command.object_pose_range = StackingCommandCfg.ObjectPoseRangeCfg(
    x=_STACK_X,
    y=(workspace.GRASP_Y_RANGE[0], -0.04),  # cube: right half
    z=(CUBE_HALF_HEIGHT, CUBE_HALF_HEIGHT),  # resting on the ground plane
    yaw=(0.0, 0.0),
  )
  stack_command.base_pose_range = StackingCommandCfg.BasePoseRangeCfg(
    x=_STACK_X,
    y=(0.04, workspace.GRASP_Y_RANGE[1]),  # cuboid base: left half
    z=(CUBOID_HALF_HEIGHT, CUBOID_HALF_HEIGHT),  # cuboid base half-height
    yaw=(0.0, 0.0),
  )

  # Franka uses the "gripper" site.
  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["stack"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms

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


def franka_peg_insertion_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka inserting a peg into a hole board (the insertion skill).

  Reuses the Stack MDP pattern (move 'object'=peg onto 'base'=hole_board, dynamic goal
  at the hole opening), keeping obs/action/reward/success identical in shape to Stack.
  Insertion is tighter than stacking: the peg must align to the ~3cm hole, so the xy
  success tolerance is smaller. This is the benchmark's insertion / contact-precision
  skill (the most fragile tier).
  """
  cfg = make_stack_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "object": get_peg_cfg(),
    "base": get_hole_board_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg))
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  stack_command = cfg.commands["stack_object"]
  assert isinstance(stack_command, StackingCommandCfg)
  stack_command.robot_asset_cfg.site_names = ("gripper",)
  # RE-DERIVED for the CL-V2 shape-sorter geometry (W1-c). StackingCommand measures
  # the peg's BODY ORIGIN against base_root + stack_height, so stack_height is the
  # peg's INSERTED centre height above the board origin, not a site offset:
  #   inserted peg centre z = ground + peg half-length = 0.050
  #   board body origin z   = board half-thickness     = 0.015
  #   => stack_height = 0.050 - 0.015 = 0.035
  # (the old 0.01 aimed the goal 25 mm BELOW the peg's resting centre, which made the
  # G5 oracle unreachable: the verifier's teleport had 25 mm still to fall and three
  # steps is 0.06 s against a 0.071 s drop — measured oracle 0.000 on the primitives).
  stack_command.stack_height = 0.035
  # xy: 6x the physical 2.5 mm per-side clearance, so "physically in the bore" implies
  # success and nothing else can be within 15 mm of the hole centre at that height.
  stack_command.success_threshold = 0.015
  # z: 0 when fully seated; 0.030 resting on the board top and 0.021 stuck on the
  # chamfer, so 0.015 separates "inserted" from "sitting on the lid" by 2x.
  stack_command.height_threshold = 0.015

  # Same lateral split as stack (see franka_stack_cube_env_cfg): peg on the right half
  # of the GRASP_* box, hole board on the left, neither overlapping and both under
  # GRASP_RADIAL_MAX (far corner sqrt(0.47^2 + 0.25^2) = 0.532).
  _PEG_X = (workspace.GRASP_RADIAL_MIN, 0.47)
  stack_command.object_pose_range = StackingCommandCfg.ObjectPoseRangeCfg(
    x=_PEG_X,
    y=(workspace.GRASP_Y_RANGE[0], -0.04),  # peg
    # The peg is a 10cm box (half-length 0.05) spawned UPRIGHT, so its centre must sit
    # a half-length above the floor. The old 0.02 buried its lower half in the ground
    # plane — invisible to smoke and to the reach audit, but a real 3cm penetration.
    z=(0.05, 0.05),
    yaw=(0.0, 0.0),
  )
  stack_command.base_pose_range = StackingCommandCfg.BasePoseRangeCfg(
    x=_PEG_X,
    y=(0.04, workspace.GRASP_Y_RANGE[1]),  # hole board
    z=(0.015, 0.015),
    yaw=(0.0, 0.0),
  )

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params["robot_asset_cfg"].site_names = (
        "gripper",
      )
  cfg.rewards["stack"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms

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
  # The default target box (x 0.40-0.70, y +-0.25, z 0.15-0.50) CONTAINS the Franka's
  # reset EE pose (0.677, 0.000, 0.382 at HOME_QPOS), so ~0.5% of resets were already
  # at the goal (measured 5/1000, a G5 failure). 0.10 m = 2x success_threshold, so
  # every episode starts with a real reach to make. The box itself is untouched: the
  # reach band still spans the envelope, which is this task's whole purpose.
  reach_command.min_gripper_clearance = 0.10

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

  # NOTE: door_pose_range above does NOT position the door — the actual mount pose is
  # written by the `reset_door_position` reset event, which is what we override here.
  # door.xml: the handle (object_site) sits at (-0.04, +0.25, 0) from door_base, so the
  # mount is shifted 0.25 to -y to bring the handle onto the robot's midline. Handle
  # then lands at x 0.44-0.48, y +-0.05 -> radial <= 0.49, under
  # workspace.MECHANISM_HANDLE_RADIAL_MAX. z is left at the asset's 0.61 wall height,
  # which the door's hinge geometry and success target depend on.
  cfg.events["reset_door_position"].params["pose_range"] = {
    "x": (0.48, 0.52),
    "y": (-0.30, -0.20),
    "z": _mech_z("door"),
  }

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

  # As for the door, placement comes from the reset event, not drawer_pose_range.
  # drawer.xml: handle at (-0.04, 0, 0) from drawer_base and the drawer resets closed,
  # so handle x = mount x - 0.04 -> 0.42-0.52 here (radial <= 0.53).
  cfg.events["reset_drawer_position"].params["pose_range"] = {
    "x": (0.46, 0.56),
    "y": (-0.10, 0.10),
    "z": _mech_z("drawer"),
  }

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

  # As for the door, placement comes from the reset event, not button_pose_range.
  # button.xml: the button cap (object_site) is directly ABOVE the mount at
  # (0, 0, +0.1), so handle radial == mount radial; only x needs pulling in.
  cfg.events["reset_button_position"].params["pose_range"] = {
    "x": (0.44, 0.48),
    "y": (-0.10, 0.10),
    "z": _mech_z("button"),
  }

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

  # Class A grasp envelope: the cuboid must be contacted top-down/side-on by the
  # fingertips, so it obeys the same reachability bound as a graspable object.
  # Spawn takes the near half of the x band and the target the far half, so every
  # episode is a genuine forward push rather than a nudge; y keeps the full spread.
  #
  # SEPARATION BAND. The two halves used to MEET at _x_mid while both spanned the
  # full y range, so a reset could draw the box and its goal within the 2 cm success
  # radius of each other: measured 2/1000 success-at-reset, which fails G5. A 4 cm
  # dead band makes the shortest possible push 0.04 m — twice the success window.
  # (Pre-existing at 1127d12; nothing to do with the asset swap.)
  #
  # The whole dead band comes out of the GOAL side, none of it out of the object
  # band. The object band is the one bound by top-down grasp density
  # (audit_workspace's `sparse-reach` check reads the OBJECT positions); a goal only
  # has to be reachable, which is a much looser bound. Splitting the band evenly
  # costs object-placement diversity for nothing — it measurably tipped Drag-Pull
  # into `sparse-reach` (3.7% -> 2.0% of comfortable top-down poses) before this.
  _SEPARATION = 0.04
  _x_lo, _x_hi = workspace.GRASP_X_RANGE
  _x_mid = (_x_lo + _x_hi) / 2
  push_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(_x_lo, _x_mid),
    y=workspace.GRASP_Y_RANGE,
    z=(CUBOID_HALF_HEIGHT, CUBOID_HALF_HEIGHT),  # rest on the ground plane
    yaw=(0.0, 0.0),  # No rotation - keep upright
  )
  push_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(_x_mid + _SEPARATION, _x_hi),
    y=workspace.GRASP_Y_RANGE,
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

  # Class A grasp envelope, split near/far in x as in the cuboid push so the disc
  # actually has to travel; y keeps the full lateral spread.
  _x_lo, _x_hi = workspace.GRASP_X_RANGE
  _x_mid = (_x_lo + _x_hi) / 2
  push_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(_x_lo, _x_mid),
    y=workspace.GRASP_Y_RANGE,
    z=(0.05, 0.05),  # TEST: Fixed height
    yaw=(0.0, 0.0),  # TEST: NO rotation - keep upright
  )
  push_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(_x_mid, _x_hi),
    y=workspace.GRASP_Y_RANGE,
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


##
# Class A motion-profile expansion (see continual_distill/docs/benchmark/
# CLASS_A_EXPANSION.md). Four new articulation profiles + four new manipulation
# reward/success shapes, all on the same 8-D joint action.
##


def _apply_franka_articulation_common(cfg, asset_name: str, command_name: str):
  """Shared Franka specialization for the new single-DoF articulation tasks.

  Factors out the per-robot wiring that is identical across lever/valve/switch/
  window/lid: EE site names, fingertip friction geoms, collision-sensor patterns,
  viewer body and env spacing.
  """
  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(
    joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg)
  )
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  command = cfg.commands[command_name]
  command.robot_asset_cfg.site_names = ("gripper",)

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params[
        "robot_asset_cfg"
      ].site_names = ("gripper",)

  for reward_name in ["reach_object"]:
    if reward_name in cfg.rewards:
      cfg.rewards[reward_name].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if isinstance(sensor, ContactSensorCfg):
      sensor.primary.pattern = "link7"

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 2.0
  return cfg


def _apply_play_test(cfg, play: bool, test: bool, extra_terminations=()):
  """Shared play/test overrides (verbatim semantics from the existing tasks)."""
  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

  if test:
    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("ee_ground_collision", None)
    for t in extra_terminations:
      cfg.terminations.pop(t, None)
  return cfg


def franka_turn_lever_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka turning a lever about the approach axis (wrist-rotation profile)."""
  cfg = make_turn_lever_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "lever": get_lever_cfg(),
    "mocap_goal": get_lever_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "lever", "turn_lever")

  # Mount placement is chosen so the HANDLE lands inside the workspace, not the mount.
  # lever.xml: object_site sits at (-0.05, +0.12, 0) from lever_base, so the mount is
  # pushed 0.12 to -y and the handle ends up at x 0.45-0.51, y +-0.08
  # (radial <= 0.52, comfortably under workspace.MECHANISM_HANDLE_RADIAL_MAX).
  cfg.events["reset_lever_position"].params["pose_range"] = {
    "x": (0.50, 0.56),
    "y": (-0.20, -0.04),
    "z": _mech_z("lever"),
  }

  return _apply_play_test(cfg, play, test)


def franka_rotate_valve_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka rotating a valve 270 deg (multi-cycle regrasp profile)."""
  cfg = make_rotate_valve_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "valve": get_valve_cfg(),
    "mocap_goal": get_valve_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "valve", "rotate_valve")

  # valve.xml: object_site (spoke tip) at (-0.04, +0.09, 0) from valve_base. Offsetting
  # the mount by -0.09 in y centres the spoke tip on the robot's midline; the spoke
  # sweeps a 0.09 radius circle in the y-z plane, which stays inside the ceiling.
  cfg.events["reset_valve_position"].params["pose_range"] = {
    "x": (0.49, 0.55),
    "y": (-0.17, -0.01),
    "z": _mech_z("valve"),
  }
  # Multi-turn task: needs a longer episode than a single-stroke articulation.
  cfg.episode_length_s = 8.0

  return _apply_play_test(cfg, play, test)


def franka_flip_switch_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka flipping a detented toggle switch (ballistic-commit profile)."""
  cfg = make_flip_switch_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "switch": get_switch_cfg(),
    "mocap_goal": get_switch_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "switch", "flip_switch")

  # switch.xml: object_site is directly ABOVE the mount at (0, 0, +0.062), so the
  # handle radial equals the mount radial and z is raised 6cm. Mount z is kept at the
  # low end of workspace.MECHANISM_Z_RANGE so the toggle tip lands mid-band.
  cfg.events["reset_switch_position"].params["pose_range"] = {
    "x": (0.46, 0.52),
    "y": (-0.08, 0.08),
    "z": _mech_z("switch"),
  }

  return _apply_play_test(cfg, play, test)


def franka_slide_window_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka sliding a window pane laterally (lateral face-push profile)."""
  cfg = make_slide_window_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "window": get_window_cfg(),
    "mocap_goal": get_window_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "window", "slide_window")

  # window.xml: object_site (grab bar) at (-0.04, -0.10, 0) from window_base, so the
  # mount is shifted +0.10 in y to bring the closed-position bar back to the midline.
  # The pane then slides +y (toward the midline and beyond), never further out.
  cfg.events["reset_window_position"].params["pose_range"] = {
    "x": (0.50, 0.56),
    "y": (0.05, 0.15),
    "z": _mech_z("window"),
  }

  return _apply_play_test(cfg, play, test)


def franka_open_lid_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka opening a hinged lid against gravity (vertical-arc profile).

  Unlike the other articulation tasks this runs with gravity ENABLED — the lid falling
  shut when released is the whole point of the task.
  """
  cfg = make_open_lid_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "lid": get_lid_cfg(),
    "mocap_goal": get_lid_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "lid", "open_lid")

  # lid.xml (CL-V2): the lid now LIFTS (hinge axis 0 -1 0), so the grab batten travels
  # from (-0.09, 0, +0.05) at rest to (+0.06, 0, +0.226) at the -75 deg target — i.e.
  # it moves AWAY from the robot and upward. The mount band is pulled in accordingly so
  # the batten stays inside workspace.MECHANISM_HANDLE_RADIAL_MAX (0.58) over the whole
  # arc AND the WRIST stays reachable: the hand sits 0.07 further out than the knob
  # along the approach axis, so at the -75 deg target it is 0.068 m beyond the knob in
  # x. Closed radial 0.31-0.37, fully open 0.46-0.52, wrist <= 0.59. The swept drop fell
  # 0.158 -> 0.070 m (nothing dips below the box any more), so _mech_z drops to 0.10
  # and the chest sits on the floor like the real object it now is.
  cfg.events["reset_lid_position"].params["pose_range"] = {
    "x": (0.40, 0.46),
    "y": (-0.08, 0.08),
    "z": _mech_z("lid"),
  }

  return _apply_play_test(cfg, play, test)


def franka_place_in_container_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka placing a cube inside an open-top bin (containment success shape)."""
  cfg = make_place_in_container_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cube": get_cube_cfg(),
    "container": get_container_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(
    joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg)
  )
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  cfg.commands["place_in_container"].robot_asset_cfg.site_names = ("gripper",)

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params[
        "robot_asset_cfg"
      ].site_names = ("gripper",)
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  # Cube spawns on the near side; the bin sits off to one side so the transport is a
  # real lateral carry rather than a vertical lift.
  place_command = cfg.commands["place_in_container"]
  assert isinstance(place_command, PlaceInContainerCommandCfg)
  # Cube on the -y half of the GRASP_* box, bin on the +y half: a genuine lateral
  # carry across the midline, both inside GRASP_RADIAL_MAX. x runs from
  # GRASP_RADIAL_MIN to 0.48, so the far corner sqrt(0.48^2 + 0.25^2) = 0.541 stays
  # under the 0.55 ceiling.
  # CL-V2 (W1-b): the cube band's far edge moved from -0.04 to -0.07 so that the real
  # basket (183 mm across in y, against the primitive's 140) can stay CLOSE to the
  # robot's centreline instead of being pushed out to make room. Keeping the bin near
  # y=0.10 rather than 0.13 is what preserves the arm's freedom to hold the cube above
  # the bin's interior site, which is the pose the whole task turns on.
  _PLACE_X = (workspace.GRASP_RADIAL_MIN, 0.48)
  place_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=_PLACE_X, y=(workspace.GRASP_Y_RANGE[0], -0.07), z=(0.03, 0.03)
  )
  # The container is a static MOCAP body: nothing resets it, so the command must
  # write it per-env (see PlaceInContainerCommand._resample_command). Before this it
  # kept the MJCF world-frame pose, leaving the bin at the world origin for every env
  # except env 0. The bin only has to be REACHED OVER, never grasped, so it uses a
  # slightly wider x band than GRASP_X_RANGE — that keeps its approach-freedom above
  # the audit's 3% floor.
  # CL-V2 (W1-b): the real basket has a 191 x 183 mm footprint, not the primitive's
  # 140 x 140, so the bands are re-derived from its half-extents (0.0956 x, 0.0913 y):
  #   * y starts at 0.10, so the basket's near wall (0.10 - 0.0913 = 0.009) clears the
  #     cube band's far edge (-0.07 + 0.023 = -0.047) by 56 mm. At the old 0.07 against
  #     a -0.04 cube band the two footprints OVERLAPPED by 4 mm and every reset would
  #     have been a spawn collision.
  #   * x keeps the primitive's 0.28-0.48: the basket's near wall
  #     (0.28 - 0.0956 = 0.184) still stands well clear of the robot base, and the far
  #     corner sqrt(0.48^2 + 0.24^2) = 0.537 respects the ceiling. Narrowing x to
  #     0.30-0.46 was tried and measured WORSE on the audit's approach-freedom metric
  #     (2.7% vs 3.3%, against the audit's 3% floor): the near-x cells are where the
  #     comfortable top-down poses live.
  #   * z = 0 because the basket's body origin is its UNDERSIDE: it sits ON the ground.
  #     (The primitive's z = 0.02 with a floor slab centred on the origin left the old
  #     bin floating 12 mm in the air.)
  place_command.container_spawn_range = _ObjectSpawnRangeCfg(
    x=(0.28, 0.48), y=(0.10, 0.24), z=(0.0, 0.0), yaw=(0.0, 0.0)
  )

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 2.0

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_reorient_object_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka standing a lying cylinder upright (orientation success shape)."""
  cfg = make_reorient_object_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cylinder": get_cylinder_cfg(),
    "mocap_goal": get_cylinder_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(
    joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg)
  )
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  cfg.commands["reorient_object"].robot_asset_cfg.site_names = ("gripper",)

  for obs_name in ["gripper_pos", "gripper_orientation", "gripper_to_object"]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params[
        "robot_asset_cfg"
      ].site_names = ("gripper",)
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  # Spawn the cylinder LYING DOWN (rolled 90 deg about x) so that standing it upright
  # is a genuine reorientation rather than a no-op.
  reorient_command = cfg.commands["reorient_object"]
  assert isinstance(reorient_command, ReorientObjectCommandCfg)
  # Inside the shared Class A grasp envelope. The bottle lies on its side, so at a
  # random yaw its footprint half-diagonal is hypot(half_length, radius) =
  # hypot(0.0266, 0.0150) = 0.0305 m; inset x/y by that so no part of it leaves the
  # envelope regardless of yaw. (The 0.05 this replaces was never a derived value: its
  # comment called 0.05 "the half-length" of a cylinder whose half-length was 0.02.)
  # Corner radial of the resulting box, hypot(0.489, 0.219) = 0.536, is still inside
  # workspace.GRASP_RADIAL_MAX (0.55).
  _pad = 0.031
  reorient_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=(workspace.GRASP_X_RANGE[0] + _pad, workspace.GRASP_X_RANGE[1] - _pad),
    y=(workspace.GRASP_Y_RANGE[0] + _pad, workspace.GRASP_Y_RANGE[1] - _pad),
    # Lying down, the body origin sits on the bottle axis, so the resting height is the
    # collision hull's max radius (0.01501 m); +1 mm so it is not born in contact.
    z=(0.016, 0.016),
    roll=(1.5707963, 1.5707963),
    yaw=(-3.14159, 3.14159),
  )

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 1.5

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_tool_pull_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka dragging an out-of-reach puck into the near zone with a stick (tool use)."""
  cfg = make_tool_pull_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "puck": get_puck_cfg(),
    "stick": get_stick_cfg(),
    "mocap_goal": get_puck_mocap_goal_cfg(),
  }

  joint_pos_action = cfg.actions["robot_joint_pos"]
  assert isinstance(
    joint_pos_action, (JointPositionActionCfg, JointDeltaPositionActionCfg)
  )
  joint_pos_action.scale = FRANKA_ACTION_SCALE

  assert cfg.commands is not None
  cfg.commands["tool_pull"].robot_asset_cfg.site_names = ("gripper",)

  # gripper_to_tool is tool-pull specific: it must be resolved to the "gripper"
  # site like the other EE-relative terms, or its SceneEntityCfg stays empty and
  # the observation build fails on a 0-width tensor.
  for obs_name in [
    "gripper_pos",
    "gripper_orientation",
    "gripper_to_object",
    "gripper_to_tool",
  ]:
    if obs_name in cfg.observations["policy"].terms:
      cfg.observations["policy"].terms[obs_name].params[
        "robot_asset_cfg"
      ].site_names = ("gripper",)
  cfg.rewards["reach_object"].params["robot_asset_cfg"].site_names = ("gripper",)

  fingertip_geoms = r"(left_finger_pad|right_finger_pad)"
  cfg.events["fingertip_friction_slide"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_spin"].params["asset_cfg"].geom_names = fingertip_geoms
  cfg.events["fingertip_friction_roll"].params["asset_cfg"].geom_names = fingertip_geoms

  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = "link7"

  # Puck spawns BEYOND direct reach — the tool is required, not optional.
  # Stick spawns within easy reach, off to the side.
  pull_command = cfg.commands["tool_pull"]
  assert isinstance(pull_command, ToolPullCommandCfg)
  # PUCK — DELIBERATELY OUTSIDE workspace.GRASP_RADIAL_MAX (0.55). Do NOT "fix" this
  # to satisfy the workspace audit: the puck being unreachable by hand is the entire
  # premise of the task, and pulling it inside the grasp envelope would make the tool
  # optional. It is still bounded: radial 0.62-0.71, which the stick's ~0.25m
  # effective extension can cover from a grasp inside the GRASP_* box, and which the
  # goal at x=0.42 is a ~0.24m drag away. (The previous 0.78-0.88 was not bounded by
  # anything: 0.1% approach freedom even WITH the tool.)
  # The puck sits on the +y side and the stick on the -y side so the stick's forward-
  # pointing hook (body +0.12 x, +0.035 y) can never spawn intersecting the puck.
  pull_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=(0.62, 0.69), y=(0.05, 0.17), z=(0.0127, 0.0127), yaw=(0.0, 0.0)
  )
  # STICK — must be grasped, so it obeys the shared GRASP_* envelope. Offset toward
  # -y so it never overlaps the puck (x bands are disjoint anyway) and stays clear of
  # the straight-line drag corridor along y=0. Far corner radial 0.519 < 0.55.
  # The +0.09 offset accounts for the stick's ``object_site`` (the grasp point) sitting
  # 0.09m BEHIND the body origin along the shaft: this puts the SITE, not the body
  # origin, in the 0.28-0.48 band (0.28 == GRASP_RADIAL_MIN, so the near edge is still
  # in front of the base; far corner sqrt(0.48^2 + 0.25^2) = 0.541 < GRASP_RADIAL_MAX).
  pull_command.tool_spawn_range = _ObjectSpawnRangeCfg(
    x=(0.28 + 0.09, 0.48 + 0.09),
    y=(workspace.GRASP_Y_RANGE[0], -0.06),
    z=(0.012, 0.012),  # dowel radius 0.011 + 1 mm, so it is not born in contact
    yaw=(0.0, 0.0),
  )
  # Two-stage task (acquire tool, then drag): needs a longer episode.
  cfg.episode_length_s = 12.0

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 2.0

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


##
# Class A Wave-1 expansion (continual_distill/docs/benchmark/CATALOG_100_TASKS.md,
# T17-T25). ``_apply_franka_articulation_common`` covers the per-robot wiring for
# these tasks too (sites, fingertip geoms, sensor patterns, viewer): despite its
# name it is task-agnostic, so it is reused rather than re-copied.
##


def franka_drag_pull_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka dragging a far cuboid back into the near zone (engagement-inverted push)."""
  cfg = make_drag_pull_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cuboid": get_cuboid_cfg(),
    "mocap_goal": get_cuboid_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "cuboid", "drag_pull")

  drag_command = cfg.commands["drag_pull"]
  assert isinstance(drag_command, PushingCommandCfg)
  # Inverse of push-cuboid: object takes the FAR half of the corner-safe grasp box,
  # the goal the NEAR half, so every episode is a genuine pull back toward the base.
  #
  # ENGAGEMENT INSET. Unlike push, drag engages the object's FAR face — the gripper
  # has to get BEYOND the object before it can pull. The reachability constraint
  # therefore applies to `object_x + cuboid_half_x`, not to the object centre. The
  # corner-safe box is computed for the contact point and the object band is then
  # inset by the same amount; without this the audit passes on the centre (radial
  # 0.537) while the contact point sits at radial 0.571, outside GRASP_RADIAL_MAX.
  _ENGAGE_INSET = CUBOID_HALF_EXTENTS[0]  # 0.0365, the cuboid's x half-extent
  _x_lo_x_hi, _y = _grasp_box_corner_safe()
  _x_lo, _x_hi = _x_lo_x_hi
  _x_hi -= _ENGAGE_INSET
  _x_mid = (_x_lo + _x_hi) / 2
  # SEPARATION BAND, same defect and same fix as push-cuboid (measured 5/1000
  # success-at-reset before this): the object half and the goal half met at _x_mid
  # while both spanned the full y range. Drag-pull's success radius is 0.03, so the
  # band is 0.05 — the shortest legal pull is then 0.05 m.
  #
  # It comes ENTIRELY out of the goal band. This one matters here more than on
  # push-cuboid: drag-pull's object band is already the FAR half of an x range that
  # `_ENGAGE_INSET` has also trimmed, so it sits where comfortable top-down poses are
  # scarcest. Taking half the dead band off it measured 2.0% grasp-pose freedom and
  # tripped audit_workspace's `sparse-reach` flag; taking none off it keeps the
  # audited 3.7%, unchanged from before the fix.
  _SEPARATION = 0.05
  drag_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(_x_mid, _x_hi),
    y=_y,
    z=(CUBOID_HALF_HEIGHT, CUBOID_HALF_HEIGHT),
    yaw=(0.0, 0.0),
  )
  drag_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(_x_lo, _x_mid - _SEPARATION),
    y=_y,
  )

  cfg.scene.env_spacing = 1.5

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_strike_slide_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka striking a puck so it slides to a goal beyond the reach envelope."""
  cfg = make_strike_slide_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "puck": get_puck_cfg(),
    "mocap_goal": get_puck_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "puck", "strike_slide")

  strike_command = cfg.commands["strike_slide"]
  assert isinstance(strike_command, PushingCommandCfg)
  # Puck spawns in the corner-safe grasp box (it must be STRUCK there); the goal
  # band starts past the arm's ~0.85 m absolute stretch — DELIBERATELY outside
  # every workspace envelope. Do not "fix" the goal to satisfy the audit: goals
  # inside reach turn this back into a quasi-static push.
  _x, _y = _grasp_box_corner_safe()
  strike_command.object_pose_range = PushingCommandCfg.ObjectPoseRangeCfg(
    x=(_x[0], _x[1] - 0.06),
    y=(-0.15, 0.15),
    z=(0.0127, 0.0127),  # puck half-thickness (regulation puck is 25.4 mm)
    yaw=(0.0, 0.0),
  )
  strike_command.target_position_range = PushingCommandCfg.TargetPositionRangeCfg(
    x=(0.88, 1.05),
    y=(-0.18, 0.18),
  )

  cfg.scene.env_spacing = 2.5  # The slide corridor extends past 1 m

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_cage_drag_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka transporting a cube caged between open fingers (pinch voids the episode)."""
  cfg = make_cage_drag_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cube": get_cube_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "cube", "cage_drag")

  cage_command = cfg.commands["cage_drag"]
  assert isinstance(cage_command, CageDragCommandCfg)
  # Both spawn and goal live in the corner-safe grasp box: caging transport works
  # in any direction, so unlike push/drag no half-split is imposed.
  _x, _y = _grasp_box_corner_safe()
  # aperture_min (0.055) is a bound on the FINGER JOINT SUM, not on the object: the
  # gap between the pads at aperture 0.055 is ~0.070 m, so the 0.046 m cube cannot
  # force the latch open-side; 24 mm of margin. Re-checked for the new asset.
  cage_command.object_pose_range = CageDragCommandCfg.ObjectPoseRangeCfg(
    x=_x,
    y=(-0.18, 0.18),
    z=(CUBE_HALF_HEIGHT, CUBE_HALF_HEIGHT),
    yaw=(0.0, 0.0),
  )
  cage_command.target_position_range = CageDragCommandCfg.TargetPositionRangeCfg(
    x=_x,
    y=(-0.2, 0.2),
  )

  cfg.scene.env_spacing = 1.5

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_topple_block_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka toppling an ungraspable standing block onto a designated face pair."""
  cfg = make_topple_block_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "block": get_block_cfg(),
    "mocap_goal": get_block_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "block", "topple_block")

  topple_command = cfg.commands["topple_block"]
  assert isinstance(topple_command, ReorientObjectCommandCfg)
  # Standing upright (half-height 0.09). The pad is ASYMMETRIC in x because a topple
  # is asymmetric: the block is poked on its near face and falls AWAY from the robot.
  #
  #   near edge: inset by the yaw-swept horizontal half-extent (0.07) only, so the
  #     poke contact at spawn_x - 0.07 stays clear of GRASP_RADIAL_MIN (0.28).
  #   far edge:  inset by half-extent + the ~0.09 topple travel, so the fallen block
  #     stays inside the out-of-bounds termination box.
  #
  # Symmetric 0.09 padding on both sides collapsed a 22 cm range to a 3.8 cm band
  # (x 0.39-0.43), which is not placement diversity — every episode saw the block in
  # essentially one spot. The near-side 0.09 was pure waste: nothing travels that way.
  # NOTE the bounds below are NOT derived from GRASP_X_RANGE. Nothing in this task is
  # grasped: the block is poked on its near face. The binding constraints are where
  # that CONTACT POINT lands, not where a top-down pinch would.
  _half = BLOCK_HALF_EXTENTS[1]  # 0.08: the larger horizontal half-extent (0.05 x 0.08)
  _travel = BLOCK_HALF_HEIGHT  # 0.105: a topple carries the block ~one half-height
  _poke_near = workspace.GRASP_RADIAL_MIN + _half  # 0.35: contact clear of the base
  _poke_far = 0.44  # contact at 0.36; fallen block reaches 0.625, inside x_bounds
  topple_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=(_poke_near + 0.01, _poke_far),
    y=(workspace.GRASP_Y_RANGE[0] + _half, workspace.GRASP_Y_RANGE[1] - _half),
    z=(BLOCK_HALF_HEIGHT, BLOCK_HALF_HEIGHT),
    yaw=(-3.14159, 3.14159),
  )
  assert _poke_far + _half + _travel < 1.0, "fallen block must stay in x_bounds"

  cfg.scene.env_spacing = 1.5

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_push_flap_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka face-pushing a handle-less flap through its hinge arc."""
  cfg = make_push_flap_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "flap": get_flap_cfg(),
    "mocap_goal": get_flap_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "flap", "push_flap")

  # flap.xml: the push point (object_site) sits 0.18 along +y from the mount, so
  # the mount is shifted -y to centre the panel on the midline. The -70 deg swing
  # carries the contact point ~0.17 further out in x (R(-70deg) @ (0, 0.18)), so
  # the mount x band is pulled IN to (0.44, 0.50) to keep the END of the arc at
  # radial ~0.61-0.68 — inside the side-approach envelope's p90 (0.727), unlike a
  # window-style 0.50-0.56 band whose arc end would leave it entirely.
  cfg.events["reset_flap_position"].params["pose_range"] = {
    "x": (0.44, 0.50),
    "y": (-0.22, -0.12),
    "z": _mech_z("flap"),
  }

  return _apply_play_test(cfg, play, test)


def franka_axial_extract_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka pulling a friction-fit plug vertically out of its socket."""
  cfg = make_axial_extract_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg_neutral(),
    "plug": get_plug_cfg(),
    "mocap_goal": get_plug_mocap_target_cfg(),
  }
  _apply_franka_articulation_common(cfg, "plug", "axial_extract")

  # plug.xml: the pinchable head (object_site) sits directly above the mount at
  # +0.062, and extraction is a TOP-DOWN pinch-and-pull — so unlike the side-on
  # mechanisms this placement obeys the GRASP envelope: head radial <= 0.49 at the
  # box corner, under the 0.55 top-down ceiling. Mount z is floor-clearance
  # dictated (_mech_z), putting the head at ~0.14 — inside the grasp height band.
  cfg.events["reset_plug_position"].params["pose_range"] = {
    "x": (0.40, 0.48),
    "y": (-0.10, 0.10),
    "z": _mech_z("plug"),
  }

  return _apply_play_test(cfg, play, test)


def franka_edge_grasp_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka sliding a plate over the ledge edge and pinching it at the overhang."""
  cfg = make_edge_grasp_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "plate": get_plate_cfg(),
    "ledge": get_ledge_cfg(),
    "mocap_goal": get_plate_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "plate", "edge_grasp")

  # EdgeGraspCommand owns both the ledge placement (static mocap, written per-env)
  # and the plate spawn on its top. Defaults put the ledge at x 0.44-0.50: the
  # plate then lives at x ~0.42-0.54, z ~0.11 — a top-down pinch inside the grasp
  # envelope (corner radial ~0.55 at the widest).
  edge_command = cfg.commands["edge_grasp"]
  assert isinstance(edge_command, EdgeGraspCommandCfg)

  cfg.scene.env_spacing = 2.0

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_pivot_lift_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka pivoting a flat board against the wall, then grasping and lifting it."""
  cfg = make_pivot_lift_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "board": get_board_cfg(),
    "wall": get_wall_cfg(),
    "mocap_goal": get_board_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "board", "pivot_lift")

  # PivotLiftCommand owns the wall placement (static mocap, per-env at x=0.57):
  # just beyond the board band (x 0.42-0.50), so pushing the board +x jams it
  # against the wall and pivots it. Board and lift goal both sit inside the grasp
  # envelope; the wall itself only has to be REACHED AGAINST, never grasped.
  pivot_command = cfg.commands["pivot_lift"]
  assert isinstance(pivot_command, PivotLiftCommandCfg)

  cfg.scene.env_spacing = 2.0

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))


def franka_throw_to_bin_env_cfg(
  play: bool = False,
  test: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Franka tossing a cube into a bin placed beyond the reach envelope.

  Reuses the place-in-container base (same containment + settle success shape);
  the bin band beyond arm stretch is what turns placing into throwing.
  """
  cfg = make_place_in_container_env_cfg()

  cfg.scene.entities = {
    "robot": get_franka_robot_cfg(),
    "cube": get_cube_cfg(),
    "container": get_container_cfg(),
    "mocap_goal": get_mocap_goal_cfg(),
  }
  _apply_franka_articulation_common(cfg, "cube", "place_in_container")

  throw_command = cfg.commands["place_in_container"]
  assert isinstance(throw_command, PlaceInContainerCommandCfg)
  # Cube in the corner-safe grasp box; bin DELIBERATELY beyond the arm's ~0.85 m
  # stretch (radial >= 0.78, up to 0.90). Do not pull it inside the envelope: a
  # reachable bin turns this back into place-in-container. The command's
  # containment + settled predicate is exactly right for a toss — a cube bounced
  # out does not count, a cube still in hand does not count.
  _x, _y = _grasp_box_corner_safe()
  throw_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=_x, y=_y, z=(0.03, 0.03)
  )
  # z = 0: the basket's body origin is its underside, so it sits ON the ground.
  throw_command.container_spawn_range = _ObjectSpawnRangeCfg(
    x=(0.78, 0.90), y=(-0.15, 0.15), z=(0.0, 0.0), yaw=(0.0, 0.0)
  )

  # Room for the flight and for overshoot.
  cfg.terminations["object_out_of_bounds"].params["x_bounds"] = (0.0, 1.4)
  cfg.terminations["object_out_of_bounds"].params["y_bounds"] = (-0.7, 0.7)
  cfg.episode_length_s = 5.0

  cfg.scene.env_spacing = 2.5

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))
