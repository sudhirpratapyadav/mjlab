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
  PlaceInContainerCommandCfg,
  ReorientObjectCommandCfg,
  ToolPullCommandCfg,
)
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

  # NOTE: these are the PRE-workspace-audit ranges, restored for the same reason as
  # the four tasks in a0cc9be — Lift-Cube is the 5th continual-distill task (P1-6) and
  # its RL teacher predates the audit. Measured on HEAD with the audited ranges the
  # teacher scores 0.023 (128 episodes); these are the ranges it was trained against.
  # obs_dim is unchanged either way, so the failure is silent rather than an error.
  # See docs/P0_EXPERIMENTS.md. Restore the audited ranges (_grasp_box_corner_safe /
  # workspace.GOAL_*) only together with a retrained teacher.
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

  # Same spawn/target ranges as the cube lift: the shared Class A grasp envelope.
  _obj_x, _obj_y = _grasp_box_corner_safe()
  lift_command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=_obj_x,
    y=_obj_y,
    z=(0.02, 0.05),
    yaw=(-3.14, 3.14),
  )
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=workspace.GOAL_X_RANGE,
    y=workspace.GOAL_Y_RANGE,
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
  lift_command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=workspace.GOAL_X_RANGE, y=workspace.GOAL_Y_RANGE, z=workspace.GOAL_Z_RANGE,
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

  # cube half-height 0.02 + cuboid half-height 0.015 => stack offset 0.035.
  assert cfg.commands is not None
  stack_command = cfg.commands["stack_object"]
  assert isinstance(stack_command, StackingCommandCfg)
  stack_command.robot_asset_cfg.site_names = ("gripper",)
  stack_command.stack_height = 0.035

  # Two objects that must not overlap: split the shared GRASP_* box laterally, one
  # either side of y=0. Each half keeps the FULL |y| extent of GRASP_Y_RANGE (that is
  # what makes the two objects' relative bearing vary), so x is squeezed instead: it
  # runs from GRASP_RADIAL_MIN out to 0.47, short of GRASP_X_RANGE's 0.52, putting the
  # far corner at sqrt(0.47^2 + 0.25^2) = 0.532 — inside GRASP_RADIAL_MAX (0.55).
  _STACK_X = (workspace.GRASP_RADIAL_MIN, 0.47)
  stack_command.object_pose_range = StackingCommandCfg.ObjectPoseRangeCfg(
    x=_STACK_X,
    y=(workspace.GRASP_Y_RANGE[0], -0.04),  # cube: right half
    z=(0.02, 0.02),  # cube half-height, resting on the ground plane
    yaw=(0.0, 0.0),
  )
  stack_command.base_pose_range = StackingCommandCfg.BasePoseRangeCfg(
    x=_STACK_X,
    y=(0.04, workspace.GRASP_Y_RANGE[1]),  # cuboid base: left half
    z=(0.015, 0.015),  # cuboid half-height
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
  # Peg tip (object_site) should reach the hole opening (board top ~+0.03 above the
  # board body origin at z=0.015 => opening ~0.015 above origin). Aim the peg site
  # slightly INTO the hole for an inserted pose.
  stack_command.stack_height = 0.01
  stack_command.success_threshold = 0.015  # tight xy alignment for insertion
  stack_command.height_threshold = 0.03

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

  # NOTE: the workspace-audit placement override (`reset_door_position` pose_range,
  # added in 2ab6d11) is deliberately NOT applied here. The continual-distill RL
  # teachers for this task were trained against the pre-audit placement, and moving
  # the mount silently drops them to ~0 success (they still see obs_dim=60, so it
  # fails silently rather than erroring). See docs/P0_EXPERIMENTS.md. Re-enable this
  # only together with retrained teachers.

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

  # NOTE: workspace-audit placement override intentionally not applied — see the
  # note in franka_open_door_env_cfg and docs/P0_EXPERIMENTS.md. The RL teacher for
  # this task predates it.

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

  # NOTE: workspace-audit placement override intentionally not applied — see the
  # note in franka_open_door_env_cfg and docs/P0_EXPERIMENTS.md. The RL teacher for
  # this task predates it.

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

  # Override object and target ranges for Franka.
  #
  # NOTE: these are the PRE-workspace-audit ranges. 2ab6d11 replaced them with
  # workspace.GRASP_X_RANGE halves (spawn near / target far); that moves the cuboid
  # off the distribution the continual-distill RL teacher was trained on and drops
  # it from ~0.83 to ~0.008 success. obs_dim stays 60, so the failure is silent.
  # See docs/P0_EXPERIMENTS.md. Restore the audit ranges only with a retrained teacher.
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

  # lid.xml: the grab lip (object_site) is at (-0.09, 0, +0.05) from lid_base — a deep
  # -x protrusion, so the mount may sit further out than the others. The box is a
  # mocap-mounted body (not floor-standing), but its z is what sets the lift arc, so
  # the original height band is preserved; only x is pulled in.
  cfg.events["reset_lid_position"].params["pose_range"] = {
    "x": (0.52, 0.59),
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
  _PLACE_X = (workspace.GRASP_RADIAL_MIN, 0.48)
  place_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=_PLACE_X, y=(workspace.GRASP_Y_RANGE[0], -0.04), z=(0.03, 0.03)
  )
  # The container is a static MOCAP body: nothing resets it, so the command must
  # write it per-env (see PlaceInContainerCommand._resample_command). Before this it
  # kept the MJCF world-frame pose, leaving the bin at the world origin for every env
  # except env 0. The +y band leaves >=0.14 of clearance to the cube band, comfortably
  # more than the bin's 0.078 half-width.
  # The bin only has to be REACHED OVER, never grasped, so it uses a slightly wider x
  # band than GRASP_X_RANGE (starting at 0.28) — that keeps its approach-freedom
  # comfortably above the audit's 3% floor while its far corner,
  # sqrt(0.48^2 + 0.24^2) = 0.537, still clears GRASP_RADIAL_MAX.
  place_command.container_spawn_range = _ObjectSpawnRangeCfg(
    x=(0.28, 0.48), y=(0.07, 0.24), z=(0.02, 0.02), yaw=(0.0, 0.0)
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
  # Inside the shared Class A grasp envelope. The cylinder lies on its side, so its
  # body extends ~0.05 either side of the spawn point along a random yaw; inset x/y by
  # that half-length so no part of it leaves the envelope regardless of yaw.
  _pad = 0.05
  reorient_command.object_spawn_range = _ObjectSpawnRangeCfg(
    x=(workspace.GRASP_X_RANGE[0] + _pad, workspace.GRASP_X_RANGE[1] - _pad),
    y=(workspace.GRASP_Y_RANGE[0] + _pad, workspace.GRASP_Y_RANGE[1] - _pad),
    z=(0.025, 0.025),
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
    x=(0.62, 0.69), y=(0.05, 0.17), z=(0.012, 0.012), yaw=(0.0, 0.0)
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
    z=(0.012, 0.012),
    yaw=(0.0, 0.0),
  )
  # Two-stage task (acquire tool, then drag): needs a longer episode.
  cfg.episode_length_s = 12.0

  cfg.viewer.body_name = "link0"
  cfg.scene.env_spacing = 2.0

  return _apply_play_test(cfg, play, test, extra_terminations=("object_out_of_bounds",))
