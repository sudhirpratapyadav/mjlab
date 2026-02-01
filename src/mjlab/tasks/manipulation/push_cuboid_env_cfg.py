from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
  CurriculumTermCfg,
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import PushingCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_push_cuboid_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base cuboid pushing task configuration."""

  policy_terms = {
    # Robot state (9 + 9 = 18 dims)
    "robot_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "robot_joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Object state (3 + 4 = 7 dims)
    "object_pos": ObservationTermCfg(
      func=manipulation_mdp.object_position,
      params={"object_asset_name": "cuboid"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_quat": ObservationTermCfg(
      func=manipulation_mdp.object_quaternion,
      params={"object_asset_name": "cuboid"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Gripper state (3 + 6 = 9 dims)
    "gripper_pos": ObservationTermCfg(
      func=manipulation_mdp.gripper_position,
      params={
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "gripper_orientation": ObservationTermCfg(
      func=manipulation_mdp.gripper_orientation,
      params={
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Object body orientation (6 dims)
    "object_orientation": ObservationTermCfg(
      func=manipulation_mdp.object_orientation,
      params={"object_asset_name": "cuboid"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Relative vectors (3 + 3 = 6 dims)
    "gripper_to_object": ObservationTermCfg(
      func=manipulation_mdp.gripper_to_object_vector,
      params={
        "object_asset_name": "cuboid",
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_to_goal": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_vector,
      params={
        "command_name": "push_cuboid",
        "object_asset_name": "cuboid",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Target orientation difference (6 dims)
    "goal_orientation_diff": ObservationTermCfg(
      func=manipulation_mdp.goal_orientation_diff,
      params={
        "command_name": "push_cuboid",
        "object_asset_name": "cuboid",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Control-qpos difference (8 dims)
    "control_qpos_diff": ObservationTermCfg(
      func=manipulation_mdp.control_qpos_difference,
      params={
        "robot_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
  }
  # Total: 18 + 7 + 9 + 6 + 6 + 6 + 8 = 60 dims

  critic_terms = {**policy_terms}

  observations = {
    "policy": ObservationGroupCfg(policy_terms, enable_corruption=True),
    "critic": ObservationGroupCfg(critic_terms, enable_corruption=False),
  }

  actions: dict[str, ActionTermCfg] = {
    "robot_joint_pos": JointPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=True,
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "push_cuboid": PushingCommandCfg(
      asset_name="cuboid",
      robot_asset_cfg=SceneEntityCfg("robot", site_names=()),  # Set per-robot
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
      success_threshold=0.02,  # 2cm threshold for success
      goal_z_height=0.015,  # Goals at ground level (cuboid half-height)
      object_pose_range=PushingCommandCfg.ObjectPoseRangeCfg(
        x=(0.2, 0.4),
        y=(-0.2, 0.2),
        z=(0.015, 0.015),  # Cuboid half-height is 0.015, spawn at ground level
        yaw=(-3.14, 3.14),
      ),
    )
  }

  events = {
    # For positioning the base of the robot at env_origins.
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="startup",
      params={
        "pose_range": {},
        "velocity_range": {},
      },
    ),
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (0.0, 0.0),
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    "fingertip_friction_slide": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.3, 1.5),
      },
    ),
    "fingertip_friction_spin": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "distribution": "log_uniform",
        "axes": [1],
        "ranges": (1e-4, 2e-2),
      },
    ),
    "fingertip_friction_roll": EventTermCfg(
      mode="startup",
      func=mdp.randomize_field,
      domain_randomization=True,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # Set per-robot.
        "operation": "abs",
        "field": "geom_friction",
        "distribution": "log_uniform",
        "axes": [2],
        "ranges": (1e-5, 5e-3),
      },
    ),
  }

  # Collision sensor for end-effector to ground contact.
  ee_ground_collision_cfg = ContactSensorCfg(
    name="ee_ground_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="",  # Set per-robot (e.g., "link_6" for all geoms in end-effector).
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  rewards = {
    # Phase 1: Reach object
    "reach_object": RewardTermCfg(
      func=manipulation_mdp.staged_manipulation_reward,
      weight=1.0,
      params={
        "command_name": "push_cuboid",
        "object_asset_name": "cuboid",
        "reaching_max_dist": 0.35,
        "bringing_max_dist": 0.2,
        "reaching_clip_dist": 0.05,  # Clip reward at 5cm - reward maxes out below this distance
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
    ),
    # Phase 2: Move object to goal
    "move_object_to_goal": RewardTermCfg(
      func=manipulation_mdp.object_at_goal_reward,
      weight=1.0,
      params={
        "command_name": "push_cuboid",
        "object_asset_name": "cuboid",
        "max_dist": 0.35,
      },
    ),
    # Regularization
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "joint_pos_limits": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    "joint_vel_penalty": RewardTermCfg(
      func=manipulation_mdp.joint_velocity_penalty,
      weight=-0.01,
      params={
        "max_vel": 0.5,
        "robot_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "ee_ground_collision": TerminationTermCfg(
      func=manipulation_mdp.illegal_contact,
      params={"sensor_name": "ee_ground_collision"},
    ),
    "object_out_of_bounds": TerminationTermCfg(
      func=manipulation_mdp.object_out_of_bounds,
      params={
        "object_name": "cuboid",
        "x_bounds": (0.0, 1.0),
        "y_bounds": (-0.5, 0.5),
      },
    ),
  }

  curriculum = {
    "joint_vel_penalty_weight": CurriculumTermCfg(
      func=manipulation_mdp.reward_weight,
      params={
        "reward_name": "joint_vel_penalty",
        "weight_stages": [
          {"step": 0, "weight": -0.01},
          {"step": 1000 * 24, "weight": -0.1},
          {"step": 1500 * 24, "weight": -1.0},
        ],
      },
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      sensors=(ee_ground_collision_cfg,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name="",  # Set per-robot.
      distance=1.5,
      elevation=-5.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=200,
      njmax=1000,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=4,
    episode_length_s=3.0,
  )
