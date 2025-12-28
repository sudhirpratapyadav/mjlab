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
from mjlab.tasks.manipulation.mdp import OpenDoorCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_open_door_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base door opening task configuration."""

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
      params={"object_asset_name": "door"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_quat": ObservationTermCfg(
      func=manipulation_mdp.object_quaternion,
      params={"object_asset_name": "door"},
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
      params={"object_asset_name": "door"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Relative vectors (3 + 3 = 6 dims)
    "gripper_to_object": ObservationTermCfg(
      func=manipulation_mdp.gripper_to_object_vector,
      params={
        "object_asset_name": "door",
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_to_goal": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_vector,
      params={
        "command_name": "open_door",
        "object_asset_name": "door",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Goal orientation difference (6 dims)
    "goal_orientation_diff": ObservationTermCfg(
      func=manipulation_mdp.goal_orientation_diff,
      params={
        "command_name": "open_door",
        "object_asset_name": "door",
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
    "open_door": OpenDoorCommandCfg(
      asset_name="door",
      resampling_time_range=(10.0, 15.0),
      debug_vis=True,
      difficulty="fixed",
      # Base [0.68, 0.0, 0.61] + offset ranges
      door_pose_range=OpenDoorCommandCfg.DoorPoseRangeCfg(
        x=(0.66, 0.70),  # 0.68 + [-0.02, 0.02]
        y=(-0.1, 0.0),   # 0.0 + [-0.1, 0.0]
        z=(0.61, 0.61),  # 0.61 + [0.0, 0.0]
        yaw=(0.0, 0.0),  # No yaw randomization
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
    # Robot arm perturbation
    # Uses ±10 degrees offset
    "reset_robot_joints": EventTermCfg(
      func=mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.174533, 0.174533),  # ±10 degrees in radians
        "velocity_range": (0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),
    # Randomize door position
    # Base pos [0.68, 0, 0.61] + offsets: X: ±2cm, Y: -10cm to 0cm, Z: no offset
    "reset_door_position": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.66, 0.70),  # 0.68 ± 0.02
          "y": (-0.1, 0.0),   # 0.0 + [-0.1, 0.0]
          "z": (0.61, 0.61),  # No vertical randomization
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("door"),
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
      pattern="",  # Set per-robot (e.g., "link7" for Franka).
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  # Collision sensor for end-effector to door body contact.
  # Matches reference XML line 20-22 (door_body is the barrier geom)
  ee_door_collision_cfg = ContactSensorCfg(
    name="ee_door_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="",  # Set per-robot (e.g., "link7" for Franka).
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern="door_body",  # Barrier geom, not door_panel
      entity="door",
    ),
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
        "command_name": "open_door",
        "object_asset_name": "door",
        "reaching_max_dist": 0.65,
        "bringing_max_dist": 0.6,
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot
      },
    ),
    # Phase 2: Move object to goal
    "move_object_to_goal": RewardTermCfg(
      func=manipulation_mdp.object_at_goal_reward,
      weight=1.0,
      params={
        "command_name": "open_door",
        "object_asset_name": "door",
        "max_dist": 0.6,
      },
    ),
    # No collision with door body
    "no_object_collision": RewardTermCfg(
      func=manipulation_mdp.no_object_body_collision_reward,
      weight=0.25,
      params={"sensor_name": "ee_door_collision"},
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
      sensors=(ee_ground_collision_cfg, ee_door_collision_cfg),
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
      distance=2.0,
      elevation=-10.0,
      azimuth=135.0,
    ),
    sim=SimulationCfg(
      nconmax=60,
      njmax=650,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
        gravity=(0.0, 0.0, 0.0),  # DISABLED for debugging
      ),
    ),
    decimation=4,
    episode_length_s=3.0,  # Matches mujoco_playground: 150 steps at 0.02 ctrl_dt
  )
