from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointDeltaPositionActionCfg
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
from mjlab.tasks.manipulation.mdp import OpenDrawerCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_open_drawer_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create base drawer opening task configuration."""

  policy_terms = {
    # Robot state (9 + 9 = 18 dims)
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Handle state (3 + 4 = 7 dims)
    "handle_pos": ObservationTermCfg(
      func=manipulation_mdp.handle_geom_position,
      params={"asset_name": "drawer"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "handle_quat": ObservationTermCfg(
      func=manipulation_mdp.handle_body_quaternion,
      params={"asset_name": "drawer"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Gripper state (3 + 6 = 9 dims)
    # Note: site_names set per-robot in Franka config
    "gripper_pos": ObservationTermCfg(
      func=manipulation_mdp.gripper_position,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "gripper_orientation": ObservationTermCfg(
      func=manipulation_mdp.gripper_orientation,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Handle body orientation (6 dims)
    "handle_body_orientation": ObservationTermCfg(
      func=manipulation_mdp.handle_body_orientation,
      params={"asset_name": "drawer"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Relative vectors (3 + 3 = 6 dims)
    # Note: site_names set per-robot in Franka config
    "gripper_to_handle": ObservationTermCfg(
      func=manipulation_mdp.gripper_to_handle_vector,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),
        "articulated_asset_name": "drawer",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "target_to_handle": ObservationTermCfg(
      func=manipulation_mdp.target_to_handle_vector,
      params={
        "command_name": "open_drawer",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Target orientation difference (6 dims)
    "target_orientation_diff": ObservationTermCfg(
      func=manipulation_mdp.target_orientation_diff,
      params={
        "command_name": "open_drawer",
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Control-qpos difference (8 dims)
    "control_qpos_diff": ObservationTermCfg(
      func=manipulation_mdp.control_qpos_difference,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
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
    "joint_pos": JointDeltaPositionActionCfg(
      asset_name="robot",
      actuator_names=(".*",),
      scale=0.04,  # Matches mujoco_playground action_scale
      offset=0.0,  # No offset for delta control
    )
  }

  commands: dict[str, CommandTermCfg] = {
    "open_drawer": OpenDrawerCommandCfg(
      asset_name="drawer",
      resampling_time_range=(10.0, 15.0),
      debug_vis=True,
      difficulty="fixed",
      # Base [0.65, 0.0, 0.5] + offset ranges
      drawer_pose_range=OpenDrawerCommandCfg.DrawerPoseRangeCfg(
        x=(0.5, 0.7),  # Reduced range for testing
        y=(0.0, 0.0),    # No Y randomization
        z=(0.5, 0.5),    # 0.5 + [0.0, 0.0]
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
    # Randomize drawer position 
    "reset_drawer_position": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (0.50, 0.70),  # 20cm X randomization
          "y": (-0.1, 0.1),   # 20cm Y randomization (±10cm)
          "z": (0.5, 0.5),    # No vertical randomization
        },
        "velocity_range": {},
        "asset_cfg": SceneEntityCfg("drawer"),
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

  # Collision sensor for end-effector to drawer body contact.
  # Matches drawer_body (the cabinet geom)
  ee_drawer_collision_cfg = ContactSensorCfg(
    name="ee_drawer_collision",
    primary=ContactMatch(
      mode="subtree",
      pattern="",  # Set per-robot (e.g., "link7" for Franka).
      entity="robot",
    ),
    secondary=ContactMatch(
      mode="geom",
      pattern="drawer_body",  # Cabinet geom, not drawer_panel
      entity="drawer",
    ),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  rewards = {
    # Gripper goes to handle
    "gripper_to_handle": RewardTermCfg(
      func=manipulation_mdp.gripper_to_handle_reward,
      weight=4.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # Set per-robot.
        "articulated_asset_name": "drawer",
      },
    ),
    # Handle goes to target
    "handle_to_target": RewardTermCfg(
      func=manipulation_mdp.handle_to_target_reward,
      weight=8.0,
      params={
        "command_name": "open_drawer",
      },
    ),
    # No collision with drawer body
    "no_drawer_collision": RewardTermCfg(
      func=manipulation_mdp.no_door_body_collision_reward,
      weight=0.25,
      params={"sensor_name": "ee_drawer_collision"},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "ee_ground_collision": TerminationTermCfg(
      func=manipulation_mdp.illegal_contact,
      params={"sensor_name": "ee_ground_collision"},
    ),
  }

  curriculum = {}  # No curriculum

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.0,
      sensors=(ee_ground_collision_cfg, ee_drawer_collision_cfg),
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
