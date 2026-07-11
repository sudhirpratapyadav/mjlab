"""Base config for the stacking task: place a free object on top of a base object.

Adds a higher-fragility pick_place instance to the benchmark: unlike lift (bring an
object to a free-space goal), stacking requires precise placement on top of another
object plus a controlled release, so a small error topples or misses. The goal is
dynamic (tracks the base object). Robot-agnostic; specialized per robot in config/.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  ActionTermCfg,
  CommandTermCfg,
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
from mjlab.tasks.manipulation.mdp import StackingCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_stack_object_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base stacking task configuration (move 'object' onto 'base')."""

  policy_terms = {
    "robot_joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "robot_joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    # Moving object state.
    "object_pos": ObservationTermCfg(
      func=manipulation_mdp.object_position,
      params={"object_asset_name": "object"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_quat": ObservationTermCfg(
      func=manipulation_mdp.object_quaternion,
      params={"object_asset_name": "object"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Base object state (the stacking target reference).
    "base_pos": ObservationTermCfg(
      func=manipulation_mdp.object_position,
      params={"object_asset_name": "base"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Gripper state.
    "gripper_pos": ObservationTermCfg(
      func=manipulation_mdp.gripper_position,
      params={"robot_asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "gripper_orientation": ObservationTermCfg(
      func=manipulation_mdp.gripper_orientation,
      params={"robot_asset_cfg": SceneEntityCfg("robot", site_names=())},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Relative task signals.
    "gripper_to_object": ObservationTermCfg(
      func=manipulation_mdp.gripper_to_object_vector,
      params={
        "object_asset_name": "object",
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "object_to_goal": ObservationTermCfg(
      func=manipulation_mdp.object_to_goal_vector,
      params={"command_name": "stack_object", "object_asset_name": "object"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "control_qpos_diff": ObservationTermCfg(
      func=manipulation_mdp.control_qpos_difference,
      params={"robot_asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
  }

  observations = {
    "policy": ObservationGroupCfg(policy_terms, enable_corruption=True),
    "critic": ObservationGroupCfg({**policy_terms}, enable_corruption=False),
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
    "stack_object": StackingCommandCfg(
      asset_name="object",
      base_asset_name="base",
      robot_asset_cfg=SceneEntityCfg("robot", site_names=()),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
    )
  }

  events = {
    "reset_base": EventTermCfg(
      func=mdp.reset_root_state_uniform,
      mode="startup",
      params={"pose_range": {}, "velocity_range": {}},
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
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "field": "geom_friction",
        "distribution": "uniform",
        "axes": [0],
        "ranges": (0.3, 1.5),
      },
    ),
  }

  ee_ground_collision_cfg = ContactSensorCfg(
    name="ee_ground_collision",
    primary=ContactMatch(mode="subtree", pattern="", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  rewards = {
    # Reach the moving object, then bring it to the (dynamic) stack target.
    "stack": RewardTermCfg(
      func=manipulation_mdp.staged_manipulation_reward,
      weight=1.0,
      params={
        "command_name": "stack_object",
        "object_asset_name": "object",
        "reaching_max_dist": 0.35,
        "bringing_max_dist": 0.35,
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
    ),
    "at_goal_bonus": RewardTermCfg(
      func=manipulation_mdp.object_at_goal_reward,
      weight=1.0,
      params={
        "command_name": "stack_object",
        "object_asset_name": "object",
        "max_dist": 0.35,
      },
    ),
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
      params={"object_name": "object", "x_bounds": (0.0, 1.0), "y_bounds": (-0.5, 0.5)},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.5,
      sensors=(ee_ground_collision_cfg,),
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      asset_name="robot",
      body_name="",
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
    episode_length_s=20.0,
  )
