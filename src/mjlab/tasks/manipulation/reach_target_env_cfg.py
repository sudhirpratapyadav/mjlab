"""Base config for the end-effector reaching task (no object).

The reach task adds the ``reach`` skill family to the benchmark: move the gripper to a
sampled 3D target in the workspace. It is deliberately the simplest skill — a dense,
monotonic distance reward with no contact — so it anchors the low-fragility end of the
axis. Robot-agnostic; specialized per robot in ``config/<robot>/env_cfgs.py``.
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
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.mdp import ReachingCommandCfg
from mjlab.tasks.velocity import mdp
from mjlab.terrains import TerrainImporterCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig


def make_reach_target_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the base end-effector reaching task configuration."""

  policy_terms = {
    # Robot state (9 + 9 = 18 dims)
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
    # Gripper state (3 + 6 = 9 dims)
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
    # Task signal: gripper -> target vector (3 dims)
    "gripper_to_target": ObservationTermCfg(
      func=manipulation_mdp.gripper_to_target_vector,
      params={
        "command_name": "reach_target",
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),
      },
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    # Control-qpos difference (8 dims)
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
    "reach_target": ReachingCommandCfg(
      robot_asset_cfg=SceneEntityCfg("robot", site_names=()),
      resampling_time_range=(8.0, 12.0),
      debug_vis=True,
      difficulty="dynamic",
      success_threshold=0.05,
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
  }

  rewards = {
    "reach_target": RewardTermCfg(
      func=manipulation_mdp.reach_target_reward,
      weight=1.0,
      params={
        "command_name": "reach_target",
        "robot_asset_cfg": SceneEntityCfg("robot", site_names=()),
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
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainImporterCfg(terrain_type="plane"),
      num_envs=1,
      env_spacing=1.5,
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
    episode_length_s=20.0,
  )
