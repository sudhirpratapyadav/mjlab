from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_manipulation_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_name: str = "object",
  reaching_std: float = 0.6,
  bringing_std: float = 0.86,
  reaching_max_dist: float = 1.0,
  bringing_max_dist: float = 1.0,
  reaching_clip_dist: float = 0.0,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward that gates manipulation bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before manipulation.

  Args:
    reaching_clip_dist: Minimum distance for reward computation. When distance
      is below this threshold, reward maxes out (clips). Default 0.0 (no clipping).

  Works for all manipulation tasks (free and articulated objects).
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  obj: Entity = env.scene[object_asset_name]
  command = env.command_manager.get_term(command_name)

  # Get gripper and object positions
  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids].squeeze(1)

  # Get object position (object_site or root link)
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    object_pos_w = obj.data.root_link_pos_w

  # Reaching phase: gripper to object
  # Compute distance and clip to minimum threshold
  reach_distance = torch.norm(gripper_pos_w - object_pos_w, dim=-1)
  reach_distance_clamped = torch.clamp(reach_distance, min=reaching_clip_dist)
  reach_error = reach_distance_clamped ** 2
  reaching = torch.exp(-reach_error / (reaching_std * reaching_max_dist)**2)

  # Manipulation phase: object to goal
  position_error = torch.sum(torch.square(command.target_pos - object_pos_w), dim=-1)
  bringing = torch.exp(-position_error / (bringing_std * bringing_max_dist)**2)

  return reaching * (1.0 + bringing)


def object_at_goal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_name: str = "object",
  std: float = 0.14,
  max_dist: float = 1.0,
) -> torch.Tensor:
  """Precise reward for object reaching goal position.

  Works for all manipulation tasks using Gaussian kernel.
  """
  obj: Entity = env.scene[object_asset_name]
  command = env.command_manager.get_term(command_name)

  # Get object position
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    object_pos_w = obj.data.root_link_pos_w

  position_error = torch.sum(
    torch.square(command.target_pos - object_pos_w), dim=-1
  )
  return torch.exp(-position_error / (std * max_dist)**2)


def joint_velocity_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel.
  Returns the squared L2 norm of the excess velocities (positive value, use negative weight).
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, robot_asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


def reach_object_reward(
  env: ManagerBasedRlEnv,
  object_asset_name: str = "object",
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  k: float = 7.0,
) -> torch.Tensor:
  """Reward for gripper approaching object.

  Uses tanh-based reward: 1 - tanh(k * ||object_pos - gripper_pos||^4)
  Works for both free and articulated objects.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  obj: Entity = env.scene[object_asset_name]

  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids].squeeze(1)

  # Get object position
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    object_pos_w = obj.data.root_link_pos_w

  distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
  reward = 1.0 - torch.tanh(k * distance**4)
  return reward


def move_object_to_goal_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_name: str = "object",
  k: float = 70.0,
  use_gating: bool = True,
) -> torch.Tensor:
  """Reward for moving object to goal position.

  Uses tanh-based reward: (1 - tanh(k * ||goal_pos - object_pos||^4)) * reached_object
  where reached_object is latched once gripper is within threshold of object.

  Args:
    env: Environment
    command_name: Name of the command
    object_asset_name: Name of the object asset
    k: Steepness parameter for tanh (default: 70.0)
    use_gating: If True, gate reward by reached_object latch (default: True)
  """
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_asset_name]

  # Get object position
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    object_pos_w = obj.data.root_link_pos_w

  # Compute object-to-goal distance reward
  goal_pos = command.target_pos
  distance = torch.norm(goal_pos - object_pos_w, dim=-1)
  reward = 1.0 - torch.tanh(k * distance**4)

  # Gate by reached_object if available and use_gating is True
  if use_gating and hasattr(command, "reached_object"):
    reward = reward * command.reached_object

  return reward


def no_object_body_collision_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward for not colliding with object body.

  Returns 1.0 if no collision, 0.0 if collision.
  Works for all articulated objects (door/drawer/button bodies).
  """
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Return 1.0 if no collision, 0.0 if collision
  collision = torch.any(sensor.data.found, dim=-1).float()
  return 1.0 - collision


def robot_init_pose_reward(
  env: ManagerBasedRlEnv,
  init_joint_pos: torch.Tensor,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for robot staying close to initial joint configuration.

  Uses: 1 - tanh(||q - q_init||)
  where q is current joint position and q_init is initial position.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  current_qpos = robot.data.joint_pos[:, robot_asset_cfg.joint_ids]

  # Compute L2 norm of joint position difference
  qpos_diff = torch.norm(current_qpos - init_joint_pos, dim=-1)
  reward = 1.0 - torch.tanh(qpos_diff)
  return reward


def reach_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  std: float = 0.1,
) -> torch.Tensor:
  """Reward for the gripper site reaching the reach-command target.

  Gaussian kernel on the gripper-to-target distance: exp(-||target - gripper||^2 / std^2).
  Monotonic in distance (dense, well-shaped, no local optima), so the reach task is
  easy to learn — it exists to add the ``reach`` skill family, not to be hard.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  command = env.command_manager.get_term(command_name)
  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids].squeeze(1)
  position_error = torch.sum(
    torch.square(command.target_pos - gripper_pos_w), dim=-1
  )
  return torch.exp(-position_error / (std**2))
