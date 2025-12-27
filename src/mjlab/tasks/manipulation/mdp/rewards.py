from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand, OpenDoorCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def staged_position_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  reaching_std: float,
  bringing_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Curriculum reward that gates lifting bonus on reaching progress.

  Returns reaching * (1 + bringing), where both terms are Gaussian kernels
  over position error. Ensures learning signal for approach before lift.
  """
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  reach_error = torch.sum(torch.square(ee_pos_w - obj_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)
  position_error = torch.sum(torch.square(command.target_pos - obj_pos_w), dim=-1)
  bringing = torch.exp(-position_error / bringing_std**2)
  return reaching * (1.0 + bringing)


def bring_object_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_name: str,
  std: float,
) -> torch.Tensor:
  obj: Entity = env.scene[object_name]
  command = cast(LiftingCommand, env.command_manager.get_term(command_name))
  position_error = torch.sum(
    torch.square(command.target_pos - obj.data.root_link_pos_w), dim=-1
  )
  return torch.exp(-position_error / std**2)


def joint_velocity_hinge_penalty(
  env: ManagerBasedRlEnv,
  max_vel: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Quadratic hinge penalty on joint velocities exceeding a symmetric limit.

  Penalizes only the amount by which |v| exceeds max_vel. Returns a negative
  penalty, shaped as the negative squared L2 norm of the excess velocities.
  """
  robot: Entity = env.scene[asset_cfg.name]
  joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
  excess = (joint_vel.abs() - max_vel).clamp_min(0.0)
  return (excess**2).sum(dim=-1)


def staged_door_opening_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  reaching_std: float,
  opening_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Staged reward for door opening: reaching * (1 + opening).

  Stages:
  1. Reaching: Gaussian kernel on distance from EE to handle
  2. Opening: Gaussian kernel on door angle error (gated by reaching)

  This ensures robot learns to reach handle before attempting to open.
  """
  robot: Entity = env.scene[asset_cfg.name]
  door: Entity = env.scene["door"]
  command = cast(OpenDoorCommand, env.command_manager.get_term(command_name))

  # Get end-effector and handle positions
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  handle_site_idx = door.site_names.index("handle_site")
  handle_pos_w = door.data.site_pos_w[:, handle_site_idx]

  # Reaching reward (distance to handle)
  reach_error = torch.sum(torch.square(ee_pos_w - handle_pos_w), dim=-1)
  reaching = torch.exp(-reach_error / reaching_std**2)

  # Opening reward (door angle error)
  hinge_idx = door.joint_names.index("door_hinge")
  door_angle = door.data.joint_pos[:, hinge_idx]
  angle_error = torch.square(command.target_angle - door_angle)
  opening = torch.exp(-angle_error / opening_std**2)

  # Combined staged reward
  return reaching * (1.0 + opening)


def door_opening_angle_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Precise reward for achieving target door angle."""
  door: Entity = env.scene["door"]
  command = cast(OpenDoorCommand, env.command_manager.get_term(command_name))

  hinge_idx = door.joint_names.index("door_hinge")
  door_angle = door.data.joint_pos[:, hinge_idx]
  angle_error = torch.square(command.target_angle - door_angle)

  return torch.exp(-angle_error / std**2)


def contact_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalty for contacts (e.g., hitting door panel)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Return negative value (penalty) when contact detected
  return -torch.any(sensor.data.found, dim=-1).float()


def gripper_to_handle_reward(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  articulated_asset_name: str = "door",
) -> torch.Tensor:
  """Reward for gripper approaching handle (mujoco_playground style).

  Uses tanh-based reward: 1 - tanh(7 * ||handle_pos - gripper_pos||^4)
  """
  robot: Entity = env.scene[asset_cfg.name]
  articulated_obj: Entity = env.scene[articulated_asset_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  handle_site_idx = articulated_obj.site_names.index("handle_site")
  handle_pos_w = articulated_obj.data.site_pos_w[:, handle_site_idx]

  distance = torch.norm(handle_pos_w - ee_pos_w, dim=-1)
  reward = 1.0 - torch.tanh(7.0 * distance**4)
  return reward


def handle_to_target_reward(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Reward for handle reaching target position, gated by gripper-handle proximity.

  Uses tanh-based reward: (1 - tanh(70 * ||target_pos - handle_pos||^4)) * reached_box
  where reached_box is latched once gripper is within distance_threshold of handle.

  Note: This requires the command to track target_pos and reached_box state.
  """
  from mjlab.tasks.manipulation.mdp.commands import OpenDrawerCommand, PushButtonCommand

  command = env.command_manager.get_term(command_name)

  # Determine asset name based on command type
  if isinstance(command, OpenDoorCommand):
    asset_name = "door"
  elif isinstance(command, OpenDrawerCommand):
    asset_name = "drawer"
  elif isinstance(command, PushButtonCommand):
    asset_name = "button"
  else:
    # Fallback - try to get asset_name from command config
    asset_name = getattr(command.cfg, "asset_name", "door")

  articulated_obj: Entity = env.scene[asset_name]

  # Get handle position
  handle_site_idx = articulated_obj.site_names.index("handle_site")
  handle_pos_w = articulated_obj.data.site_pos_w[:, handle_site_idx]

  # Use reached_box from command (which is latched)
  reached_box = command.reached_box

  # Compute handle-to-target distance reward
  target_pos = command.target_pos

  distance = torch.norm(target_pos - handle_pos_w, dim=-1)
  box_target_reward = 1.0 - torch.tanh(70.0 * distance**4)

  return box_target_reward * reached_box


def no_door_body_collision_reward(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Reward for not colliding with door body (1.0 if no collision, 0.0 if collision)."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  # Return 1.0 if no collision, 0.0 if collision
  collision = torch.any(sensor.data.found, dim=-1).float()
  return 1.0 - collision


def robot_init_pose_reward(
  env: ManagerBasedRlEnv,
  init_joint_pos: torch.Tensor,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for robot staying close to initial joint configuration.

  Uses: 1 - tanh(||q - q_init||)
  where q is current joint position and q_init is initial position.
  """
  robot: Entity = env.scene[asset_cfg.name]
  current_qpos = robot.data.joint_pos[:, asset_cfg.joint_ids]

  # Compute L2 norm of joint position difference
  qpos_diff = torch.norm(current_qpos - init_joint_pos, dim=-1)
  reward = 1.0 - torch.tanh(qpos_diff)
  return reward
