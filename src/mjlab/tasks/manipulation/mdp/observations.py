from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def gripper_to_object_vector(
  env: ManagerBasedRlEnv,
  object_asset_name: str = "object",
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Vector from gripper to object in world frame (3D).

  Works for both free objects (cube) and articulated objects (door/drawer/button).
  For articulated objects, uses the 'object_site' which should be at the manipulation point.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  obj: Entity = env.scene[object_asset_name]

  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids].squeeze(1)

  # For free objects, use root_link_pos; for articulated, use object_site
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    # Fallback for objects without object_site (e.g., cube with root link)
    object_pos_w = obj.data.root_link_pos_w

  return object_pos_w - gripper_pos_w


def object_to_goal_vector(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_name: str = "object",
) -> torch.Tensor:
  """Vector from object to goal position in world frame (3D).

  Works for all manipulation tasks. Returns goal_pos - object_pos.
  For articulated objects, uses object_site position; for free objects, uses root_link position.
  """
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_asset_name]

  # Get object position
  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    object_pos_w = obj.data.site_pos_w[:, object_site_idx]
  else:
    # Fallback for free objects
    object_pos_w = obj.data.root_link_pos_w

  # Get goal position from command
  goal_pos = command.target_pos

  return goal_pos - object_pos_w




def gripper_position(
  env: ManagerBasedRlEnv,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gripper position in world frame (3D)."""
  robot: Entity = env.scene[robot_asset_cfg.name]
  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids]  # (N, num_sites, 3)
  # Handle both single site and multiple sites
  if gripper_pos_w.dim() == 3:
    gripper_pos_w = gripper_pos_w.squeeze(1)  # (N, 3)
  return gripper_pos_w


def gripper_orientation(
  env: ManagerBasedRlEnv,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gripper orientation as 6D rotation matrix representation.

  Returns flattened rotation matrix excluding first row (6D: mat[3:]).
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  # Access site_xmat with Entity's filtered site indices
  gripper_mat = robot.data.data.site_xmat[:, robot.data.indexing.site_ids]  # (N, num_entity_sites, 3, 3)
  # Then further filter by robot_asset_cfg.site_ids if specified
  gripper_mat = gripper_mat[:, robot_asset_cfg.site_ids]  # (N, num_sites, 3, 3)
  # Handle both single site and multiple sites
  if gripper_mat.dim() == 4:  # (N, num_sites, 3, 3)
    gripper_mat = gripper_mat.squeeze(1)  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  gripper_mat_flat = gripper_mat.reshape(gripper_mat.shape[0], -1)  # (N, 9)
  return gripper_mat_flat[:, 3:]  # Last 6 values (excluding first row)


def object_position(
  env: ManagerBasedRlEnv,
  object_asset_name: str = "object",
) -> torch.Tensor:
  """Object position in world frame (3D).

  For articulated objects, uses object_site position.
  For free objects, uses root_link position.
  """
  obj: Entity = env.scene[object_asset_name]

  if "object_site" in obj.site_names:
    object_site_idx = obj.site_names.index("object_site")
    return obj.data.site_pos_w[:, object_site_idx]
  else:
    return obj.data.root_link_pos_w


def object_quaternion(
  env: ManagerBasedRlEnv,
  object_asset_name: str = "object",
) -> torch.Tensor:
  """Object orientation as quaternion (4D: w, x, y, z).

  Returns the orientation of the object's root body.
  """
  obj: Entity = env.scene[object_asset_name]
  return obj.data.root_link_quat_w


def object_orientation(
  env: ManagerBasedRlEnv,
  object_asset_name: str = "object",
) -> torch.Tensor:
  """Object orientation as 6D rotation matrix representation.

  Returns last 6 values of rotation matrix (mat[3:]).
  """
  obj: Entity = env.scene[object_asset_name]
  # Get rotation matrix for object root body
  body_mat = obj.data.data.xmat[:, obj.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  body_mat_flat = body_mat.reshape(body_mat.shape[0], -1)  # (N, 9)
  return body_mat_flat[:, 3:]  # Last 6 values




def goal_orientation_diff(
  env: ManagerBasedRlEnv,
  command_name: str,
  object_asset_name: str = "object",
) -> torch.Tensor:
  """Difference between goal orientation and object orientation (6D).

  Returns goal_mat[:6] - object_mat[:6] (first two rows of rotation matrix).
  Reads goal orientation from mocap_goal body.
  """
  command = env.command_manager.get_term(command_name)
  obj: Entity = env.scene[object_asset_name]

  # Current object orientation (N, 3, 3)
  object_mat = obj.data.data.xmat[:, obj.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  object_mat_flat = object_mat.reshape(object_mat.shape[0], -1)  # (N, 9)

  # Get goal orientation from mocap_goal
  mocap_goal: Entity = env.scene["mocap_goal"]
  goal_mat = mocap_goal.data.data.xmat[:, mocap_goal.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  goal_mat_flat = goal_mat.reshape(goal_mat.shape[0], -1)  # (N, 9)

  # Return difference of FIRST 6 values (first two rows)
  return goal_mat_flat[:, :6] - object_mat_flat[:, :6]


def control_qpos_difference(
  env: ManagerBasedRlEnv,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Difference between control commands and joint positions (excluding last finger joint).

  Returns ctrl - qpos[:-1] (8D for Franka: 7 arm + 1 finger).
  Provides feedback signal for delta position control.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  # Get current joint positions (9: 7 arm + 2 fingers)
  joint_pos = robot.data.joint_pos[:, robot_asset_cfg.joint_ids]
  # Get control commands from raw MuJoCo data
  ctrl = robot.data.data.ctrl[:, robot.data.indexing.ctrl_ids]  # (N, num_actuators)

  # Return ctrl - qpos[:-1] (exclude last finger joint)
  # Assuming ctrl is 8D and qpos is 9D
  if ctrl.shape[-1] == 8 and joint_pos.shape[-1] == 9:
    return ctrl - joint_pos[:, :-1]
  else:
    # Fallback: return zeros if dimensions don't match
    return torch.zeros(joint_pos.shape[0], 8, device=joint_pos.device)


def gripper_to_target_vector(
  env: ManagerBasedRlEnv,
  command_name: str,
  robot_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
  """Vector from the gripper site to the reach-command target (world frame, 3D).

  Returns target_pos - gripper_pos. This is the primary task signal for the reach
  task (offset-free relative vector — the FINDINGS "use relative obs only" lesson).
  """
  command = env.command_manager.get_term(command_name)
  robot: Entity = env.scene[robot_asset_cfg.name]
  gripper_pos_w = robot.data.site_pos_w[:, robot_asset_cfg.site_ids].squeeze(1)
  return command.target_pos - gripper_pos_w
