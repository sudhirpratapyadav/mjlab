from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.manipulation.mdp.commands import LiftingCommand
from mjlab.utils.lab_api.math import quat_apply, quat_inv

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def ee_to_object_distance(
  env: ManagerBasedRlEnv,
  object_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end effector to object in robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  obj: Entity = env.scene[object_name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  obj_pos_w = obj.data.root_link_pos_w
  distance_vec_w = obj_pos_w - ee_pos_w
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)
  return distance_vec_b


def object_position_error(
  env: ManagerBasedRlEnv,
  object_name: str,
  command_name: str,
) -> torch.Tensor:
  """3D position error between object and target position (target - current)."""
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, LiftingCommand):
    raise TypeError(
      f"Command '{command_name}' must be a LiftingCommand, got {type(command)}"
    )
  obj: Entity = env.scene[object_name]
  obj_pos_w = obj.data.root_link_pos_w
  position_error = command.target_pos - obj_pos_w
  return position_error


def ee_to_handle_distance(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Distance vector from end-effector to door handle in robot base frame."""
  robot: Entity = env.scene[asset_cfg.name]
  door: Entity = env.scene["door"]

  # Get end-effector position
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)

  # Get handle position (from handle_site)
  handle_site_idx = door.site_names.index("handle_site")
  handle_pos_w = door.data.site_pos_w[:, handle_site_idx]

  # Distance vector in world frame
  distance_vec_w = handle_pos_w - ee_pos_w

  # Transform to robot base frame
  base_quat_w = robot.data.root_link_quat_w
  distance_vec_b = quat_apply(quat_inv(base_quat_w), distance_vec_w)

  return distance_vec_b


def door_hinge_position(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current door hinge angle (1D: rotation in radians)."""
  door: Entity = env.scene["door"]
  hinge_idx = door.joint_names.index("door_hinge")
  return door.data.joint_pos[:, hinge_idx : hinge_idx + 1]


def door_hinge_velocity(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Current door hinge angular velocity (1D: rad/s)."""
  door: Entity = env.scene["door"]
  hinge_idx = door.joint_names.index("door_hinge")
  return door.data.joint_vel[:, hinge_idx : hinge_idx + 1]


def gripper_position(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gripper (end-effector) position in world frame (3D)."""
  robot: Entity = env.scene[asset_cfg.name]
  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids]  # (N, num_sites, 3)
  # Handle both single site and multiple sites
  if ee_pos_w.dim() == 3:
    ee_pos_w = ee_pos_w.squeeze(1)  # (N, 3)
  return ee_pos_w


def gripper_orientation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Gripper orientation as 6D rotation matrix representation (last 6 values of 3x3 matrix).

  Returns flattened rotation matrix excluding first row (6D: mat[3:]).
  """
  robot: Entity = env.scene[asset_cfg.name]
  # Access site_xmat with Entity's filtered site indices
  ee_mat = robot.data.data.site_xmat[:, robot.data.indexing.site_ids]  # (N, num_entity_sites, 3, 3)
  # Then further filter by asset_cfg.site_ids if specified
  ee_mat = ee_mat[:, asset_cfg.site_ids]  # (N, num_sites, 3, 3)
  # Handle both single site and multiple sites
  if ee_mat.dim() == 4:  # (N, num_sites, 3, 3)
    ee_mat = ee_mat.squeeze(1)  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  ee_mat_flat = ee_mat.reshape(ee_mat.shape[0], -1)  # (N, 9)
  return ee_mat_flat[:, 3:]  # Last 6 values (excluding first row)


def handle_geom_position(env: ManagerBasedRlEnv, asset_name: str = "door") -> torch.Tensor:
  """Handle geom position in world frame (3D).

  Uses geom position which may be offset from body origin.
  """
  articulated_obj: Entity = env.scene[asset_name]
  # Assuming handle geom is named "handle" or similar - adjust as needed
  # For now, we'll use the handle site position as proxy
  handle_site_idx = articulated_obj.site_names.index("handle_site")
  handle_pos_w = articulated_obj.data.site_pos_w[:, handle_site_idx]
  return handle_pos_w


def handle_body_quaternion(env: ManagerBasedRlEnv, asset_name: str = "door") -> torch.Tensor:
  """Handle body orientation as quaternion (4D: w, x, y, z)."""
  articulated_obj: Entity = env.scene[asset_name]
  # Get handle body index - assuming it's a child body with the hinge joint
  # The door entity should have a body that contains the hinge
  # We need to get the quaternion of the door body/frame
  # This returns the orientation of the door's root body
  return articulated_obj.data.root_link_quat_w


def handle_body_orientation(env: ManagerBasedRlEnv, asset_name: str = "door") -> torch.Tensor:
  """Handle body orientation as 6D rotation matrix representation.

  Returns last 6 values of rotation matrix (mat[3:]).
  """
  articulated_obj: Entity = env.scene[asset_name]
  # Get rotation matrix for handle body from raw MuJoCo data
  # Access xmat for root body (N, 3, 3) - 3x3 rotation matrix
  body_mat = articulated_obj.data.data.xmat[:, articulated_obj.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  body_mat_flat = body_mat.reshape(body_mat.shape[0], -1)  # (N, 9)
  return body_mat_flat[:, 3:]  # Last 6 values


def gripper_to_handle_vector(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  articulated_asset_name: str = "door",
) -> torch.Tensor:
  """Vector from gripper to handle in world frame (3D)."""
  robot: Entity = env.scene[asset_cfg.name]
  articulated_obj: Entity = env.scene[articulated_asset_name]

  ee_pos_w = robot.data.site_pos_w[:, asset_cfg.site_ids].squeeze(1)
  handle_site_idx = articulated_obj.site_names.index("handle_site")
  handle_pos_w = articulated_obj.data.site_pos_w[:, handle_site_idx]

  return handle_pos_w - ee_pos_w


def target_to_handle_vector(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Vector from target position to handle in world frame (3D)."""
  from mjlab.tasks.manipulation.mdp.commands import OpenDoorCommand, OpenDrawerCommand, PushButtonCommand
  from typing import cast

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

  handle_site_idx = articulated_obj.site_names.index("handle_site")
  handle_pos_w = articulated_obj.data.site_pos_w[:, handle_site_idx]

  # Get target position from command (computed in _resample_command)
  target_pos = command.target_pos

  # Return target - handle (matching mujoco_playground)
  return target_pos - handle_pos_w


def target_orientation_diff(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Difference between target orientation and handle body orientation (6D).

  Returns target_mat[:6] - handle_mat[:6] (first two rows of rotation matrix).

  Reads target orientation directly from mocap_target body (matching mujoco_playground).
  """
  from mjlab.tasks.manipulation.mdp.commands import OpenDoorCommand, OpenDrawerCommand, PushButtonCommand

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

  # Current handle body orientation (N, 3, 3)
  handle_mat = articulated_obj.data.data.xmat[:, articulated_obj.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  handle_mat_flat = handle_mat.reshape(handle_mat.shape[0], -1)  # (N, 9)

  # Get target orientation from mocap_target (direct capture - mujoco_playground style ✅)
  target_mat = command.mocap_target.data.data.xmat[:, command.mocap_target.data.indexing.root_body_id]  # (N, 3, 3)
  # Flatten to 9D: (N, 3, 3) -> (N, 9)
  target_mat_flat = target_mat.reshape(target_mat.shape[0], -1)  # (N, 9)

  # Return difference of FIRST 6 values (first two rows) - matches mujoco_playground
  return target_mat_flat[:, :6] - handle_mat_flat[:, :6]


def control_qpos_difference(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Difference between control commands and joint positions (excluding last finger joint).

  Returns ctrl - qpos[:-1] (8D for Franka: 7 arm + 1 finger).
  """
  robot: Entity = env.scene[asset_cfg.name]
  # Get current joint positions (9: 7 arm + 2 fingers)
  joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
  # Get control commands from raw MuJoCo data
  ctrl = robot.data.data.ctrl[:, robot.data.indexing.ctrl_ids]  # (N, num_actuators)

  # Return ctrl - qpos[:-1] (exclude last finger joint)
  # Assuming ctrl is 8D and qpos is 9D
  if ctrl.shape[-1] == 8 and joint_pos.shape[-1] == 9:
    return ctrl - joint_pos[:, :-1]
  else:
    # Fallback: return zeros if dimensions don't match
    return torch.zeros(joint_pos.shape[0], 8, device=joint_pos.device)
