from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .task_geometry import between_fingers, grasped, tracking_goal, tracking_position


def _goal_orientation_factor(command):
  """Square insertion needs uprightness and yaw agreement, modulo square symmetry."""
  if not getattr(command.cfg, "insertion", False):
    return 1.0
  from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

  n = command.num_envs
  x = quat_apply_inverse(
    command.base.data.root_link_quat_w,
    quat_apply(
      command.object.data.root_link_quat_w,
      torch.tensor([1.0, 0.0, 0.0], device=command.device).expand(n, 3),
    ),
  )
  z = quat_apply_inverse(
    command.base.data.root_link_quat_w,
    quat_apply(
      command.object.data.root_link_quat_w,
      torch.tensor([0.0, 0.0, 1.0], device=command.device).expand(n, 3),
    ),
  )
  yaw = torch.atan2(x[:, 1], x[:, 0])
  yaw_error = torch.atan2(torch.sin(4 * yaw), torch.cos(4 * yaw)) / 4
  tilt = torch.acos(z[:, 2].clamp(-1, 1))
  return torch.exp(-((tilt / 0.2) ** 2) - (yaw_error / 0.15) ** 2)


def _grasp_factor(command):
  if not getattr(command.cfg, "require_grasp", False):
    return 1.0
  valid = grasped(command)
  if hasattr(command, "pivoted"):
    valid &= command.pivoted
  return valid.float()


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
  reach_error = reach_distance_clamped**2
  reaching = torch.exp(-reach_error / (reaching_std * reaching_max_dist) ** 2)

  # Manipulation phase: object to goal
  position_error = torch.sum(
    torch.square(tracking_goal(command) - object_pos_w), dim=-1
  )
  bringing = torch.exp(-position_error / (bringing_std * bringing_max_dist) ** 2)
  bringing = bringing * _goal_orientation_factor(command) * _grasp_factor(command)

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
    torch.square(tracking_goal(command) - object_pos_w), dim=-1
  )
  return (
    torch.exp(-position_error / (std * max_dist) ** 2)
    * _goal_orientation_factor(command)
    * _grasp_factor(command)
  )


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


def joint_target_error_penalty(
  env: ManagerBasedRlEnv,
  max_error: float,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Soft cost for physical position targets far from current joint positions.

  This provides command-dependent reward even when nearby commands produce the
  same force-limited motion. It never modifies targets, velocities or physics.
  Select arm joints explicitly to exclude the gripper. Units are radians squared.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  target = robot.data.joint_pos_target[:, robot_asset_cfg.joint_ids]
  position = robot.data.joint_pos[:, robot_asset_cfg.joint_ids]
  excess = ((target - position).abs() - max_error).clamp_min(0.)
  return excess.square().sum(dim=-1)


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
  goal_pos = tracking_goal(command)
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
  position_error = torch.sum(torch.square(command.target_pos - gripper_pos_w), dim=-1)
  return torch.exp(-position_error / (std**2))


def gripper_closure_penalty(
  env: ManagerBasedRlEnv,
  threshold: float,
  robot_asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  finger_joint_names: tuple[str, str] = ("finger_joint1", "finger_joint2"),
) -> torch.Tensor:
  """1.0 while the combined finger opening is below ``threshold``.

  Used with a NEGATIVE weight by the caging task: its success predicate voids the
  episode the moment ``min_aperture`` dips under the cage threshold, which a sparse
  predicate only reports after the fact — this is the per-step signal that steers
  the policy away from closing at all.
  """
  robot: Entity = env.scene[robot_asset_cfg.name]
  i0 = robot.joint_names.index(finger_joint_names[0])
  i1 = robot.joint_names.index(finger_joint_names[1])
  aperture = robot.data.joint_pos[:, i0] + robot.data.joint_pos[:, i1]
  return (aperture <= threshold).float()


def cage_transport_reward(env, command_name, object_asset_name="cube", **kwargs):
  """Approach an open cage, then pay transport only while the cube is enclosed."""
  command = env.command_manager.get_term(command_name)
  valid = (
    torch.minimum(command.min_aperture, command._aperture()) > command.cfg.aperture_min
  )
  cage = between_fingers(command).float()
  precise = object_at_goal_reward(env, command_name, object_asset_name, max_dist=0.35)
  if "robot_asset_cfg" not in kwargs:
    return precise * cage * valid.float()
  robot_cfg = kwargs["robot_asset_cfg"]
  approach = reach_object_reward(env, object_asset_name, robot_cfg, k=30.0)
  return (approach + cage * precise) * valid.float()


def orientation_task_reward(env, command_name, object_asset_name="object", **kwargs):
  """Reward the same body-axis orientation and positional drift as success."""
  from mjlab.utils.lab_api.math import quat_apply

  command = env.command_manager.get_term(command_name)
  obj = env.scene[object_asset_name]
  axis = quat_apply(
    obj.data.root_link_quat_w, command.body_axis.expand(env.num_envs, 3)
  )
  alignment = (axis * command.target_axis).sum(-1).clamp(-1, 1)
  if command.cfg.symmetric_axis:
    alignment = alignment.abs()
  angle = torch.acos(alignment)
  drift = torch.linalg.vector_norm(
    tracking_position(obj)[:, :2] - command.target_pos[:, :2], dim=-1
  )
  score = torch.exp(-((angle / 0.5) ** 2)) * (drift < command.cfg.max_drift).float()
  if "robot_asset_cfg" in kwargs:
    reaching = reach_object_reward(
      env, object_asset_name, kwargs["robot_asset_cfg"], k=30.0
    )
    return reaching * (1 + score)
  return score


def tool_transport_reward(env, command_name, object_asset_name="puck", **kwargs):
  """Reach the tool and reward puck transport only after tool-mediated contact."""
  from .task_geometry import grasped, touching

  command = env.command_manager.get_term(command_name)
  held = grasped(command, command.tool)
  used = (command.tool_used > 0) | (
    held & touching(command, command.tool, command.object)
  )
  valid = ~(command.direct_contact | touching(command, command.robot, command.object))
  bringing = object_at_goal_reward(env, command_name, object_asset_name, max_dist=0.35)
  if "robot_asset_cfg" in kwargs:
    approach = reach_object_reward(
      env, command.cfg.tool_asset_name, kwargs["robot_asset_cfg"], k=30.0
    )
    return (approach + held.float() + used.float() * bringing) * valid.float()
  return bringing * used.float() * valid.float()


def articulation_task_reward(env, command_name, object_asset_name="object", approach_scale=None, **kwargs):
  """Joint-space progress avoids periodic Cartesian shortcuts (e.g. a 270° valve)."""
  command = env.command_manager.get_term(command_name)
  if hasattr(command, "target_value"):
    target, value = command.target_value, command._joint_value()
  elif hasattr(command, "target_angle"):
    target = command.target_angle
    value = command.door.data.joint_pos[:, command.hinge_idx]
  else:
    target = command.target_distance
    asset = command.drawer if hasattr(command, "drawer") else command.button
    value = asset.data.joint_pos[:, command.slide_idx]
  error = (target - value).abs()
  if getattr(command.cfg, "directional", False):
    error = ((target - value) * target.sign()).clamp_min(0)
  if "robot_asset_cfg" in kwargs:
    if approach_scale is None:
      approach = reach_object_reward(
        env, object_asset_name, kwargs["robot_asset_cfg"], k=30.0
      )
    else:
      robot_cfg = kwargs["robot_asset_cfg"]
      gripper = env.scene[robot_cfg.name].data.site_pos_w[:, robot_cfg.site_ids].squeeze(1)
      distance = torch.linalg.vector_norm(tracking_position(env.scene[object_asset_name]) - gripper, dim=-1)
      approach = torch.exp(-distance / approach_scale)
    progress = torch.exp(-error / (target.abs() * 0.5).clamp_min(0.01))
    return approach * (1 + progress)
  return torch.exp(-0.5 * (error / command.cfg.success_threshold) ** 2)
