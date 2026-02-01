from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm
from mjlab.managers.manager_term_config import CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  quat_from_euler_xyz,
  random_orientation,
  sample_uniform,
)

if TYPE_CHECKING:
  from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class LiftingCommand(CommandTerm):
  cfg: LiftingCommandCfg

  def __init__(self, cfg: LiftingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Mocap goal for visualization (used by goal_orientation_diff observation)
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    # Common metrics (all tasks)
    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gripper_object_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Task-specific metrics
    self.metrics["cube_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]
    goal_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (goal_error < self.cfg.success_threshold).float()

    # Latch episode_success to 1 once goal is reached
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    # Update reached_object state (latch when gripper is close to object)
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()  # 10 cm threshold
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    # Common metrics
    self.metrics["goal_error"] = goal_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance

    # Task-specific metrics
    self.metrics["cube_height"] = object_height

  def compute_success(self) -> torch.Tensor:
    goal_error = self.metrics["goal_error"]
    return goal_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset episode success and reached_object for resampled envs
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    # Set target position based on difficulty mode.
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]

      # Sample object orientation (yaw only, keep upright).
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

      # Sample goal orientation independently (yaw only, keep upright).
      goal_yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      target_quats = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        goal_yaw,
      )
    else:
      # Default to identity quaternion if object pose not randomized
      target_quats = torch.zeros(n, 4, device=self.device)
      target_quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity)

    # Update mocap_goal visualization (for goal_orientation_diff observation)
    # Goal has independently sampled orientation
    mocap_pos = self.target_pos[env_ids].clone()

    mocap_pose = torch.cat([mocap_pos, target_quats], dim=-1)
    self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # Visualize targets for all environments
    for env_idx in range(self.num_envs):
      target_pos = self.target_pos[env_idx].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{env_idx}",
      )


@dataclass(kw_only=True)
class LiftingCommandCfg(CommandTermCfg):
  asset_name: str
  robot_asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot", site_names=()))
  class_type: type[CommandTerm] = LiftingCommand
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    z: tuple[float, float] = (0.2, 0.4)

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.3, 0.35)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)


class OpenDoorCommand(CommandTerm):
  """Command for door opening task.

  Tracks:
  - Target door opening angle (0 to 90 degrees)
  - Door hinge position/velocity
  - Success when door opens past threshold
  """

  cfg: OpenDoorCommandCfg

  def __init__(self, cfg: OpenDoorCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.door: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.target_angle = torch.zeros(self.num_envs, device=self.device)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Get mocap goal entity for visualization and orientation tracking
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    # Common metrics (all tasks)
    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gripper_object_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Task-specific metrics
    self.metrics["door_angle"] = torch.zeros(self.num_envs, device=self.device)

    # Find door hinge joint index
    joint_names = self.door.joint_names
    self.hinge_idx = joint_names.index("door_hinge")


  @property
  def command(self) -> torch.Tensor:
    """Return target angle."""
    return self.target_angle.unsqueeze(-1)

  def _update_metrics(self) -> None:
    # Get current hinge angle
    door_angle = self.door.data.joint_pos[:, self.hinge_idx]
    angle_error = torch.abs(self.target_angle - door_angle)
    at_goal = (angle_error < self.cfg.success_threshold).float()

    # Latch success
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    # Update reached_object state (latch when gripper is close to object)
    # Get object and gripper positions
    object_site_idx = self.door.site_names.index("object_site") if "object_site" in self.door.site_names else self.door.site_names.index("handle_site")
    object_pos_w = self.door.data.site_pos_w[:, object_site_idx]
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)

    # Compute distance
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()  # 10 cm threshold
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    # Common metrics
    self.metrics["goal_error"] = angle_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance

    # Task-specific metrics
    self.metrics["door_angle"] = door_angle

  def compute_success(self) -> torch.Tensor:
    goal_error = self.metrics["goal_error"]
    return goal_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset success tracking
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    # Set target angle based on difficulty
    if self.cfg.difficulty == "fixed":
      self.target_angle[env_ids] = 1.5708  # 90 degrees
    else:
      # Dynamic: random angle between 60-90 degrees
      target = sample_uniform(1.0472, 1.5708, (n,), device=self.device)
      self.target_angle[env_ids] = target

    # Set door hinge to initial position (can be randomized)
    # For now using 0.0 (closed), but this could be randomized
    joint_pos = torch.full((n, 1), 0.0, device=self.device)
    joint_vel = torch.zeros(n, 1, device=self.device)

    self.door.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    self.door.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

    # Get base_site position (positioned at handle location when joint=0)
    # This reflects base randomization but not joint position
    base_site_idx = self.door.site_names.index("base_site")
    base_site_pos = self.door.data.site_pos_w[env_ids, base_site_idx]

    # Calculate target: base_site is at handle's closed position
    # Hinge is at [0, -0.3, 0] in door_base frame, handle at base_site is at [-0.04, 0.25, 0]
    # So hinge relative to base_site is [0.04, -0.55, 0]
    handle_to_hinge_dist = 0.55  # Distance from handle to hinge

    for i, env_id in enumerate(env_ids):
      # Hinge position = base_site + offset to hinge
      hinge_pos = base_site_pos[i] + torch.tensor([0.04, -handle_to_hinge_dist, 0.0], device=self.device)

      # Calculate target handle position by rotating around hinge
      angle = self.target_angle[env_id]
      cos_a = torch.cos(angle)
      sin_a = torch.sin(angle)

      # Vector from hinge to handle at target angle (rotating around Z axis)
      # Initial vector is [-0.04, 0.55, 0.0] (handle relative to hinge when closed)
      rotated_x = -0.04 * cos_a - handle_to_hinge_dist * sin_a
      rotated_y = -0.04 * sin_a + handle_to_hinge_dist * cos_a

      self.target_pos[env_id] = hinge_pos + torch.tensor(
        [rotated_x, rotated_y, 0.0], device=self.device
      )

    if self.mocap_goal is not None:
      target_quats = torch.zeros(n, 4, device=self.device)
      for i, env_id in enumerate(env_ids):
        angle = self.target_angle[env_id]
        half_angle = angle / 2.0
        target_quats[i] = torch.tensor(
          [torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)],
          device=self.device
        )
      mocap_pos = self.target_pos[env_ids].clone()
      mocap_pose = torch.cat([mocap_pos, target_quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)


  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # No additional debug visualization needed - orange mocap_goal box already shows target
    pass


@dataclass(kw_only=True)
class OpenDoorCommandCfg(CommandTermCfg):
  """Configuration for door opening command."""

  asset_name: str = "door"
  robot_asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot", site_names=()))
  class_type: type[CommandTerm] = OpenDoorCommand
  success_threshold: float = 0.1  # 0.1 radians (~6 degrees)
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class DoorPoseRangeCfg:
    """Door position/orientation randomization."""

    x: tuple[float, float] = (0.6, 0.8)
    y: tuple[float, float] = (0.2, 0.4)
    z: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-0.5, 0.5)

  door_pose_range: DoorPoseRangeCfg | None = field(default_factory=DoorPoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)


class OpenDrawerCommand(CommandTerm):
  """Command for drawer opening task.

  Tracks:
  - Target drawer opening distance (0 to 0.3 meters)
  - Drawer slide position/velocity
  - Success when drawer opens past threshold
  """

  cfg: OpenDrawerCommandCfg

  def __init__(self, cfg: OpenDrawerCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.drawer: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.target_distance = torch.zeros(self.num_envs, device=self.device)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Get mocap target entity for visualization
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    # Common metrics (all tasks)
    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gripper_object_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Task-specific metrics
    self.metrics["drawer_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Find drawer slide joint index
    joint_names = self.drawer.joint_names
    self.slide_idx = joint_names.index("drawer_slide")


  @property
  def command(self) -> torch.Tensor:
    """Return target distance."""
    return self.target_distance.unsqueeze(-1)

  def _update_metrics(self) -> None:
    # Get current slide distance
    drawer_distance = self.drawer.data.joint_pos[:, self.slide_idx]
    distance_error = torch.abs(self.target_distance - drawer_distance)
    at_goal = (distance_error < self.cfg.success_threshold).float()

    # Latch success
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    # Update reached_object state (latch when gripper is close to object)
    object_site_idx = self.drawer.site_names.index("object_site") if "object_site" in self.drawer.site_names else self.drawer.site_names.index("handle_site")
    object_pos_w = self.drawer.data.site_pos_w[:, object_site_idx]
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)

    # Compute distance
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()  # 10 cm threshold
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    # Common metrics
    self.metrics["goal_error"] = distance_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance

    # Task-specific metrics
    self.metrics["drawer_distance"] = drawer_distance

  def compute_success(self) -> torch.Tensor:
    goal_error = self.metrics["goal_error"]
    return goal_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset success tracking
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    # Set target distance based on difficulty
    if self.cfg.difficulty == "fixed":
      self.target_distance[env_ids] = -0.25  # -0.25m (fully open)
    else:
      # Dynamic: random distance between -0.15 and -0.25
      target = sample_uniform(-0.25, -0.15, (n,), device=self.device)
      self.target_distance[env_ids] = target

    # Set drawer slide to initial position (can be randomized)
    # For now using 0.0 (closed), but this could be randomized
    joint_pos = torch.full((n, 1), 0.0, device=self.device)
    joint_vel = torch.zeros(n, 1, device=self.device)

    self.drawer.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    self.drawer.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

    # Get base_site position (positioned at handle location when joint=0)
    # This reflects base randomization but not joint position
    base_site_idx = self.drawer.site_names.index("base_site")
    base_site_pos = self.drawer.data.site_pos_w[env_ids, base_site_idx]

    # Calculate target: base_site is at handle's closed position
    # Target is at open position (joint = target_distance along X)
    for i, env_id in enumerate(env_ids):
      distance = self.target_distance[env_id]
      # Target = base_site + slide distance along X
      self.target_pos[env_id] = base_site_pos[i] + torch.tensor(
        [distance, 0.0, 0.0], device=self.device
      )

    if self.mocap_goal is not None:
      # For drawer, no rotation - just translation
      target_quats = torch.zeros(n, 4, device=self.device)
      target_quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)

      mocap_pos = self.target_pos[env_ids].clone()
      mocap_pose = torch.cat([mocap_pos, target_quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)


  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # No additional debug visualization needed - orange mocap_goal box already shows target
    pass


@dataclass(kw_only=True)
class OpenDrawerCommandCfg(CommandTermCfg):
  """Configuration for drawer opening command."""

  asset_name: str = "drawer"
  robot_asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot", site_names=()))
  class_type: type[CommandTerm] = OpenDrawerCommand
  success_threshold: float = 0.02  # 2cm threshold
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class DrawerPoseRangeCfg:
    """Drawer position/orientation randomization."""

    x: tuple[float, float] = (0.6, 0.8)
    y: tuple[float, float] = (0.2, 0.4)
    z: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-0.5, 0.5)

  drawer_pose_range: DrawerPoseRangeCfg | None = field(default_factory=DrawerPoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)


class PushButtonCommand(CommandTerm):
  """Command for button pushing task.

  Tracks:
  - Target button push distance (0.25 to 0 meters)
  - Button slide position/velocity
  - Success when button is pushed past threshold
  """

  cfg: PushButtonCommandCfg

  def __init__(self, cfg: PushButtonCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.button: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.target_distance = torch.zeros(self.num_envs, device=self.device)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Get mocap target entity for visualization
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    # Common metrics (all tasks)
    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gripper_object_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Task-specific metrics
    self.metrics["button_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Find button slide joint index
    joint_names = self.button.joint_names
    self.slide_idx = joint_names.index("button_slide")


  @property
  def command(self) -> torch.Tensor:
    """Return target distance."""
    return self.target_distance.unsqueeze(-1)

  def _update_metrics(self) -> None:
    # Get current slide distance
    button_distance = self.button.data.joint_pos[:, self.slide_idx]
    distance_error = torch.abs(self.target_distance - button_distance)
    at_goal = (distance_error < self.cfg.success_threshold).float()

    # Latch success
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    # Update reached_object state (latch when gripper is close to object)
    object_site_idx = self.button.site_names.index("object_site") if "object_site" in self.button.site_names else self.button.site_names.index("handle_site")
    object_pos_w = self.button.data.site_pos_w[:, object_site_idx]
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)

    # Compute distance
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()  # 10 cm threshold
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    # Common metrics
    self.metrics["goal_error"] = distance_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance

    # Task-specific metrics
    self.metrics["button_distance"] = button_distance

  def compute_success(self) -> torch.Tensor:
    goal_error = self.metrics["goal_error"]
    return goal_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset success tracking
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    # Set target distance based on difficulty
    # Joint range is -0.05 to 0.0, where 0.0 is unpressed and -0.05 is fully pressed
    if self.cfg.difficulty == "fixed":
      self.target_distance[env_ids] = -0.05  # -0.05m (fully pressed)
    else:
      # Dynamic: random distance between -0.05 (fully pressed) and -0.02 (partially pressed)
      target = sample_uniform(-0.05, -0.02, (n,), device=self.device)
      self.target_distance[env_ids] = target

    # Set button slide to initial position (unpressed)
    joint_pos = torch.full((n, 1), 0.0, device=self.device)
    joint_vel = torch.zeros(n, 1, device=self.device)

    self.button.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    self.button.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

    # Get base_site position (positioned at handle location when joint=0)
    # This reflects base randomization but not joint position
    base_site_idx = self.button.site_names.index("base_site")
    base_site_pos = self.button.data.site_pos_w[env_ids, base_site_idx]

    # Calculate target: base_site is at handle's position when joint=0
    # Target is at pressed position (joint = target_distance along Z)
    for i, env_id in enumerate(env_ids):
      distance = self.target_distance[env_id]
      # Target = base_site + slide distance along Z
      self.target_pos[env_id] = base_site_pos[i] + torch.tensor(
        [0.0, 0.0, distance], device=self.device
      )

    if self.mocap_goal is not None:
      # For button, no rotation - just translation
      target_quats = torch.zeros(n, 4, device=self.device)
      target_quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity quaternion)

      mocap_pos = self.target_pos[env_ids].clone()
      mocap_pose = torch.cat([mocap_pos, target_quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)


  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # No additional debug visualization needed - orange mocap_goal box already shows target
    pass


@dataclass(kw_only=True)
class PushButtonCommandCfg(CommandTermCfg):
  """Configuration for button pushing command."""

  asset_name: str = "button"
  robot_asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot", site_names=()))
  class_type: type[CommandTerm] = PushButtonCommand
  success_threshold: float = 0.02  # 2cm threshold
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  @dataclass
  class ButtonPoseRangeCfg:
    """Button position/orientation randomization."""

    x: tuple[float, float] = (0.6, 0.8)
    y: tuple[float, float] = (0.2, 0.4)
    z: tuple[float, float] = (0.0, 0.0)
    yaw: tuple[float, float] = (-0.5, 0.5)

  button_pose_range: ButtonPoseRangeCfg | None = field(default_factory=ButtonPoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)


class PushingCommand(CommandTerm):
  """Command for pushing objects on the ground.

  Similar to LiftingCommand but constrains goal positions to ground level (x-y plane only).
  """

  cfg: PushingCommandCfg

  def __init__(self, cfg: PushingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Mocap goal for visualization (used by goal_orientation_diff observation)
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    # Common metrics (all tasks)
    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["gripper_object_distance"] = torch.zeros(self.num_envs, device=self.device)

    # Task-specific metrics
    self.metrics["object_height"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _update_metrics(self) -> None:
    object_pos_w = self.object.data.root_link_pos_w
    object_height = object_pos_w[:, 2]
    goal_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    at_goal = (goal_error < self.cfg.success_threshold).float()

    # Latch episode_success to 1 once goal is reached
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    # Update reached_object state (latch when gripper is close to object)
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()  # 10 cm threshold
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    # Common metrics
    self.metrics["goal_error"] = goal_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance

    # Task-specific metrics
    self.metrics["object_height"] = object_height

  def compute_success(self) -> torch.Tensor:
    goal_error = self.metrics["goal_error"]
    return goal_error < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)

    # Reset episode success and reached_object for resampled envs
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    # Set target position based on difficulty mode.
    # Key difference from LiftingCommand: z is fixed at ground level
    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.4, 0.0, self.cfg.goal_z_height], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      # Sample x and y, but keep z constant at ground level
      lower = torch.tensor([r.x[0], r.y[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1]], device=self.device)
      target_pos_xy = sample_uniform(lower, upper, (n, 2), device=self.device)
      # Add fixed z height (ground level)
      z_height = torch.full((n, 1), self.cfg.goal_z_height, device=self.device)
      target_pos = torch.cat([target_pos_xy, z_height], dim=-1)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    # Reset object to new position.
    if self.cfg.object_pose_range is not None:
      r = self.cfg.object_pose_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      pos = pos + self._env.scene.env_origins[env_ids]

      # Sample object orientation (yaw only, keep upright).
      yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        yaw,
      )
      pose = torch.cat([pos, quat], dim=-1)

      velocity = torch.zeros(n, 6, device=self.device)

      self.object.write_root_link_pose_to_sim(pose, env_ids=env_ids)
      self.object.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)

      # Sample goal orientation independently (yaw only, keep upright).
      goal_yaw = sample_uniform(r.yaw[0], r.yaw[1], (n,), device=self.device)
      target_quats = quat_from_euler_xyz(
        torch.zeros(n, device=self.device),  # roll
        torch.zeros(n, device=self.device),  # pitch
        goal_yaw,
      )
    else:
      # Default to identity quaternion if object pose not randomized
      target_quats = torch.zeros(n, 4, device=self.device)
      target_quats[:, 0] = 1.0  # w=1, x=y=z=0 (identity)

    # Update mocap_goal visualization (for goal_orientation_diff observation)
    # Goal has independently sampled orientation
    mocap_pos = self.target_pos[env_ids].clone()

    mocap_pose = torch.cat([mocap_pos, target_quats], dim=-1)
    self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    # Visualize targets for all environments
    for env_idx in range(self.num_envs):
      target_pos = self.target_pos[env_idx].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"target_position_{env_idx}",
      )


@dataclass(kw_only=True)
class PushingCommandCfg(CommandTermCfg):
  """Configuration for pushing objects on the ground (x-y plane only)."""

  asset_name: str
  robot_asset_cfg: SceneEntityCfg = field(default_factory=lambda: SceneEntityCfg("robot", site_names=()))
  class_type: type[CommandTerm] = PushingCommand
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "fixed"

  # Fixed z-height for goals (ground level)
  goal_z_height: float = 0.03

  @dataclass
  class TargetPositionRangeCfg:
    """Configuration for target position sampling in dynamic mode (x-y only)."""

    x: tuple[float, float] = (0.3, 0.5)
    y: tuple[float, float] = (-0.2, 0.2)
    # Note: z is not included here - it's fixed at goal_z_height

  # Only used in dynamic mode.
  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class ObjectPoseRangeCfg:
    """Configuration for object pose sampling when resampling commands."""

    x: tuple[float, float] = (0.3, 0.35)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.02, 0.05)
    yaw: tuple[float, float] = (-math.pi, math.pi)

  object_pose_range: ObjectPoseRangeCfg | None = field(
    default_factory=ObjectPoseRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.5, 0.0, 0.3)

  viz: VizCfg = field(default_factory=VizCfg)
