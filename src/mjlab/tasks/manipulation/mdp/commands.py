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


class ReachingCommand(CommandTerm):
  """Command for the end-effector reaching task.

  Samples a 3D target position in the robot's workspace and tracks whether the gripper
  site has reached it. Success is latched (via ``torch.maximum``) once the gripper
  comes within ``success_threshold`` of the target — matching the eval-success
  convention used across the benchmark (see continual_distill/docs FINDINGS.md).

  Unlike the object-manipulation commands, there is no object to move: the target IS
  the command, and success depends only on the gripper reaching it. The ``mocap_goal``
  entity visualizes the target and provides a stable body for goal observations.
  """

  cfg: ReachingCommandCfg

  def __init__(self, cfg: ReachingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg
    self.mocap_goal: Entity = env.scene["mocap_goal"]

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)

    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _gripper_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)

  def _update_metrics(self) -> None:
    goal_error = torch.norm(self.target_pos - self._gripper_pos_w(), dim=-1)
    at_goal = (goal_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)
    self.metrics["goal_error"] = goal_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success

  def compute_success(self) -> torch.Tensor:
    return self.metrics["goal_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0

    if self.cfg.difficulty == "fixed":
      target_pos = torch.tensor(
        [0.5, 0.0, 0.3], device=self.device, dtype=torch.float32
      ).expand(n, 3)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]
    else:
      assert self.cfg.difficulty == "dynamic"
      r = self.cfg.target_position_range
      lower = torch.tensor([r.x[0], r.y[0], r.z[0]], device=self.device)
      upper = torch.tensor([r.x[1], r.y[1], r.z[1]], device=self.device)
      target_pos = sample_uniform(lower, upper, (n, 3), device=self.device)
      self.target_pos[env_ids] = target_pos + self._env.scene.env_origins[env_ids]

    # Identity orientation for the mocap goal (reach is position-only).
    quat = torch.zeros(n, 4, device=self.device)
    quat[:, 0] = 1.0
    mocap_pose = torch.cat([self.target_pos[env_ids], quat], dim=-1)
    self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for env_idx in range(self.num_envs):
      target_pos = self.target_pos[env_idx].cpu().numpy()
      visualizer.add_sphere(
        center=target_pos,
        radius=0.03,
        color=self.cfg.viz.target_color,
        label=f"reach_target_{env_idx}",
      )


@dataclass(kw_only=True)
class ReachingCommandCfg(CommandTermCfg):
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  class_type: type[CommandTerm] = ReachingCommand
  success_threshold: float = 0.05
  difficulty: Literal["fixed", "dynamic"] = "dynamic"

  @dataclass
  class TargetPositionRangeCfg:
    """Workspace box the reach target is sampled from (dynamic mode)."""

    x: tuple[float, float] = (0.4, 0.7)
    y: tuple[float, float] = (-0.25, 0.25)
    z: tuple[float, float] = (0.15, 0.5)

  target_position_range: TargetPositionRangeCfg = field(
    default_factory=TargetPositionRangeCfg
  )

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (0.0, 0.8, 1.0, 0.4)

  viz: VizCfg = field(default_factory=VizCfg)


class StackingCommand(CommandTerm):
  """Command for stacking one free object on top of another.

  ``asset_name`` is the object to move (e.g. cube); ``base_asset_name`` is the object
  to stack ONTO (e.g. cuboid). The goal position is DYNAMIC: it tracks the base
  object's current position plus a stacking height offset, so the target follows the
  base if it is nudged. Success latches when the moved object is within
  ``success_threshold`` of the stack target in xy AND resting at the correct height
  (i.e. actually stacked, not just hovering).

  Mirrors LiftingCommand's structure (reach-then-bring), but the target is computed
  from the base object each step rather than sampled once, and success also checks
  the height so a hovering object does not count.
  """

  cfg: StackingCommandCfg

  def __init__(self, cfg: StackingCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]
    self.base: Entity = env.scene[cfg.base_asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    self.metrics["goal_error"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["at_goal"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["episode_success"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["reached_object"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["stack_height_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _stack_target(self) -> torch.Tensor:
    """Target = base object position, raised by the stacking height offset."""
    base_pos = self.base.data.root_link_pos_w.clone()
    base_pos[:, 2] = base_pos[:, 2] + self.cfg.stack_height
    return base_pos

  def _update_metrics(self) -> None:
    self.target_pos = self._stack_target()
    object_pos_w = self.object.data.root_link_pos_w

    goal_error = torch.norm(self.target_pos - object_pos_w, dim=-1)
    xy_error = torch.norm(self.target_pos[:, :2] - object_pos_w[:, :2], dim=-1)
    height_error = torch.abs(self.target_pos[:, 2] - object_pos_w[:, 2])

    at_goal = (
      (xy_error < self.cfg.success_threshold)
      & (height_error < self.cfg.height_threshold)
    ).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    reached = (torch.norm(object_pos_w - gripper_pos_w, dim=-1) < 0.10).float()
    self.reached_object = torch.maximum(self.reached_object, reached)

    self.metrics["goal_error"] = goal_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["stack_height_error"] = height_error

  def compute_success(self) -> torch.Tensor:
    return self.metrics["at_goal"] > 0.5

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    r = self.cfg.object_pose_range
    origins = self._env.scene.env_origins[env_ids]

    def _sample_pose(rng) -> torch.Tensor:
      lower = torch.tensor([rng.x[0], rng.y[0], rng.z[0]], device=self.device)
      upper = torch.tensor([rng.x[1], rng.y[1], rng.z[1]], device=self.device)
      pos = sample_uniform(lower, upper, (n, 3), device=self.device) + origins
      yaw = sample_uniform(rng.yaw[0], rng.yaw[1], (n,), device=self.device)
      quat = quat_from_euler_xyz(
        torch.zeros(n, device=self.device), torch.zeros(n, device=self.device), yaw
      )
      return torch.cat([pos, quat], dim=-1)

    # Place the moving object and the base object at separated random poses.
    obj_pose = _sample_pose(r)
    base_pose = _sample_pose(self.cfg.base_pose_range)
    vel = torch.zeros(n, 6, device=self.device)
    self.object.write_root_link_pose_to_sim(obj_pose, env_ids=env_ids)
    self.object.write_root_link_velocity_to_sim(vel, env_ids=env_ids)
    self.base.write_root_link_pose_to_sim(base_pose, env_ids=env_ids)
    self.base.write_root_link_velocity_to_sim(vel, env_ids=env_ids)

    self.target_pos[env_ids] = self._stack_target()[env_ids]

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    for env_idx in range(self.num_envs):
      visualizer.add_sphere(
        center=self.target_pos[env_idx].cpu().numpy(),
        radius=0.02,
        color=self.cfg.viz.target_color,
        label=f"stack_target_{env_idx}",
      )


@dataclass(kw_only=True)
class StackingCommandCfg(CommandTermCfg):
  asset_name: str
  base_asset_name: str
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  class_type: type[CommandTerm] = StackingCommand
  success_threshold: float = 0.03  # xy tolerance (m)
  height_threshold: float = 0.02   # vertical tolerance (m)
  stack_height: float = 0.035      # base-top + moved-object-half-height (m)

  @dataclass
  class ObjectPoseRangeCfg:
    x: tuple[float, float] = (0.55, 0.7)
    y: tuple[float, float] = (-0.15, -0.05)
    z: tuple[float, float] = (0.02, 0.02)
    yaw: tuple[float, float] = (0.0, 0.0)

  object_pose_range: ObjectPoseRangeCfg = field(default_factory=ObjectPoseRangeCfg)

  @dataclass
  class BasePoseRangeCfg:
    x: tuple[float, float] = (0.55, 0.7)
    y: tuple[float, float] = (0.05, 0.15)
    z: tuple[float, float] = (0.015, 0.015)
    yaw: tuple[float, float] = (0.0, 0.0)

  base_pose_range: BasePoseRangeCfg = field(default_factory=BasePoseRangeCfg)

  @dataclass
  class VizCfg:
    target_color: tuple[float, float, float, float] = (1.0, 0.0, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)


##
# Class A motion-profile expansion (see continual_distill/docs/benchmark/
# CLASS_A_EXPANSION.md). Eight new commands covering four new articulation profiles
# and four new manipulation reward/success shapes.
##


class _ArticulationJointCommand(CommandTerm):
  """Shared base for single-DoF articulation tasks (lever/valve/switch/window/lid).

  Generalizes the OpenDrawerCommand pattern: track ONE joint of an articulated asset
  toward a scalar target, latch success, and place a Cartesian mocap marker for viz.

  Subclasses declare ``joint_name`` and the Cartesian offset used to place the goal
  marker; everything else (metrics, latching, reached_object) is shared. Success is
  measured on the JOINT VALUE, not on a Cartesian distance — for rotary joints the
  grasp site can return near its start while the joint has moved a long way.
  """

  cfg: _ArticulationJointCommandCfg

  # Subclass hooks.
  joint_name: str = ""

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.asset: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg

    self.target_value = torch.zeros(self.num_envs, device=self.device)
    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    self.mocap_goal: Entity | None = env.scene["mocap_goal"]

    # Common metrics (same keys as every other manipulation command, so the
    # existing eval/logging path works unchanged).
    for k in (
      "goal_error",
      "at_goal",
      "episode_success",
      "reached_object",
      "gripper_object_distance",
      "joint_value",
    ):
      self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

    self.joint_idx = self.asset.joint_names.index(self.cfg.joint_name)

  @property
  def command(self) -> torch.Tensor:
    return self.target_value.unsqueeze(-1)

  def _joint_value(self) -> torch.Tensor:
    return self.asset.data.joint_pos[:, self.joint_idx]

  def _update_metrics(self) -> None:
    value = self._joint_value()
    # Signed progress toward the target: for a target of -90deg, having gone
    # further than the target still counts as success (clamped error).
    if self.cfg.directional:
      # Error only counts the shortfall in the direction of travel.
      sign = torch.sign(self.target_value)
      shortfall = (self.target_value - value) * sign
      distance_error = shortfall.clamp_min(0.0)
    else:
      distance_error = torch.abs(self.target_value - value)
    at_goal = (distance_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    object_site_idx = self.asset.site_names.index("object_site")
    object_pos_w = self.asset.data.site_pos_w[:, object_site_idx]
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    gripper_object_distance = torch.norm(object_pos_w - gripper_pos_w, dim=-1)
    currently_reached = (gripper_object_distance < 0.10).float()
    self.reached_object = torch.maximum(self.reached_object, currently_reached)

    self.metrics["goal_error"] = distance_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance
    self.metrics["joint_value"] = value

  def compute_success(self) -> torch.Tensor:
    return self.metrics["goal_error"] < self.cfg.success_threshold

  def _goal_marker_offset(self) -> torch.Tensor:
    """Cartesian offset from base_site to the goal marker. Viz only."""
    return torch.tensor(self.cfg.goal_marker_offset, device=self.device)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    self.target_value[env_ids] = self.cfg.target_value

    # Reset the mechanism to its start pose.
    joint_pos = torch.full((n, 1), self.cfg.init_value, device=self.device)
    joint_vel = torch.zeros(n, 1, device=self.device)
    self.asset.write_joint_position_to_sim(joint_pos, env_ids=env_ids)
    self.asset.write_joint_velocity_to_sim(joint_vel, env_ids=env_ids)

    base_site_idx = self.asset.site_names.index("base_site")
    base_site_pos = self.asset.data.site_pos_w[env_ids, base_site_idx]
    self.target_pos[env_ids] = base_site_pos + self._goal_marker_offset()

    if self.mocap_goal is not None:
      target_quats = torch.zeros(n, 4, device=self.device)
      target_quats[:, 0] = 1.0
      mocap_pose = torch.cat([self.target_pos[env_ids].clone(), target_quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(mocap_pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class _ArticulationJointCommandCfg(CommandTermCfg):
  """Shared cfg for the single-DoF articulation commands."""

  asset_name: str = "object"
  joint_name: str = ""
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  target_value: float = 0.0
  init_value: float = 0.0
  success_threshold: float = 0.05
  directional: bool = True
  """If True, overshooting the target still counts as success (shortfall-only error)."""
  goal_marker_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

  @dataclass
  class PoseRangeCfg:
    x: tuple[float, float] = (0.65, 0.65)
    y: tuple[float, float] = (-0.1, 0.1)
    z: tuple[float, float] = (0.5, 0.5)
    yaw: tuple[float, float] = (0.0, 0.0)

  pose_range: PoseRangeCfg | None = field(default_factory=PoseRangeCfg)


class TurnLeverCommand(_ArticulationJointCommand):
  """Turn a lever whose hinge axis points along the approach direction.

  Motion profile: WRIST ROTATION about the approach axis (contrast: the door is a
  whole-arm pull). Target is -90 deg.
  """

  cfg: TurnLeverCommandCfg


@dataclass(kw_only=True)
class TurnLeverCommandCfg(_ArticulationJointCommandCfg):
  asset_name: str = "lever"
  joint_name: str = "lever_hinge"
  class_type: type[CommandTerm] = TurnLeverCommand
  target_value: float = -1.5707963  # -90 deg (fully turned)
  init_value: float = 0.0
  success_threshold: float = 0.15  # rad (~8.6 deg)
  goal_marker_offset: tuple[float, float, float] = (0.0, -0.12, -0.12)


class RotateValveCommand(_ArticulationJointCommand):
  """Rotate a valve wheel through a large angle, forcing a regrasp.

  Motion profile: MULTI-CYCLE rotation. The 270 deg target exceeds the wrist range
  from any single grasp, so the policy must turn, release, re-grasp on the opposite
  spoke, and continue. Success is angular; a Cartesian goal would be ambiguous since
  the spoke tip returns to its start every full turn.
  """

  cfg: RotateValveCommandCfg


@dataclass(kw_only=True)
class RotateValveCommandCfg(_ArticulationJointCommandCfg):
  asset_name: str = "valve"
  joint_name: str = "valve_hinge"
  class_type: type[CommandTerm] = RotateValveCommand
  target_value: float = 4.712389  # 270 deg
  init_value: float = 0.0
  success_threshold: float = 0.2  # rad
  goal_marker_offset: tuple[float, float, float] = (0.0, -0.09, 0.09)


class FlipSwitchCommand(_ArticulationJointCommand):
  """Flip a detented toggle switch past centre.

  Motion profile: BALLISTIC COMMIT. The detent spring pushes the toggle back until it
  crosses centre, so quasi-static servoing fails; success is a state flip, not a
  displacement threshold.
  """

  cfg: FlipSwitchCommandCfg


@dataclass(kw_only=True)
class FlipSwitchCommandCfg(_ArticulationJointCommandCfg):
  asset_name: str = "switch"
  joint_name: str = "switch_hinge"
  class_type: type[CommandTerm] = FlipSwitchCommand
  target_value: float = 0.5235988  # +30 deg — past centre, into the ON state
  init_value: float = -0.7853982  # -45 deg (OFF stop, at the spring rest pose)
  success_threshold: float = 0.15
  goal_marker_offset: tuple[float, float, float] = (0.0, 0.0, 0.04)


class SlideWindowCommand(_ArticulationJointCommand):
  """Slide a window pane along the robot's lateral axis.

  Motion profile: LATERAL FACE-PUSH. Same joint type as the drawer but a different
  axis, and the drawer's hook strategy does not transfer.
  """

  cfg: SlideWindowCommandCfg


@dataclass(kw_only=True)
class SlideWindowCommandCfg(_ArticulationJointCommandCfg):
  asset_name: str = "window"
  joint_name: str = "window_slide"
  class_type: type[CommandTerm] = SlideWindowCommand
  target_value: float = 0.22  # m, near the 0.25 open stop
  init_value: float = 0.0
  success_threshold: float = 0.03  # m
  goal_marker_offset: tuple[float, float, float] = (0.0, 0.22, 0.0)


class OpenLidCommand(_ArticulationJointCommand):
  """Open a hinged box lid against gravity.

  Motion profile: VERTICAL ARC UNDER GRAVITY. Gravity opposes the motion throughout
  and the lid falls shut if released — unlike the door, whose vertical hinge axis
  makes its swing gravity-neutral. Runs with gravity ENABLED.
  """

  cfg: OpenLidCommandCfg


@dataclass(kw_only=True)
class OpenLidCommandCfg(_ArticulationJointCommandCfg):
  asset_name: str = "lid"
  joint_name: str = "lid_hinge"
  class_type: type[CommandTerm] = OpenLidCommand
  target_value: float = -1.308997  # -75 deg (open)
  init_value: float = 0.0
  success_threshold: float = 0.2
  goal_marker_offset: tuple[float, float, float] = (0.10, 0.0, 0.12)


@dataclass
class _ObjectSpawnRangeCfg:
  """Object spawn range for the new Class A manipulation commands.

  Like ``LiftingCommandCfg.ObjectPoseRangeCfg`` but also exposes roll/pitch, which the
  reorient task needs (it must spawn the cylinder LYING DOWN).
  """

  x: tuple[float, float] = (0.45, 0.60)
  y: tuple[float, float] = (-0.15, 0.15)
  z: tuple[float, float] = (0.02, 0.05)
  roll: tuple[float, float] = (0.0, 0.0)
  pitch: tuple[float, float] = (0.0, 0.0)
  yaw: tuple[float, float] = (-math.pi, math.pi)


def _spawn_object(command, entity: Entity, rng, env_ids: torch.Tensor) -> None:
  """Write a uniformly-sampled root pose (and zero velocity) for ``entity``."""
  if rng is None:
    return
  n = len(env_ids)
  device = command.device
  lower = torch.tensor([rng.x[0], rng.y[0], rng.z[0]], device=device)
  upper = torch.tensor([rng.x[1], rng.y[1], rng.z[1]], device=device)
  pos = sample_uniform(lower, upper, (n, 3), device=device)
  pos = pos + command._env.scene.env_origins[env_ids]
  quat = quat_from_euler_xyz(
    sample_uniform(rng.roll[0], rng.roll[1], (n,), device=device),
    sample_uniform(rng.pitch[0], rng.pitch[1], (n,), device=device),
    sample_uniform(rng.yaw[0], rng.yaw[1], (n,), device=device),
  )
  entity.write_root_link_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
  entity.write_root_link_velocity_to_sim(
    torch.zeros(n, 6, device=device), env_ids=env_ids
  )


class PlaceInContainerCommand(CommandTerm):
  """Place a carried object INSIDE an open-top container.

  New success shape: CONTAINMENT, not proximity. The object must be within the bin's
  lateral footprint AND below the rim AND resting (low speed) — i.e. actually released
  into the bin, not merely held above it or clipped through a wall. A plain distance
  threshold would fire while the object is still in the gripper.
  """

  cfg: PlaceInContainerCommandCfg

  def __init__(self, cfg: PlaceInContainerCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]
    self.container: Entity = env.scene[cfg.container_asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Scene has no .get(); mocap goal is optional so probe by key.
    try:
      self.mocap_goal: Entity | None = env.scene["mocap_goal"]
    except KeyError:
      self.mocap_goal = None

    for k in (
      "goal_error",
      "at_goal",
      "episode_success",
      "reached_object",
      "gripper_object_distance",
      "contained",
    ):
      self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _object_pos(self) -> torch.Tensor:
    if "object_site" in self.object.site_names:
      idx = self.object.site_names.index("object_site")
      return self.object.data.site_pos_w[:, idx]
    return self.object.data.root_link_pos_w

  def _update_metrics(self) -> None:
    object_pos = self._object_pos()
    delta = object_pos - self.target_pos

    lateral = torch.norm(delta[:, :2], dim=-1)
    vertical = delta[:, 2]

    # Containment: inside the footprint, below the rim, and above the floor.
    inside_xy = lateral < self.cfg.lateral_tolerance
    below_rim = vertical < self.cfg.rim_height
    above_floor = vertical > -self.cfg.floor_tolerance
    contained = inside_xy & below_rim & above_floor

    # Released and settled (not still being carried).
    speed = torch.norm(self.object.data.root_link_lin_vel_w, dim=-1)
    settled = speed < self.cfg.settle_speed

    at_goal = (contained & settled).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    gripper_object_distance = torch.norm(object_pos - gripper_pos_w, dim=-1)
    self.reached_object = torch.maximum(
      self.reached_object, (gripper_object_distance < 0.05).float()
    )

    # Dense error for logging/obs: 3D distance to the bin interior centre.
    self.metrics["goal_error"] = torch.norm(delta, dim=-1)
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance
    self.metrics["contained"] = contained.float()

  def compute_success(self) -> torch.Tensor:
    return self.metrics["at_goal"] > 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    _spawn_object(self, self.object, self.cfg.object_spawn_range, env_ids)

    # The container is a STATIC MOCAP body. Mocap bodies are not reset by the scene,
    # so unless we write them ourselves every env shares the single world-frame pose
    # baked into the MJCF — i.e. the bin sits at the world origin and only env 0's
    # robot has it in reach. Write it per-env, relative to that env's origin.
    origins = self._env.scene.env_origins[env_ids]
    crng = self.cfg.container_spawn_range
    container_pos = origins + sample_uniform(
      torch.tensor([crng.x[0], crng.y[0], crng.z[0]], device=self.device),
      torch.tensor([crng.x[1], crng.y[1], crng.z[1]], device=self.device),
      (n, 3),
      device=self.device,
    )
    quats = torch.zeros(n, 4, device=self.device)
    quats[:, 0] = 1.0
    self.container.write_mocap_pose_to_sim(
      torch.cat([container_pos, quats], dim=-1), env_ids=env_ids
    )

    # Goal = the container's interior reference site. Computed analytically from the
    # pose we just wrote rather than read back from ``site_pos_w``: forward kinematics
    # has not re-run yet this step, so the cached site position is still the old one.
    site_idx = self.container.site_names.index("object_site")
    site_offset = torch.tensor(
      self.container.spec.sites[site_idx].pos, device=self.device, dtype=torch.float
    )
    self.target_pos[env_ids] = container_pos + site_offset

    if self.mocap_goal is not None:
      pose = torch.cat([self.target_pos[env_ids].clone(), quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class PlaceInContainerCommandCfg(CommandTermCfg):
  asset_name: str = "object"
  container_asset_name: str = "container"
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  class_type: type[CommandTerm] = PlaceInContainerCommand
  lateral_tolerance: float = 0.055
  """Inside the bin footprint (walls are at ~0.062 from centre)."""
  rim_height: float = 0.05
  """Object centre must be below this height above the interior reference site."""
  floor_tolerance: float = 0.04
  """Guard against a below-floor (tunnelled) false positive."""
  settle_speed: float = 0.12
  """Object must be moving slower than this — i.e. released, not carried."""
  container_spawn_range: _ObjectSpawnRangeCfg = field(
    default_factory=lambda: _ObjectSpawnRangeCfg(
      x=(0.55, 0.55), y=(0.20, 0.20), z=(0.02, 0.02), yaw=(0.0, 0.0)
    )
  )
  """Container body placement in ENV-LOCAL coordinates, written per-env every resample.

  The container is a mocap body, so nothing else resets it; the MJCF's own ``pos``
  would otherwise leave it at the world origin for every env but env 0. Defaults to
  the MJCF value so behaviour is unchanged for env 0.
  """
  object_spawn_range: _ObjectSpawnRangeCfg | None = field(
    default_factory=_ObjectSpawnRangeCfg
  )


class ReorientObjectCommand(CommandTerm):
  """Reorient an object in place to a target orientation (topple / stand-up).

  New success shape: ORIENTATION. Success is an ANGULAR alignment of the object's body
  z-axis with a target axis, with only a loose position constraint (the object must
  stay on the table near where it started). Every other Class A task's success is
  positional; this is the only rotational one.
  """

  cfg: ReorientObjectCommandCfg

  def __init__(self, cfg: ReorientObjectCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_axis = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)

    # Scene has no .get(); mocap goal is optional so probe by key.
    try:
      self.mocap_goal: Entity | None = env.scene["mocap_goal"]
    except KeyError:
      self.mocap_goal = None

    for k in (
      "goal_error",
      "at_goal",
      "episode_success",
      "reached_object",
      "gripper_object_distance",
      "axis_alignment",
    ):
      self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_axis

  def _object_pos(self) -> torch.Tensor:
    if "object_site" in self.object.site_names:
      idx = self.object.site_names.index("object_site")
      return self.object.data.site_pos_w[:, idx]
    return self.object.data.root_link_pos_w

  def _update_metrics(self) -> None:
    # Object body z-axis in world frame = third column of its rotation matrix.
    mat = self.object.data.data.xmat[:, self.object.data.indexing.root_body_id]
    obj_z = mat[:, :, 2]

    alignment = torch.sum(obj_z * self.target_axis, dim=-1).clamp(-1.0, 1.0)
    angle_error = torch.acos(alignment)

    object_pos = self._object_pos()
    # Loose positional constraint: the object must not have been flung away.
    drift = torch.norm(object_pos[:, :2] - self.target_pos[:, :2], dim=-1)

    at_goal = (
      (angle_error < self.cfg.angle_threshold) & (drift < self.cfg.max_drift)
    ).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)
    gripper_object_distance = torch.norm(object_pos - gripper_pos_w, dim=-1)
    self.reached_object = torch.maximum(
      self.reached_object, (gripper_object_distance < 0.08).float()
    )

    self.metrics["goal_error"] = angle_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance
    self.metrics["axis_alignment"] = alignment

  def compute_success(self) -> torch.Tensor:
    return self.metrics["at_goal"] > 0.0

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0

    _spawn_object(self, self.object, self.cfg.object_spawn_range, env_ids)

    self.target_pos[env_ids] = self._object_pos()[env_ids]
    axis = torch.tensor(self.cfg.target_axis, device=self.device, dtype=torch.float32)
    self.target_axis[env_ids] = axis / torch.norm(axis)

    if self.mocap_goal is not None:
      quats = torch.zeros(n, 4, device=self.device)
      quats[:, 0] = 1.0
      pos = self.target_pos[env_ids].clone()
      pos[:, 2] += self.cfg.marker_z_offset
      pose = torch.cat([pos, quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class ReorientObjectCommandCfg(CommandTermCfg):
  asset_name: str = "object"
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  class_type: type[CommandTerm] = ReorientObjectCommand
  target_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
  """Desired world-frame direction of the object's body z-axis. The cylinder spawns
  lying down (z-axis horizontal); standing it up means aligning z with world +z."""
  angle_threshold: float = 0.35  # rad (~20 deg)
  max_drift: float = 0.18  # m — object must stay near its start
  marker_z_offset: float = 0.12
  object_spawn_range: _ObjectSpawnRangeCfg | None = field(
    default_factory=_ObjectSpawnRangeCfg
  )


class ToolPullCommand(CommandTerm):
  """Drag an out-of-reach object back into the near zone using a grasped tool.

  New success shape: TWO-STAGE, TOOL-MEDIATED. Success requires the puck to be pulled
  within a target radius of the robot base. The task is only solvable via the stick —
  the puck spawns beyond direct reach — so ``tool_grasped`` is tracked separately and
  used to gate the pulling reward.
  """

  cfg: ToolPullCommandCfg

  def __init__(self, cfg: ToolPullCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.object: Entity = env.scene[cfg.asset_name]  # the puck
    self.tool: Entity = env.scene[cfg.tool_asset_name]  # the stick
    self.robot: Entity = env.scene[cfg.robot_asset_cfg.name]
    self.robot_cfg = cfg.robot_asset_cfg

    self.target_pos = torch.zeros(self.num_envs, 3, device=self.device)
    self.episode_success = torch.zeros(self.num_envs, device=self.device)
    self.reached_object = torch.zeros(self.num_envs, device=self.device)
    self.tool_grasped = torch.zeros(self.num_envs, device=self.device)

    # Scene has no .get(); mocap goal is optional so probe by key.
    try:
      self.mocap_goal: Entity | None = env.scene["mocap_goal"]
    except KeyError:
      self.mocap_goal = None

    for k in (
      "goal_error",
      "at_goal",
      "episode_success",
      "reached_object",
      "gripper_object_distance",
      "tool_grasped",
      "gripper_tool_distance",
    ):
      self.metrics[k] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.target_pos

  def _site_pos(self, entity: Entity, name: str = "object_site") -> torch.Tensor:
    if name in entity.site_names:
      return entity.data.site_pos_w[:, entity.site_names.index(name)]
    return entity.data.root_link_pos_w

  def _update_metrics(self) -> None:
    puck_pos = self._site_pos(self.object)
    tool_pos = self._site_pos(self.tool)
    gripper_pos_w = self.robot.data.site_pos_w[:, self.robot_cfg.site_ids].squeeze(1)

    # Stage 1: tool acquired (latched).
    gripper_tool_distance = torch.norm(tool_pos - gripper_pos_w, dim=-1)
    self.tool_grasped = torch.maximum(
      self.tool_grasped, (gripper_tool_distance < self.cfg.grasp_threshold).float()
    )

    # Stage 2: puck dragged into the near zone.
    distance_error = torch.norm(puck_pos - self.target_pos, dim=-1)
    at_goal = (distance_error < self.cfg.success_threshold).float()
    self.episode_success = torch.maximum(self.episode_success, at_goal)

    gripper_object_distance = torch.norm(puck_pos - gripper_pos_w, dim=-1)
    self.reached_object = torch.maximum(
      self.reached_object, (gripper_object_distance < 0.10).float()
    )

    self.metrics["goal_error"] = distance_error
    self.metrics["at_goal"] = at_goal
    self.metrics["episode_success"] = self.episode_success
    self.metrics["reached_object"] = self.reached_object
    self.metrics["gripper_object_distance"] = gripper_object_distance
    self.metrics["tool_grasped"] = self.tool_grasped
    self.metrics["gripper_tool_distance"] = gripper_tool_distance

  def compute_success(self) -> torch.Tensor:
    return self.metrics["goal_error"] < self.cfg.success_threshold

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    self.episode_success[env_ids] = 0.0
    self.reached_object[env_ids] = 0.0
    self.tool_grasped[env_ids] = 0.0

    _spawn_object(self, self.object, self.cfg.object_spawn_range, env_ids)
    _spawn_object(self, self.tool, self.cfg.tool_spawn_range, env_ids)

    # Goal is a fixed near-zone point in each env's local frame.
    origins = self._env.scene.env_origins[env_ids]
    offset = torch.tensor(self.cfg.goal_offset, device=self.device)
    self.target_pos[env_ids] = origins + offset

    if self.mocap_goal is not None:
      quats = torch.zeros(n, 4, device=self.device)
      quats[:, 0] = 1.0
      pose = torch.cat([self.target_pos[env_ids].clone(), quats], dim=-1)
      self.mocap_goal.write_mocap_pose_to_sim(pose, env_ids=env_ids)

  def _update_command(self) -> None:
    pass

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    pass


@dataclass(kw_only=True)
class ToolPullCommandCfg(CommandTermCfg):
  asset_name: str = "puck"
  tool_asset_name: str = "stick"
  robot_asset_cfg: SceneEntityCfg = field(
    default_factory=lambda: SceneEntityCfg("robot", site_names=())
  )
  class_type: type[CommandTerm] = ToolPullCommand
  goal_offset: tuple[float, float, float] = (0.42, 0.0, 0.012)
  """Near-zone goal in env-local coordinates (well within direct reach).

  The z MUST match the puck's resting height on the ground plane (its half-thickness,
  0.012). An earlier value of 0.31 put the goal in mid-air, 0.3m above where the puck
  physically rests — with a 0.07 threshold the task was unsatisfiable. Keep this tied
  to the puck geometry if either changes.
  """
  success_threshold: float = 0.07
  grasp_threshold: float = 0.06
  object_spawn_range: _ObjectSpawnRangeCfg | None = field(
    default_factory=_ObjectSpawnRangeCfg
  )
  tool_spawn_range: _ObjectSpawnRangeCfg | None = field(
    default_factory=_ObjectSpawnRangeCfg
  )
