"""Versioned, task-independent Franka observation and action contract.

Positions are environment-local meters; rotations retain world-aligned axes.
The working 60D observation layout is shared across every task.
"""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.envs.mdp.actions.joint_actions import JointPositionAction
from mjlab.managers.manager_term_config import ObservationGroupCfg, ObservationTermCfg
from mjlab.tasks.manipulation.mdp.task_geometry import tracking_goal, tracking_position

VERSION = "franka_shared_60_v2"
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8)) + (
  "finger_joint1",
  "finger_joint2",
)
# Panda physical position/control limits. A single mapping regardless of reset pose.
LOWER = (-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0)
UPPER = (2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04)
CENTER = tuple((a + b) / 2 for a, b in zip(LOWER, UPPER, strict=True))
HALF_RANGE = tuple((b - a) / 2 for a, b in zip(LOWER, UPPER, strict=True))
FIELDS = {
  "joint_position": 9,
  "joint_velocity": 9,
  "object_position": 3,
  "object_quaternion": 4,
  "gripper_position": 3,
  "gripper_rotation": 6,
  "object_rotation": 6,
  "gripper_to_object": 3,
  "object_to_goal": 3,
  "goal_orientation_diff": 6,
  "control_error": 8,
}
SLICES = {}
OBS_DIM = 0
for _name, _width in FIELDS.items():
  SLICES[_name] = slice(OBS_DIM, OBS_DIM + _width)
  OBS_DIM += _width


class NormalizedFrankaPositionAction(JointPositionAction):
  """Clip in normalized space once per control step, then set absolute targets."""

  def process_actions(self, actions):
    super().process_actions(actions.clamp(-1.0, 1.0))


@dataclass(kw_only=True)
class NormalizedFrankaPositionActionCfg(JointPositionActionCfg):
  class_type: type = NormalizedFrankaPositionAction


def apply_shared_interface(cfg):
  """Finalize a registered train/play/test config after task-specific construction."""
  if len(cfg.commands or {}) != 1:
    raise ValueError("The shared Franka interface requires exactly one task command")
  command_name = next(iter(cfg.commands))
  command = cfg.commands[command_name]
  params = dict(
    command_name=command_name,
    object_name=getattr(command, "asset_name", None),
  )
  # Exact state for this first RL phase, including precision insertion. Domain
  # randomization/noise must later use one shared, unit-aware model, not task layouts.
  cfg.observations = {
    group: ObservationGroupCfg(
      {VERSION: ObservationTermCfg(func=shared_observation, params=dict(params))},
      enable_corruption=False,
    )
    for group in ("policy", "critic")
  }
  cfg.actions = {
    "robot_joint_pos": NormalizedFrankaPositionActionCfg(
      asset_name="robot",
      actuator_names=JOINT_NAMES[:8],
      scale=dict(zip(JOINT_NAMES[:8], HALF_RANGE[:8], strict=True)),
      offset=dict(zip(JOINT_NAMES[:8], CENTER[:8], strict=True)),
      use_default_offset=False,
    )
  }


def _rotation(entity):
  return entity.data.data.xmat[:, entity.data.indexing.root_body_id].reshape(-1, 9)


def shared_observation(env, command_name, object_name=None):
  """Working 60D layout, with positions translated into each environment's frame.

  Joint state is relative to the robot defaults in physical rad/m and rad/s/m/s,
  as in the original working policy. No task-specific state or extra channels.
  Reach uses its target as a virtual object; Tool-Pull observes the puck only.
  """
  robot = env.scene["robot"]
  command = env.command_manager.get_term(command_name)
  ids = [robot.joint_names.index(n) for n in JOINT_NAMES]
  q = robot.data.joint_pos[:, ids]
  grip_id = robot.site_names.index("gripper")
  grip = robot.data.site_pos_w[:, grip_id]
  grip_mat = robot.data.data.site_xmat[
    :, robot.data.indexing.site_ids[grip_id]
  ].reshape(-1, 9)
  goal = tracking_goal(command)
  goal_entity = env.scene.entities.get("mocap_goal")
  goal_mat = (
    _rotation(goal_entity)
    if goal_entity is not None
    else torch.eye(3, device=env.device).reshape(1, 9).expand(env.num_envs, -1)
  )
  if object_name is None:
    # The reach marker supplies the same object/goal fields as manipulation tasks.
    pos = goal
    obj_mat = goal_mat
    quat = (
      goal_entity.data.root_link_quat_w
      if goal_entity is not None
      else q.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(env.num_envs, -1)
    )
  else:
    obj = env.scene[object_name]
    pos = tracking_position(obj)
    quat = obj.data.root_link_quat_w
    obj_mat = _rotation(obj)
  ctrl = robot.data.data.ctrl[:, robot.data.indexing.ctrl_ids]
  return torch.cat(
    (
      q - robot.data.default_joint_pos[:, ids],
      robot.data.joint_vel[:, ids] - robot.data.default_joint_vel[:, ids],
      pos - env.scene.env_origins,
      quat,
      grip - env.scene.env_origins,
      grip_mat[:, 3:],
      obj_mat[:, 3:],
      pos - grip,
      goal - pos,
      goal_mat[:, :6] - obj_mat[:, :6],
      ctrl - q[:, :8],
    ),
    dim=-1,
  )
