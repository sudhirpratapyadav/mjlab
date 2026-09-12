"""Translate shared policy state/actions at the legacy scripted-teacher boundary."""

import numpy as np

from mjlab.tasks.manipulation.franka_interface import CENTER, HALF_RANGE, SLICES


def legacy_observation(obs, default_qpos, layout):
  """Adapt shared 60D inputs to the existing scripted controller layouts.

  The old Stack/Peg base-position slot is unused by those controllers and is zero.
  Tool's independently randomized pose cannot be reconstructed from puck state.
  Explicit 69D privileged teacher inputs are still accepted for Tool.
  """
  if layout == "tool":
    if obs.shape[-1] == 69:
      return obs
    raise ValueError(
      "Tool-Pull's scripted teacher requires the separate tool pose (legacy 69D input). "
      "The shared 60D policy observation contains only puck and goal state; "
      "it cannot reconstruct the tool pose."
    )
  if obs.shape[-1] != 60:
    raise ValueError(f"Expected shared 60D observation, got {obs.shape[-1]}")
  if layout == "standard":
    return obs

  def field(name):
    return obs[:, SLICES[name]]

  common = [field("joint_position"), field("joint_velocity")]
  grip = [field("gripper_position"), field("gripper_rotation")]
  if layout == "reach":
    parts = (
      common
      + grip
      + [field("gripper_to_object") + field("object_to_goal"), field("control_error")]
    )
  elif layout == "stack":
    parts = common + [
      field("object_position"),
      field("object_quaternion"),
      np.zeros_like(field("object_position")),
    ]
    parts += grip + [
      field("gripper_to_object"),
      field("object_to_goal"),
      field("control_error"),
    ]
  else:
    raise ValueError(f"Unknown teacher layout: {layout}")
  return np.concatenate(parts, axis=-1)


def shared_teacher(policy_cls):
  """Opt registry teachers into the new contract without guessing from width.

  Both old and new standard observations are 60D, but their action units differ.
  Direct legacy policy classes retain their original behavior.
  """
  return type(
    f"Shared{policy_cls.__name__}",
    (policy_cls,),
    {
      "shared_observations": True,
      "__module__": policy_cls.__module__,
    },
  )


def normalized_actions(actions, default_qpos):
  targets = np.asarray(actions, dtype=np.float64) * 0.04 + np.asarray(default_qpos[:8])
  # Legacy open +1 requests .08m; preserve its physical .04m saturated meaning.
  return np.clip(
    (targets - np.asarray(CENTER[:8])) / np.asarray(HALF_RANGE[:8]), -1, 1
  ).astype(np.float32)
