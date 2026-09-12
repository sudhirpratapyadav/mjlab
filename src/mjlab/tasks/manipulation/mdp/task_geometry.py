"""Shared coordinate conventions and geometric task predicates."""

import torch

from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse


def tracking_position(entity):
  if "object_site" in entity.site_names:
    return entity.data.site_pos_w[:, entity.site_names.index("object_site")]
  return entity.data.root_link_pos_w


def tracking_goal(command):
  """Desired tracking-site position; target_pos may instead specify a root pose."""
  return getattr(command, "target_site_pos", command.target_pos)


def finger_aperture(robot):
  names = ("finger_joint1", "finger_joint2")
  if not all(name in robot.joint_names for name in names):
    return None
  return robot.data.joint_pos[:, [robot.joint_names.index(n) for n in names]].sum(-1)


def released(command):
  """No actual robot/object collision contact; slow carrying is not release."""
  return ~touching(command, command.robot, command.object)


def settled(entity, linear=0.03, angular=0.3):
  return (
    torch.linalg.vector_norm(entity.data.root_link_lin_vel_w, dim=-1) < linear
  ) & (torch.linalg.vector_norm(entity.data.root_link_ang_vel_w, dim=-1) < angular)


def touching(command, first, second, first_geom_names=None):
  """Per-world actual collision contact, excluding unused contact-buffer slots."""
  data = command._env.sim.data
  contact = data.contact
  a = first.data.indexing.geom_ids
  if first_geom_names is not None:
    a = a[[first.geom_names.index(name) for name in first_geom_names]]
  b = second.data.indexing.geom_ids
  geoms = contact.geom[:]
  pairs = (torch.isin(geoms[:, 0], a) & torch.isin(geoms[:, 1], b)) | (
    torch.isin(geoms[:, 1], a) & torch.isin(geoms[:, 0], b)
  )
  valid = (
    (torch.arange(len(geoms), device=command.device) < data.nacon[0])
    & (contact.dist[:] <= 0.001)
    & pairs
  )
  worlds = contact.worldid[:].long().clamp(0, command.num_envs - 1)
  hits = torch.zeros(command.num_envs, dtype=torch.long, device=command.device)
  hits.scatter_add_(0, worlds, valid.long())
  return hits > 0


def grasped(command, entity=None):
  """Both finger pads contact the object; free-flight proximity does not qualify."""
  entity = command.object if entity is None else entity
  robot = command.robot
  pads = ("left_finger_pad", "right_finger_pad")
  if not all(name in robot.geom_names for name in pads):
    return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
  return touching(command, robot, entity, (pads[0],)) & touching(
    command, robot, entity, (pads[1],)
  )


def object_corners(entity):
  """Collider support points in world coordinates, cached in the entity frame."""
  if not hasattr(entity, "_task_support_points"):
    from mjlab.asset_zoo.objects.goal import object_support_points

    entity._task_support_points = torch.as_tensor(
      object_support_points(entity.cfg.spec_fn()),
      device=entity.data.root_link_pos_w.device,
      dtype=torch.float32,
    )
  points = entity._task_support_points
  n = entity.data.root_link_pos_w.shape[0]
  q = entity.data.root_link_quat_w[:, None, :].expand(n, len(points), 4)
  p = points[None].expand(n, -1, -1)
  return (
    quat_apply(q.reshape(-1, 4), p.reshape(-1, 3)).reshape(n, -1, 3)
    + tracking_position(entity)[:, None]
  )


def resting_site_height(entity):
  """Upright tracking-site height above a supporting plane, from collider extents."""
  object_corners(entity)  # populate the collider cache
  return -entity._task_support_points[:, 2].min()


def between_fingers(command, entity=None):
  """Object spans the pad height and overlaps their width inside the open gap.

  Uses actual Panda pad centers and the object's collider support in the gripper
  frame. This is a geometric cage relation, not a force-closure estimate.
  """
  entity = command.object if entity is None else entity
  robot = command.robot
  if not all(n in robot.geom_names for n in ("left_finger_pad", "right_finger_pad")):
    return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
  origin = robot.data.site_pos_w[:, command.robot_cfg.site_ids].squeeze(1)
  q = robot.data.site_quat_w[:, command.robot_cfg.site_ids].squeeze(1)
  world = object_corners(entity)
  n, count = world.shape[:2]
  local = quat_apply_inverse(
    q[:, None].expand(n, count, 4).reshape(-1, 4),
    (world - origin[:, None]).reshape(-1, 3),
  ).reshape(n, count, 3)
  lo, hi = local.amin(1), local.amax(1)
  ids = [
    robot.geom_names.index(name) for name in ("left_finger_pad", "right_finger_pad")
  ]
  pads = robot.data.geom_pos_w[:, ids]
  pads = quat_apply_inverse(
    q[:, None].expand(n, 2, 4).reshape(-1, 4), (pads - origin[:, None]).reshape(-1, 3)
  ).reshape(n, 2, 3)
  # 8.8 mm pad half-width; 8.2 mm half-height; pad y thickness 7.6 mm.
  ylo = pads[:, :, 1].amin(1) + 0.0076
  yhi = pads[:, :, 1].amax(1) - 0.0076
  center = pads.mean(1)
  return (
    (lo[:, 1] >= ylo - 0.002)
    & (hi[:, 1] <= yhi + 0.002)
    & (lo[:, 0] < center[:, 0] + 0.0088)
    & (hi[:, 0] > center[:, 0] - 0.0088)
    & (lo[:, 2] < center[:, 2] + 0.0082)
    & (hi[:, 2] > center[:, 2] - 0.0082)
  )
