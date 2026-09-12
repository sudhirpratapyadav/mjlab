"""Training-only grasp geometry from opposing pad contacts and local ray width."""
import torch
from mjlab.tasks.manipulation.mdp.task_geometry import object_corners, tracking_position
from mjlab.utils.lab_api.math import quat_apply_inverse


def opposed_contact_mask(geoms, normals, distances, worlds, count, pad_ids, object_ids, pad_positions):
  """Each pad needs a real object contact whose normal points into the gap.

  MJWarp contact.frame[:,0] points from geom[0] to geom[1]. Unused slots and
  contacts belonging to another world/object cannot establish a grasp.
  """
  n = pad_positions.shape[0]
  valid = ((torch.arange(len(geoms),device=geoms.device)<count)
           & (distances<=.001) & (worlds>=0) & (worlds<n))
  ids = worlds.long().clamp(0,n-1)
  sides=[]
  for side in (0,1):
    first = (geoms[:,0]==pad_ids[side]) & torch.isin(geoms[:,1],object_ids)
    second = (geoms[:,1]==pad_ids[side]) & torch.isin(geoms[:,0],object_ids)
    inward = pad_positions[:,1-side]-pad_positions[:,side]
    inward = inward/torch.linalg.vector_norm(inward,dim=-1,keepdim=True).clamp_min(1e-8)
    normal = normals*torch.where(first,1.,-1.)[:,None]
    opposed = (normal*inward[ids]).sum(-1)>.5
    hits = torch.zeros(n,device=geoms.device,dtype=torch.long)
    hits.scatter_add_(0,ids,(valid & (first|second) & opposed).long())
    sides.append(hits>0)
  return sides[0] & sides[1]


def opposed_grasp(command):
  robot = command.robot
  names = ('left_finger_pad','right_finger_pad')
  local_ids = [robot.geom_names.index(name) for name in names]
  pad_ids = robot.data.indexing.geom_ids[local_ids]
  data = command._env.sim.data
  c = data.contact
  return opposed_contact_mask(c.geom[:],c.frame[:,0,:],c.dist[:],c.worldid[:],data.nacon[0],
                              pad_ids,command.object.data.indexing.geom_ids,robot.data.geom_pos_w[:,local_ids])


def box_ray_width(origin, direction, lower, upper):
  """Span of a line through a local collider box, with explicit miss handling."""
  parallel = direction.abs()<1e-6
  safe = torch.where(parallel,torch.ones_like(direction),direction)
  a,b = (lower-origin)/safe,(upper-origin)/safe
  near,far = torch.minimum(a,b),torch.maximum(a,b)
  near = torch.where(parallel,torch.full_like(near,-float('inf')),near)
  far = torch.where(parallel,torch.full_like(far,float('inf')),far)
  entry,exit = near.amax(-1),far.amin(-1)
  valid = (exit>=entry) & ~(parallel & ((origin<lower)|(origin>upper))).any(-1)
  valid &= direction.square().sum(-1)>1e-8
  return torch.where(valid,(exit-entry).clamp_min(0),torch.zeros_like(entry)),valid


def pad_centerline_width(command):
  """Local box cross-section at pad center; full projection is a miss fallback.

  Collider bounds are a shaping approximation for meshes, never a contact or
  completion predicate. Actual opposed contacts determine grasp credit.
  """
  robot,obj = command.robot,command.object
  local_ids = [robot.geom_names.index(name) for name in ('left_finger_pad','right_finger_pad')]
  pads = robot.data.geom_pos_w[:,local_ids]
  direction = pads[:,1]-pads[:,0]
  direction /= torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(1e-8)
  q = obj.data.root_link_quat_w
  local_origin = quat_apply_inverse(q,pads.mean(1)-tracking_position(obj))
  local_direction = quat_apply_inverse(q,direction)
  corners = object_corners(obj)
  support = obj._task_support_points
  width,valid = box_ray_width(local_origin,local_direction,support.amin(0),support.amax(0))
  projected = (corners*direction[:,None]).sum(-1)
  fallback = projected.amax(1)-projected.amin(1)
  return torch.where(valid,width,fallback)
