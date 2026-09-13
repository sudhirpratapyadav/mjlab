"""Grasp, transport and released completion shaping under the native predicates."""
import torch
from mjlab.tasks.manipulation.mdp.task_geometry import (
  between_fingers, finger_aperture, grasped, object_corners, released, resting_site_height,
  touching, tracking_goal, tracking_position,
)
from mjlab.tasks.manipulation.mdp.rewards import _goal_orientation_factor
from mjlab.utils.lab_api.math import quat_apply


def safe_grasp_target(command):
  target = tracking_position(command.object).clone()
  if getattr(command.cfg, 'insertion', False):
    # The peg tracking site is its insertion tip. Grasp the upper body instead.
    corners = object_corners(command.object)
    low, high = corners[:,:,2].amin(1), corners[:,:,2].amax(1)
    target[:,2] = low + 0.70*(high-low)
  else:
    target[:,2] += 0.006
  return target


def completion_score(approach, aperture_match, held, lift, transport, released_near_support, strict, lift_weight=2.):
  """Native completion dominates every physically valid noncompleted stage."""
  return (approach*(1+0.5*aperture_match) + 2*held + lift_weight*lift*held
          + 5*transport*held + 6*released_near_support*(1-held) + 15*strict)


def smooth_closure_bonus(distance, aperture):
  """Monotone approach/closure incentive without an open-to-close reward cliff."""
  return torch.exp(-distance/0.04)*(1-aperture/0.08).clamp(0,1)


def enclosed_grasp(command):
  """Training grasp credit requires opposed contact, not two pads pressing the top."""
  return grasped(command) & between_fingers(command)


def aperture_fit_score(distance, gap, object_width, squeeze=False):
  """Smooth capture clearance; closing empty fingers should not win approach."""
  clearance = (-.004 if squeeze else .003)+.020*(1-torch.exp(-distance/.05))
  return torch.exp(-distance/.15)*torch.exp(-((gap-object_width-clearance)/.025).square())


def capture_aperture_bonus(command, distance, squeeze=False, centerline=False, centered_fallback=False):
  robot, obj = command.robot, command.object
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  closing_axis = quat_apply(quat,torch.tensor([0.,1.,0.],device=command.device).expand(command.num_envs,3))
  projection = ((object_corners(obj)-tracking_position(obj)[:,None])*closing_axis[:,None]).sum(-1)
  width = projection.amax(1)-projection.amin(1)
  if centerline:
    from contact_grasp import pad_centerline_width
    width = pad_centerline_width(command,centered_fallback=centered_fallback)
  ids = [robot.geom_names.index(n) for n in ('left_finger_pad','right_finger_pad')]
  pads = robot.data.geom_pos_w[:,ids]
  gap = torch.linalg.vector_norm(pads[:,0]-pads[:,1],dim=-1)-.0152
  return aperture_fit_score(distance,gap,width,squeeze=squeeze)


def completion_reward(env, command_name, object_asset_name='object', smooth_closure=False, require_enclosure=False, geometry_aperture=False, squeeze_capture=False, contact_geometry=False, centered_fallback=False, lift_weight=2., **kwargs):
  command = env.command_manager.get_term(command_name)
  robot, obj = command.robot, command.object
  gripper = robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  distance = torch.linalg.vector_norm(gripper-safe_grasp_target(command),dim=-1)
  down = -quat_apply(quat,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))[:,2]
  approach = (0.7*torch.exp(-distance/0.25)+0.3*torch.exp(-distance/0.04))*(0.25+0.75*down.clamp(0,1))
  aperture = finger_aperture(robot)
  if smooth_closure:
    aperture_match = smooth_closure_bonus(distance,aperture)
  else:
    desired = torch.where(distance>0.035,0.07,0.025)
    aperture_match = torch.exp(-((aperture-desired)/0.025).square())
  if geometry_aperture:
    aperture_match = capture_aperture_bonus(command,distance,squeeze=squeeze_capture,centerline=contact_geometry,centered_fallback=centered_fallback)
  held = (enclosed_grasp(command) if require_enclosure else grasped(command)).float()
  if contact_geometry:
    from contact_grasp import opposed_grasp
    held = opposed_grasp(command).float()
  pos = tracking_position(obj)
  height = pos[:,2]-env.scene.env_origins[:,2]-resting_site_height(obj)
  lift = (height/0.10).clamp(0,1)
  error = torch.linalg.vector_norm(tracking_goal(command)-pos,dim=-1)
  orientation = _goal_orientation_factor(command)
  transport = torch.exp(-error/0.15)*orientation
  support = command.base if hasattr(command,'base') else command.container
  released_near_support = (released(command)&touching(command,obj,support)).float()*torch.exp(-error/0.04)*orientation
  # These tasks have stateless completion predicates plus reporting latches.
  # Refresh them at the current physics state so the bonus uses the exact rule,
  # including release, support, containment/bore fit and settling.
  command._update_metrics()
  strict = command.compute_success().float()
  return completion_score(approach,aperture_match,held,lift,transport,released_near_support,strict,lift_weight=lift_weight)
