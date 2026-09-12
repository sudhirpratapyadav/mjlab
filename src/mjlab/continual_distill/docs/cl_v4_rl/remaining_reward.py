"""Training-only precursor and flight shaping; native completion stays authoritative.

Contact dimensions below come from the registered plate/ledge/board/puck assets.
Predictions are shaping approximations, never replacement success predicates.
"""
import math
import torch
from mjlab.tasks.manipulation.mdp.task_geometry import (
  finger_aperture, grasped, object_corners, released, touching,
  tracking_goal, tracking_position,
)
from mjlab.utils.lab_api.math import quat_apply
from completion_reward import safe_grasp_target, smooth_closure_bonus


def sliding_endpoint(position, velocity, friction, gravity=9.81):
  speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
  return position + velocity*speed/(2*friction*gravity)


def ballistic_endpoint(position, velocity, target_height, gravity=9.81):
  """Descending crossing of the target plane, with an explicit reachability mask."""
  discriminant = velocity[:,2].square()+2*gravity*(position[:,2]-target_height)
  time = (velocity[:,2]+discriminant.clamp_min(0).sqrt())/gravity
  valid = (discriminant >= 0) & (time > 0)
  return position[:,:2]+velocity[:,:2]*time.clamp_min(0)[:,None], valid


def hand_geometry(env, command):
  robot = command.robot
  pos = robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  axes = [quat_apply(quat,torch.tensor(v,device=env.device).expand(env.num_envs,3))
          for v in ([1.,0.,0.],[0.,1.,0.],[0.,0.,1.])]
  return pos, axes, finger_aperture(robot)


def native_success(command):
  command._update_metrics()
  return command.compute_success().float()


def strike_reward(env, command_name, **kwargs):
  from mjlab.tasks.manipulation.config.franka.env_cfgs import STRIKE_PUCK_MU
  command = env.command_manager.get_term(command_name)
  pos, goal = tracking_position(command.object), tracking_goal(command)
  velocity = command.object.data.root_link_lin_vel_w
  direction = goal[:,:2]-pos[:,:2]
  direction = direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(.001)
  grip, axes, aperture = hand_geometry(env,command)
  target = pos.clone()
  target[:,:2] -= .0353*direction  # Symmetric partially open paddle behind the rim.
  target[:,2] = env.scene.env_origins[:,2]+.028
  distance = torch.linalg.vector_norm(grip-target,dim=-1)
  down = (-axes[2][:,2]).clamp(0,1)
  square = (axes[0][:,:2]*direction).sum(-1).abs().clamp(0,1)
  wedge = torch.exp(-((aperture-.060)/.025).square())
  approach = torch.exp(-distance/.20)*(.25+.75*down)*(.25+.75*square)*(1+.5*wedge)
  predicted = sliding_endpoint(pos[:,:2],velocity[:,:2],STRIKE_PUCK_MU)
  prediction_error = torch.linalg.vector_norm(predicted-goal[:,:2],dim=-1)
  actual_error = torch.linalg.vector_norm(pos-goal,dim=-1)
  return approach+4*torch.exp(-prediction_error/.25)+3*torch.exp(-actual_error/.20)+20*native_success(command)


def throw_reward(env, command_name, **kwargs):
  command = env.command_manager.get_term(command_name)
  obj = command.object
  grip, axes, aperture = hand_geometry(env,command)
  pos, goal = tracking_position(obj), tracking_goal(command)
  distance = torch.linalg.vector_norm(grip-safe_grasp_target(command),dim=-1)
  approach = torch.exp(-distance/.20)*(.25+.75*(-axes[2][:,2]).clamp(0,1))
  closure = smooth_closure_bonus(distance,aperture)
  held = grasped(command).float()
  height = (pos[:,2]-env.scene.env_origins[:,2]-.023).clamp_min(0)
  lift = (height/.25).clamp(0,1)
  # Predict the rim crossing, not a trajectory through a bin wall below the rim.
  rim_plane = command.container.data.root_link_pos_w[:,2]+command.cfg.inner_rim_z+.03
  landing, valid = ballistic_endpoint(pos,obj.data.root_link_lin_vel_w,rim_plane)
  aim = torch.exp(-torch.linalg.vector_norm(landing-goal[:,:2],dim=-1)/.15)*valid.float()
  airborne = (released(command) & (height>.10)).float()
  return (approach*(1-held)+2*closure*(1-held)+4*held+3*lift*held
          +6*aim*held+12*aim*airborne+25*native_success(command))


def edge_reward(env, command_name, **kwargs):
  command = env.command_manager.get_term(command_name)
  obj = command.object
  pos = tracking_position(obj)
  grip, axes, aperture = hand_geometry(env,command)
  far_axis = quat_apply(command.ledge.data.root_link_quat_w,
                       torch.tensor([1.,0.,0.],device=env.device).expand(env.num_envs,3))
  relative_x = ((pos-command.ledge_center)*far_axis).sum(-1)
  # Near rim overhang = .075 - .100 - relative_x. Aim for 5cm of exposure.
  exposure = (-relative_x/.075).clamp(0,1)
  top = command.ledge_center[:,2]+command.cfg.ledge_top_height
  at_ledge = (pos[:,2]>top+.004).float()
  push_target = pos+.075*far_axis
  push_target[:,2] += .010
  pinch_target = pos-.065*far_axis
  push_distance = torch.linalg.vector_norm(grip-push_target,dim=-1)
  pinch_distance = torch.linalg.vector_norm(grip-pinch_target,dim=-1)
  push = torch.exp(-push_distance/.15)*(.25+.75*(-axes[2][:,2]).clamp(0,1))
  pinch = torch.exp(-pinch_distance/.10)*(.25+.75*axes[1][:,2].abs())
  pinch *= 1+smooth_closure_bonus(pinch_distance,aperture)
  precursor = ((1-exposure)*push+exposure*pinch+4*exposure)*at_ledge
  held = grasped(command).float()
  lift = ((pos[:,2]-top)/.10).clamp(0,1)
  goal = torch.exp(-torch.linalg.vector_norm(pos-tracking_goal(command),dim=-1)/.15)
  return precursor*(1-held)+8*held+6*lift*held+4*goal*held+25*native_success(command)


def pivot_reward(env, command_name, **kwargs):
  command = env.command_manager.get_term(command_name)
  obj = command.object
  pos = tracking_position(obj)
  grip, axes, aperture = hand_geometry(env,command)
  board_x = quat_apply(obj.data.root_link_quat_w,torch.tensor([1.,0.,0.],device=env.device).expand(env.num_envs,3))
  board_z = quat_apply(obj.data.root_link_quat_w,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))
  tilt = (1-board_z[:,2].abs()).clamp(0,1)
  psi = math.radians(70)-math.radians(15)*(tilt/.43).clamp(0,1)
  expected_closing = torch.stack([psi.sin(),torch.zeros_like(psi),psi.cos()],dim=-1)
  expected_approach = torch.stack([psi.cos(),torch.zeros_like(psi),-psi.sin()],dim=-1)
  corner = pos-.05*board_x+.010*board_z
  ramp_target = corner+.5*aperture[:,None]*expected_closing-.005*expected_approach
  distance = torch.linalg.vector_norm(grip-ramp_target,dim=-1)
  alignment = (axes[1]*expected_closing).sum(-1).abs().clamp(0,1)
  approach = torch.exp(-distance/.15)*(.25+.75*alignment)
  wall_face = command.wall.data.root_link_pos_w[:,0]-.0525
  board_far = object_corners(obj)[:,:,0].amax(1)
  wall_near = torch.exp(-(wall_face-board_far).clamp_min(0)/.08)
  contact = touching(command,obj,command.wall).float()
  held = grasped(command).float()
  goal = torch.exp(-torch.linalg.vector_norm(pos-tracking_goal(command),dim=-1)/.20)
  lift = ((pos[:,2]-env.scene.env_origins[:,2]-.01)/.12).clamp(0,1)
  precursor = approach+2*wall_near+4*tilt*contact
  return precursor*(1-held)+8*held+5*lift*held+5*goal*held+25*native_success(command)
