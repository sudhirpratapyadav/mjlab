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
from completion_reward import capture_aperture_bonus, enclosed_grasp, safe_grasp_target, smooth_closure_bonus


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


def strike_approach_target(position, direction, floor_height, contact_drive=0.):
  """Reward waypoint measured from the pre-contact paddle seating position.

  With the60mm wedge, the old35.3mm standoff stops about3mm before pad
  contact. A positive drive advances only the reward target, never body poses.
  """
  target = position.clone()
  target[:,:2] -= (.0353-contact_drive)*direction
  target[:,2] = floor_height+.028
  return target


def strike_endpoint_precision(prediction_error, actual_error, speed):
  """Fine endpoint credit; retain broad shaping and native success separately."""
  return torch.exp(-prediction_error/.05)+torch.exp(-actual_error/.05)/(1+speed/.05)


def strike_reward(env, command_name, contact_drive=0., precision_weight=0., endpoint_precision_weight=0., native_weight=20., **kwargs):
  from mjlab.tasks.manipulation.config.franka.env_cfgs import STRIKE_PUCK_MU
  command = env.command_manager.get_term(command_name)
  pos, goal = tracking_position(command.object), tracking_goal(command)
  velocity = command.object.data.root_link_lin_vel_w
  direction = goal[:,:2]-pos[:,:2]
  direction = direction/torch.linalg.vector_norm(direction,dim=-1,keepdim=True).clamp_min(.001)
  grip, axes, aperture = hand_geometry(env,command)
  target = strike_approach_target(pos,direction,env.scene.env_origins[:,2],contact_drive)
  distance = torch.linalg.vector_norm(grip-target,dim=-1)
  down = (-axes[2][:,2]).clamp(0,1)
  square = (axes[0][:,:2]*direction).sum(-1).abs().clamp(0,1)
  wedge = torch.exp(-((aperture-.060)/.025).square())
  # The recorded weak/nonlaunch policies stop6-8cm above the contact height.
  # A near-target term increases the value of completing that last approach.
  approach = (torch.exp(-distance/.20)+precision_weight*torch.exp(-distance/.05))
  approach *= (.25+.75*down)*(.25+.75*square)*(1+.5*wedge)
  predicted = sliding_endpoint(pos[:,:2],velocity[:,:2],STRIKE_PUCK_MU)
  prediction_error = torch.linalg.vector_norm(predicted-goal[:,:2],dim=-1)
  actual_error = torch.linalg.vector_norm(pos-goal,dim=-1)
  reward = approach+4*torch.exp(-prediction_error/.25)+3*torch.exp(-actual_error/.20)+native_weight*native_success(command)
  if endpoint_precision_weight:
    reward += endpoint_precision_weight*strike_endpoint_precision(prediction_error,actual_error,torch.linalg.vector_norm(velocity[:,:2],dim=-1))
  return reward


def throw_reward(env, command_name, require_enclosure=False, geometry_aperture=False, contact_geometry=False, **kwargs):
  command = env.command_manager.get_term(command_name)
  obj = command.object
  grip, axes, aperture = hand_geometry(env,command)
  pos, goal = tracking_position(obj), tracking_goal(command)
  distance = torch.linalg.vector_norm(grip-safe_grasp_target(command),dim=-1)
  approach = torch.exp(-distance/.20)*(.25+.75*(-axes[2][:,2]).clamp(0,1))
  closure = smooth_closure_bonus(distance,aperture)
  if geometry_aperture:
    closure = capture_aperture_bonus(command,distance,squeeze=contact_geometry,centerline=contact_geometry)
  held = (enclosed_grasp(command) if require_enclosure else grasped(command)).float()
  if contact_geometry:
    from contact_grasp import opposed_grasp
    held = opposed_grasp(command).float()
  height = (pos[:,2]-env.scene.env_origins[:,2]-.023).clamp_min(0)
  lift = (height/.25).clamp(0,1)
  # Predict the rim crossing, not a trajectory through a bin wall below the rim.
  rim_plane = command.container.data.root_link_pos_w[:,2]+command.cfg.inner_rim_z+.03
  landing, valid = ballistic_endpoint(pos,obj.data.root_link_lin_vel_w,rim_plane)
  aim = torch.exp(-torch.linalg.vector_norm(landing-goal[:,:2],dim=-1)/.15)*valid.float()
  airborne = (released(command) & (height>.10)).float()
  return (approach*(1-held)+2*closure*(1-held)+4*held+3*lift*held
          +6*aim*held+12*aim*airborne+25*native_success(command))


def edge_side_alignment(axes, far_axis):
  """Best full wrist orientation among three checked side-approach headings."""
  side_axis = torch.stack([-far_axis[:,1],far_axis[:,0],torch.zeros_like(far_axis[:,0])],dim=-1)
  down = torch.zeros_like(far_axis)
  down[:,2] = -1
  pitch = math.radians(20)
  scores = []
  for angle in (0.,math.radians(70),-math.radians(70)):
    heading = math.cos(angle)*far_axis+math.sin(angle)*side_axis
    desired = (torch.cross(down,heading,dim=-1),
               math.cos(pitch)*down-math.sin(pitch)*heading,
               math.cos(pitch)*heading+math.sin(pitch)*down)
    # (trace(R_expected^T R_actual)+1)/4 = cos(rotation_error/2)^2.
    scores.append(((sum((actual*expected).sum(-1) for actual,expected in zip(axes,desired))+1)/4).clamp(0,1))
  return torch.stack(scores).amax(0)


def edge_side_pinch_score(distance, alignment, orientation_power=1.):
  alignment = alignment.pow(orientation_power)
  return (torch.exp(-distance/.15)*(.25+.75*alignment)
          +.5*torch.exp(-distance/.04)*alignment)


def edge_arrival_aperture_score(distance, aperture, capture_fit):
  wide = torch.exp(-((aperture-.070)/.025).square())
  arrival = torch.sigmoid((distance-.015)/.005)
  return arrival*wide+(1-arrival)*capture_fit


def edge_clearance_waypoint(pinch_target, hand_position, far_axis, clearance):
  """Training waypoint: clear the exposed rim before descending into the pinch."""
  high = torch.sigmoid((hand_position[:,2]-pinch_target[:,2]-.020)/.005)
  return pinch_target-clearance*high[:,None]*far_axis


def edge_reward(env, command_name, require_enclosure=False, side_wrist=False, contact_geometry=False, pinch_weight=1., orientation_power=1., held_weight=8., wide_approach=False, arrival_clearance=0., **kwargs):
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
  if side_wrist:
    pinch_target[:,2] += .005
  if arrival_clearance:
    pinch_target = edge_clearance_waypoint(pinch_target,grip,far_axis,arrival_clearance)
  push_distance = torch.linalg.vector_norm(grip-push_target,dim=-1)
  pinch_distance = torch.linalg.vector_norm(grip-pinch_target,dim=-1)
  push = torch.exp(-push_distance/.15)*(.25+.75*(-axes[2][:,2]).clamp(0,1))
  pinch = torch.exp(-pinch_distance/.10)*(.25+.75*axes[1][:,2].abs())
  if side_wrist:
    alignment = edge_side_alignment(axes,far_axis)
    pinch = edge_side_pinch_score(pinch_distance,alignment,orientation_power)
  closure = smooth_closure_bonus(pinch_distance,aperture)
  if contact_geometry:
    closure = capture_aperture_bonus(command,pinch_distance,squeeze=True,centerline=True)
  if wide_approach:
    closure = edge_arrival_aperture_score(pinch_distance,aperture,closure)
  pinch *= pinch_weight*(1+closure)
  precursor = ((1-exposure)*push+exposure*pinch+4*exposure)*at_ledge
  held = (enclosed_grasp(command) if require_enclosure else grasped(command)).float()
  if contact_geometry:
    from contact_grasp import opposed_grasp
    held = opposed_grasp(command).float()
  lift = ((pos[:,2]-top)/.10).clamp(0,1)
  goal = torch.exp(-torch.linalg.vector_norm(pos-tracking_goal(command),dim=-1)/.15)
  return precursor*(1-held)+held_weight*held+6*lift*held+4*goal*held+25*native_success(command)


def pivot_ramp_target(corner, aperture, closing, approach, floor):
  """Reward waypoint with a1mm floor margin for the registered Panda pads."""
  target = corner+.5*aperture[:,None]*closing-.005*approach
  lowest = floor+.001+(.5*aperture+.0152)*closing[:,2].abs()+.0119*approach[:,2].abs()
  target[:,2] = torch.maximum(target[:,2],lowest)
  return target


def pivot_approach_score(distance, axes, closing, approach, aperture, tilt):
  lateral = torch.cross(closing,approach,dim=-1)
  alignment = ((sum((actual*expected).sum(-1) for actual,expected in zip(axes,(lateral,closing,approach)))+1)/4).clamp(0,1)
  # Wide jaws let the lower pad reach the end face while the upper clears the
  # flat board. As the board rises toward55deg, reduce toward its20mm thickness.
  desired = .080-.060*(tilt/.43).clamp(0,1)
  opening = torch.exp(-((aperture-desired)/.025).square())
  return (torch.exp(-distance/.15)+.5*torch.exp(-distance/.04))*(.25+.75*alignment)*(.5+.5*opening)


def pivot_wrist_angle(tilt, early_transition=False):
  scale = 1-math.cos(math.radians(10)) if early_transition else .43
  return math.radians(70)-math.radians(15)*(tilt/scale).clamp(0,1)


def pivot_wall_tilt_score(tilt, contact):
  return (tilt/(1-math.cos(math.radians(20)))).clamp(0,1)*contact


def pivot_reward(env, command_name, require_enclosure=False, ramp_geometry=False, contact_geometry=False, wrist_transition=False, **kwargs):
  command = env.command_manager.get_term(command_name)
  obj = command.object
  pos = tracking_position(obj)
  grip, axes, aperture = hand_geometry(env,command)
  board_x = quat_apply(obj.data.root_link_quat_w,torch.tensor([1.,0.,0.],device=env.device).expand(env.num_envs,3))
  board_z = quat_apply(obj.data.root_link_quat_w,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))
  tilt = (1-board_z[:,2].abs()).clamp(0,1)
  psi = pivot_wrist_angle(tilt, wrist_transition)
  expected_closing = torch.stack([psi.sin(),torch.zeros_like(psi),psi.cos()],dim=-1)
  expected_approach = torch.stack([psi.cos(),torch.zeros_like(psi),-psi.sin()],dim=-1)
  corner = pos-.05*board_x+.010*board_z
  if wrist_transition:
    # Actual board collider half-width60mm plus an outside2mm reward standoff.
    corner = pos-.062*board_x+.010*board_z
  ramp_target = corner+.5*aperture[:,None]*expected_closing-.005*expected_approach
  if ramp_geometry:
    ramp_target = pivot_ramp_target(corner,aperture,expected_closing,expected_approach,env.scene.env_origins[:,2])
  distance = torch.linalg.vector_norm(grip-ramp_target,dim=-1)
  alignment = (axes[1]*expected_closing).sum(-1).abs().clamp(0,1)
  approach = torch.exp(-distance/.15)*(.25+.75*alignment)
  if ramp_geometry:
    approach = pivot_approach_score(distance,axes,expected_closing,expected_approach,aperture,tilt)
  wall_face = command.wall.data.root_link_pos_w[:,0]-.0525
  board_far = object_corners(obj)[:,:,0].amax(1)
  wall_near = torch.exp(-(wall_face-board_far).clamp_min(0)/.08)
  contact = touching(command,obj,command.wall).float()
  held = (enclosed_grasp(command) if require_enclosure else grasped(command)).float()
  if contact_geometry:
    from contact_grasp import opposed_grasp
    held = opposed_grasp(command).float()
  goal = torch.exp(-torch.linalg.vector_norm(pos-tracking_goal(command),dim=-1)/.20)
  lift = ((pos[:,2]-env.scene.env_origins[:,2]-.01)/.12).clamp(0,1)
  precursor = approach+2*wall_near+4*tilt*contact
  if wrist_transition:
    precursor += 8*pivot_wall_tilt_score(tilt,contact)
  return precursor*(1-held)+8*held+5*lift*held+5*goal*held+25*native_success(command)
