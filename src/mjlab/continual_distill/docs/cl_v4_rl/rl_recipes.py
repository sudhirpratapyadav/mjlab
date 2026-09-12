"""Explicit training variants; benchmark geometry, success and interface stay fixed."""

from mjlab.tasks.manipulation.mdp.rewards import articulation_task_reward
from mjlab.tasks.manipulation.mdp.task_geometry import finger_aperture, grasped, tracking_goal, tracking_position
from mjlab.utils.lab_api.math import quat_apply
import torch

RECIPES = ("baseline", "baseline_long", "stable_v1", "mechanism_v1", "lid_v1", "lid_v2", "lift_v1", "reorient_v1", "lift_v2", "lift_v3", "reorient_v2", "reorient_v3", "reorient_v4", "cage_v1", "cage_v2", "completion_v1", "completion_v2")
RECIPES += ("edge_v1", "pivot_v1", "strike_v1", "throw_v1", "completion_v3")
RECIPES += ("edge_v2", "pivot_v2", "throw_v2", "lift_v4", "reorient_v5")
RECIPES += ("cage_v3",)
RECIPES += ("lift_v5", "reorient_v6", "throw_v3")
RECIPES += ("completion_v4",)


def grasp_components(env, command_name, object_asset_name="object", require_enclosure=False, geometry_aperture=False, **kwargs):
  command = env.command_manager.get_term(command_name)
  robot = command.robot
  obj = env.scene[object_asset_name]
  gripper = robot.data.site_pos_w[:, command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:, command.robot_cfg.site_ids].squeeze(1)
  target = tracking_position(obj).clone()
  target[:,2] += 0.006
  distance = torch.linalg.vector_norm(gripper-target, dim=-1)
  down = -quat_apply(quat,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))[:,2]
  approach = torch.exp(-distance/0.10) * (0.5 + 0.5 * down.clamp(0,1))
  aperture = finger_aperture(robot)
  closing = torch.exp(-distance/0.025) * (1-aperture/0.08).clamp(0,1)
  if geometry_aperture:
    from completion_reward import capture_aperture_bonus
    closing = capture_aperture_bonus(command,distance)
  if require_enclosure:
    from completion_reward import enclosed_grasp
    held = enclosed_grasp(command).float()
  else:
    held = grasped(command).float()
  return command, obj, approach, closing, held


def lift_grasp_reward(env, command_name, object_asset_name="object", closure_weight=0.5, **kwargs):
  command, obj, approach, closing, held = grasp_components(env,command_name,object_asset_name,**kwargs)
  # Height relative to the scene's floor; above 12cm is an unmistakable lift.
  height = obj.data.root_link_pos_w[:,2] - env.scene.env_origins[:,2]
  lift = ((height-0.02)/0.10).clamp(0,1)*held
  goal_error = torch.linalg.vector_norm(tracking_goal(command)-tracking_position(obj),dim=-1)
  goal = torch.exp(-goal_error/0.12)*held
  grasp_bonus = 4 if kwargs.get('geometry_aperture',False) else 2
  return approach + closure_weight*closing + grasp_bonus*held + 3*lift + 5*goal


def reorient_grasp_reward(env, command_name, object_asset_name="object", **kwargs):
  command, obj, approach, closing, held = grasp_components(env,command_name,object_asset_name)
  axis = quat_apply(obj.data.root_link_quat_w,command.body_axis.expand(env.num_envs,3))
  alignment = (axis*command.target_axis).sum(-1).clamp(-1,1)
  if command.cfg.symmetric_axis:
    alignment = alignment.abs()
  drift = torch.linalg.vector_norm(tracking_position(obj)[:,:2]-command.target_pos[:,:2],dim=-1)
  valid = (drift<command.cfg.max_drift).float()
  orientation = ((alignment+1)*0.5).square()*valid
  return approach + 0.5*closing + held + 4*orientation


def apply_recipe(cfg, recipe):
  if recipe == "baseline":
    return
  parent = {"lift_v5":"lift_v4", "reorient_v6":"reorient_v5", "throw_v3":"throw_v2", "completion_v4":"completion_v3"}.get(recipe)
  if parent:
    apply_recipe(cfg,parent)
    name = "stack" if "stack" in cfg.env.rewards else "reach_object"
    cfg.env.rewards[name].params["geometry_aperture"] = True
    cfg.agent.policy.initial_mean = (*cfg.agent.policy.initial_mean[:7], .5)
    cfg.agent.policy.initial_gripper_std = .15
    return
  if recipe not in RECIPES:
    raise ValueError(recipe)
  if cfg.env.curriculum:
    cfg.env.curriculum.pop("joint_vel_penalty_weight", None)
  if recipe == "baseline_long":
    return
  # The old scalar std can become nonpositive. Log std keeps it positive;
  # a fixed, smaller LR avoids the adaptive schedule's high-LR excursions.
  cfg.agent.policy.noise_std_type = "log"
  cfg.agent.algorithm.learning_rate = 3e-4
  cfg.agent.algorithm.schedule = "fixed"
  cfg.agent.algorithm.gamma = 0.995
  # Freeze the initial regularization for learning beyond the old 1000/1500
  # update boundaries instead of abruptly increasing it by 10x and 100x.
  if recipe in ("mechanism_v1", "lid_v1", "lid_v2"):
    reach = cfg.env.rewards["reach_object"]
    if reach.func is not articulation_task_reward:
      raise ValueError("mechanism_v1 requires an articulation task")
    reach.params["approach_scale"] = 0.3
    cfg.agent.policy.init_noise_std = 0.1
    if "no_object_collision" in cfg.env.rewards:
      cfg.env.rewards["no_object_collision"].weight = 0.0
    cfg.env.rewards["action_rate_l2"].weight = -0.005
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    if recipe in ("lid_v1", "lid_v2"):
      if reach.params["command_name"] != "open_lid":
        raise ValueError("lid_v1 requires Open-Lid")
      reach.func = lid_grasp_progress_reward
      if recipe == "lid_v2":
        bounded_initialization(cfg)
        cfg.agent.policy.initial_mean = (*cfg.agent.policy.initial_mean[:7], .5)
        cfg.agent.policy.initial_gripper_std = .15
  if recipe in ("lift_v1", "reorient_v1", "lift_v2", "lift_v3", "lift_v4", "reorient_v2", "reorient_v3", "reorient_v4", "reorient_v5"):
    reach = cfg.env.rewards["reach_object"]
    expected = "lift_object" if recipe.startswith("lift_") else "reorient_object"
    if reach.params["command_name"] != expected:
      raise ValueError(f"{recipe} is restricted to {expected}")
    reach.func = lift_grasp_reward if recipe.startswith("lift_") else reorient_grasp_reward
    if recipe in ("lift_v3", "lift_v4"):
      reach.params["closure_weight"] = 2.0
    if recipe in ("lift_v4", "reorient_v5"):
      reach.params["require_enclosure"] = True
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    cfg.env.rewards["action_rate_l2"].weight = -0.005
  if recipe in ("lift_v2", "lift_v3", "lift_v4", "reorient_v2", "reorient_v3", "reorient_v4", "reorient_v5", "cage_v1", "cage_v2", "cage_v3"):
    bounded_initialization(cfg)
    if recipe in ("reorient_v2", "reorient_v3", "reorient_v4", "reorient_v5"):
      cfg.env.rewards["reach_object"].func = reorient_endface_reward
      if recipe in ("reorient_v3", "reorient_v4", "reorient_v5"):
        cfg.env.rewards["reach_object"].params["smooth_closure"] = True
      if recipe in ("reorient_v4", "reorient_v5"):
        cfg.env.rewards["reach_object"].params["closure_weight"] = 3.0
    if recipe in ("cage_v1", "cage_v2", "cage_v3"):
      reach = cfg.env.rewards["reach_object"]
      if reach.params["command_name"] != "cage_drag":
        raise ValueError("cage_v1 requires the cage task")
      reach.func = cage_approach_reward
      if recipe == "cage_v2":
        reach.params["transport_guidance"] = True
      if recipe == "cage_v3":
        reach.func = cage_contact_reward
      cfg.env.rewards["joint_vel_penalty"].weight = -0.001
      cfg.env.rewards["action_rate_l2"].weight = -0.005
  if recipe in ("completion_v1", "completion_v2", "completion_v3"):
    from completion_reward import completion_reward
    if cfg.agent.experiment_name not in ("franka_stack_cube", "franka_peg_insertion", "franka_place_in_container"):
      raise ValueError("completion_v1 is restricted to Stack/Place/Peg")
    bounded_initialization(cfg)
    cfg.agent.policy.initial_mean = (*cfg.agent.policy.initial_mean[:7], 0.5)
    cfg.agent.policy.initial_gripper_std = 0.1
    name = "stack" if "stack" in cfg.env.rewards else "reach_object"
    cfg.env.rewards[name].func = completion_reward
    if recipe in ("completion_v2", "completion_v3"):
      cfg.env.rewards[name].params["smooth_closure"] = True
    if recipe == "completion_v3":
      cfg.env.rewards[name].params["require_enclosure"] = True
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    cfg.env.rewards["action_rate_l2"].weight = -0.005
  if recipe in ("edge_v1", "pivot_v1", "strike_v1", "throw_v1", "edge_v2", "pivot_v2", "throw_v2"):
    from remaining_reward import edge_reward, pivot_reward, strike_reward, throw_reward
    choices = {
      "edge_v1": ("franka_edge_grasp", edge_reward),
      "pivot_v1": ("franka_pivot_lift", pivot_reward),
      "strike_v1": ("franka_strike_slide", strike_reward),
      "throw_v1": ("franka_throw_to_bin", throw_reward),
    }
    expected, reward = choices[recipe.replace('_v2','_v1')]
    if cfg.agent.experiment_name != expected:
      raise ValueError(f"{recipe} requires {expected}")
    bounded_initialization(cfg)
    cfg.agent.policy.initial_mean = (*cfg.agent.policy.initial_mean[:7], .5)
    cfg.agent.policy.initial_gripper_std = .15
    cfg.env.rewards["reach_object"].func = reward
    if recipe.endswith('_v2'):
      cfg.env.rewards["reach_object"].params["require_enclosure"] = True
    cfg.env.rewards["joint_vel_penalty"].weight = -.0001 if recipe in ("strike_v1", "throw_v1", "throw_v2") else -.001
    cfg.env.rewards["action_rate_l2"].weight = -.002


def bounded_initialization(cfg):
  from dataclasses import asdict
  from bounded_policy import BoundedPolicyCfg, register
  register()
  action = cfg.env.actions['robot_joint_pos']
  joints = cfg.env.scene.entities['robot'].init_state.joint_pos
  names = [f'joint{i}' for i in range(1,8)] + ['finger_joint1']
  mean = tuple(max(-0.99,min(0.99,(joints[name]-action.offset[name])/action.scale[name])) for name in names)
  values = asdict(cfg.agent.policy)
  values['class_name'] = 'BoundedActorCritic'
  cfg.agent.policy = BoundedPolicyCfg(**values, initial_mean=mean, initial_gripper_std=0.03)
  cfg.agent.policy.init_noise_std = 0.10
  cfg.agent.algorithm.learning_rate = 1e-4
  cfg.agent.algorithm.entropy_coef = 0.0
  cfg.agent.algorithm.max_grad_norm = 0.5


def reorient_endface_reward(env, command_name, object_asset_name='object', smooth_closure=False, closure_weight=1.0, require_enclosure=False, geometry_aperture=False, **kwargs):
  command, obj, _, _, held = grasp_components(env,command_name,object_asset_name,require_enclosure=require_enclosure)
  robot = command.robot
  gripper = robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  target = tracking_position(obj).clone()
  target[:,2] += 0.006
  distance = torch.linalg.vector_norm(gripper-target,dim=-1)
  down = -quat_apply(quat,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))[:,2]
  closing_axis = quat_apply(quat,torch.tensor([0.,1.,0.],device=env.device).expand(env.num_envs,3))
  object_axis = quat_apply(obj.data.root_link_quat_w,command.body_axis.expand(env.num_envs,3))
  parallel = (closing_axis*object_axis).sum(-1).abs().clamp(0,1)
  approach = torch.exp(-distance/0.12)*(0.2+0.8*down.clamp(0,1))*(0.2+0.8*parallel)
  aperture = finger_aperture(robot)
  # Keep pads open during approach, then close on the bottle's end faces.
  desired = torch.where(distance>0.035,0.075,0.050)
  aperture_score = torch.exp(-((aperture-desired)/0.02).square())
  if smooth_closure:
    from completion_reward import smooth_closure_bonus
    aperture_score = smooth_closure_bonus(distance,aperture)
  if geometry_aperture:
    from completion_reward import capture_aperture_bonus
    aperture_score = capture_aperture_bonus(command,distance)
  height = obj.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
  lift = ((height-0.015)/0.10).clamp(0,1)*held
  aligned = (object_axis*command.target_axis).sum(-1).clamp(0,1)
  valid = torch.linalg.vector_norm(tracking_position(obj)[:,:2]-command.target_pos[:,:2],dim=-1)<command.cfg.max_drift
  orientation = aligned.square()*held*valid.float()
  # Actual contact must still dominate the maximum noncontact closure score.
  grasp_bonus = max(2.,1+closure_weight)+(2 if geometry_aperture else 0)
  return approach*(1-held)*(1+closure_weight*aperture_score) + grasp_bonus*held + 3*lift + 6*orientation


def cage_approach_reward(env, command_name, object_asset_name='cube', transport_guidance=False, **kwargs):
  from mjlab.tasks.manipulation.mdp.task_geometry import between_fingers
  command = env.command_manager.get_term(command_name)
  valid = torch.minimum(command.min_aperture,command._aperture())>command.cfg.aperture_min
  robot = command.robot
  gripper = robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  target = tracking_position(command.object).clone()
  target[:,2] += 0.008
  distance = torch.linalg.vector_norm(gripper-target,dim=-1)
  down = -quat_apply(quat,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))[:,2]
  approach = torch.exp(-distance/0.15)*(0.25+0.75*down.clamp(0,1))
  caged = between_fingers(command).float()
  error = torch.linalg.vector_norm(tracking_position(command.object)-command.target_pos,dim=-1)
  transport = caged*(1+3*torch.exp(-error/0.10))
  if transport_guidance:
    grip_goal = command.target_pos.clone()
    grip_goal[:,2] += 0.008
    hand_error = torch.linalg.vector_norm(gripper-grip_goal,dim=-1)
    progress = (command.caged_progress/command.required_progress.clamp_min(0.001)).clamp(0,1)
    transport += caged*(2*torch.exp(-hand_error/0.15)+2*progress)
    transport += 15*command.compute_success().float()
  return (approach+transport)*valid.float()


def lid_grasp_progress_reward(env, command_name, object_asset_name='object', **kwargs):
  from completion_reward import smooth_closure_bonus
  command = env.command_manager.get_term(command_name)
  base = articulation_task_reward(env,command_name,object_asset_name,**kwargs)
  grip = command.robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  distance = torch.linalg.vector_norm(grip-tracking_position(command.asset),dim=-1)
  closure = smooth_closure_bonus(distance,finger_aperture(command.robot))
  held = grasped(command,command.asset).float()*torch.exp(-distance/0.05)
  error = (command.target_value-command._joint_value()).abs()
  progress = (1-error/command.target_value.abs().clamp_min(.01)).clamp(0,1)
  return base+2*closure+3*held+6*progress


def cage_contact_target(position, corners, pads, grip, direction):
  """Put the trailing inner pad at the cube's rear face, keeping an open cage."""
  projected = ((corners-position[:,None])*direction[:,None]).sum(-1)
  half_width = .5*(projected.amax(1)-projected.amin(1))
  half_gap = .5*torch.linalg.vector_norm(pads[:,0]-pads[:,1],dim=-1)-.0076
  # One millimetre of contact drive fits the native two-millimetre cage tolerance.
  shift = (half_gap-half_width).clamp(0,.06)+.001
  return position+shift[:,None]*direction-(pads.mean(1)-grip)


def cage_contact_reward(env, command_name, **kwargs):
  from mjlab.tasks.manipulation.mdp.task_geometry import object_corners, touching
  command = env.command_manager.get_term(command_name)
  robot, obj = command.robot, command.object
  pos = tracking_position(obj)
  grip = robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
  quat = robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
  ids = [robot.geom_names.index(n) for n in ('left_finger_pad','right_finger_pad')]
  pads = robot.data.geom_pos_w[:,ids]
  direction = command.target_pos-pos
  direction = direction.clone(); direction[:,2] = 0
  error = torch.linalg.vector_norm(direction,dim=-1)
  direction = direction/error[:,None].clamp_min(.001)
  target = cage_contact_target(pos,object_corners(obj),pads,grip,direction)
  distance = torch.linalg.vector_norm(grip-target,dim=-1)
  down_axis = quat_apply(quat,torch.tensor([0.,0.,1.],device=env.device).expand(env.num_envs,3))
  close_axis = quat_apply(quat,torch.tensor([0.,1.,0.],device=env.device).expand(env.num_envs,3))
  alignment = (.25+.75*(-down_axis[:,2]).clamp(0,1))*(.25+.75*(close_axis*direction).sum(-1).abs())
  approach = (torch.exp(-distance/.15)+2*torch.exp(-distance/.03))*alignment
  side_height = torch.exp(-((pads.mean(1)[:,2]-pos[:,2])/.02).square())
  contact = touching(command,robot,obj,('left_finger_pad','right_finger_pad')).float()*side_height*alignment
  progress = (command.caged_progress/command.required_progress.clamp_min(.001)).clamp(0,1)
  valid = torch.minimum(command.min_aperture,command._aperture())>command.cfg.aperture_min
  # Do not remove all transport guidance when stochastic contact briefly crosses
  # the geometric cage boundary. Actual native history still gates completion.
  return (approach+3*contact+4*torch.exp(-error/.15)+4*progress+20*command.compute_success().float())*valid.float()
