"""Explicit training variants; benchmark geometry, success and interface stay fixed."""

from mjlab.tasks.manipulation.mdp.rewards import articulation_task_reward
from mjlab.tasks.manipulation.mdp.task_geometry import finger_aperture, grasped, tracking_goal, tracking_position
from mjlab.utils.lab_api.math import quat_apply
import torch

RECIPES = ("baseline", "baseline_long", "stable_v1", "mechanism_v1", "lift_v1", "reorient_v1", "lift_v2", "reorient_v2", "reorient_v3", "cage_v1", "completion_v1", "completion_v2")


def grasp_components(env, command_name, object_asset_name="object", **kwargs):
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
  held = grasped(command).float()
  return command, obj, approach, closing, held


def lift_grasp_reward(env, command_name, object_asset_name="object", **kwargs):
  command, obj, approach, closing, held = grasp_components(env,command_name,object_asset_name)
  # Height relative to the scene's floor; above 12cm is an unmistakable lift.
  height = obj.data.root_link_pos_w[:,2] - env.scene.env_origins[:,2]
  lift = ((height-0.02)/0.10).clamp(0,1)*held
  goal_error = torch.linalg.vector_norm(tracking_goal(command)-tracking_position(obj),dim=-1)
  goal = torch.exp(-goal_error/0.12)*held
  return approach + 0.5*closing + 2*held + 3*lift + 5*goal


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
  if recipe == "mechanism_v1":
    reach = cfg.env.rewards["reach_object"]
    if reach.func is not articulation_task_reward:
      raise ValueError("mechanism_v1 requires an articulation task")
    reach.params["approach_scale"] = 0.3
    cfg.agent.policy.init_noise_std = 0.1
    if "no_object_collision" in cfg.env.rewards:
      cfg.env.rewards["no_object_collision"].weight = 0.0
    cfg.env.rewards["action_rate_l2"].weight = -0.005
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
  if recipe in ("lift_v1", "reorient_v1", "lift_v2", "reorient_v2", "reorient_v3"):
    reach = cfg.env.rewards["reach_object"]
    expected = "lift_object" if recipe.startswith("lift_") else "reorient_object"
    if reach.params["command_name"] != expected:
      raise ValueError(f"{recipe} is restricted to {expected}")
    reach.func = lift_grasp_reward if recipe.startswith("lift_") else reorient_grasp_reward
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    cfg.env.rewards["action_rate_l2"].weight = -0.005
  if recipe in ("lift_v2", "reorient_v2", "reorient_v3", "cage_v1"):
    bounded_initialization(cfg)
    if recipe in ("reorient_v2", "reorient_v3"):
      cfg.env.rewards["reach_object"].func = reorient_endface_reward
      if recipe == "reorient_v3":
        cfg.env.rewards["reach_object"].params["smooth_closure"] = True
    if recipe == "cage_v1":
      reach = cfg.env.rewards["reach_object"]
      if reach.params["command_name"] != "cage_drag":
        raise ValueError("cage_v1 requires the cage task")
      reach.func = cage_approach_reward
      cfg.env.rewards["joint_vel_penalty"].weight = -0.001
      cfg.env.rewards["action_rate_l2"].weight = -0.005
  if recipe in ("completion_v1", "completion_v2"):
    from completion_reward import completion_reward
    if cfg.agent.experiment_name not in ("franka_stack_cube", "franka_peg_insertion", "franka_place_in_container"):
      raise ValueError("completion_v1 is restricted to Stack/Place/Peg")
    bounded_initialization(cfg)
    cfg.agent.policy.initial_mean = (*cfg.agent.policy.initial_mean[:7], 0.5)
    cfg.agent.policy.initial_gripper_std = 0.1
    name = "stack" if "stack" in cfg.env.rewards else "reach_object"
    cfg.env.rewards[name].func = completion_reward
    if recipe == "completion_v2":
      cfg.env.rewards[name].params["smooth_closure"] = True
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    cfg.env.rewards["action_rate_l2"].weight = -0.005


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


def reorient_endface_reward(env, command_name, object_asset_name='object', smooth_closure=False, **kwargs):
  command, obj, _, _, held = grasp_components(env,command_name,object_asset_name)
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
  height = obj.data.root_link_pos_w[:,2]-env.scene.env_origins[:,2]
  lift = ((height-0.015)/0.10).clamp(0,1)*held
  aligned = (object_axis*command.target_axis).sum(-1).clamp(0,1)
  valid = torch.linalg.vector_norm(tracking_position(obj)[:,:2]-command.target_pos[:,:2],dim=-1)<command.cfg.max_drift
  orientation = aligned.square()*held*valid.float()
  return approach*(1-held)*(1+aperture_score) + 2*held + 3*lift + 6*orientation


def cage_approach_reward(env, command_name, object_asset_name='cube', **kwargs):
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
  return (approach+transport)*valid.float()
