"""Explicit training variants; benchmark geometry, success and interface stay fixed."""

from mjlab.tasks.manipulation.mdp.rewards import articulation_task_reward
from mjlab.tasks.manipulation.mdp.task_geometry import finger_aperture, grasped, tracking_goal, tracking_position
from mjlab.utils.lab_api.math import quat_apply
import torch

RECIPES = ("baseline", "baseline_long", "stable_v1", "mechanism_v1", "lift_v1", "reorient_v1")


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
  if recipe in ("lift_v1", "reorient_v1"):
    reach = cfg.env.rewards["reach_object"]
    expected = "lift_object" if recipe == "lift_v1" else "reorient_object"
    if reach.params["command_name"] != expected:
      raise ValueError(f"{recipe} is restricted to {expected}")
    reach.func = lift_grasp_reward if recipe == "lift_v1" else reorient_grasp_reward
    cfg.env.rewards["joint_vel_penalty"].weight = -0.001
    cfg.env.rewards["action_rate_l2"].weight = -0.005
