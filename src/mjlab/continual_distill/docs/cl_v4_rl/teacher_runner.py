"""PPO diagnostics that stop at the first nonfinite rollout or optimizer update."""

import copy
import json
import math
from collections import deque
from pathlib import Path
from unittest.mock import patch

import torch
from rsl_rl.runners import OnPolicyRunner


def separately_clipped_update(policy, update):
  """Bound actor/noise and critic gradients independently for one PPO update.

  The optional intervention is confined to this process and update. The shared
  library, objective, Adam state and KL rollback logic are unchanged.
  """
  groups = [[], []]
  for name, parameter in policy.named_parameters():
    if name.startswith('actor.') or name in ('log_std', 'std'):
      groups[0].append(parameter)
    elif name.startswith('critic.'):
      groups[1].append(parameter)
    else:
      raise ValueError(f'Unclassified PPO parameter: {name}')
  expected = {id(parameter) for group in groups for parameter in group}
  if not all(groups):
    raise ValueError('Separate clipping requires disjoint actor and critic parameters')
  original = torch.nn.utils.clip_grad_norm_

  def clip(parameters, max_norm, norm_type=2., **kwargs):
    values = list(parameters)
    if len(values) != len(expected) or {id(p) for p in values} != expected:
      raise ValueError('Separate clipping expects exactly this policy parameter set')
    if norm_type != 2.:
      raise ValueError('Separate PPO clipping currently requires L2 norm')
    norms = [original(group, max_norm, norm_type=norm_type, **kwargs) for group in groups]
    # Preserve the normal API's pre-clipping combined norm return value.
    return torch.stack(norms).norm(2)

  with patch('torch.nn.utils.clip_grad_norm_', clip):
    return update()


def scale_policy_exploration(policy, optimizer, factor):
  """Scale learned action noise once; preserve all deterministic behavior."""
  if not math.isfinite(factor) or factor <= 0:
    raise ValueError("Exploration scale must be finite and positive")
  if policy.noise_std_type != 'log' or policy.log_std.shape != (8,):
    raise ValueError("Expected independent log std for the approved eight actions")
  with torch.no_grad():
    policy.log_std.add_(math.log(factor))
    for name in ('exp_avg', 'exp_avg_sq', 'max_exp_avg_sq'):
      moment = optimizer.state.get(policy.log_std, {}).get(name)
      if moment is not None:
        moment.zero_()


def reset_gripper_exploration(policy, optimizer, std):
  """Change only the eighth action's exploration and its Adam moments."""
  if not math.isfinite(std) or std <= 0:
    raise ValueError("Gripper exploration std must be finite and positive")
  parameter = policy.log_std if policy.noise_std_type == "log" else policy.std
  if parameter.shape != (8,):
    raise ValueError("Expected the approved eight-action policy")
  with torch.no_grad():
    parameter[-1] = math.log(std) if policy.noise_std_type == "log" else std
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
      moment = optimizer.state.get(parameter, {}).get(name)
      if moment is not None:
        moment[-1] = 0


def reset_arm_exploration(policy, optimizer, std):
  """Reset seven arm noise scales; retain gripper noise and policy means."""
  if not math.isfinite(std) or std <= 0:
    raise ValueError("Arm exploration std must be finite and positive")
  parameter = policy.log_std if policy.noise_std_type == "log" else policy.std
  if parameter.shape != (8,):
    raise ValueError("Expected the approved eight-action policy")
  with torch.no_grad():
    parameter[:7] = math.log(std) if policy.noise_std_type == "log" else std
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
      moment = optimizer.state.get(parameter, {}).get(name)
      if moment is not None:
        moment[:7] = 0


def reset_gripper_output(policy, optimizer, mean):
  """Reopen a saturated gripper output while preserving the seven arm outputs."""
  if not math.isfinite(mean) or not -1 < mean < 1:
    raise ValueError("Gripper mean must be finite and strictly between -1 and 1")
  if not isinstance(policy.actor[-1],torch.nn.Tanh):
    raise ValueError("Gripper output reset requires a bounded-mean actor")
  head = policy.actor[-2]
  if not isinstance(head,torch.nn.Linear) or head.out_features != 8:
    raise ValueError("Expected the approved eight-action output layer")
  with torch.no_grad():
    head.weight[-1].zero_()
    head.bias[-1] = math.atanh(mean)
    for parameter in (head.weight,head.bias):
      for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        moment = optimizer.state.get(parameter,{}).get(name)
        if moment is not None:
          moment[-1].zero_()


@torch.no_grad()
def distribution_drift(policy, observations, old_mean, old_std):
  """Read-only KL(old || current) on a fixed set of actual rollout states."""
  mean = policy.act_inference(observations)
  std = policy.log_std.exp()
  kl = (torch.log(std / old_std) + (old_std.square() + (old_mean-mean).square()) / (2*std.square()) - .5).sum(-1)
  return {"kl_mean": float(kl.mean()), "kl_max": float(kl.max()),
          "mean_step_in_old_std_rms": float(((mean-old_mean)/old_std).square().mean().sqrt())}


def guarded_update(algorithm, update, measure, maximum_kl, max_attempts=8, learning_rate_ceiling=None):
  """Backtrack a fixed-schedule PPO update, retaining only an accepted state."""
  if algorithm.schedule != 'fixed' or not math.isfinite(maximum_kl) or maximum_kl <= 0:
    raise ValueError('KL guard requires a fixed schedule and positive finite limit')
  if learning_rate_ceiling is not None and (not math.isfinite(learning_rate_ceiling) or learning_rate_ceiling <= 0):
    raise ValueError('Learning rate ceiling must be finite and positive')
  state = copy.deepcopy(algorithm.policy.state_dict())
  optimizer = copy.deepcopy(algorithm.optimizer.state_dict())
  storage_step = algorithm.storage.step
  cpu_rng = torch.random.get_rng_state()
  cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
  rates = [group['lr'] for group in algorithm.optimizer.param_groups]
  if learning_rate_ceiling is not None:
    # Retry the explicit ceiling for each new rollout; retain the same within-update
    # rollback/KL acceptance rule. Failed updates restore the pre-call LR as well.
    rates = [learning_rate_ceiling] * len(rates)
  saved_rate = algorithm.learning_rate
  distribution = getattr(algorithm.policy, 'distribution', None)
  def restore():
    # Rollout normalization can replace _std with an inference tensor.
    with torch.inference_mode():
      algorithm.policy.load_state_dict(state)
    # Adam may alias loaded tensors on the same device; keep the snapshot pristine.
    algorithm.optimizer.load_state_dict(copy.deepcopy(optimizer))
    if hasattr(algorithm.policy, 'distribution'):
      algorithm.policy.distribution = distribution
    algorithm.storage.step = storage_step
    algorithm.learning_rate = saved_rate
    torch.random.set_rng_state(cpu_rng)
    if cuda_rng is not None:
      torch.cuda.set_rng_state_all(cuda_rng)
  attempted = []
  for attempt in range(max_attempts):
    if attempt:
      restore()
    for group, rate in zip(algorithm.optimizer.param_groups, rates):
      group['lr'] = rate * .5**attempt
    algorithm.learning_rate = algorithm.optimizer.param_groups[0]['lr']
    try:
      loss = update()
      kl = measure()['kl_mean']
    except Exception:
      restore()
      raise
    attempted.append(kl)
    if math.isfinite(kl) and kl <= maximum_kl:
      return loss, {'Diagnostics/kl_guard_backtracks': attempt,
                    'Diagnostics/kl_guard_attempted_learning_rate': rates[0],
                    'Diagnostics/kl_guard_accepted_learning_rate': algorithm.learning_rate,
                    'Diagnostics/kl_guard_first_kl': attempted[0],
                    'Diagnostics/kl_guard_accepted_kl': kl}
  restore()
  raise RuntimeError(f'PPO KL guard exhausted {max_attempts} attempts: {attempted}')


class TeacherRunner(OnPolicyRunner):
  def __init__(self, *args, resume_gripper_std=None, resume_gripper_mean=None, resume_noise_scale=None, resume_arm_std=None, capture_pre_step=False, learning_rate_override=None, update_kl_diagnostics=False, max_update_kl=None, update_learning_rate_ceiling=None, separate_gradient_clipping=False, **kwargs):
    self.resume_gripper_std = resume_gripper_std
    self.resume_gripper_mean = resume_gripper_mean
    self.resume_noise_scale = resume_noise_scale
    self.resume_arm_std = resume_arm_std
    self.learning_rate_override = learning_rate_override
    super().__init__(*args, **kwargs)
    update_kl_diagnostics |= max_update_kl is not None
    original_step = self.env.step
    original_update = self.alg.update
    if separate_gradient_clipping:
      shared_update = original_update
      original_update = lambda: separately_clipped_update(self.alg.policy, shared_update)
    original_log = self.logger.log
    self._saturation = []
    self._update_metrics = {}
    previous_state = {}
    state_history = deque(maxlen=8)

    def fail(stage, observations=None, rewards=None):
      directory = Path(self.logger.log_dir)
      policy = self.alg.policy
      details = {"stage": stage, "last_completed_iteration": self.current_learning_iteration,
                 "nonfinite_parameters": [name for name,p in policy.named_parameters() if not torch.isfinite(p).all()]}
      data = self.env.unwrapped.sim.data
      bad = (~torch.isfinite(data.qpos).all(-1)) | (~torch.isfinite(data.qvel).all(-1))
      details["nonfinite_simulator_env_ids"] = bad.nonzero().flatten().cpu().tolist()
      details["pre_step_captured"] = bool(previous_state)
      details["captured_history_steps"] = len(state_history)
      if observations is not None:
        details["nonfinite_observation_env_ids"] = (~torch.isfinite(observations).all(-1)).nonzero().flatten().cpu().tolist()
      if rewards is not None:
        details["nonfinite_reward_env_ids"] = (~torch.isfinite(rewards)).nonzero().flatten().cpu().tolist()
      (directory / "numerical_failure.json").write_text(json.dumps(details, indent=2) + "\n")
      torch.save({"model_state_dict":policy.state_dict(), "optimizer_state_dict":self.alg.optimizer.state_dict(),
                  "qpos":self.env.unwrapped.sim.data.qpos.detach().cpu(),
                  "qvel":self.env.unwrapped.sim.data.qvel.detach().cpu(),
                  "previous_state":{name:value.cpu() for name,value in previous_state.items()},
                  "state_history":[{name:value.cpu() for name,value in state.items()} for state in state_history],
                  "observations":observations.detach().cpu() if observations is not None else None,
                  "rewards":rewards.detach().cpu() if rewards is not None else None}, directory / "numerical_failure.pt")
      raise RuntimeError(f"Numerical failure at {stage}; see {directory}/numerical_failure.json")

    def checked_step(actions):
      if not torch.isfinite(actions).all():
        fail("policy_action")
      if capture_pre_step:
        data = self.env.unwrapped.sim.data
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat"):
          previous_state[name] = getattr(data,name).detach().clone()
        previous_state["actions"] = actions.detach().clone()
        state_history.append(dict(previous_state))
      result = original_step(actions)
      obs, rewards = result[:2]
      if not (torch.isfinite(obs["policy"]).all() & torch.isfinite(rewards).all()
              & torch.isfinite(self.env.unwrapped.sim.data.qpos).all()
              & torch.isfinite(self.env.unwrapped.sim.data.qvel).all()):
        fail("rollout_state_or_reward", obs["policy"], rewards)
      self._saturation.append(float((actions.abs() >= 1).float().mean()))
      return result

    def checked_update():
      guard_metrics = {}
      if update_kl_diagnostics:
        storage = self.alg.storage
        # Fixed evenly spaced samples consume no RNG and span all rollout times.
        count = storage.mu.shape[0]*storage.mu.shape[1]
        stride = max(1, math.ceil(count/1024))
        observations = storage.observations.flatten(0,1)[::stride].clone()
        old_mean = storage.mu.flatten(0,1)[::stride].clone()
        old_std = storage.sigma.flatten(0,1)[::stride].clone()
        before = distribution_drift(self.alg.policy, observations, old_mean, old_std)
      try:
        if max_update_kl is None:
          loss = original_update()
        else:
          measure = lambda: distribution_drift(self.alg.policy, observations, old_mean, old_std)
          loss, guard_metrics = guarded_update(self.alg, original_update, measure, max_update_kl,
                                               learning_rate_ceiling=update_learning_rate_ceiling)
      except (RuntimeError, ValueError):
        fail("ppo_update_exception")
      policy = self.alg.policy
      if not all(torch.isfinite(p).all() for p in policy.parameters()):
        fail("ppo_updated_parameters")
      std = policy.std if policy.noise_std_type == "scalar" else policy.log_std.exp()
      if not (torch.isfinite(std).all() and (std > 0).all()):
        fail("nonpositive_policy_std")
      if not all(torch.isfinite(torch.as_tensor(value)).all() for value in loss.values()):
        fail("ppo_loss")
      if update_kl_diagnostics:
        after = distribution_drift(policy, observations, old_mean, old_std)
        self._update_metrics = {f"Diagnostics/{phase}_{key}": value
                               for phase, metrics in (("before_update",before),("after_update",after))
                               for key,value in metrics.items()}
        self._update_metrics.update(guard_metrics)
      return loss

    def log_with_diagnostics(**values):
      original_log(**values)
      policy = self.alg.policy
      std = policy.std if policy.noise_std_type == "scalar" else policy.log_std.exp()
      metrics = {"Diagnostics/std_min":float(std.min().detach()),
                 "Diagnostics/std_max":float(std.max().detach()),
                 "Diagnostics/gripper_std":float(std[-1].detach()),
                 "Diagnostics/gripper_action_mean":float(policy.action_mean[:,-1].mean().detach()),
                 "Diagnostics/mean_abs_max":float(policy.action_mean.abs().max().detach()),
                 "Diagnostics/action_saturation":sum(self._saturation)/max(1,len(self._saturation)),
                 "Diagnostics/peak_gpu_memory_mib":torch.cuda.max_memory_allocated()/1024**2}
      env = self.env.unwrapped
      names = env.command_manager.active_terms
      if len(names) == 1:
        command = env.command_manager.get_term(names[0])
        if hasattr(command, "object"):
          from mjlab.tasks.manipulation.mdp.task_geometry import between_fingers, finger_aperture, grasped, touching, tracking_position
          from contact_grasp import opposed_grasp
          with torch.no_grad():
            contact = grasped(command)
            enclosure = between_fingers(command)
            linear_speed = torch.linalg.vector_norm(command.object.data.root_link_lin_vel_w,dim=-1)
            angular_speed = torch.linalg.vector_norm(command.object.data.root_link_ang_vel_w,dim=-1)
            quiet = (linear_speed<.10)&(angular_speed<.5)
            metrics.update({
              "Diagnostics/two_pad_contact_fraction":float(contact.float().mean()),
              "Diagnostics/robot_object_contact_fraction":float(touching(command,command.robot,command.object).float().mean()),
              "Diagnostics/enclosed_fraction":float(enclosure.float().mean()),
              "Diagnostics/enclosed_grasp_fraction":float((contact&enclosure).float().mean()),
              "Diagnostics/opposed_grasp_fraction":float(opposed_grasp(command).float().mean()),
              "Diagnostics/aperture_mean":float(finger_aperture(command.robot).mean()),
              "Diagnostics/object_height_mean":float((tracking_position(command.object)[:,2]-env.scene.env_origins[:,2]).mean()),
              "Diagnostics/object_linear_speed_mean":float(linear_speed.mean()),
              "Diagnostics/object_angular_speed_mean":float(angular_speed.mean()),
              "Diagnostics/settled_at_lift_limits_fraction":float(quiet.float().mean()),
              "Diagnostics/grasped_and_settled_at_lift_limits_fraction":float((contact&quiet).float().mean()),
            })
      self._saturation.clear()
      metrics.update(self._update_metrics)
      for name,value in metrics.items():
        self.logger.writer.add_scalar(name,value,values["it"])
      with (Path(self.logger.log_dir)/"diagnostics.jsonl").open("a") as stream:
        stream.write(json.dumps({"iteration":values["it"],**metrics})+"\n")

    self.env.step = checked_step
    self.alg.update = checked_update
    self.logger.log = log_with_diagnostics

  def load(self, path, load_optimizer=True, map_location=None):
    infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
    if self.learning_rate_override is not None:
      for group in self.alg.optimizer.param_groups:
        group['lr'] = self.learning_rate_override
    # Adam's restored parameter groups override the constructor LR. Keep PPO's
    # scheduler/logger aligned with the optimizer actually used for updates.
    self.alg.learning_rate = self.alg.optimizer.param_groups[0]['lr']
    if self.resume_noise_scale is not None:
      scale_policy_exploration(self.alg.policy, self.alg.optimizer, self.resume_noise_scale)
    if self.resume_arm_std is not None:
      reset_arm_exploration(self.alg.policy, self.alg.optimizer, self.resume_arm_std)
    if self.resume_gripper_std is not None:
      reset_gripper_exploration(self.alg.policy, self.alg.optimizer, self.resume_gripper_std)
    if self.resume_gripper_mean is not None:
      reset_gripper_output(self.alg.policy, self.alg.optimizer, self.resume_gripper_mean)
    return infos
