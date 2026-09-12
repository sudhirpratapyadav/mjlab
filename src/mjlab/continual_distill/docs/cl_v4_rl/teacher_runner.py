"""PPO diagnostics that stop at the first nonfinite rollout or optimizer update."""

import json
import math
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner


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


class TeacherRunner(OnPolicyRunner):
  def __init__(self, *args, resume_gripper_std=None, capture_pre_step=False, **kwargs):
    self.resume_gripper_std = resume_gripper_std
    super().__init__(*args, **kwargs)
    original_step = self.env.step
    original_update = self.alg.update
    original_log = self.logger.log
    self._saturation = []
    previous_state = {}

    def fail(stage):
      directory = Path(self.logger.log_dir)
      policy = self.alg.policy
      details = {"stage": stage, "last_completed_iteration": self.current_learning_iteration,
                 "nonfinite_parameters": [name for name,p in policy.named_parameters() if not torch.isfinite(p).all()]}
      data = self.env.unwrapped.sim.data
      bad = (~torch.isfinite(data.qpos).all(-1)) | (~torch.isfinite(data.qvel).all(-1))
      details["nonfinite_simulator_env_ids"] = bad.nonzero().flatten().cpu().tolist()
      details["pre_step_captured"] = bool(previous_state)
      (directory / "numerical_failure.json").write_text(json.dumps(details, indent=2) + "\n")
      torch.save({"model_state_dict":policy.state_dict(), "optimizer_state_dict":self.alg.optimizer.state_dict(),
                  "qpos":self.env.unwrapped.sim.data.qpos.detach().cpu(),
                  "qvel":self.env.unwrapped.sim.data.qvel.detach().cpu(),
                  "previous_state":{name:value.cpu() for name,value in previous_state.items()}}, directory / "numerical_failure.pt")
      raise RuntimeError(f"Numerical failure at {stage}; see {directory}/numerical_failure.json")

    def checked_step(actions):
      if not torch.isfinite(actions).all():
        fail("policy_action")
      if capture_pre_step:
        data = self.env.unwrapped.sim.data
        for name in ("qpos", "qvel", "ctrl", "qacc_warmstart", "mocap_pos", "mocap_quat"):
          previous_state[name] = getattr(data,name).detach().clone()
        previous_state["actions"] = actions.detach().clone()
      result = original_step(actions)
      obs, rewards = result[:2]
      if not (torch.isfinite(obs["policy"]).all() & torch.isfinite(rewards).all()
              & torch.isfinite(self.env.unwrapped.sim.data.qpos).all()
              & torch.isfinite(self.env.unwrapped.sim.data.qvel).all()):
        fail("rollout_state_or_reward")
      self._saturation.append(float((actions.abs() >= 1).float().mean()))
      return result

    def checked_update():
      try:
        loss = original_update()
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
      return loss

    def log_with_diagnostics(**values):
      original_log(**values)
      policy = self.alg.policy
      std = policy.std if policy.noise_std_type == "scalar" else policy.log_std.exp()
      metrics = {"Diagnostics/std_min":float(std.min().detach()),
                 "Diagnostics/std_max":float(std.max().detach()),
                 "Diagnostics/mean_abs_max":float(policy.action_mean.abs().max().detach()),
                 "Diagnostics/action_saturation":sum(self._saturation)/max(1,len(self._saturation)),
                 "Diagnostics/peak_gpu_memory_mib":torch.cuda.max_memory_allocated()/1024**2}
      self._saturation.clear()
      for name,value in metrics.items():
        self.logger.writer.add_scalar(name,value,values["it"])
      with (Path(self.logger.log_dir)/"diagnostics.jsonl").open("a") as stream:
        stream.write(json.dumps({"iteration":values["it"],**metrics})+"\n")

    self.env.step = checked_step
    self.alg.update = checked_update
    self.logger.log = log_with_diagnostics

  def load(self, path, load_optimizer=True, map_location=None):
    infos = super().load(path, load_optimizer=load_optimizer, map_location=map_location)
    if self.resume_gripper_std is not None:
      reset_gripper_exploration(self.alg.policy, self.alg.optimizer, self.resume_gripper_std)
    return infos
