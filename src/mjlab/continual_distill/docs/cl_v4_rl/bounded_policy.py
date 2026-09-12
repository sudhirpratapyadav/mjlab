"""A bounded Gaussian mean and explicit initialization; action units stay unchanged."""
from dataclasses import dataclass
import torch
from torch import nn
from rsl_rl.modules import ActorCritic
from mjlab.rl.config import RslRlPpoActorCriticCfg

@dataclass
class BoundedPolicyCfg(RslRlPpoActorCriticCfg):
  class_name: str = 'BoundedActorCritic'
  initial_mean: tuple[float, ...] = (0.,) * 8
  initial_gripper_std: float = 0.03

class BoundedActorCritic(ActorCritic):
  def __init__(self, *args, initial_mean=(0.,) * 8, initial_gripper_std=0.03, **kwargs):
    super().__init__(*args, **kwargs)
    if self.state_dependent_std or self.noise_std_type != 'log':
      raise ValueError('Bounded policy requires independent log std')
    output = self.actor[-1]
    mean = torch.as_tensor(initial_mean, dtype=output.bias.dtype)
    if mean.shape != output.bias.shape or not (mean.abs() < 1).all():
      raise ValueError('Initial mean must have eight values strictly inside [-1,1]')
    nn.init.orthogonal_(output.weight, gain=0.01)
    with torch.no_grad():
      output.bias.copy_(torch.atanh(mean))
      self.log_std[-1] = torch.log(torch.tensor(initial_gripper_std))
    self.actor.add_module(str(len(self.actor)), nn.Tanh())

# Installed RSL-RL resolves policy class names in the runner module's namespace.
# Register only this process-local experiment class for both training/evaluation.
def register():
  import rsl_rl.runners.on_policy_runner as runner
  runner.BoundedActorCritic = BoundedActorCritic
