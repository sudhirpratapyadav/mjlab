"""Single-episode evaluation without counting retries after automatic resets."""

from contextlib import AbstractContextManager

import torch


class TerminalSuccessCapture(AbstractContextManager):
  """Preserve success at the last physics state before the environment resets.

  ManagerBasedRlEnv resets command latches before returning a terminal step.
  This evaluation-only hook observes that last state; it does not feed simulator
  state to the teacher or alter the task's success predicate.
  """

  def __init__(self, env, command, terminal_callback=None):
    self.env = env
    self.command = command
    self.callback = terminal_callback
    self.terminal_success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self._stepping = False
    self._original_reset = env._reset_idx

  def __enter__(self):
    def capture(env_ids):
      if self._stepping:
        self.command._update_metrics()
        self.terminal_success[env_ids] = self.command.episode_success[env_ids] > 0
        if self.callback is not None:
          self.callback(env_ids)
      self._original_reset(env_ids)
    self.env._reset_idx = capture
    return self

  def step(self, actions):
    self.terminal_success.zero_()
    self._stepping = True
    try:
      return self.env.step(actions)
    finally:
      self._stepping = False

  def __exit__(self, *exc):
    self.env._reset_idx = self._original_reset
    return False
