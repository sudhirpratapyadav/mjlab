"""Terminal success survives reset, while success from a later episode does not."""

import torch

from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture
from mjlab.continual_distill.classical import render_rollout


class Command:
  def __init__(self):
    self.episode_success = torch.zeros(1)
  def _update_metrics(self):
    pass


class Env:
  num_envs = 1
  device = 'cpu'
  max_episode_length = 3
  def __init__(self, terminal_success):
    self.command = Command()
    self.terminal_success = terminal_success
    self.steps = 0
  def _reset_idx(self, env_ids):
    # A different episode may start already at its goal; it must not be counted.
    self.command.episode_success[:] = 1
  def reset(self):
    return {'policy': torch.zeros((1, 60))}, {}
  def step(self, actions):
    self.steps += 1
    self.command.episode_success[:] = self.terminal_success
    self._reset_idx(torch.tensor([0]))
    return self.reset()[0], torch.zeros(1), torch.ones(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool), {}
  def close(self):
    pass


def test_captures_last_state_and_restores_reset_method():
  env = Env(1)
  original = env._reset_idx
  with TerminalSuccessCapture(env, env.command) as capture:
    capture.step(torch.zeros((1, 8)))
    assert capture.terminal_success.item()
  assert env._reset_idx == original


def test_stats_stop_on_first_episode_and_ignore_respawn_success(monkeypatch):
  env = Env(0)
  class Policy:
    def __init__(self, num_envs):
      pass
    def reset(self):
      pass
    def __call__(self, obs):
      return obs[:, :8].astype('float32')
  monkeypatch.setattr(render_rollout, '_build_env', lambda *a, **kw: env)
  monkeypatch.setattr(render_rollout, '_success_term', lambda env: env.command)
  monkeypatch.setitem(render_rollout.CLASSICAL_POLICIES, 'test', Policy)
  stats = render_rollout.run_stats_phase('test', 1, 1, 'cpu')
  assert stats.n == 1 and stats.num_success == 0
  assert env.steps == 1


def test_resample_preserves_duration_above_and_below_control_rate():
  import numpy as np
  frames = [np.array([i]) for i in range(100)]
  for fps in (10, 30, 60):
    sampled = render_rollout._resample_fps(frames, .05, fps)
    assert len(sampled) / fps == 5.0
    assert sampled[0][0] == 0
    assert sampled[-1][0] <= 99


def test_inactive_teacher_rows_do_not_advance_state():
  import numpy as np
  from mjlab.continual_distill.classical.base import ClassicalPolicyBase
  policy = object.__new__(ClassicalPolicyBase)
  policy._phase_steps = np.zeros(3, dtype=int)
  visited = []
  def act(i, obs):
    visited.append(i)
    return np.full(8, i, dtype=np.float32)
  policy._act_single = act
  result = policy(np.zeros((3, 60)), active_env_ids=np.array([1]))
  assert visited == [1]
  assert policy._phase_steps.tolist() == [0, 1, 0]
  assert result[1].tolist() == [1] * 8
