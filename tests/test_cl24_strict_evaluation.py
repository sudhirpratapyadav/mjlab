"""Exercise the evaluator's real rollout loop against adversarial auto-resets."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest


@pytest.mark.parametrize('gyro_compat', [False, True])
def test_terminal_capture_subset_resets_and_retry_exclusion(monkeypatch, tmp_path, gyro_compat):
  stage = Path(__file__).resolve().parents[1] / "src/mjlab/continual_distill/docs/cl_v4_rl"
  monkeypatch.syspath_prepend(str(stage))
  evaluation = importlib.import_module("evaluate_teacher")
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

  class Environment:
    def __init__(self, cfg, device):
      assert cfg.sim.free_body_implicitfast_compat is gyro_compat
      self.num_envs = 4
      self.max_episode_length = 4
      self.cfg = cfg
      self.lengths = torch.tensor([1, 3, 4, 2])
      self.steps = torch.zeros(4, dtype=torch.long)
      self.episode = torch.zeros(4, dtype=torch.long)
      self.state_success = torch.zeros(4, dtype=torch.bool)
      self.metrics = torch.zeros(4, dtype=torch.bool)
      self.command = SimpleNamespace(
        _update_metrics=lambda: self.metrics.copy_(self.state_success),
        compute_success=lambda: self.metrics,
      )
      self.command_manager = SimpleNamespace(active_terms=["task"], get_term=lambda name:self.command)
      self.terms = {name:torch.zeros(4,dtype=torch.bool) for name in ("time_out", "collision")}
      self.termination_manager = SimpleNamespace(active_terms=list(self.terms), get_term=lambda name:self.terms[name])
      self.reset_time_outs = self.terms["time_out"]

    def _reset_idx(self, ids):
      self.steps[ids] = 0
      self.episode[ids] += 1
      # Deliberately make reset states report success: the terminal snapshot
      # must be read before this mutation, even for a single-instance reset.
      self.state_success[ids] = True
      self.metrics[ids] = True

    def step(self, action):
      self.steps += 1
      done = self.steps >= self.lengths
      first = self.episode == 0
      expected = torch.tensor([False, True, False, True])
      self.state_success[done] = torch.where(first, expected, True)[done]
      collision = done & torch.tensor([True, False, False, True])
      self.terms["collision"].copy_(collision)
      self.terms["time_out"].copy_(done & ~collision)
      if done.any():
        self._reset_idx(done.nonzero().flatten())
      return {"policy":torch.zeros(4,60)}, torch.ones(4), done, {}

    def close(self):
      pass

  class Wrapper:
    def __init__(self, env, clip_actions):
      self.env = env
      self.num_actions = 8
    def get_observations(self):
      return {"policy":torch.zeros(4,60)}
    def step(self, action):
      return self.env.step(action)
    def close(self):
      self.env.close()

  class Runner:
    def __init__(self, *args, **kwargs):
      pass
    def load(self, *args, **kwargs):
      pass
    def get_inference_policy(self):
      return lambda obs:torch.zeros(4,8)

  monkeypatch.setattr(evaluation,"ManagerBasedRlEnv",Environment)
  monkeypatch.setattr(evaluation,"RslRlVecEnvWrapper",Wrapper)
  monkeypatch.setattr(evaluation,"OnPolicyRunner",Runner)
  checkpoint = tmp_path / "fake.pt"
  checkpoint.write_bytes(b"fake-checkpoint")
  (tmp_path/'manifest.json').write_text(json.dumps({'task':'Mjlab-Reach-Target-Franka',
                                                  'free_body_implicitfast_compat':gyro_compat}))
  result = evaluation.evaluate("Mjlab-Reach-Target-Franka",checkpoint,4,123)
  assert result['free_body_implicitfast_compat'] is gyro_compat
  assert result["successes"] == 2
  assert [row["success"] for row in result["records"]] == [False,True,False,True]
  assert [row["steps"] for row in result["records"]] == [1,3,4,2]
  assert [row["return"] for row in result["records"]] == [1,3,4,2]
  assert result["termination_counts"] == {"time_out":2,"collision":2}
