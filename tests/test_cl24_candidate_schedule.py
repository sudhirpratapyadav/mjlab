"""A preregistered confirmation seed stays separate from validation and is gated."""
import importlib
import json
from pathlib import Path
import sys
import pytest


@pytest.mark.parametrize("successes,expected_seeds",[(116,[20260914,20260916]),(115,[20260914])])
def test_candidate_uses_distinct_preregistered_confirmation(monkeypatch,tmp_path,successes,expected_seeds):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  candidate=importlib.import_module('evaluate_candidate')
  checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'not loaded by orchestration')
  (tmp_path/'evidence').mkdir()
  monkeypatch.setattr(candidate,'HERE',tmp_path)
  monkeypatch.setattr(sys,'argv',['evaluate_candidate','--task','test','--checkpoint',str(checkpoint),'--label','candidate','--confirmation-seed','20260916'])
  seeds=[]
  def run(cmd,check):
    if cmd[1].endswith('evaluate_teacher.py'):
      seed=int(cmd[cmd.index('--seed')+1]);seeds.append(seed)
      Path(cmd[cmd.index('--output')+1]).write_text(json.dumps(dict(successes=successes,episodes=128,success_rate=successes/128)))
  monkeypatch.setattr(candidate.subprocess,'run',run)
  candidate.main()
  assert seeds==expected_seeds
  assert (tmp_path/'evidence/candidate-confirm-20260916.json').exists()==(successes>=116)
  monkeypatch.setattr(sys,'argv',['evaluate_candidate','--task','test','--checkpoint',str(checkpoint),'--label','candidate','--confirmation-seed','20260914'])
  with pytest.raises(SystemExit):candidate.main()
  assert seeds==expected_seeds
