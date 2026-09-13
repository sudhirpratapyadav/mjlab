"""Motion comparisons must exclude reset artifacts and retain failed episodes."""
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_motion_audit_uses_terminal_boundary_and_applied_controls(tmp_path,monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));module=importlib.import_module('audit_motion_quality')
  model=SimpleNamespace(opt=SimpleNamespace(timestep=.005),jnt_dofadr=np.arange(7),
    joint=lambda name:SimpleNamespace(id=int(name[-1])-1),actuator=lambda name:SimpleNamespace(id=int(name[-1])-1))
  cfg=SimpleNamespace(scene=SimpleNamespace(num_envs=1),sim=SimpleNamespace(mujoco=SimpleNamespace(apply=lambda m:None)),decimation=4)
  monkeypatch.setattr(module,'load_env_cfg',lambda task:cfg)
  monkeypatch.setattr(module,'Scene',lambda *args:SimpleNamespace(compile=lambda:model))
  vel=np.ones((4,2,9));ctrl=np.zeros((4,2,8));ctrl[0]=999
  ctrl[2:,:,:7]=.02;vel[3,0]=999;ctrl[3,0]=999
  vel[:,:,7:]=10000 # Finger quantities must not leak into arm metrics.
  np.savez(tmp_path/'trace.npz',qvel=vel,ctrl=ctrl)
  ev=dict(task='Mjlab-Lift-Cube-Franka',trace_dir=str(tmp_path),checkpoint_sha256='test',seed=14,episodes=2,successes=1,records=[dict(env_id=0,steps=2,success=True),dict(env_id=1,steps=3,success=False)])
  path=tmp_path/'evaluation.json';path.write_text(json.dumps(ev))
  result=module.audit(path)
  assert len(result['rows'])==2 and result['rows'][1]['success'] is False
  for row in result['rows']:
    assert row['max_joint_speed_rad_s']==1
    assert row['sampled_acceleration_rms_rad_s2']==0
    assert abs(row['max_consecutive_target_jump_rad']-.02)<1e-8
  assert result['rows'][0]['duration_s']==.04
