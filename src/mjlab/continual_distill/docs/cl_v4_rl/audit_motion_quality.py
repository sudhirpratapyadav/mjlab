"""Measure complete recorded episodes without changing or rerunning a policy."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def audit(evaluation):
  ev=json.loads(evaluation.read_text())
  with np.load(Path(ev['trace_dir'])/'trace.npz') as z:t={k:z[k] for k in z.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m)
  dt=cfg.decimation*m.opt.timestep;assert abs(dt-.02)<1e-12
  joints=[m.joint(f'robot/joint{i}').id for i in range(1,8)]
  dofs=m.jnt_dofadr[joints];acts=[m.actuator(f'robot/actuator{i}').id for i in range(1,8)]
  rows=[]
  for outcome in ev['records']:
    lane,end=outcome['env_id'],outcome['steps']
    # Initial reset ctrl is not necessarily an applied policy target.
    vel=t['qvel'][1:end+1,lane][:,dofs];ctrl=t['ctrl'][1:end+1,lane][:,acts]
    assert np.isfinite(vel).all() and np.isfinite(ctrl).all()
    acceleration=np.diff(vel,axis=0)/dt;delta=np.diff(ctrl,axis=0)
    moving=np.max(np.abs(vel),axis=1)>.1
    rows.append(dict(env_id=lane,success=outcome['success'],duration_s=end*dt,
      max_joint_speed_rad_s=float(np.abs(vel).max()),
      rms_joint_speed_rad_s=float(np.sqrt(np.mean(vel**2))),
      moving_p95_joint_speed_rad_s=float(np.quantile(np.abs(vel[moving]),.95)) if moving.any() else 0.,
      sampled_acceleration_rms_rad_s2=float(np.sqrt(np.mean(acceleration**2))) if len(acceleration) else 0.,
      max_consecutive_target_jump_rad=float(np.abs(delta).max()) if len(delta) else 0.,
      target_change_rms_rad=float(np.sqrt(np.mean(delta**2))) if len(delta) else 0.,
      moving_fraction=float(moving.mean())))
  keys=['max_joint_speed_rad_s','rms_joint_speed_rad_s','moving_p95_joint_speed_rad_s','sampled_acceleration_rms_rad_s2','max_consecutive_target_jump_rad','target_change_rms_rad','moving_fraction']
  summary={key:{'median':float(np.median([r[key] for r in rows])), 'p95':float(np.quantile([r[key] for r in rows],.95)), 'max':float(max(r[key] for r in rows))} for key in keys}
  return dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],seed=ev['seed'],episodes=ev['episodes'],successes=ev['successes'],control_dt_s=dt,summary=summary,rows=rows,
    method='All recorded first episodes, including failures; actual arm joint speed at20ms snapshots. Acceleration is a finite-difference proxy, not native5ms acceleration/jerk peak. Target changes compare consecutive applied physical arm targets, excluding reset ctrl. No policy rerun, smoothing, hardware-limit claim or success substitution.')


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--baseline',type=Path)
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args()
  if a.output.exists():p.error('Output exists')
  report=audit(a.evaluation)
  if a.baseline:
    baseline=audit(a.baseline);assert baseline['task']==report['task'] and baseline['seed']==report['seed']
    keys=report['summary']
    report['baseline']=baseline
    report['ratios_to_baseline']={k:{q:(report['summary'][k][q]/baseline['summary'][k][q] if baseline['summary'][k][q] else None) for q in ['median','p95','max']} for k in keys}
    report['provisional_motion_gates']=dict(p95_episode_peak_speed_under3=report['summary']['max_joint_speed_rad_s']['p95']<=3.,p95_episode_acceleration_halved=report['ratios_to_baseline']['sampled_acceleration_rms_rad_s2']['p95']<=.5,p95_episode_max_target_jump_under025=report['summary']['max_consecutive_target_jump_rad']['p95']<=.25)
  a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print(json.dumps({k:report[k] for k in ['task','successes','episodes','summary','provisional_motion_gates'] if k in report},indent=2))


if __name__=='__main__':main()
