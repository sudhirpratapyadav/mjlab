"""Read exact terminal velocities and controls alongside the orientation audit."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--progress-audit',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  e=json.loads(args.evaluation.read_text());a=json.loads(args.progress_audit.read_text())
  assert e['task']=='Mjlab-Reorient-Object-Franka' and e['checkpoint_sha256']==a['checkpoint_sha256']
  cfg=load_env_cfg(e['task']);cfg.scene.num_envs=1;cmd=next(iter(cfg.commands.values()))
  m=Scene(cfg.scene,'cpu').compile()
  with np.load(Path(e['trace_dir'])/'trace.npz') as z:t={k:z[k] for k in z.files}
  joints=[j for j in range(m.njnt) if m.joint(j).name.startswith(cmd.asset_name+'/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
  assert len(joints)==1
  v=int(m.jnt_dofadr[joints[0]])
  q=[m.jnt_qposadr[m.joint(f'robot/joint{i}').id] for i in range(1,8)]
  u=[m.actuator(f'robot/actuator{i}').id for i in range(1,8)]
  rows=[]
  for rec,ar in zip(e['records'],a['rows'],strict=True):
    lane,end=rec['env_id'],rec['steps'];assert lane==ar['env_id']
    vel=t['qvel'][end,lane,v:v+6];term=ar['terminal'];ctrl=t['ctrl'][end,lane,u];pos=t['qpos'][end,lane,q]
    rows.append(dict(env_id=lane,angle_deg=term['angle_deg'],drift_m=term['drift_m'],valid_drift=term['valid_drift'],
      linear_speed=float(np.linalg.norm(vel[:3])),angular_speed=float(np.linalg.norm(vel[3:])),
      arm_target_rms=float(np.sqrt(np.mean((ctrl-pos)**2))),
      arm_target_step_rms=float(np.sqrt(np.mean((ctrl-t['ctrl'][end-1,lane,u])**2)))))
  near=[r for r in rows if r['angle_deg']<np.rad2deg(cmd.angle_threshold) and r['valid_drift']]
  summary=dict(episodes=len(rows),orientation_and_drift_pass=len(near),
    angle_pass=int(sum(r['angle_deg']<np.rad2deg(cmd.angle_threshold) for r in rows)),
    drift_pass=sum(r['valid_drift'] for r in rows),
    near_linear_fail=sum(r['linear_speed']>=.03 for r in near),
    near_angular_fail=sum(r['angular_speed']>=.3 for r in near))
  for name in ('linear_speed','angular_speed','arm_target_rms','arm_target_step_rms'):
    summary['near_median_'+name]=float(np.median([r[name] for r in near])) if near else None
  report=dict(task=e['task'],checkpoint_sha256=e['checkpoint_sha256'],
    method='Exact recorded terminal qvel/ctrl plus original-anchor orientation/drift audit. All lanes, no intervention or success substitution.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
