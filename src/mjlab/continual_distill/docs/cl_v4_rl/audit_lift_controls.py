"""Quantify actual actuator targets near the Lift goal in first-episode traces."""
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
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());assert ev['task']=='Mjlab-Lift-Cube-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  assert 'ctrl' in t, 'Requires actual controls, not a reconstructed policy query'
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1;m=Scene(cfg.scene,'cpu').compile()
  armq=[int(m.jnt_qposadr[m.joint(f'robot/joint{i}').id]) for i in range(1,8)]
  armu=[m.actuator(f'robot/actuator{i}').id for i in range(1,8)]
  fingerq=int(m.jnt_qposadr[m.joint('robot/finger_joint1').id]);fingeru=m.actuator('robot/actuator8').id
  joint=m.joint('cube/cube_joint').id;qadr=int(m.jnt_qposadr[joint]);vadr=int(m.jnt_dofadr[joint])
  rows=[];near=[]
  for rec in ev['records']:
    lane,end=rec['env_id'],rec['steps'];start=max(1,end-99)
    pos=t['qpos'][start:end+1,lane,qadr:qadr+3]
    goal=t['mocap_pos'][start:end+1,lane,1]
    error=np.linalg.norm(pos-goal,axis=1)
    arm_error=t['ctrl'][start:end+1,lane][:,armu]-t['qpos'][start:end+1,lane][:,armq]
    target_delta=t['ctrl'][start:end+1,lane][:,armu]-t['ctrl'][start-1:end,lane][:,armu]
    closure=t['qpos'][start:end+1,lane,fingerq]-t['ctrl'][start:end+1,lane,fingeru]
    vel=t['qvel'][start:end+1,lane,vadr:vadr+6]
    values=np.column_stack([error,np.sqrt((arm_error**2).mean(1)),np.sqrt((target_delta**2).mean(1)),closure,
                            np.linalg.norm(vel[:,:3],axis=1),np.linalg.norm(vel[:,3:],axis=1)])
    near.extend(values[error<.05].tolist())
    rows.append(dict(env_id=lane,terminal_native_success=rec['success'],terminal_goal_error_m=float(error[-1]),
                     terminal_arm_target_error_rad=arm_error[-1].tolist(),terminal_arm_target_rms_rad=float(values[-1,1]),
                     terminal_loaded_finger_closure_m=float(closure[-1]),last100_near_goal_states=int((error<.05).sum()),
                     last100_arm_target_delta_rms_rad_median=float(np.median(values[:,2]))))
  a=np.asarray(near)
  summary=dict(episodes=len(rows),native_successes=ev['successes'],terminal_inside5cm=sum(r['terminal_goal_error_m']<.05 for r in rows),
               last100_near_goal_states=len(near),near_goal_median_goal_error_m=float(np.median(a[:,0])) if len(a) else None,
               near_goal_median_arm_target_rms_rad=float(np.median(a[:,1])) if len(a) else None,
               near_goal_median_target_step_rms_rad=float(np.median(a[:,2])) if len(a) else None,
               near_goal_median_loaded_closure_m=float(np.median(a[:,3])) if len(a) else None,
               near_goal_median_linear_speed_m_s=float(np.median(a[:,4])) if len(a) else None,
               near_goal_median_angular_speed_rad_s=float(np.median(a[:,5])) if len(a) else None)
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],method='Actual applied ctrl and joint state in the last100 control states of each uninterrupted first episode; near-goal filter is position<5cm, not a contact or success proxy. Controls are recorded before terminal reset. No integration, policy query or success reclassification.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
