"""Audit launch speed and endpoint calibration from recorded first episodes."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.config.franka.env_cfgs import STRIKE_PUCK_MU


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());assert ev['task']=='Mjlab-Strike-Slide-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as archive:trace={k:archive[k] for k in archive.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  m=Scene(cfg.scene,'cpu').compile()
  joints=[i for i in range(m.njnt) if m.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE and not m.joint(i).name.startswith('robot/')]
  assert len(joints)==1
  qadr,vadr=int(m.jnt_qposadr[joints[0]]),int(m.jnt_dofadr[joints[0]])
  goal_bodies=[i for i in range(m.nbody) if m.body(i).name.startswith('mocap_goal/') and m.body_mocapid[i]>=0]
  assert len(goal_bodies)==1
  goal_id=int(m.body_mocapid[goal_bodies[0]])
  rows=[]
  for r in ev['records']:
    lane,end=r['env_id'],r['steps']
    pos=trace['qpos'][:end+1,lane,qadr:qadr+3];vel=trace['qvel'][:end+1,lane,vadr:vadr+3]
    goal=trace['mocap_pos'][0,lane,goal_id];speed=np.linalg.norm(vel[:,:2],axis=-1)
    direction=goal[:2]-pos[0,:2];direction/=np.linalg.norm(direction)
    residual=pos[-1,:2]-goal[:2];along=float(residual@direction)
    prediction=pos[:,:2]+vel[:,:2]*speed[:,None]/(2*STRIKE_PUCK_MU*9.81)
    peak=int(speed.argmax())
    rows.append(dict(env_id=lane,success=r['success'],max_speed_m_s=float(speed[peak]),final_speed_m_s=float(speed[-1]),
      final_error_m=float(np.linalg.norm(pos[-1]-goal)),signed_longitudinal_error_m=along,
      lateral_error_m=float(np.linalg.norm(residual-along*direction)),
      peak_speed_prediction_error_m=float(np.linalg.norm(prediction[peak]-goal[:2])),
      peak_prediction_vs_actual_endpoint_m=float(np.linalg.norm(prediction[peak]-pos[-1,:2]))))
  summary=dict(episodes=len(rows),successes=ev['successes'],median_final_error_m=float(np.median([r['final_error_m'] for r in rows])),
    negligible_launches=sum(r['max_speed_m_s']<.05 for r in rows),substantial_launches=sum(r['max_speed_m_s']>=.4 for r in rows),
    undershot_more_than8cm=sum(r['signed_longitudinal_error_m']<-.08 for r in rows),overshot_more_than8cm=sum(r['signed_longitudinal_error_m']>.08 for r in rows),
    lateral_error_more_than8cm=sum(r['lateral_error_m']>.08 for r in rows),
    median_peak_prediction_vs_endpoint_m=float(np.median([r['peak_prediction_vs_actual_endpoint_m'] for r in rows if r['max_speed_m_s']>=.4])))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],qpos_address=qadr,qvel_address=vadr,goal_mocap_id=goal_id,
    method='Recorded first episodes only, exact terminal indices and compiled free-joint/goal addresses. Peak XY speed is a launch proxy. Constant-friction sliding prediction at peak speed is compared with measured endpoint; peak may still have hand contact, so prediction discrepancy is not a physics defect claim.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
