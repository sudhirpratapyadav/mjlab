"""Recorded pre-launch hand velocity and task-distance calibration; no rollout."""
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
  a=p.parse_args()
  if a.output.exists():p.error('Output exists')
  ev=json.loads(a.evaluation.read_text());assert ev['task']=='Mjlab-Strike-Slide-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as z:t={k:z[k] for k in z.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  obj=m.site_bodyid[m.site('puck/object_site').id];joint=m.body_jntadr[obj]
  assert m.jnt_type[joint]==mujoco.mjtJoint.mjJNT_FREE
  q,v=int(m.jnt_qposadr[joint]),int(m.jnt_dofadr[joint])
  goals=[m.body_mocapid[i] for i in range(m.nbody) if m.body(i).name.startswith('mocap_goal/') and m.body_mocapid[i]>=0]
  assert len(goals)==1
  grip=m.site('robot/gripper').id;rows=[]
  for record in ev['records']:
    lane,end=record['env_id'],record['steps']
    pos=t['qpos'][:end+1,lane,q:q+3];vel=t['qvel'][:end+1,lane,v:v+3]
    goal=t['mocap_pos'][0,lane,goals[0]];delta=goal[:2]-pos[0,:2]
    distance=float(np.linalg.norm(delta));direction=delta/distance
    speed=np.linalg.norm(vel[:,:2],axis=-1);launch=np.flatnonzero(speed>.05)
    row=dict(env_id=lane,success=record['success'],initial_distance_m=distance,
      initial_puck_local_xy_m=(pos[0,:2]-t['origins'][lane,:2]).tolist(),
      initial_goal_local_xy_m=(goal[:2]-t['origins'][lane,:2]).tolist(),
      signed_endpoint_error_m=float((pos[-1,:2]-goal[:2])@direction),
      ideal_slide_speed_m_s=float(np.sqrt(2*STRIKE_PUCK_MU*9.81*distance)),peak_speed_m_s=float(speed.max()))
    if len(launch):
      before=max(int(launch[0])-1,0)
      for f in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,f)[:]=t[f][before,lane]
      for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
        if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=t['origins'][lane]
      d.mocap_pos[:]-=t['origins'][lane]
      mujoco.mj_forward(m,d);jac=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jac,None,grip)
      hand=jac@d.qvel
      row.update(pre_launch_sample_step=before,pre_launch_hand_along_speed_m_s=float(hand[:2]@direction),
        pre_launch_hand_vertical_speed_m_s=float(hand[2]),
        pre_launch_hand_to_puck_m=float(np.linalg.norm(d.site_xpos[grip]-(pos[before]-t['origins'][lane]))))
    rows.append(row)
  def corr(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>1e-8 and np.std(y)>1e-8 else None
  launched=[r for r in rows if 'pre_launch_sample_step' in r]
  errors=[r['signed_endpoint_error_m'] for r in rows]
  summary=dict(episodes=len(rows),launched=len(launched),
    endpoint_error_vs_initial_distance_correlation=corr([r['initial_distance_m'] for r in rows],errors),
    endpoint_error_vs_initial_puck_x_correlation=corr([r['initial_puck_local_xy_m'][0] for r in rows],errors),
    endpoint_error_vs_initial_puck_y_correlation=corr([r['initial_puck_local_xy_m'][1] for r in rows],errors),
    peak_speed_vs_ideal_speed_correlation=corr([r['peak_speed_m_s'] for r in rows],[r['ideal_slide_speed_m_s'] for r in rows]),
    peak_speed_vs_pre_launch_hand_speed_correlation=corr([r['peak_speed_m_s'] for r in launched],[r['pre_launch_hand_along_speed_m_s'] for r in launched]))
  a.output.write_text(json.dumps(dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],
    method='All128 recorded first episodes. FK/Jacobian at the20ms sample immediately before first puckXYspeed>.05m/s; this can precede/miss exact contact and is not an impact law. Ideal slide speed assumes the registered constant friction and initial distance. Descriptive correlations do not establish causality. No integration, actions, resets or policy data.',summary=summary,rows=rows),indent=2,allow_nan=False)+'\n')
  print(json.dumps(summary))


if __name__=='__main__':main()
