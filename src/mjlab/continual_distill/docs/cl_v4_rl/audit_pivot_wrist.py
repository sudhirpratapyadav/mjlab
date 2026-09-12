"""Recorded Pivot wrist geometry and bounded endpoint IK; no policy actions."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def frame(tilt):
  psi = np.deg2rad(70)-np.deg2rad(15)*np.clip(tilt/.43,0,1)
  y = np.array([np.sin(psi),0.,np.cos(psi)])
  z = np.array([np.cos(psi),0.,-np.sin(psi)])
  return np.column_stack([np.cross(y,z),y,z])


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--floor-clearance',type=float)
  parser.add_argument('--finger-position',type=float,default=.035)
  args=parser.parse_args()
  evaluation=json.loads(args.evaluation.read_text())
  assert evaluation['task']=='Mjlab-Pivot-Lift-Franka'
  with np.load(Path(evaluation['trace_dir'])/'trace.npz') as archive:
    trace={k:archive[k] for k in archive.files}
  cfg=load_env_cfg(evaluation['task']);cfg.scene.num_envs=1
  model=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(model)
  data=mujoco.MjData(model)
  grip=model.site('robot/gripper').id
  obj=model.body('board/board').id
  site=model.site('board/object_site').id
  wall=model.body('wall/wall_base').id
  arm=[model.jnt_qposadr[model.joint(f'robot/joint{i}').id] for i in range(1,8)]
  joints=[model.joint(f'robot/joint{i}').id for i in range(1,8)]
  bounds=(model.jnt_range[joints,0]+1e-6,model.jnt_range[joints,1]-1e-6)
  fingers=[model.jnt_qposadr[model.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  robot={i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')}
  obstacles={i for i in range(model.ngeom) if model.geom(i).name.startswith('wall/') or model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}

  def restore(step,lane):
    for field in ['qpos','qvel','mocap_pos','mocap_quat']:
      getattr(data,field)[:]=trace[field][step,lane]
    for adr,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:data.qpos[adr:adr+3]-=trace['origins'][lane]
    data.mocap_pos[:]-=trace['origins'][lane]
    mujoco.mj_kinematics(model,data)

  def geometry():
    board=data.xmat[obj].reshape(3,3)
    tilt=1-abs(board[2,2]);desired=frame(tilt)
    corner=data.site_xpos[site]-.05*board[:,0]+.010*board[:,2]
    target=corner+.5*data.qpos[fingers].sum()*desired[:,1]-.005*desired[:,2]
    actual=data.site_xmat[grip].reshape(3,3)
    return target,desired,dict(distance_m=float(np.linalg.norm(data.site_xpos[grip]-target)),
      closing_alignment=float(abs(actual[:,1]@desired[:,1])),
      approach_alignment=float(actual[:,2]@desired[:,2]),
      full_rotation_error_deg=float(np.rad2deg(np.arccos(np.clip((np.trace(desired.T@actual)-1)/2,-1,1)))),
      tilt_deg=float(np.rad2deg(np.arccos(np.clip(abs(board[2,2]),0,1)))) ,
      wall_gap_m=float(data.xpos[wall,0]-.0525-data.site_xpos[site,0]-np.abs(board[0])@np.array([.06,.1094,.01])),
      aperture_m=float(data.qpos[fingers].sum()))

  rows=[]
  for record in evaluation['records']:
    lane=record['env_id'];end=record['steps'];samples=[]
    for step in sorted(set(range(0,end+1,10))|{end}):
      restore(step,lane);_,_,g=geometry();samples.append(dict(step=step,**g))
    closest=min(samples,key=lambda r:r['distance_m'])
    row=dict(env_id=lane,closest=closest,terminal=samples[-1],max_sampled_tilt_deg=max(r['tilt_deg'] for r in samples))
    if lane%4==0:
      restore(closest['step'],lane);data.qpos[fingers]=args.finger_position
      target,desired,_=geometry()
      if args.floor_clearance is not None:
        # Panda XML pad support along world z for this expected ramp frame.
        target[2]=max(target[2],args.floor_clearance+(args.finger_position+.0152)*abs(desired[2,1])+.0119*abs(desired[2,2]))
      # Numerical branch seeds only; these never initialize or supervise PPO.
      seeds=[data.qpos[arm].copy(),np.array([-.47,.39,-.13,-2.81,1.35,1.52,-.85]),np.array([.45,.38,.11,-2.83,-1.40,1.53,-.71])]
      def residual(q):
        data.qpos[arm]=q;mujoco.mj_kinematics(model,data)
        return np.r_[data.site_xpos[grip]-target,.08*(data.site_xmat[grip].reshape(3,3)-desired).ravel()]
      cases=[]
      for seed in seeds:
        result=least_squares(residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=120)
        residual(result.x);error=float(np.linalg.norm(data.site_xpos[grip]-target))
        angle=float(np.rad2deg(np.arccos(np.clip((np.trace(desired.T@data.site_xmat[grip].reshape(3,3))-1)/2,-1,1))))
        mujoco.mj_collision(model,data)
        bad=[dict(geoms=[model.geom(int(i)).name for i in c.geom],depth_m=float(-c.dist)) for c in data.contact if c.dist<-.003 and ((c.geom[0] in robot and c.geom[1] in obstacles) or (c.geom[1] in robot and c.geom[0] in obstacles))]
        cases.append(dict(position_error_m=error,orientation_error_deg=angle,penetrations=bad,passes=error<.02 and angle<15 and not bad,qpos=result.x.tolist()))
      row['endpoint_ik']=dict(target_m=target.tolist(),**min(cases,key=lambda r:(not r['passes'],r['position_error_m']+.001*r['orientation_error_deg'])))
    rows.append(row)
  summary={k:float(np.median([r['closest'][k] for r in rows])) for k in ['distance_m','closing_alignment','approach_alignment','full_rotation_error_deg','tilt_deg','wall_gap_m','aperture_m']}
  summary.update(episodes=len(rows),sampled_tilt_over20_count=sum(r['max_sampled_tilt_deg']>20 for r in rows),endpoint_ik_cases=sum('endpoint_ik' in r for r in rows),endpoint_ik_passes=sum(r.get('endpoint_ik',{}).get('passes',False) for r in rows))
  report=dict(task=evaluation['task'],checkpoint_sha256=evaluation['checkpoint_sha256'],finger_position_m=args.finger_position,floor_clearance_m=args.floor_clearance,method='CPU FK every tenth recorded20ms state plus terminal,128 first episodes.32 evenly indexed closest ramp poses additionally use bounded three-seed endpoint IK at the reported opening and optional floor guard. Endpoint gate position<2cm, orientation<15deg and no robot/floor/wall penetration>3mm. No integration, policy supervision, trajectory feasibility or RL success measurement.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
