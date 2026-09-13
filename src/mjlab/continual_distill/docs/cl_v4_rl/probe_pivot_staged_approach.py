"""Diagnostic staged side arrival from actual Pivot states; never policy data."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import LOWER, UPPER
from probe_pivot_geometry_dynamics import contact_wrench


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--standoff-audit',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--extended-climb',action='store_true')
  p.add_argument('--incremental-climb',action='store_true')
  p.add_argument('--wrist-transition',action='store_true')
  p.add_argument('--wrist-angles',type=float,nargs='+',default=[0.,-30.,-45.])
  p.add_argument('--save-controls',type=Path,help='Export initial states and every physical control for a single wrist variant')
  args=p.parse_args()
  if sum([args.extended_climb,args.incremental_climb,args.wrist_transition])>1:p.error('Select one diagnostic variant')
  if not all(np.isfinite(a) for a in args.wrist_angles):p.error('Wrist angles must be finite')
  if not args.wrist_transition and args.wrist_angles!=[0.,-30.,-45.]:p.error('Wrist angles require wrist transition')
  if args.output.exists():p.error('Output exists')
  if args.save_controls and (args.save_controls.exists() or not args.wrist_transition or len(args.wrist_angles)!=1):
    p.error('Control export requires one wrist variant and a new archive path')
  ev=json.loads(args.evaluation.read_text());audit=json.loads(args.standoff_audit.read_text())
  assert ev['task']=='Mjlab-Pivot-Lift-Franka' and ev['checkpoint_sha256']==audit['checkpoint_sha256']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:trace={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  board=template.body('board/board').id;wallbody=template.body('wall/wall_base').id;grip=template.site('robot/gripper').id
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)]
  arm=template.jnt_qposadr[joints];dof=template.jnt_dofadr[joints]
  fingers=[template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  assert template.nu==8 and all(template.actuator_trnid[i,0]==joints[i] for i in range(7))
  robot={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith('board/')}
  wall={i for i in range(template.ngeom) if template.geom(i).name.startswith('wall/')}
  obstacle=wall|{i for i in range(template.ngeom) if template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  rows=[];exports=[]
  for record in audit['rows']:
    lane,step=record['env_id'],record['source_step'];corrected=next(c for c in record['cases'] if c['label']=='half_0.060_press_-0.002')
    m=copy.copy(template);m.geom_friction[:]=trace['initial_model_geom_friction'][lane];d=mujoco.MjData(m)
    initial={f:trace[f][step,lane].copy() for f in ['qpos','qvel','mocap_pos','mocap_quat']}
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:initial['qpos'][adr:adr+3]-=trace['origins'][lane]
    initial['mocap_pos']-=trace['origins'][lane]
    def restore():
      mujoco.mj_resetData(m,d)
      for f,v in initial.items():getattr(d,f)[:]=v
      mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0
    restore();rotation=d.site_xmat[grip].reshape(3,3).copy();axis=d.xmat[board].reshape(3,3)[:,0].copy();start_hand=d.site_xpos[grip].copy()
    target=np.asarray(corrected['target_m']);cases=[];kp=m.actuator_gainprm[:7,0]
    measured_finger=float(initial['qpos'][fingers].mean())
    finger_target=float(np.clip(measured_finger,LOWER[7],UPPER[7]))
    assert np.allclose(m.actuator_gear[:7,0],1) and np.allclose(m.actuator_gear[:7,1:],0) and np.allclose(m.actuator_biasprm[:7,1],-kp) and (kp>0).all()
    modes=[(.02,.04,angle) for angle in args.wrist_angles] if args.wrist_transition else [(p,c,0.) for p,c in ([(.04,.08),(.06,.12)] if args.extended_climb or args.incremental_climb else [(.01,.02),(.02,.04)])]
    for press,climb,wrist_angle in modes:
      positions=[start_hand,target-.03*axis+np.array([0.,0.,.03]),target-.03*axis,target]
      increments=round(press/.02) if args.incremental_climb else 1
      positions.extend(target+press*k/increments*axis+np.array([0.,0.,climb*k/increments]) for k in range(1,increments+1))
      times=np.array([0.,.5,1.,1.5,*[1.5+k for k in range(1,increments+1)],2.+increments]);commands=[];ik_errors=[];rotation_errors=[];clipped=0;seed=initial['qpos'][arm].copy()
      if args.wrist_transition:
        positions.extend([positions[-1].copy(),target+2*press*axis+np.array([0.,0.,2*climb])])
        times=np.array([0.,.5,1.,1.5,2.5,3.5,4.5,5.])
      for i,position in enumerate(positions):
        d.qpos[:]=initial['qpos']
        angle=np.deg2rad(wrist_angle) if args.wrist_transition and i>=5 else 0.
        rotated=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])@rotation
        if i:
          def residual(q):
            d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
            return np.r_[d.site_xpos[grip]-position,.08*(d.site_xmat[grip].reshape(3,3)-rotated).ravel()]
          result=least_squares(residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=120);seed=result.x;residual(seed)
          ik_errors.append(float(np.linalg.norm(d.site_xpos[grip]-position)))
          rotation_errors.append(float(np.rad2deg(np.arccos(np.clip((np.trace(rotated.T@d.site_xmat[grip].reshape(3,3))-1)/2,-1,1)))))
        else:d.qpos[arm]=seed
        d.qvel[:]=0;mujoco.mj_forward(m,d)
        raw=seed+(d.qfrc_bias[dof]-d.qfrc_passive[dof])/kp
        bounded=np.clip(raw,np.asarray(LOWER[:7]),np.asarray(UPPER[:7]));clipped+=int(np.count_nonzero(raw!=bounded));commands.append(bounded)
      positions.append(positions[-1]);commands.append(commands[-1]);restore()
      samples=[];applied_controls=[];maximum=0.;pivoted=False;minimum=0.;finite=True
      for tick in range(round(times[-1]/m.opt.timestep)):
        time=(tick+1)*m.opt.timestep;phase=min(int(np.searchsorted(times,time,side='right')-1),len(times)-2)
        fraction=np.clip((time-times[phase])/(times[phase+1]-times[phase]),0,1)
        d.ctrl[:7]=(1-fraction)*commands[phase]+fraction*commands[phase+1];d.ctrl[7]=finger_target
        if args.save_controls:applied_controls.append(d.ctrl.copy())
        mujoco.mj_step(m,d)
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:break
        if tick%4==3:
          mujoco.mj_forward(m,d);tilt=float(np.rad2deg(np.arccos(np.clip(abs(d.xmat[board].reshape(3,3)[2,2]),0,1))));maximum=max(maximum,tilt)
          contact=False;origin=np.array([d.xpos[wallbody,0]-.0525,d.xpos[board,1],0.]);force=np.zeros(6)
          for index,c in enumerate(d.contact):
            a,b=c.geom
            if (a in robot and b in obstacle) or (b in robot and a in obstacle):minimum=min(minimum,float(c.dist))
            if c.dist<=.001 and ((a in wall and b in objects) or (b in wall and a in objects)):contact=True
            if (a in robot and b in objects) or (b in robot and a in objects):force+=contact_wrench(m,d,index,objects,origin)
          pivoted|=tilt>20 and contact
          nominal=(1-fraction)*positions[phase]+fraction*positions[phase+1]
          samples.append(dict(time_s=time,tilt_deg=tilt,wall_contact=contact,robot_wrench=force.tolist(),hand_error_m=float(np.linalg.norm(d.site_xpos[grip]-nominal)),hand_position_m=d.site_xpos[grip].tolist(),board_position_m=d.xpos[board].tolist()))
      cases.append(dict(press_m=press,climb_m=climb,wrist_transition_deg=wrist_angle,waypoint_times_s=times.tolist(),waypoints_m=[v.tolist() for v in positions],finite=finite,warnings=d.warning.number.tolist(),max_tilt_deg=maximum,pivoted_with_wall_contact=bool(pivoted),min_robot_obstacle_distance_m=minimum,max_waypoint_ik_error_m=max(ik_errors),max_waypoint_rotation_error_deg=max(rotation_errors),clipped_targets=clipped,samples=samples))
      cases[-1].update(measured_finger_mean_m=measured_finger,finger_target_m=finger_target,finger_target_clipped=finger_target!=measured_finger)
      if args.save_controls:exports.append(dict(initial,controls=np.asarray(applied_controls),geom_friction=m.geom_friction.copy()))
    rows.append(dict(env_id=lane,source_step=step,cases=cases))
  summary={}
  for i in range(len(rows[0]['cases'])):
    c=[r['cases'][i] for r in rows]
    key=str(c[0]['wrist_transition_deg'] if args.wrist_transition else c[0]['press_m'])
    summary[key]=dict(pivoted_with_wall_contact=sum(x['pivoted_with_wall_contact'] for x in c),tilted_over20=sum(x['max_tilt_deg']>20 for x in c),median_max_tilt_deg=float(np.median([x['max_tilt_deg'] for x in c])),all_finite=all(x['finite'] for x in c),warning_cases=sum(any(x['warnings']) for x in c),obstacle_penetrations_over3mm=sum(x['min_robot_obstacle_distance_m']<-.003 for x in c),max_waypoint_ik_error_m=max(x['max_waypoint_ik_error_m'] for x in c),max_waypoint_rotation_error_deg=max(x['max_waypoint_rotation_error_deg'] for x in c),clipped_cases=sum(x['clipped_targets']>0 for x in c))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],episodes=len(rows),incremental_climb=args.incremental_climb,method='Native CPU from32 original recorded poses/velocities/friction, no IK reset. Withdraw30mm outward+30mm up0.5s, lower outside0.5s, side approach0.5s, press/climb1s per recorded segment, hold0.5s. Explicit waypoint positions/times recorded per case. Original wrist/fingers, bounded native joint-position controls with static gravity compensation. Wrenches every20ms; positive20deg+wall contact is a mechanism diagnostic, not RL success. No policy supervision or model change.',summary=summary,rows=rows)
  if args.save_controls:
    args.save_controls.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.save_controls,**{k:np.stack([v[k] for v in exports]) for k in exports[0]},
      lanes=np.asarray([r['env_id'] for r in rows]),source_steps=np.asarray([r['source_step'] for r in rows]),
      timestep=np.asarray(template.opt.timestep),checkpoint_sha256=np.asarray(ev['checkpoint_sha256']))
    report['control_archive']=str(args.save_controls.resolve())
  if args.wrist_transition:
    archive=Path(__file__).resolve().parent/'runs'/f'{args.output.stem}-samples.npz';arrays={}
    for row in rows:
      for case in row['cases']:
        arrays[f"lane{row['env_id']}_wrist{case['wrist_transition_deg']}"]=np.asarray([
          [v['time_s'],v['tilt_deg'],v['wall_contact'],*v['robot_wrench'],v['hand_error_m'],*v['hand_position_m'],*v['board_position_m']]
          for v in case.pop('samples')])
    archive.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(archive,**arrays)
    report.update(wrist_transition=True,wrist_angles_deg=args.wrist_angles,sample_archive=str(archive),sample_columns=['time_s','tilt_deg','wall_contact','force_x','force_y','force_z','torque_x','torque_y','torque_z','hand_error_m','hand_x','hand_y','hand_z','board_x','board_y','board_z'],method=report['method'].replace('Original wrist/fingers','Original fingers; after first20/40mm climb, apply each reported wrist angle around worldY in1s at fixed hand target, then another20/40mm climb in1s. Same first segment and bounds/friction; wrist commands are diagnostic only'))
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
