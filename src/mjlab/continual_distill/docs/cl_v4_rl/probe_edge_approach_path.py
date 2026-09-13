"""Compare actual-state Edge arrival paths; diagnostic controls never train PPO."""
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


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--approach-audit',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--approach-opening',type=float,help='Diagnostic finger target during approach, in metres per finger')
  p.add_argument('--pinch-offset',type=float,default=.065,help='Diagnostic near-side target distance from plate center')
  p.add_argument('--ideal-wrist',action='store_true',help='Rotate during arrival to the nearest canonical side-pinch wrist')
  args=p.parse_args()
  if args.approach_opening is not None and not 0<=args.approach_opening<=.04:p.error('Opening must be within approved0–40mm per finger')
  if not 0<args.pinch_offset<=.1:p.error('Pinch offset must be in(0,0.1]m')
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());audit=json.loads(args.approach_audit.read_text())
  assert ev['task']=='Mjlab-Edge-Grasp-Franka' and ev['checkpoint_sha256']==audit['checkpoint_sha256']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  grip=template.site('robot/gripper').id;site=template.site('plate/object_site').id
  plate=template.site_bodyid[site];ledge=template.body('ledge/ledge_base').id
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)]
  arm=template.jnt_qposadr[joints];dof=template.jnt_dofadr[joints]
  fingers=[template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  assert template.nu==8 and all(template.actuator_trnid[i,0]==joints[i] for i in range(7))
  pads=[template.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith('plate/')}
  robots={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  obstacles={i for i in range(template.ngeom) if template.geom(i).name.startswith('ledge/') or template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  rows=[]
  for record in audit['rows'][::4]:
    lane=record['env_id'];step=record['closest_exposed_pinch']['step']
    m=copy.copy(template);m.geom_friction[:]=t['initial_model_geom_friction'][lane];d=mujoco.MjData(m)
    initial={f:t[f][step,lane].copy() for f in ['qpos','qvel','mocap_pos','mocap_quat','ctrl']}
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:initial['qpos'][adr:adr+3]-=t['origins'][lane]
    initial['mocap_pos']-=t['origins'][lane]
    def restore():
      mujoco.mj_resetData(m,d)
      for f,v in initial.items():getattr(d,f)[:]=v
      mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0
    restore();rotation=d.site_xmat[grip].reshape(3,3).copy();start=d.site_xpos[grip].copy()
    far=d.xmat[ledge].reshape(3,3)[:,0].copy();target=d.site_xpos[site]-args.pinch_offset*far+np.array([0,0,.005])
    desired_rotation=rotation
    if args.ideal_wrist:
      from probe_edge_wrist import side_frame
      ledge_rotation=d.xmat[ledge].reshape(3,3)
      candidates=[side_frame(ledge_rotation@np.array([np.cos(np.deg2rad(h)),np.sin(np.deg2rad(h)),0.]),np.deg2rad(20)) for h in [0.,70.,-70.]]
      desired_rotation=max(candidates,key=lambda r:np.trace(r.T@rotation))
    plate_height=float(d.xpos[plate,2]);opening=float(initial['qpos'][fingers].mean()) if args.approach_opening is None else args.approach_opening;cases=[]
    for mode in ['frozen_recorded','direct_close_lift','staged_close_lift','staged_open_lift']:
      staged=mode.startswith('staged')
      outside=target-.03*far;outside_high=outside.copy();outside_high[2]=start[2]
      positions=[start,outside_high,outside,target,target,target+np.array([0,0,.05]),target+np.array([0,0,.05])] if staged else [start,target,target,target+np.array([0,0,.05]),target+np.array([0,0,.05])]
      times=np.array([0,.5,1,1.5,2,3,3.5] if staged else [0,1.5,2,3,3.5])
      controls=[];max_ik=0.;clipped=0;seed=initial['qpos'][arm].copy()
      if mode!='frozen_recorded':
        for index,position in enumerate(positions):
          waypoint_rotation=rotation if index==0 else desired_rotation
          d.qpos[:]=initial['qpos']
          def residual(q):
            d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
            return np.r_[d.site_xpos[grip]-position,.08*(d.site_xmat[grip].reshape(3,3)-waypoint_rotation).ravel()]
          sol=least_squares(residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=120)
          seed=sol.x;residual(seed);max_ik=max(max_ik,float(np.linalg.norm(d.site_xpos[grip]-position)))
          d.qvel[:]=0;mujoco.mj_forward(m,d);kp=m.actuator_gainprm[:7,0]
          assert np.all(kp>0) and np.allclose(m.actuator_biasprm[:7,1],-kp)
          raw=seed+(d.qfrc_bias[dof]-d.qfrc_passive[dof])/kp
          bounded=np.clip(raw,np.asarray(LOWER[:7]),np.asarray(UPPER[:7]));clipped+=int(np.count_nonzero(raw!=bounded));controls.append(bounded)
      restore();samples=[];snapshots=[];minimum=0.;finite=True
      for tick in range(round(3.5/m.opt.timestep)):
        time=(tick+1)*m.opt.timestep
        phase=min(int(np.searchsorted(times,time,side='right')-1),len(times)-2)
        fraction=np.clip((time-times[phase])/(times[phase+1]-times[phase]),0,1)
        if mode!='frozen_recorded':
          d.ctrl[:7]=(1-fraction)*controls[phase]+fraction*controls[phase+1]
          closure=np.clip((time-1.5)/.5,0,1) if mode!='staged_open_lift' else 0.
          d.ctrl[7]=(1-closure)*opening+closure*.003
        mujoco.mj_step(m,d);finite&=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:break
        if tick%4==3:
          mujoco.mj_forward(m,d);held=[False,False];robot_plate=False
          for c in d.contact:
            a,b=c.geom
            if (a in robots and b in obstacles) or (b in robots and a in obstacles):minimum=min(minimum,float(c.dist))
            if c.dist>.001:continue
            robot_plate|=(a in robots and b in objects) or (b in robots and a in objects)
            for side,pad in enumerate(pads):
              if a==pad and b in objects:normal=c.frame[:3]
              elif b==pad and a in objects:normal=-c.frame[:3]
              else:continue
              inward=d.geom_xpos[pads[1-side]]-d.geom_xpos[pad];inward/=np.linalg.norm(inward)
              held[side]|=bool(normal@inward>.5)
          nominal=(1-fraction)*positions[phase]+fraction*positions[phase+1]
          samples.append(dict(time_s=time,hand_error_m=float(np.linalg.norm(d.site_xpos[grip]-nominal)),
            plate_lift_m=float(d.xpos[plate,2]-plate_height),opposed_contact=all(held),robot_plate_contact=bool(robot_plate),
            hand_position_m=d.site_xpos[grip].tolist(),plate_position_m=d.xpos[plate].tolist()))
          if abs(time-1.5)<m.opt.timestep/2 or abs(time-2.)<m.opt.timestep/2:
            contacts=[]
            for c in d.contact:
              a,b=c.geom
              if (a in robots and b in objects) or (b in robots and a in objects):
                contacts.append(dict(geoms=[m.geom(a).name,m.geom(b).name],distance_m=float(c.dist),position_m=c.pos.tolist(),normal=c.frame[:3].tolist()))
            snapshots.append(dict(time_s=time,finger_qpos_m=d.qpos[fingers].tolist(),pad_positions_m=d.geom_xpos[pads].tolist(),plate_position_m=d.xpos[plate].tolist(),plate_rotation=d.xmat[plate].reshape(3,3).tolist(),contacts=contacts))
      cases.append(dict(mode=mode,finite=finite,warnings=int(d.warning.number.sum()),max_waypoint_ik_error_m=max_ik,
        clipped_targets=clipped,min_robot_obstacle_distance_m=minimum,
        ever_opposed=any(x['opposed_contact'] for x in samples),held_lift_above2cm=any(x['opposed_contact'] and x['plate_lift_m']>.02 for x in samples),
        terminal=samples[-1],contact_snapshots=snapshots,samples=samples))
    rows.append(dict(env_id=lane,source_step=step,opening_m=opening,target_m=target.tolist(),cases=cases))
  summary={}
  for mode in ['frozen_recorded','direct_close_lift','staged_close_lift','staged_open_lift']:
    c=[next(c for c in r['cases'] if c['mode']==mode) for r in rows]
    summary[mode]=dict(episodes=len(c),ever_opposed=sum(x['ever_opposed'] for x in c),held_lift_above2cm=sum(x['held_lift_above2cm'] for x in c),
      median_terminal_hand_error_m=float(np.median([x['terminal']['hand_error_m'] for x in c])),
      max_waypoint_ik_error_m=max(x['max_waypoint_ik_error_m'] for x in c),clipped_cases=sum(x['clipped_targets']>0 for x in c),
      obstacle_penetrations_over3mm=sum(x['min_robot_obstacle_distance_m']<-.003 for x in c),all_finite=all(x['finite'] for x in c),warnings=sum(x['warnings'] for x in c))
  archive=Path(__file__).resolve().parent/'runs'/f'{args.output.stem}-samples.npz'
  columns=['time_s','hand_error_m','plate_lift_m','opposed_contact','robot_plate_contact','hand_x','hand_y','hand_z','plate_x','plate_y','plate_z']
  arrays={}
  for row in rows:
    for case in row['cases']:
      arrays[f"lane{row['env_id']}_{case['mode']}"]=np.asarray([
        [v['time_s'],v['hand_error_m'],v['plate_lift_m'],v['opposed_contact'],v['robot_plate_contact'],*v['hand_position_m'],*v['plate_position_m']]
        for v in case.pop('samples')])
  archive.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(archive,**arrays)
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],ideal_wrist=args.ideal_wrist,pinch_offset_m=args.pinch_offset,approach_opening_override_m=args.approach_opening,sample_archive=str(archive),sample_columns=columns,
    method='32 every-fourth closest exposed recorded states, all retained; native CPU, original velocities/friction; the reported ideal-wrist variant rotates from actual wrist during the first segment. Frozen actual controls versus1.5s direct or three0.5s outward/lower/approach segments,0.5s close3mm/finger,1s50mm lift,0.5s hold. Approach opening is original recorded position unless an explicit override is reported; staged open keeps that approach target throughout. Initial finger states are never reset. Bounded gravity-compensated IK controls, no IK initialization, policy supervision or certification. Frozen case hand error is relative to the direct nominal path, not a tracking instruction.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
