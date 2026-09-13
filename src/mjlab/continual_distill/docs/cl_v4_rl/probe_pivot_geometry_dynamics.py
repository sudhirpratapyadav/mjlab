"""Matched native CPU dynamics for preregistered Pivot contact-target diagnostics."""
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


def contact_wrench(model, data, index, body_geoms, origin):
  """World wrench on the selected geom side, about origin.

  MuJoCo contact force acts geom0 -> geom1; frame axes are stored as rows.
  https://mujoco.readthedocs.io/en/latest/computation/#contact
  """
  contact=data.contact[index]
  assert (contact.geom[0] in body_geoms) != (contact.geom[1] in body_geoms)
  local=np.zeros(6);mujoco.mj_contactForce(model,data,index,local)
  rotation=contact.frame.reshape(3,3).T
  sign=1 if contact.geom[1] in body_geoms else -1
  force=sign*(rotation@local[:3])
  torque=sign*(rotation@local[3:])+np.cross(contact.pos-origin,force)
  return np.r_[force,torque]


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--legacy-audit',type=Path,required=True)
  p.add_argument('--standoff-audit',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--gravity-compensate',action='store_true')
  p.add_argument('--contact-wrenches',action='store_true')
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());old=json.loads(args.legacy_audit.read_text());new=json.loads(args.standoff_audit.read_text())
  assert ev['task']=='Mjlab-Pivot-Lift-Franka' and ev['checkpoint_sha256']==old['checkpoint_sha256']==new['checkpoint_sha256']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  board=template.body('board/board').id;grip=template.site('robot/gripper').id
  wall_body=template.body('wall/wall_base').id
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)];arm=template.jnt_qposadr[joints]
  arm_dof=template.jnt_dofadr[joints]
  fingers=[template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  assert template.nu==8 and all(template.actuator_trnid[i,0]==joints[i] for i in range(7))
  robot={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith('board/')}
  wall={i for i in range(template.ngeom) if template.geom(i).name.startswith('wall/')}
  obstacle=wall|{i for i in range(template.ngeom) if template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  modes=[('legacy_hold',0.,0.),('native_standoff_hold',0.,0.),('native_press10_climb20',.01,.02),('native_press20_climb40',.02,.04)]
  rows=[]
  for previous,record in zip(old['rows'],new['rows'],strict=True):
    assert previous['env_id']==record['env_id'] and previous['source_step']==record['source_step']
    lane,step=record['env_id'],record['source_step']
    legacy=next(c for c in previous['cases'] if c['label']=='half_0.050_press_0.000')
    corrected=next(c for c in record['cases'] if c['label']=='half_0.060_press_-0.002')
    m=copy.copy(template);m.geom_friction[:]=t['initial_model_geom_friction'][lane];d=mujoco.MjData(m)
    initial={f:t[f][step,lane].copy() for f in ['qpos','qvel','mocap_pos','mocap_quat']}
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:initial['qpos'][adr:adr+3]-=t['origins'][lane]
    initial['mocap_pos']-=t['origins'][lane]
    def restore():
      mujoco.mj_resetData(m,d)
      for f,v in initial.items():getattr(d,f)[:]=v
      mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0
    restore();rotation=d.site_xmat[grip].reshape(3,3).copy();axis=d.xmat[board].reshape(3,3)[:,0].copy();cases=[]
    for label,press,climb in modes:
      start=np.asarray(legacy['qpos'] if label=='legacy_hold' else corrected['qpos'])
      finish=start.copy();ik_error=0.
      start_hand=np.asarray(legacy["target_m"] if label=="legacy_hold" else corrected["target_m"])
      finish_hand=start_hand.copy()
      if press:
        target=np.asarray(corrected['target_m'])+press*axis+np.array([0.,0.,climb]);finish_hand=target.copy();d.qpos[:]=initial['qpos']
        def residual(q):
          d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
          return np.r_[d.site_xpos[grip]-target,.08*(d.site_xmat[grip].reshape(3,3)-rotation).ravel()]
        result=least_squares(residual,np.clip(start,*bounds),bounds=bounds,max_nfev=100);residual(result.x)
        finish=result.x;ik_error=float(np.linalg.norm(d.site_xpos[grip]-target))
      control_start,control_finish=start.copy(),finish.copy();clipped=0
      if args.gravity_compensate:
        assert np.allclose(m.actuator_gear[:7,0],1) and np.allclose(m.actuator_gear[:7,1:],0)
        kp=m.actuator_gainprm[:7,0]
        assert (kp>0).all() and np.allclose(m.actuator_biasprm[:7,1],-kp)
        commands=[]
        for nominal in (start,finish):
          d.qpos[:]=initial['qpos'];d.qpos[arm]=nominal;d.qvel[:]=0
          mujoco.mj_forward(m,d)
          value=nominal+(d.qfrc_bias[arm_dof]-d.qfrc_passive[arm_dof])/kp
          bounded=np.clip(value,np.asarray(LOWER[:7]),np.asarray(UPPER[:7]));clipped+=int(np.count_nonzero(value!=bounded));commands.append(bounded)
        control_start,control_finish=commands
      restore();maximum=0.;minimum_obstacle=0.;minimum_object=0.;pivoted=False;finite=True;samples=[];wrench_history=[]
      for tick in range(400):
        fraction=np.clip(((tick+1)*m.opt.timestep-.5)/1.,0,1) if press else 0.
        d.ctrl[:7]=(1-fraction)*control_start+fraction*control_finish;d.ctrl[7]=initial['qpos'][fingers].mean()
        mujoco.mj_step(m,d)
        finite=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:break
        if tick%4==3:
          mujoco.mj_forward(m,d);tilt=float(np.rad2deg(np.arccos(np.clip(abs(d.xmat[board].reshape(3,3)[2,2]),0,1))));maximum=max(maximum,tilt);contact=False
          origin=np.array([d.xpos[wall_body,0]-.0525,d.xpos[board,1],0.])
          wrenches={name:np.zeros(6) for name in ('robot','wall','floor')}
          for index,c in enumerate(d.contact):
            a,b=c.geom
            if (a in robot and b in obstacle) or (b in robot and a in obstacle):minimum_obstacle=min(minimum_obstacle,float(c.dist))
            if (a in robot and b in objects) or (b in robot and a in objects):minimum_object=min(minimum_object,float(c.dist))
            if c.dist<=.001 and ((a in wall and b in objects) or (b in wall and a in objects)):contact=True
            if args.contact_wrenches and ((a in objects) != (b in objects)):
              other=b if a in objects else a
              category='robot' if other in robot else 'wall' if other in wall else 'floor' if other in obstacle else None
              if category:wrenches[category]+=contact_wrench(m,d,index,objects,origin)
          if args.contact_wrenches:
            weight=m.body_mass[board]*m.opt.gravity
            wrenches['gravity']=np.r_[weight,np.cross(d.xipos[board]-origin,weight)]
            wrench_history.append(dict(time_s=(tick+1)*m.opt.timestep,**{k:v.tolist() for k,v in wrenches.items()}))
          pivoted|=contact and tilt>20
          if tick%40==39:samples.append(dict(time_s=(tick+1)*m.opt.timestep,tilt_deg=tilt,wall_contact=contact,board_position_m=d.xpos[board].tolist(),hand_position_m=d.site_xpos[grip].tolist(),nominal_hand_target_m=((1-fraction)*start_hand+fraction*finish_hand).tolist(),hand_target_error_m=float(np.linalg.norm(d.site_xpos[grip]-((1-fraction)*start_hand+fraction*finish_hand))),arm_target_rms_error_rad=float(np.sqrt(np.mean((d.ctrl[:7]-d.qpos[arm])**2)))))
      cases.append(dict(mode=label,gravity_target_clipped_values=clipped,start_bias_rad=(control_start-start).tolist(),finish_bias_rad=(control_finish-finish).tolist(),finite=finite,warnings=d.warning.number.tolist(),max_tilt_deg=maximum,pivoted_with_wall_contact=bool(pivoted),min_robot_obstacle_distance_m=minimum_obstacle,min_robot_object_distance_m=minimum_object,finish_ik_error_m=ik_error,samples=samples,contact_wrenches=wrench_history))
    rows.append(dict(env_id=lane,source_step=step,corrected_static_gate=corrected['passes_contact_gate'],cases=cases))
  summary={}
  for i in range(len(modes)):
    cases=[r['cases'][i] for r in rows]
    summary[modes[i][0]]=dict(tilted_over20=sum(c['max_tilt_deg']>20 for c in cases),pivoted_with_wall_contact=sum(c['pivoted_with_wall_contact'] for c in cases),median_max_tilt_deg=float(np.median([c['max_tilt_deg'] for c in cases])),all_finite=all(c['finite'] for c in cases),warning_cases=sum(any(c['warnings']) for c in cases),gravity_target_clipped_cases=sum(c['gravity_target_clipped_values']>0 for c in cases),obstacle_penetrations_over3mm=sum(c['min_robot_obstacle_distance_m']<-.003 for c in cases),max_finish_ik_error_m=max(c['finish_ik_error_m'] for c in cases),median_hand_error_at04s_m=float(np.median([c['samples'][1]['hand_target_error_m'] for c in cases])),median_final_hand_error_m=float(np.median([c['samples'][-1]['hand_target_error_m'] for c in cases])))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],episodes=len(rows),gravity_compensated_targets=args.gravity_compensate,contact_wrenches=args.contact_wrenches,wrench_convention='World forceXYZ and torqueXYZ on board about wall near-face/floor point at boardY; contact-frame force acts geom0 toward geom1, axes stored as rows. Recomputed native solver forces every20ms, grouped by robot/wall/floor; gravity reported separately. https://mujoco.readthedocs.io/en/latest/computation/#contact',method='Native CPU400x5ms steps from all32 original recorded states and velocities with source friction/cold solver cache. Restore original hand state, never IK endpoint initialization. Compare legacy absolute hold, corrected60mm extent+2mm outward hold, and corrected hold followed after0.5s by one-second linear joint-target interpolation to10mm press/20mm climb or20mm press/40mm climb. Optional flagged gravity compensation changes only bounded joint-position targets using native zero-velocity bias/passive forces divided by verified actuator kp. Keep original wrist and finger target. Report20deg tilt with actual wall contact as a mechanism diagnostic only; these controls never enter training or RL evaluation.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
