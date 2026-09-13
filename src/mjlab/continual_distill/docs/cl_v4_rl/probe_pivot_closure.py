"""Native bounded closing/lift feasibility after the retained diagnostic pivot path."""
import argparse
import copy
import json
from pathlib import Path
import mujoco
import numpy as np
from scipy.optimize import least_squares
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import LOWER,UPPER


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--controls',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--inward-shift',type=float,help='Separate diagnostic: move along board+x during closure; keep original modes unchanged when omitted')
  a=p.parse_args()
  if a.inward_shift is not None and not 0<a.inward_shift<=.04:p.error('Inward shift must be in(0,.04]m')
  modes=(['fixed_hand_close','lower_pad_preserving_close'] if a.inward_shift is None else ['inward_fixed_hand_close','inward_lower_pad_preserving_close'])
  archive=Path(__file__).resolve().parent/'runs'/f'{a.output.stem}.npz'
  if a.output.exists() or archive.exists():p.error('Output or state archive exists')
  with np.load(a.controls) as z:source={k:z[k] for k in z.files}
  assert np.isfinite(source['controls']).all()
  assert np.all(source['controls']>=np.r_[LOWER[:7],0.]-1e-6)
  assert np.all(source['controls']<=np.r_[UPPER[:7],.04]+1e-6)
  cfg=load_env_cfg('Mjlab-Pivot-Lift-Franka');cfg.scene.num_envs=1
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)]
  assert template.nu==8 and all(template.actuator_trnid[i,0]==joints[i] for i in range(7))
  assert abs(template.opt.timestep-float(source['timestep']))<1e-12
  arm=template.jnt_qposadr[joints];dof=template.jnt_dofadr[joints]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  fingers=[template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  grip=template.site('robot/gripper').id;board=template.body('board/board').id
  pads=[template.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith('board/')}
  robot={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  obstacle={i for i in range(template.ngeom) if template.geom(i).name.startswith('wall/') or template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  rows=[];archives={}
  for lane in range(len(source['lanes'])):
    m=copy.copy(template);m.geom_friction[:]=source['geom_friction'][lane];d=mujoco.MjData(m)
    fields=['qpos','qvel','mocap_pos','mocap_quat']
    for f in fields:getattr(d,f)[:]=source[f][lane]
    mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0
    for ctrl in source['controls'][lane].astype(np.float32):d.ctrl[:]=ctrl;mujoco.mj_step(m,d)
    mujoco.mj_forward(m,d)
    initial={f:getattr(d,f).copy() for f in fields};start=d.site_xpos[grip].copy();rotation=d.site_xmat[grip].reshape(3,3).copy()
    opening=float(np.clip(d.qpos[fingers].mean(),0,.04) if a.inward_shift is None else d.qpos[fingers].mean());height=float(d.xpos[board,2]);cases=[]
    board_x=d.xmat[board].reshape(3,3)[:,0].copy()
    for mode in modes:
      closed=start.copy()
      if mode.endswith('lower_pad_preserving_close'):closed-=(opening-.003)*rotation[:,1]
      if a.inward_shift is not None:closed+=a.inward_shift*board_x
      positions=[start,closed,closed+np.array([0,0,.10]),closed+np.array([0,0,.10])]
      commands=[];seed=initial['qpos'][arm].copy();error=0.;clipped=0
      for point in positions:
        d.qpos[:]=initial['qpos'];d.qvel[:]=0
        def residual(q):
          d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
          return np.r_[d.site_xpos[grip]-point,.08*(d.site_xmat[grip].reshape(3,3)-rotation).ravel()]
        sol=least_squares(residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=120)
        seed=sol.x;residual(seed);error=max(error,float(np.linalg.norm(d.site_xpos[grip]-point)))
        mujoco.mj_forward(m,d);kp=m.actuator_gainprm[:7,0]
        assert (kp>0).all() and np.allclose(m.actuator_biasprm[:7,1],-kp)
        raw=seed+(d.qfrc_bias[dof]-d.qfrc_passive[dof])/kp
        bounded=np.clip(raw,np.asarray(LOWER[:7]),np.asarray(UPPER[:7]));clipped+=int(np.count_nonzero(raw!=bounded));commands.append(bounded)
      mujoco.mj_resetData(m,d)
      for f in fields:getattr(d,f)[:]=initial[f]
      mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0;samples=[];minimum=0.;finite=True;states={f:[] for f in ['qpos','qvel','ctrl']}
      for tick in range(600):
        time=(tick+1)*m.opt.timestep;phase=min(int(time),2);fraction=np.clip(time-phase,0,1)
        d.ctrl[:7]=(1-fraction)*commands[phase]+fraction*commands[phase+1]
        d.ctrl[7]=np.clip((1-min(time,1))*opening+min(time,1)*.003,0,.04)
        mujoco.mj_step(m,d);finite &=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:break
        if tick%4==3:
          mujoco.mj_forward(m,d);held=[False,False]
          for c in d.contact:
            x,y=c.geom
            if (x in robot and y in obstacle) or (y in robot and x in obstacle):minimum=min(minimum,float(c.dist))
            if c.dist>.001:continue
            for side,pad in enumerate(pads):
              if x==pad and y in objects:normal=c.frame[:3]
              elif y==pad and x in objects:normal=-c.frame[:3]
              else:continue
              inward=d.geom_xpos[pads[1-side]]-d.geom_xpos[pad];inward/=np.linalg.norm(inward)
              held[side]|=bool(normal@inward>.5)
          samples.append([time,float(d.xpos[board,2]-height),float(all(held)),float(np.rad2deg(np.arccos(np.clip(abs(d.xmat[board].reshape(3,3)[2,2]),0,1)))),*d.qpos[fingers]])
          if a.inward_shift is not None:
            for f in states:states[f].append(getattr(d,f).copy())
      samples=np.asarray(samples);archives[f'lane{int(source["lanes"][lane])}_{mode}']=samples
      if a.inward_shift is not None:
        for f,values in states.items():archives[f'lane{int(source["lanes"][lane])}_{mode}_{f}']=np.asarray(values)
      cases.append(dict(mode=mode,all_finite=finite,warnings=int(d.warning.number.sum()),ever_opposed=bool(samples[:,2].any()),held_lift_above2cm=bool(((samples[:,1]>.02)&(samples[:,2]>0)).any()),max_lift_m=float(samples[:,1].max()),max_ik_error_m=error,clipped_targets=clipped,min_robot_obstacle_distance_m=minimum))
    rows.append(dict(env_id=int(source['lanes'][lane]),starting_opening_m=opening,cases=cases))
  summary={}
  for mode in modes:
    cs=[next(c for c in r['cases'] if c['mode']==mode) for r in rows]
    summary[mode]=dict(episodes=len(cs),ever_opposed=sum(c['ever_opposed'] for c in cs),held_lift_above2cm=sum(c['held_lift_above2cm'] for c in cs),all_finite=all(c['all_finite'] for c in cs),warnings=sum(c['warnings'] for c in cs),max_ik_error_m=max(c['max_ik_error_m'] for c in cs),obstacle_penetrations_over3mm=sum(c['min_robot_obstacle_distance_m']<-.003 for c in cs))
  np.savez_compressed(archive,**archives)
  a.output.write_text(json.dumps(dict(task='Mjlab-Pivot-Lift-Franka',checkpoint_sha256=str(source['checkpoint_sha256']),source_controls=str(a.controls.resolve()),sample_archive=str(archive),sample_columns=['time_s','lift_m','opposed','tilt_deg','finger1','finger2'],inward_shift_m=a.inward_shift,method='All32 original bounded-path cases, native CPU, original states/friction. After full5s diagnostic pivot path:1s close to3mm,1s100mm lift,1s hold. Compare fixed hand and lower-pad-preserving hand shift, with reported optional board+x insertion during closing. In insertion modes actual loaded opening determines geometry while commands stay bounded; full sampled states retained. Bounded IK physical controls, no policy data/reset demonstrations/certification.',summary=summary,rows=rows),indent=2,allow_nan=False)+'\n')
  print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
