"""Bounded vertical-lift interventions from recorded Peg states; never RL data."""
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
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  if args.output.exists(): p.error('Output exists')
  ev = json.loads(args.evaluation.read_text()); assert ev['task']=='Mjlab-Peg-Insertion-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a: t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1;cmd=next(iter(cfg.commands.values()))
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  grip=template.site('robot/gripper').id;site=template.site(cmd.asset_name+'/object_site').id
  body=template.site_bodyid[site]
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)]
  arm=template.jnt_qposadr[joints];dof=template.jnt_dofadr[joints]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  finger=template.jnt_qposadr[template.joint('robot/finger_joint1').id]
  assert template.nu==8 and all(template.actuator_trnid[i,0]==joints[i] for i in range(7))
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith(cmd.asset_name+'/')}
  robots={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  rows=[]
  for rec in ev['records'][::4]:
    lane,end=rec['env_id'],rec['steps'];m=copy.copy(template);m.geom_friction[:]=t['initial_model_geom_friction'][lane]
    d=mujoco.MjData(m);initial={f:t[f][end,lane].copy() for f in ['qpos','qvel','mocap_pos','mocap_quat','ctrl']}
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:initial['qpos'][adr:adr+3]-=t['origins'][lane]
    initial['mocap_pos']-=t['origins'][lane]
    def restore():
      mujoco.mj_resetData(m,d)
      for f,v in initial.items():getattr(d,f)[:]=v
      mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0
    restore();start=d.site_xpos[grip].copy();rotation=d.site_xmat[grip].reshape(3,3).copy();root_height=float(d.xpos[body,2])
    nominal=initial['qpos'][arm].copy();waypoints=[];max_ik=0.;clipped=0
    for dz in np.linspace(0,.05,11):
      target=start+np.array([0,0,dz])
      def residual(q):
        d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
        return np.r_[d.site_xpos[grip]-target,.08*(d.site_xmat[grip].reshape(3,3)-rotation).ravel()]
      solution=least_squares(residual,np.clip(nominal,*bounds),bounds=bounds,max_nfev=100)
      nominal=solution.x;residual(nominal);max_ik=max(max_ik,float(np.linalg.norm(d.site_xpos[grip]-target)))
      d.qvel[:]=0;mujoco.mj_forward(m,d)
      kp=m.actuator_gainprm[:7,0]
      assert np.all(kp>0) and np.allclose(m.actuator_biasprm[:7,1],-kp)
      value=nominal+(d.qfrc_bias[dof]-d.qfrc_passive[dof])/kp
      bounded=np.clip(value,np.asarray(LOWER[:7]),np.asarray(UPPER[:7]));clipped+=int(np.count_nonzero(bounded!=value));waypoints.append(bounded)
    cases=[]
    for mode in ['frozen_recorded','vertical_recorded_grip','vertical_gentle_grip']:
      restore();samples=[];finite=True
      for tick in range(round(2/m.opt.timestep)):
        time=(tick+1)*m.opt.timestep
        if mode!='frozen_recorded':
          phase=min(time,1)*10;lo=min(int(phase),9);frac=phase-lo
          d.ctrl[:7]=(1-frac)*waypoints[lo]+frac*waypoints[lo+1]
          if mode=='vertical_gentle_grip':d.ctrl[7]=max(initial['qpos'][finger]-.002,0)
        mujoco.mj_step(m,d)
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if tick%4==3:
          mujoco.mj_forward(m,d);force=np.zeros(6)
          for index,c in enumerate(d.contact):
            a,b=c.geom
            if (a in objects and b in robots) or (b in objects and a in robots):force+=contact_wrench(m,d,index,objects,d.xpos[body])
          desired=start+np.array([0,0,.05*min(time,1)]) if mode!='frozen_recorded' else start
          samples.append(dict(time_s=time,root_lift_m=float(d.xpos[body,2]-root_height),tip_height_m=float(d.site_xpos[site,2]),
            hand_error_m=float(np.linalg.norm(d.site_xpos[grip]-desired)),robot_force_N=force[:3].tolist()))
      cases.append(dict(mode=mode,finite=finite,warnings=int(d.warning.number.sum()),max_root_lift_m=max(x['root_lift_m'] for x in samples),terminal=samples[-1],samples=samples))
    rows.append(dict(env_id=lane,source_step=end,max_waypoint_ik_error_m=max_ik,clipped_targets=clipped,cases=cases))
  summary={}
  for mode in ['frozen_recorded','vertical_recorded_grip','vertical_gentle_grip']:
    cases=[next(c for c in r['cases'] if c['mode']==mode) for r in rows]
    summary[mode]=dict(episodes=len(cases),lifted_above2cm=sum(c['max_root_lift_m']>.02 for c in cases),
      median_max_lift_m=float(np.median([c['max_root_lift_m'] for c in cases])),
      median_terminal_hand_error_m=float(np.median([c['terminal']['hand_error_m'] for c in cases])),
      finite=all(c['finite'] for c in cases),warnings=sum(c['warnings'] for c in cases))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],
    method='All32 every-fourth recorded terminal states; native CPU with original velocities/friction. Frozen actual controls versus1s vertical50mm IK ramp plus1s hold, original wrist, gravity-compensated bounded targets. Recorded finger target versus current position minus2mm. No IK reset, policy supervision or certification.',
    summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
