"""Compare registered Pivot reward targets against actual collider contact geometry."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--audit',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--standoff-grid',action='store_true')
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());audit=json.loads(args.audit.read_text())
  assert ev['task']=='Mjlab-Pivot-Lift-Franka' and ev['checkpoint_sha256']==audit['checkpoint_sha256']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  template=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(template)
  board=template.body('board/board').id;wall=template.body('wall/wall_base').id
  grip=template.site('robot/gripper').id;geom=template.geom('board/board_geom').id
  assert template.geom_type[geom]==mujoco.mjtGeom.mjGEOM_BOX
  extent=float(template.geom_size[geom,0]);half_z=float(template.geom_size[geom,2])
  joints=[template.joint(f'robot/joint{i}').id for i in range(1,8)]
  arm=template.jnt_qposadr[joints]
  bounds=(template.jnt_range[joints,0]+1e-6,template.jnt_range[joints,1]-1e-6)
  fingers=[template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  pads={template.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']}
  objects={i for i in range(template.ngeom) if template.geom(i).name.startswith('board/')}
  robots={i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  obstacles={i for i in range(template.ngeom) if template.geom(i).name.startswith('wall/') or template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  rows=[]
  for rec in audit['rows'][::4]:
    lane,step=rec['env_id'],rec['closest']['step']
    m=copy.copy(template);m.geom_friction[:]=t['initial_model_geom_friction'][lane]
    d=mujoco.MjData(m)
    for field in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,field)[:]=t[field][step,lane]
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=t['origins'][lane]
    d.mocap_pos[:]-=t['origins'][lane];mujoco.mj_forward(m,d)
    initial=d.qpos.copy();rotation=d.site_xmat[grip].reshape(3,3).copy()
    board_rotation=d.xmat[board].reshape(3,3).copy();pos=d.xpos[board].copy()
    tilt=1-abs(board_rotation[2,2]);psi=np.deg2rad(70-15*np.clip(tilt/.43,0,1))
    closing=np.array([np.sin(psi),0,np.cos(psi)]);approach=np.array([np.cos(psi),0,-np.sin(psi)])
    aperture=initial[fingers].sum();cases=[]
    for half_x in ([extent] if args.standoff_grid else [.05,extent]):
      corner=pos-half_x*board_rotation[:,0]+half_z*board_rotation[:,2]
      base=corner+.5*aperture*closing-.005*approach
      base[2]=max(base[2],.001+(.5*aperture+.0152)*abs(closing[2])+.0119*abs(approach[2]))
      for press in ([-.001,-.002,-.003,-.004] if args.standoff_grid else [0.,.002,.004,.006]):
        d.qpos[:]=initial;target=base+press*board_rotation[:,0]
        def residual(q):
          d.qpos[arm]=q;mujoco.mj_kinematics(m,d)
          return np.r_[d.site_xpos[grip]-target,.08*(d.site_xmat[grip].reshape(3,3)-rotation).ravel()]
        result=least_squares(residual,np.clip(initial[arm],*bounds),bounds=bounds,max_nfev=100)
        residual(result.x)
        error=float(np.linalg.norm(d.site_xpos[grip]-target))
        angle=float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T@d.site_xmat[grip].reshape(3,3))-1)/2,-1,1))))
        mujoco.mj_forward(m,d);contacts=[];obstacle_min=0.;object_min=0.
        for c in d.contact:
          a,b=c.geom
          if (a in robots and b in obstacles) or (b in robots and a in obstacles):obstacle_min=min(obstacle_min,float(c.dist))
          if (a in robots and b in objects) or (b in robots and a in objects):object_min=min(object_min,float(c.dist))
          if (a in pads and b in objects) or (b in pads and a in objects):
            n=c.frame[:3]*(1 if a in pads else -1)
            moment=c.pos[2]*n[0]-(c.pos[0]-(d.xpos[wall,0]-.0525))*n[2]
            contacts.append(dict(geoms=[m.geom(int(i)).name for i in c.geom],distance_m=float(c.dist),pad_to_board_normal=n.tolist(),tipping_moment_arm_m=float(moment)))
        useful=any(-.003<=c['distance_m']<=.001 and c['pad_to_board_normal'][0]>.5 and c['tipping_moment_arm_m']>.001 for c in contacts)
        upward=any(-.003<=c['distance_m']<=.001 and c['pad_to_board_normal'][0]>.5 and c['pad_to_board_normal'][2]>.1 for c in contacts)
        cases.append(dict(label=f'half_{half_x:.3f}_press_{press:.3f}',half_extent_m=half_x,pressure_m=press,target_m=target.tolist(),qpos=result.x.tolist(),position_error_m=error,rotation_error_deg=angle,contacts=contacts,min_obstacle_distance_m=obstacle_min,min_object_distance_m=object_min,side_tipping_contact=useful,upward_contact=upward,passes_contact_gate=error<.002 and angle<3 and obstacle_min>=-.003 and object_min>=-.003 and useful))
    rows.append(dict(env_id=lane,source_step=step,cases=cases))
  summary={}
  for i in range(len(rows[0]['cases'])):
    cases=[r['cases'][i] for r in rows]
    summary[cases[0]['label']]=dict(passed=sum(c['passes_contact_gate'] for c in cases),side_tipping_contacts=sum(c['side_tipping_contact'] for c in cases),upward_contacts=sum(c['upward_contact'] for c in cases),object_penetrations_over3mm=sum(c['min_object_distance_m']<-.003 for c in cases),obstacle_penetrations_over3mm=sum(c['min_obstacle_distance_m']<-.003 for c in cases),median_position_error_m=float(np.median([c['position_error_m'] for c in cases])))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],episodes=len(rows),standoff_grid=args.standoff_grid,collider_half_extents_m=template.geom_size[geom].tolist(),method='Native CPU bounded IK at absolute reward waypoints for all32 every-fourth recorded closest states; original wrist, finger opening, object/wall poses and trace friction. Compare50mm reward half-extent with60mm collider extent, each at0/2/4/6mm inward pressure, or native extent with1/2/3/4mm outward standoff when flagged. Contact gate requires IK<2mm/<3deg, inward side contact with positive tipping moment>1mm, and no robot/object/obstacle penetration>3mm. Static geometry only: no dynamics, policy demonstrations, training or RL success claim.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
