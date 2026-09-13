"""Native CPU constant-target interventions from recorded Pivot states; not RL scores."""
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
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation', type=Path, required=True)
  p.add_argument('--audit', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  p.add_argument('--grid', action='store_true')
  args = p.parse_args()
  if args.output.exists():
    p.error('Output exists')
  ev = json.loads(args.evaluation.read_text())
  audit = json.loads(args.audit.read_text())
  assert ev['task'] == 'Mjlab-Pivot-Lift-Franka'
  assert ev['checkpoint_sha256'] == audit['checkpoint_sha256']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as archive:
    trace = {k:archive[k] for k in archive.files}
  cfg = load_env_cfg(ev['task']); cfg.scene.num_envs = 1
  template = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(template)
  grip = template.site('robot/gripper').id
  board = template.body('board/board').id
  armj = [template.joint(f'robot/joint{i}').id for i in range(1,8)]
  arm = template.jnt_qposadr[armj]
  fingers = [template.jnt_qposadr[template.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  bounds = (template.jnt_range[armj,0]+1e-6,template.jnt_range[armj,1]-1e-6)
  assert template.nu == 8 and all(template.actuator_trnid[i,0]==armj[i] for i in range(7))
  robot = {i for i in range(template.ngeom) if template.geom(i).name.startswith('robot/')}
  obstacle = {i for i in range(template.ngeom) if template.geom(i).name.startswith('wall/') or template.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  modes = [('hold_actual',(0.,0.),(0.,0.)),('near_5mm',(.005,0.),(.005,0.)),('near_10mm',(.010,0.),(.010,0.)),('near_5mm_then_push_10mm',(.005,0.),(-.005,0.))]
  if args.grid:
    modes = [(f'near_{x:.3f}_z_{z:.3f}',(x,z),(x,z)) for x in [-.005,0.,.003,.005,.007] for z in [-.002,.002,.005,.010]]
  rows = []
  for record in audit['rows'][::4]:
    lane = record['env_id']; step = record['closest']['step']
    m = copy.copy(template)
    m.geom_friction[:] = trace['initial_model_geom_friction'][lane]
    d = mujoco.MjData(m)
    initial = {field:trace[field][step,lane].copy() for field in ['qpos','qvel','mocap_pos','mocap_quat']}
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        initial['qpos'][adr:adr+3] -= trace['origins'][lane]
    initial['mocap_pos'] -= trace['origins'][lane]
    def restore():
      mujoco.mj_resetData(m,d)
      for field,value in initial.items():
        getattr(d,field)[:] = value
      mujoco.mj_forward(m,d)
      d.qacc_warmstart[:] = 0
    restore()
    origin = d.site_xpos[grip].copy()
    rotation = d.site_xmat[grip].reshape(3,3).copy()
    near = -d.xmat[board].reshape(3,3)[:,0].copy()
    targets = {}
    for shift,zshift in sorted({v for _,a,b in modes for v in (a,b)}):
      target = origin + shift*near + np.array([0.,0.,zshift])
      def residual(q):
        d.qpos[arm] = q
        mujoco.mj_kinematics(m,d)
        return np.r_[d.site_xpos[grip]-target,.08*(d.site_xmat[grip].reshape(3,3)-rotation).ravel()]
      result = least_squares(residual,np.clip(initial['qpos'][arm],*bounds),bounds=bounds,max_nfev=80)
      residual(result.x)
      targets[(shift,zshift)] = dict(q=result.x,position_error_m=float(np.linalg.norm(d.site_xpos[grip]-target)))
    cases = []
    for label,first,second in modes:
      restore()
      ctrl = np.r_[targets[first]['q'],initial['qpos'][fingers].mean()]
      max_tilt = 0.; min_obstacle_dist = 0.; samples = []
      finite = True
      for tick in range(400):
        # Staged diagnostic moves outward for0.5s, then applies an inward target.
        ctrl[:7] = targets[first if tick<100 else second]['q']
        d.ctrl[:] = ctrl
        mujoco.mj_step(m,d)
        finite &= bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:
          break
        if tick % 4 == 3:
          mujoco.mj_forward(m,d)
          tilt = float(np.rad2deg(np.arccos(np.clip(abs(d.xmat[board].reshape(3,3)[2,2]),0,1))))
          max_tilt = max(max_tilt,tilt)
          for c in d.contact:
            a,b = c.geom
            if (a in robot and b in obstacle) or (b in robot and a in obstacle):
              min_obstacle_dist = min(min_obstacle_dist,float(c.dist))
          if tick % 40 == 39:
            samples.append(dict(time_s=(tick+1)*m.opt.timestep,tilt_deg=tilt,board_position_m=d.xpos[board].tolist()))
      cases.append(dict(mode=label,max_tilt_deg=max_tilt,final_tilt_deg=samples[-1]['tilt_deg'] if samples else None,
                        min_robot_obstacle_distance_m=min_obstacle_dist,finite=finite,warnings=d.warning.number.tolist(),samples=samples))
    rows.append(dict(env_id=lane,source_step=step,ik_max_position_error_m=max(v['position_error_m'] for v in targets.values()),cases=cases))
  summary = {}
  for i in range(len(modes)):
    cases = [r['cases'][i] for r in rows]
    summary[cases[0]['mode']] = dict(tilted_over20=sum(c['max_tilt_deg']>20 for c in cases),
      median_max_tilt_deg=float(np.median([c['max_tilt_deg'] for c in cases])),
      final_over20=sum(c['final_tilt_deg'] is not None and c['final_tilt_deg']>20 for c in cases),
      all_finite=all(c['finite'] for c in cases),warning_episodes=sum(any(c['warnings']) for c in cases),
      robot_obstacle_penetration_over3mm=sum(c['min_robot_obstacle_distance_m']<-.003 for c in cases))
  report = dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],episodes=len(rows),grid=args.grid,
    method='Native CPU400×5ms steps from every fourth recorded closest approach state, preserving initial velocities and exact trace friction, cold solver cache. Bounded IK produces constant joint actuator targets at0/5/10mm outward hand shifts with original wrist/opening; one staged diagnostic returns inward after0.5s. No policy actions, demonstrations, training, first-episode evaluation or RL success claim. Optional grid varies outward shift[-5,0,3,5,7]mm and vertical shift[-2,2,5,10]mm. Max tilt is diagnostic only; obstacle penetration reported separately.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


if __name__ == '__main__':
  main()
