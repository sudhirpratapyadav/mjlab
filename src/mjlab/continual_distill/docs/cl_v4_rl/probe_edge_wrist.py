"""Endpoint IK/contact audit of side-grasp frames; never policy demonstrations."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def side_frame(heading, pitch):
  down = np.array([0.,0.,-1.])
  return np.column_stack([np.cross(down,heading),
                          np.cos(pitch)*down-np.sin(pitch)*heading,
                          np.cos(pitch)*heading+np.sin(pitch)*down])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation',type=Path,required=True)
  parser.add_argument('--approach-audit',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--recorded-opening',action='store_true',help='Keep the original recorded finger positions for an aperture ablation')
  args = parser.parse_args()
  evaluation = json.loads(args.evaluation.read_text())
  audit = json.loads(args.approach_audit.read_text())
  assert evaluation['task']=='Mjlab-Edge-Grasp-Franka'
  assert evaluation['checkpoint_sha256']==audit['checkpoint_sha256']
  with np.load(Path(evaluation['trace_dir'])/'trace.npz') as archive:
    trace = {key:archive[key] for key in archive.files}
  cfg = load_env_cfg(evaluation['task'])
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene,'cpu').compile()
  cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)
  grip,plate = [model.site(name).id for name in ['robot/gripper','plate/object_site']]
  ledge = model.body('ledge/ledge_base').id
  joints = [model.joint(f'robot/joint{i}').id for i in range(1,8)]
  addresses = model.jnt_qposadr[joints]
  bounds = (model.jnt_range[joints,0]+1e-6,model.jnt_range[joints,1]-1e-6)
  finger_addresses = [model.jnt_qposadr[model.joint('robot/'+name).id] for name in ['finger_joint1','finger_joint2']]
  robot_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')}
  ledge_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith('ledge/')}
  floor_geoms = {i for i in range(model.ngeom) if model.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE}
  # Solver seeds locate possible IK branches only. These are not used to
  # initialize, supervise or command the RL policy.
  branch_seeds = [np.array([-.47,.39,-.13,-2.81,1.35,1.52,-.85]),
                  np.array([.45,.38,.11,-2.83,-1.40,1.53,-.71])]
  rows = []
  for record in audit['rows'][::4]:
    lane = record['env_id']
    step = record['closest_exposed_pinch']['step']
    for field in ['qpos','qvel','mocap_pos','mocap_quat']:
      getattr(data,field)[:] = trace[field][step,lane]
    for address,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[address:address+3] -= trace['origins'][lane]
    data.mocap_pos[:] -= trace['origins'][lane]
    if not args.recorded_opening:
      data.qpos[finger_addresses] = .035
    opening = data.qpos[finger_addresses].copy()
    mujoco.mj_kinematics(model,data)
    ledge_rotation = data.xmat[ledge].reshape(3,3).copy()
    target = data.site_xpos[plate]-.065*ledge_rotation[:,0]+np.array([0.,0.,.005])
    seeds = [data.qpos[addresses].copy(),*branch_seeds]
    cases = []
    for angle in [0.,70.,-70.]:
      heading = ledge_rotation@np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.])
      rotation = side_frame(heading,np.deg2rad(20.))
      def residual(q):
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model,data)
        return np.concatenate([data.site_xpos[grip]-target,
                               .08*(data.site_xmat[grip].reshape(3,3)-rotation).ravel()])
      solutions = []
      for seed in seeds:
        result = least_squares(residual,np.clip(seed,*bounds),bounds=bounds,max_nfev=120,
                               ftol=1e-8,xtol=1e-8,gtol=1e-8)
        residual(result.x)
        error = float(np.linalg.norm(data.site_xpos[grip]-target))
        angular = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T@data.site_xmat[grip].reshape(3,3))-1)/2,-1,1))))
        mujoco.mj_collision(model,data)
        bad = []
        for contact in data.contact:
          a,b = contact.geom
          if contact.dist<-.003 and ((a in robot_geoms and b in ledge_geoms|floor_geoms) or (b in robot_geoms and a in ledge_geoms|floor_geoms)):
            bad.append(dict(geoms=[model.geom(a).name,model.geom(b).name],depth_m=float(-contact.dist)))
        solutions.append(dict(position_error_m=error,orientation_error_deg=angular,
                              endpoint_robot_ledge_floor_penetrations=bad,
                              passes_endpoint_gate=error<.02 and angular<15 and not bad,
                              qpos=result.x.tolist(),solver_evaluations=result.nfev))
      best = min(solutions,key=lambda row:(not row['passes_endpoint_gate'],row['position_error_m']+.001*row['orientation_error_deg']))
      cases.append(dict(heading_deg=angle,**best))
    rows.append(dict(env_id=lane,target_m=target.tolist(),finger_qpos_m=opening.tolist(),cases=cases))
    if len(rows)%8==0:
      print(f'Checked {len(rows)} exposed poses',flush=True)
  summary = {str(angle):sum(next(case for case in row['cases'] if case['heading_deg']==angle)['passes_endpoint_gate'] for row in rows) for angle in [0.,70.,-70.]}
  summary['either_angled_heading'] = sum(any(case['passes_endpoint_gate'] for case in row['cases'] if case['heading_deg']!=0) for row in rows)
  report = dict(task=evaluation['task'],checkpoint_sha256=evaluation['checkpoint_sha256'],
                recorded_opening=args.recorded_opening,
                method='32 evenly indexed recorded exposed-plate poses. Bounded endpoint IK, three seeds per frame; headings0,+70,-70deg in ledge frame,20deg downward pitch. Finger positions recorded in each row: original when recorded_opening is true, otherwise35mm/finger. Pass: position<2cm, orientation<15deg, no recomputed robot/ledge/floor penetration deeper than3mm. No integration, trajectory feasibility, demonstrations or RL success claim.',
                endpoint_pass_counts=summary,poses=len(rows),rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(summary,indent=2))


if __name__=='__main__':
  main()
