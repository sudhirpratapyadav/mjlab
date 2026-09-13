"""Bounded FK/contact counterfactuals around recorded Pivot approach poses."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares

from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation', type=Path, required=True)
  parser.add_argument('--audit', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  evaluation = json.loads(args.evaluation.read_text())
  audit = json.loads(args.audit.read_text())
  assert evaluation['task'] == 'Mjlab-Pivot-Lift-Franka'
  assert evaluation['checkpoint_sha256'] == audit['checkpoint_sha256']
  with np.load(Path(evaluation['trace_dir'])/'trace.npz') as archive:
    trace = {k:archive[k] for k in archive.files}
  cfg = load_env_cfg(evaluation['task']); cfg.scene.num_envs = 1
  m = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(m)
  d = mujoco.MjData(m)
  grip = m.site('robot/gripper').id
  board = m.body('board/board').id
  wall = m.body('wall/wall_base').id
  joints = [m.joint(f'robot/joint{i}').id for i in range(1, 8)]
  arm = m.jnt_qposadr[joints]
  bounds = (m.jnt_range[joints, 0]+1e-6, m.jnt_range[joints, 1]-1e-6)
  pads = {m.geom('robot/'+n).id for n in ['left_finger_pad', 'right_finger_pad']}
  objects = {i for i in range(m.ngeom) if m.geom(i).name.startswith('board/')}
  robot = {i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
  obstacles = {i for i in range(m.ngeom) if m.geom(i).name.startswith('wall/') or m.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE}
  shifts = np.arange(0, .0401, .005)
  rows = []
  for record in audit['rows'][::4]:
    lane = record['env_id']; step = record['closest']['step']
    for field in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
      getattr(d, field)[:] = trace[field][step, lane]
    for adr, kind in zip(m.jnt_qposadr, m.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[adr:adr+3] -= trace['origins'][lane]
    d.mocap_pos[:] -= trace['origins'][lane]
    mujoco.mj_kinematics(m, d)
    initial = d.qpos.copy()
    origin = d.site_xpos[grip].copy()
    rotation = d.site_xmat[grip].reshape(3, 3).copy()
    near = -d.xmat[board].reshape(3, 3)[:, 0].copy()
    cases = []
    for shift in shifts:
      d.qpos[:] = initial
      target = origin + shift*near
      def residual(q):
        d.qpos[arm] = q
        mujoco.mj_kinematics(m, d)
        return np.r_[d.site_xpos[grip]-target, .08*(d.site_xmat[grip].reshape(3, 3)-rotation).ravel()]
      result = least_squares(residual, np.clip(initial[arm], *bounds), bounds=bounds, max_nfev=80)
      residual(result.x)
      error = float(np.linalg.norm(d.site_xpos[grip]-target))
      angle = float(np.rad2deg(np.arccos(np.clip((np.trace(rotation.T@d.site_xmat[grip].reshape(3, 3))-1)/2, -1, 1))))
      mujoco.mj_collision(m, d)
      contacts = []; bad = []
      for contact in d.contact:
        a, b = contact.geom
        if (a in robot and b in obstacles) or (b in robot and a in obstacles):
          if contact.dist < -.003:
            bad.append(dict(geoms=[m.geom(int(i)).name for i in contact.geom], distance_m=float(contact.dist)))
        if (a in pads and b in objects) or (b in pads and a in objects):
          normal = contact.frame[:3]*(1 if a in pads else -1)
          contacts.append(dict(geoms=[m.geom(int(i)).name for i in contact.geom], distance_m=float(contact.dist),
                               pad_to_board_normal=normal.tolist(),
                               tipping_moment_arm_m=float(contact.pos[2]*normal[0]-(contact.pos[0]-(d.xpos[wall,0]-.0525))*normal[2])))
      useful = any(-.003 <= c['distance_m'] <= .001 and c['pad_to_board_normal'][0] > .5 and c['pad_to_board_normal'][2] > .1 for c in contacts)
      tipping = any(-.003 <= c['distance_m'] <= .001 and c['pad_to_board_normal'][0] > .5 and c['tipping_moment_arm_m'] > .001 for c in contacts)
      cases.append(dict(nearward_shift_m=float(shift), position_error_m=error, rotation_error_deg=angle,
                        obstacle_penetrations=bad, contacts=contacts,
                        inward_upward_contact=useful, valid=useful and error < .002 and angle < 3 and not bad,
                        valid_tipping_contact=tipping and error < .002 and angle < 3 and not bad))
    rows.append(dict(env_id=lane, source_step=step, cases=cases))
  summary = {f'{shift:.3f}':sum(r['cases'][i]['valid'] for r in rows) for i,shift in enumerate(shifts)}
  report = dict(task=evaluation['task'], checkpoint_sha256=evaluation['checkpoint_sha256'], poses=len(rows),
                method='Every fourth closest recorded pose; bounded seven-joint IK translates hand0..40mm toward board near edge, preserving recorded wrist/finger opening and object/wall states. Contact criterion: pad-to-board normal x>0.5,z>0.1, distance[-3,+1]mm; IK<2mm/<3deg and no robot/floor/wall penetration deeper3mm. No integration, policy supervision, demonstrations or RL success claim.',
                valid_counts=summary, rows=rows)
  report['valid_tipping_counts']={f'{shift:.3f}':sum(r['cases'][i]['valid_tipping_contact'] for r in rows) for i,shift in enumerate(shifts)}
  report['tipping_criterion']='Additionally report inward side contacts whose unit force has positive y moment arm>1mm around wall-facing floor edge. This includes horizontal pushing above the support plane; no dynamic tipping claim.'
  args.output.write_text(json.dumps(report, indent=2)+'\n'); print(summary)


if __name__ == '__main__':
  main()
