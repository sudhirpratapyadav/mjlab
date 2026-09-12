"""Measure approach to the side-pinch waypoint in actual recorded RL states."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from probe_edge_wrist import side_frame


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  evaluation = json.loads(args.evaluation.read_text())
  assert evaluation['task'] == 'Mjlab-Edge-Grasp-Franka'
  with np.load(Path(evaluation['trace_dir'])/'trace.npz') as archive:
    trace = {k:archive[k] for k in archive.files}
  cfg = load_env_cfg(evaluation['task'])
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, 'cpu').compile()
  cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)
  grip, plate = [model.site(n).id for n in ['robot/gripper', 'plate/object_site']]
  ledge = model.body('ledge/ledge_base').id
  rows = []
  for record in evaluation['records']:
    lane = record['env_id']
    samples = []
    # Trace index0 is the initial state; terminal index equals episode steps.
    for step in sorted(set(range(0, record['steps']+1, 10)) | {record['steps']}):
      for name in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
        getattr(data, name)[:] = trace[name][step, lane]
      for address, kind in zip(model.jnt_qposadr, model.jnt_type):
        if kind == mujoco.mjtJoint.mjJNT_FREE:
          data.qpos[address:address+3] -= trace['origins'][lane]
      data.mocap_pos[:] -= trace['origins'][lane]
      mujoco.mj_kinematics(model, data)
      rotation = data.xmat[ledge].reshape(3, 3)
      far = rotation[:, 0]
      pos = data.site_xpos[plate]
      exposure = float(np.clip(-np.dot(pos-data.xpos[ledge], far)/.075, 0, 1))
      target = pos-.065*far+np.array([0, 0, .005])
      error = data.site_xpos[grip]-target
      actual = data.site_xmat[grip].reshape(3, 3)
      angles = []
      for heading in [0., 70., -70.]:
        desired = side_frame(rotation@np.array([np.cos(np.deg2rad(heading)), np.sin(np.deg2rad(heading)), 0]), np.deg2rad(20))
        angles.append(np.rad2deg(np.arccos(np.clip((np.trace(desired.T@actual)-1)/2, -1, 1))))
      samples.append(dict(step=step, exposure=exposure, distance_m=float(np.linalg.norm(error)),
                          vertical_error_m=float(error[2]), horizontal_error_m=float(np.linalg.norm(error[:2])),
                          rotation_error_deg=float(min(angles)), approach_into_ledge=float(actual[:, 2]@far),
                          hand_height_m=float(data.site_xpos[grip, 2]), plate_height_m=float(pos[2])))
    exposed = [v for v in samples if v['exposure'] >= .8]
    rows.append(dict(env_id=lane, closest_exposed_pinch=min(exposed, key=lambda v:v['distance_m']) if exposed else None))
  points = [r['closest_exposed_pinch'] for r in rows if r['closest_exposed_pinch']]
  report = dict(task=evaluation['task'], checkpoint_sha256=evaluation['checkpoint_sha256'],
                method='CPU FK every10th recorded20ms state plus terminal. Existing edge_v3 side-pinch reward waypoint and nearest of three20deg-pitched wrist frames. No integration, demonstrations or success reclassification.',
                episodes=len(rows), episodes_reaching_exposure_gate=len(points),
                median_closest_exposed_pinch={k:float(np.median([p[k] for p in points])) for k in points[0] if k!='step'} if points else {},
                rows=rows)
  args.output.write_text(json.dumps(report, indent=2)+'\n')
  print({k:v for k,v in report.items() if k!='rows'})


if __name__ == '__main__':
  main()
