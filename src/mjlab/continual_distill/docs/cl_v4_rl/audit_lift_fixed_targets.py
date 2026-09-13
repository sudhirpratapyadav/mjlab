"""Single-state native CPU hold diagnostic; never used as policy or training data."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg

HERE = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--recorded-replay', action='store_true')
  parser.add_argument('--endpoints', action='store_true', help='Save fixed-target task geometry as an additional diagnostic')
  args = parser.parse_args()
  stem = 'lift_s1_recorded_target_audit' if args.recorded_replay else 'lift_s1_fixed_target_audit'
  if args.endpoints:
    stem = stem.removesuffix('_audit') + '_endpoint_audit'
  output = HERE / 'evidence' / (stem + '.json')
  assert not output.exists()
  ev = json.loads((HERE / 'evidence/RL-002-S1-m23489-val-20260914.json').read_text())
  with np.load(Path(ev['trace_dir']) / 'trace.npz') as z:
    trace = {k: z[k] for k in z.files}
  lane, start = 0, 500
  assert ev['records'][lane]['env_id'] == lane and ev['records'][lane]['steps'] >= 600
  cfg = load_env_cfg(ev['task']); cfg.scene.num_envs = 1
  m = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(m)
  assert abs(m.opt.timestep - .005) < 1e-12
  joints = [m.joint(f'robot/joint{i}').id for i in range(1, 8)]
  dofs, qadr = m.jnt_dofadr[joints], m.jnt_qposadr[joints]
  acts = [m.actuator(f'robot/actuator{i}').id for i in range(1, 8)]
  def metrics(v, dt):
    return dict(speed_rms=float(np.sqrt(np.mean(v**2))), speed_max=float(np.abs(v).max()),
                acceleration_rms=float(np.sqrt(np.mean((np.diff(v, axis=0)/dt)**2))))
  recorded = trace['qvel'][start+51:start+101, lane][:, dofs]
  rows = {'recorded_policy_11_to_12s': metrics(recorded, .02)}
  archived = {}
  modes = ['replay_recorded_targets'] if args.recorded_replay else ['hold_recorded_target', 'hold_measured_arm_position']
  for mode in modes:
    d = mujoco.MjData(m); m.geom_friction[:] = trace['initial_model_geom_friction'][lane]
    for field in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'ctrl']:
      getattr(d, field)[:] = trace[field][start, lane]
    for adr, kind in zip(m.jnt_qposadr, m.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[adr:adr+3] -= trace['origins'][lane]
    d.mocap_pos[:] -= trace['origins'][lane]
    if mode == 'hold_measured_arm_position':
      d.ctrl[acts] = d.qpos[qadr]
    mujoco.mj_forward(m, d)
    vel = []
    for step in range(400):
      if mode == 'replay_recorded_targets':
        d.ctrl[:] = trace['ctrl'][start + 1 + step//4, lane]
      mujoco.mj_step(m, d)
      assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
      vel.append(d.qvel[dofs].copy())
    v = np.asarray(vel)
    rows[mode] = dict(native_5ms=metrics(v[200:], .005), sampled_20ms=metrics(v[203::4], .02),
                      warnings=int(d.warning.number.sum()))
    if mode == 'replay_recorded_targets':
      expected = trace['qvel'][start+1:start+101, lane][:, dofs]
      rows[mode]['velocity_rmse_to_original_gpu'] = float(np.sqrt(np.mean((v[3::4] - expected)**2)))
    archived[mode] = v
    if args.endpoints:
      mujoco.mj_forward(m, d)
      body = m.body('cube/cube').id; goal = m.body('mocap_goal/mocap_goal').id
      cube_dof = int(m.jnt_dofadr[m.body_jntadr[body]])
      pads = [m.geom('robot/' + n).id for n in ['left_finger_pad', 'right_finger_pad']]
      objects = {i for i in range(m.ngeom) if m.geom(i).name.startswith('cube/')}
      touches = [any(c.dist <= .001 and ((c.geom[0] == pad and c.geom[1] in objects) or
                 (c.geom[1] == pad and c.geom[0] in objects)) for c in d.contact) for pad in pads]
      rows[mode]['endpoint'] = dict(cube_height_m=float(d.xpos[body, 2]),
        goal_error_m=float(np.linalg.norm(d.xpos[body]-d.xpos[goal])),
        linear_speed_m_s=float(np.linalg.norm(d.qvel[cube_dof:cube_dof+3])),
        angular_speed_rad_s=float(np.linalg.norm(d.qvel[cube_dof+3:cube_dof+6])),
        pad_contacts=[bool(v) for v in touches])
      for field in ['qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat']:
        archived[mode + '_' + field] = getattr(d, field).copy()
  np.savez_compressed(HERE / 'runs' / (stem + '.npz'), **archived)
  report = dict(checkpoint_sha256=ev['checkpoint_sha256'], env_id=lane, source_step=start,
    source_success=ev['records'][lane]['success'], rows=rows,
    method='Preregistered single native CPU cold-state diagnostic. Same recorded state/friction; two fixed arm-target interventions keeping original finger target, or replay original full controls at20ms boundaries for2s. Analyze last1s. No policy rerun, training data, success claim or backend-general conclusion.')
  output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
