"""Recompute Lift's gates from saved endpoints, without replaying live buffers."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  ev = json.loads(args.evaluation.read_text())
  assert ev['task'] == 'Mjlab-Lift-Cube-Franka'
  with np.load(Path(ev['trace_dir']) / 'trace.npz') as archive:
    trace = {key: archive[key] for key in archive.files}
  cfg = load_env_cfg(ev['task']); cfg.scene.num_envs = 1
  cmd = next(iter(cfg.commands.values()))
  assert cmd.require_grasp
  model = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)
  body = model.body('cube/cube').id
  goal = model.body('mocap_goal/mocap_goal').id
  dof = int(model.jnt_dofadr[model.body_jntadr[body]])
  pads = [model.geom('robot/' + name).id for name in ('left_finger_pad', 'right_finger_pad')]
  objects = {i for i in range(model.ngeom) if model.geom(i).name.startswith('cube/')}
  rows = []
  for rec in ev['records']:
    lane, step = rec['env_id'], rec['steps']
    mujoco.mj_resetData(model, data)
    model.geom_friction[:] = trace['initial_model_geom_friction'][lane]
    for field in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat'):
      getattr(data, field)[:] = trace[field][step, lane]
    for address, kind in zip(model.jnt_qposadr, model.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        data.qpos[address:address + 3] -= trace['origins'][lane]
    data.mocap_pos[:] -= trace['origins'][lane]
    mujoco.mj_forward(model, data)
    touches = [any(c.dist <= .001 and ((c.geom[0] == pad and c.geom[1] in objects)
               or (c.geom[1] == pad and c.geom[0] in objects)) for c in data.contact) for pad in pads]
    error = float(np.linalg.norm(data.xpos[body] - data.xpos[goal]))
    linear = float(np.linalg.norm(data.qvel[dof:dof + 3]))
    angular = float(np.linalg.norm(data.qvel[dof + 3:dof + 6]))
    native = error < cmd.success_threshold and all(touches) and linear < .10 and angular < .5
    rows.append(dict(env_id=lane, position_error_m=error, linear_speed_m_s=linear,
      angular_speed_rad_s=angular, pad_contacts=[bool(v) for v in touches],
      cpu_native_success=bool(native), recorded_gpu_success=rec['success']))
  summary = dict(episodes=len(rows), cpu_native_successes=sum(r['cpu_native_success'] for r in rows),
    recorded_gpu_successes=ev['successes'], label_agreement=sum(r['cpu_native_success'] == r['recorded_gpu_success'] for r in rows))
  report = dict(task=ev['task'], checkpoint_sha256=ev['checkpoint_sha256'], seed=ev['seed'],
    method='Exact recorded terminal states, original friction, CPU FK/contact queries only; no integration. Lift gate: root position within configured threshold, both pads touching at <=1mm (no opposing-normal requirement), linear speed <0.10m/s and angular speed <0.5rad/s. Norms of free-joint velocities are rotation invariant. No success substitution.',
    limitation='The live predicate uses cached xpos/contact/cvel before the last physics integration; saved qpos/qvel are post-integration. Those original derived buffers were not archived, so this recomputation can differ and does not replace the measured native labels.',
    summary=summary, rows=rows)
  args.output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(summary))


if __name__ == '__main__':
  main()
