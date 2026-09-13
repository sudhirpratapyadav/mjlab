"""Quantify the reset reference in the existing action-change reward, read only."""
import argparse
import json
from pathlib import Path

import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import CENTER, HALF_RANGE


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  assert not args.output.exists()
  ev = json.loads(args.evaluation.read_text())
  assert ev['task'] == 'Mjlab-Lift-Cube-Franka'
  with np.load(Path(ev['trace_dir']) / 'trace.npz') as archive:
    ctrl, qpos = archive['ctrl'], archive['qpos']
  cfg = load_env_cfg(ev['task'])
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, 'cpu').compile()
  acts = [model.actuator(f'robot/actuator{i}').id for i in range(1, 9)]
  joints = [model.joint(f'robot/joint{i}').id for i in range(1, 8)]
  joints.append(model.joint('robot/finger_joint1').id)
  addresses = model.jnt_qposadr[joints]
  center, half = np.array(CENTER[:8]), np.array(HALF_RANGE[:8])
  rows = []
  for episode in ev['records']:
    lane, end = episode['env_id'], episode['steps']
    actions = (ctrl[1:end+1, lane][:, acts] - center) / half
    assert np.max(np.abs(actions)) < 1.00001
    first = float(np.square(actions[0]).sum())
    subsequent = float(np.square(np.diff(actions, axis=0)).sum())
    hold = (qpos[0, lane, addresses] - center) / half
    rows.append(dict(env_id=lane, steps=end, success=episode['success'],
      first_action_squared_distance_from_zero=first,
      subsequent_action_change_squared_sum=subsequent,
      first_fraction_of_total=first/(first+subsequent) if first+subsequent else 0.,
      measured_initial_pose_squared_distance_from_zero=float(np.square(hold).sum()),
      first_action_squared_distance_from_initial_pose=float(np.square(actions[0]-hold).sum())))
  fields = [name for name in rows[0] if name not in ('env_id', 'steps', 'success')]
  result = dict(checkpoint_sha256=ev['checkpoint_sha256'], evaluation=str(args.evaluation),
    method='All recorded deterministic first episodes. Invert applied8D controls through the approved affine mapping. Existing action_rate_l2 compares the first action with reset-zero action history; later terms compare consecutive actions. These are unweighted squared distances, not a rerun or an exact stochastic-training reward audit. Initial-pose comparison is diagnostic only, never applied to the policy or reward.',
    summary={name:dict(median=float(np.median([r[name] for r in rows])),
                       p95=float(np.quantile([r[name] for r in rows], .95))) for name in fields},
    rows=rows)
  args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
  print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
  main()
