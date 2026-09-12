"""Diagnose a captured physics transition without policy training or recovery.

Restores the recorded positions, velocities, controls, warm start and mocap state,
then applies the recorded normalized action through the unchanged action manager.
CPU MuJoCo follows the same applied controls for comparison. This is not a full
bitwise replay: uncaptured solver caches and actuator histories are not restored.
"""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from rl_recipes import apply_recipe


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--failure', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--lane', type=int, required=True)
  parser.add_argument('--all-lanes', action='store_true')
  parser.add_argument('--zero-warmstart', action='store_true', help='Diagnostic ablation only; no training configuration changes')
  parser.add_argument('--elliptic-hessian', action='store_true', help='Test stable native-equivalent dense cone Hessian')
  parser.add_argument('--full-hessian', action='store_true', help='Process-local diagnostic: rebuild the Newton Hessian instead of incremental updates')
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Output already exists')
  if args.full_hessian:
    from mujoco_warp._src import solver
    solver._use_incremental = lambda model: False
  saved = torch.load(args.failure, map_location='cpu', weights_only=False)
  previous = saved['previous_state']
  manifest = json.loads((args.failure.parent/'manifest.json').read_text())
  cfg = TrainConfig.from_task(manifest['task'])
  apply_recipe(cfg, manifest['recipe'])
  cfg.env.sim.elliptic_hessian_compat = args.elliptic_hessian or manifest.get('elliptic_hessian_compat', False)
  cfg.env.sim.free_body_implicitfast_compat = manifest.get('free_body_implicitfast_compat', False)
  cfg.env.scene.num_envs = len(previous['qpos']) if args.all_lanes else 1
  cfg.env.seed = manifest['seed']
  env = ManagerBasedRlEnv(cfg.env, device='cuda:0')
  index = args.lane if args.all_lanes else 0
  samples = []
  try:
    env.reset()
    data = env.sim.data
    for name in ('qpos','qvel','ctrl','qacc_warmstart','mocap_pos','mocap_quat'):
      value = previous[name] if args.all_lanes else previous[name][args.lane:args.lane+1]
      getattr(data,name)[:] = value.to(env.device)
    # Rebuild derived transforms/contacts for the restored state. Preserve the
    # recorded warm start after this operation for the actual stepping test.
    env.sim.forward()
    warm = previous['qacc_warmstart'] if args.all_lanes else previous['qacc_warmstart'][args.lane:args.lane+1]
    data.qacc_warmstart[:] = warm.to(env.device)
    if args.zero_warmstart:
      env.sim.reset_solver_state()
    cpu = mujoco.MjData(env.sim.mj_model)
    for name in ('qpos','qvel','ctrl','mocap_pos','mocap_quat'):
      getattr(cpu,name)[:] = previous[name][args.lane].numpy()
    mujoco.mj_forward(env.sim.mj_model,cpu)
    cpu.qacc_warmstart[:] = 0 if args.zero_warmstart else previous['qacc_warmstart'][args.lane].numpy()
    action = previous['actions'] if args.all_lanes else previous['actions'][args.lane:args.lane+1]
    env.action_manager.process_action(action.to(env.device))
    for substep in range(cfg.env.decimation):
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      ctrl = data.ctrl[index].detach().cpu().numpy().copy()
      cpu.ctrl[:] = ctrl
      mujoco.mj_step(env.sim.mj_model,cpu)
      env.sim.step()
      env.scene.update(dt=env.physics_dt)
      qpos = data.qpos[index].detach().cpu().numpy().copy()
      qvel = data.qvel[index].detach().cpu().numpy().copy()
      finite = bool(np.isfinite(qpos).all() and np.isfinite(qvel).all())
      bad = (~torch.isfinite(data.qpos).all(-1) | ~torch.isfinite(data.qvel).all(-1)).nonzero().flatten().cpu().tolist()
      samples.append({'substep':substep+1,'gpu_finite':finite,'nonfinite_lanes':bad,
                      'cpu_finite':bool(np.isfinite(cpu.qpos).all() and np.isfinite(cpu.qvel).all()),
                      'cpu_warning_counts':cpu.warning.number.tolist(),
                      'gpu_qpos':qpos.tolist() if finite else None,'gpu_qvel':qvel.tolist() if finite else None,
                      'cpu_qpos':cpu.qpos.tolist(),'cpu_qvel':cpu.qvel.tolist(),
                      'applied_ctrl':ctrl.tolist(),
                      'maximum_qpos_difference':float(np.max(np.abs(qpos-cpu.qpos))) if finite else None})
      if bad:
        break
    report = {'task':manifest['task'],'failure':str(args.failure.resolve()),
              'failure_sha256':hashlib.sha256(args.failure.read_bytes()).hexdigest(),
              'source_lane':args.lane,'num_envs':env.num_envs,'zero_warmstart_ablation':args.zero_warmstart,
              'full_hessian_ablation':args.full_hessian,
              'physics_changed':False,'initial_state_restored':list(previous),
              'elliptic_hessian_compat':cfg.env.sim.elliptic_hessian_compat,
              'free_body_implicitfast_compat':cfg.env.sim.free_body_implicitfast_compat,
              'limitations':'Derived solver caches and actuator histories not captured; CPU arithmetic differs. CPU warnings are reported because MuJoCo may reset invalid state.',
              'physics_timestep':env.physics_dt,'control_substeps':cfg.env.decimation,'samples':samples}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'samples':len(samples),'gpu_failure':any(x['nonfinite_lanes'] for x in samples),
                      'cpu_warnings':samples[-1]['cpu_warning_counts']}),flush=True)
  finally:
    env.close()


if __name__ == '__main__':
  main()
