"""Replay captured history on GPU with/without the optional free-body gyro solve."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from mjlab.sim.free_body_implicitfast import FreeBodyImplicitFastStepper
from rl_recipes import apply_recipe


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--failure',type=Path,required=True)
  parser.add_argument('--lane',type=int,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--num-envs',type=int,default=1)
  parser.add_argument('--corrected-only',action='store_true',help='Use the production Simulation/CUDA-graph path')
  args=parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  saved=torch.load(args.failure,map_location='cpu',weights_only=False)
  history=saved['state_history'];manifest=json.loads((args.failure.parent/'manifest.json').read_text())
  cfg=TrainConfig.from_task(manifest['task']);apply_recipe(cfg,manifest['recipe']);cfg.env.scene.num_envs=args.num_envs
  cfg.env.sim.free_body_implicitfast_compat=args.corrected_only
  env=ManagerBasedRlEnv(cfg.env,device='cuda:0')
  try:
    env.reset();data=env.sim.data;m=env.sim.mj_model
    asset=next(iter(cfg.env.commands.values())).asset_name
    joint=next(j for j in range(m.njnt) if m.joint(j).name.startswith(asset+'/') and m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE)
    adr=int(m.jnt_dofadr[joint])
    with wp.ScopedDevice(env.sim.wp_device):
      corrected_step=FreeBodyImplicitFastStepper(m,env.sim.wp_device)
    cases=[]
    for corrected in ([True] if args.corrected_only else [False,True]):
      env.reset()
      for name in ['qpos','qvel','ctrl','qacc_warmstart','mocap_pos','mocap_quat']:
        getattr(data,name)[:]=history[0][name][args.lane:args.lane+1].to(env.device)
      env.sim.forward();data.qacc_warmstart[:]=history[0]['qacc_warmstart'][args.lane:args.lane+1].to(env.device)
      cpu=mujoco.MjData(m)
      for name in ['qpos','qvel','ctrl','qacc_warmstart','mocap_pos','mocap_quat']:
        getattr(cpu,name)[:]=history[0][name][args.lane].numpy()
      mujoco.mj_forward(m,cpu);cpu.qacc_warmstart[:]=history[0]['qacc_warmstart'][args.lane].numpy()
      samples=[]
      for i in range(len(history)-1):
        ctrl=history[i+1]['ctrl'][args.lane]
        data.ctrl[:]=ctrl.to(env.device);cpu.ctrl[:]=ctrl.numpy()
        for substep in range(cfg.env.decimation):
          mujoco.mj_step(m,cpu)
          if corrected and not args.corrected_only:
            with wp.ScopedDevice(env.sim.wp_device):
              corrected_step(env.sim.wp_model,env.sim.wp_data)
          else:
            env.sim.step()
          velocity=data.qvel[0].cpu().numpy().copy()
          position=data.qpos[0].cpu().numpy().copy()
          finite=bool(torch.isfinite(data.qpos).all() and torch.isfinite(data.qvel).all())
          samples.append({'history_index':i,'substep':substep+1,'gpu_finite':finite,
                          'gpu_angular_speed':float(np.linalg.norm(velocity[adr+3:adr+6])) if finite else None,
                          'cpu_angular_speed':float(np.linalg.norm(cpu.qvel[adr+3:adr+6])),
                          'gpu_qvel':velocity.tolist() if finite else None,
                          'cpu_qvel':cpu.qvel.tolist(),
                          'max_position_error':float(np.max(np.abs(position-cpu.qpos))) if finite else None,
                          'cpu_warnings':cpu.warning.number.tolist()})
          if not finite:
            break
        if not finite:
          break
      cases.append({'gyro_correction':corrected,'samples':samples})
    report={'task':manifest['task'],'lane':args.lane,'num_envs':env.num_envs,'training_physics_changed':False,
            'production_cuda_graph':bool(args.corrected_only and env.sim.use_cuda_graph),
            'method':'Restored oldest history state; replay recorded applied controls without policy, native CPU implicitfast comparison.',
            'cases':cases}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for case in cases:
      print('correction',case['gyro_correction'],'final',case['samples'][-1])
  finally:
    env.close()


if __name__=='__main__':
  main()
