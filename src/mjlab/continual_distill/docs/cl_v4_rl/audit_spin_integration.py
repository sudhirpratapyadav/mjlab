"""CPU integration ablations on captured history; never policy success evidence.

Only these isolated diagnostic models change integrator/timestep. Registered
environments, training recipes and evaluation configurations remain untouched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--failure',type=Path,required=True)
  parser.add_argument('--lane',type=int,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  saved=torch.load(args.failure,map_location='cpu',weights_only=False)
  history=saved['state_history']
  assert len(history)>1
  manifest=json.loads((args.failure.parent/'manifest.json').read_text())
  cfg=load_env_cfg(manifest['task']);cfg.scene.num_envs=1
  model=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(model)
  command=next(iter(cfg.commands.values()));asset=command.asset_name
  joints=[j for j in range(model.njnt) if model.joint(j).name.startswith(asset+'/') and model.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE]
  assert len(joints)==1
  joint=joints[0];adr=int(model.jnt_dofadr[joint]);body=int(model.jnt_bodyid[joint])
  geom_ids={i for i in range(model.ngeom) if model.geom(i).name.startswith(asset+'/')}
  recorded=[{'history_index':i,'linear_speed':float(torch.linalg.vector_norm(state['qvel'][args.lane,adr:adr+3])),
             'angular_speed':float(torch.linalg.vector_norm(state['qvel'][args.lane,adr+3:adr+6]))} for i,state in enumerate(history)]
  cases=[]
  for name,integrator,subdivide in [('native_implicitfast',mujoco.mjtIntegrator.mjINT_IMPLICITFAST,1),
                                  ('implicit_diagnostic',mujoco.mjtIntegrator.mjINT_IMPLICIT,1),
                                  ('rk4_diagnostic',mujoco.mjtIntegrator.mjINT_RK4,1),
                                  ('implicitfast_dt10_reference',mujoco.mjtIntegrator.mjINT_IMPLICITFAST,10)]:
    m=copy.copy(model);m.opt.integrator=integrator;m.opt.timestep/=subdivide
    d=mujoco.MjData(m)
    for field in ['qpos','qvel','ctrl','mocap_pos','mocap_quat']:
      getattr(d,field)[:]=history[0][field][args.lane].numpy()
    mujoco.mj_forward(m,d)
    d.qacc_warmstart[:]=history[0]['qacc_warmstart'][args.lane].numpy()
    matrix=np.empty((m.nv,m.nv));samples=[]
    def sample(index,step):
      mujoco.mj_fullM(m,d,matrix)
      velocity=d.qvel[adr:adr+6]
      return {'history_index':index,'substep':step,'angular_speed':float(np.linalg.norm(velocity[3:])),
              'linear_speed':float(np.linalg.norm(velocity[:3])),
              'object_kinetic_energy':float(.5*velocity@matrix[adr:adr+6,adr:adr+6]@velocity),
              'object_contacts':sum(int(c.geom[0]) in geom_ids or int(c.geom[1]) in geom_ids for c in d.contact),
              'warning_counts':d.warning.number.tolist()}
    samples.append(sample(0,0))
    for i in range(len(history)-1):
      # The following pre-step snapshot contains the applied controls of the
      # preceding control interval. These position targets are constant within it.
      d.ctrl[:]=history[i+1]['ctrl'][args.lane].numpy()
      for step in range(cfg.decimation*subdivide):
        mujoco.mj_step(m,d)
        samples.append(sample(i,step+1))
        if d.warning.number.any():
          break
      if d.warning.number.any():
        break
    cases.append({'name':name,'timestep':m.opt.timestep,'samples':samples})
  report={'task':manifest['task'],'mujoco_version':mujoco.__version__,'source_lane':args.lane,
          'failure_sha256':hashlib.sha256(args.failure.read_bytes()).hexdigest(),
          'method':'CPU replay from oldest captured state, next-state recorded controls; integrator/timestep ablations only on isolated diagnostic models.',
          'limitations':'Not a bitwise GPU replay; derived solver and actuator histories were not captured. Contact trajectories can diverge across integrators.',
          'training_physics_changed':False,'body':model.body(body).name,
          'body_inertia':model.body_inertia[body].tolist(),
          'children':[model.body(i).name for i in range(model.nbody) if model.body_parentid[i]==body],
          'recorded_history':recorded,'cases':cases}
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  for case in cases:
    print(case['name'],'initial',case['samples'][0],'final',case['samples'][-1])


if __name__=='__main__':
  main()
