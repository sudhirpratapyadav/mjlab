"""Check the production optional stepper against CPU MuJoCo on free spins."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

XML='''<mujoco><worldbody>
<body name="spinner" pos="0 0 1"><freejoint/>
<inertial pos=".01 .02 0" quat=".9238795 0 .3826834 0" mass=".2" diaginertia=".000004 .000039 .000042"/>
<geom type="box" size=".03 .02 .01" contype="0" conaffinity="0"/></body>
<body name="hinge" pos="2 0 1"><joint name="hinge" damping=".1"/>
<geom type="box" size=".1 .03 .03" contype="0" conaffinity="0"/></body>
</worldbody><actuator><position joint="hinge" kp="1" kv=".1"/></actuator></mujoco>'''


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--jacobian',choices=['sparse','dense'],default='sparse')
  args=parser.parse_args()
  if args.output.exists():
    parser.error('Output exists')
  cases=[];worlds=64
  for corrected in [False,True]:
    m=mujoco.MjModel.from_xml_string(XML)
    cfg=SimulationCfg(mujoco=MujocoCfg(timestep=.005,gravity=(0,0,0),jacobian=args.jacobian),
                      free_body_implicitfast_compat=corrected)
    sim=Simulation(worlds,cfg,m,'cuda:0')
    initial=np.tile(m.qpos0,(worlds,1));velocity=np.zeros((worlds,m.nv))
    for i in range(worlds):
      velocity[i,:3]=[.1,-.2,.3]
      velocity[i,3:6]=np.array([.3,-1.,.7])*[0.,1.,10.,100.][i%4]
    sim.data.qpos[:]=torch.tensor(initial,device='cuda:0',dtype=torch.float32)
    sim.data.qvel[:]=torch.tensor(velocity,device='cuda:0',dtype=torch.float32)
    sim.data.ctrl[:]=.3
    sim.reset_solver_state();sim.forward()
    reference=[]
    for i in range(worlds):
      data=mujoco.MjData(m);data.qpos[:]=initial[i];data.qvel[:]=velocity[i];data.ctrl[:]=.3
      mujoco.mj_forward(m,data);reference.append(data)
    samples=[]
    for step in range(100):
      for data in reference:
        mujoco.mj_step(m,data)
      sim.step()
      position=sim.data.qpos.cpu().numpy().copy();vel=sim.data.qvel.cpu().numpy().copy()
      finite=bool(np.isfinite(position).all() and np.isfinite(vel).all())
      cpu_position=np.stack([d.qpos for d in reference]);cpu_velocity=np.stack([d.qvel for d in reference])
      samples.append({'step':step+1,'gpu_finite':finite,
                      'max_position_error':float(np.max(np.abs(position-cpu_position))) if finite else None,
                      'max_velocity_error':float(np.max(np.abs(vel-cpu_velocity))) if finite else None,
                      'hinge_velocity_error':float(np.max(np.abs(vel[:,-1]-cpu_velocity[:,-1]))) if finite else None,
                      'cpu_warning_count':sum(int(d.warning.number.sum()) for d in reference)})
      if not finite:
        break
    cases.append({'correction':corrected,'cuda_graph':sim.use_cuda_graph,'samples':samples})
    if corrected:
      assert len(samples)==100 and all(x['gpu_finite'] and x['cpu_warning_count']==0 for x in samples)
      assert max(x['max_position_error'] for x in samples)<.001
      assert max(x['max_velocity_error'] for x in samples)<.01
      assert max(x['hinge_velocity_error'] for x in samples)<1e-5
  report={'num_envs':worlds,'jacobian':args.jacobian,'steps':100,'mujoco_version':mujoco.__version__,
          'method':'Production Simulation with CUDA graphs;64 free-body spins including zero, low and high angular velocity, off-center/rotated inertia, and an independent actuated hinge.',
          'cases':cases}
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
  print('PASS',args.jacobian,'corrected final',cases[-1]['samples'][-1])


if __name__=='__main__':
  main()
