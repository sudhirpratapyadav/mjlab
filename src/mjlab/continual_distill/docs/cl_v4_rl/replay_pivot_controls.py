"""Matched CPU/GPU replay of exported diagnostic controls, never policy data."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import LOWER, UPPER
from mjlab.tasks.manipulation.mdp.task_geometry import touching
from mjlab.utils.lab_api.math import quat_apply


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--controls',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--elliptic-hessian',action='store_true')
  p.add_argument('--primitive-box-box',action='store_true')
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  with np.load(args.controls) as z:a={k:z[k] for k in z.files}
  controls=a['controls'].astype(np.float32);n,ticks,nu=controls.shape
  assert nu==8 and np.isfinite(controls).all()
  assert (controls>=np.asarray(LOWER[:8])-1e-6).all() and (controls<=np.asarray(UPPER[:8])+1e-6).all()
  cfg=load_env_cfg('Mjlab-Pivot-Lift-Franka');cfg.scene.num_envs=n;cfg.seed=20260914
  assert not any((cfg.sim.free_body_implicitfast_compat,cfg.sim.elliptic_hessian_compat,cfg.sim.primitive_box_box_compat))
  cfg.sim.elliptic_hessian_compat=args.elliptic_hessian
  cfg.sim.primitive_box_box_compat=args.primitive_box_box
  env=ManagerBasedRlEnv(cfg,device='cuda:0')
  try:
    env.reset();m=env.sim.mj_model
    assert abs(env.physics_dt-float(a['timestep']))<1e-12
    assert {e.params.get('field') for e in cfg.events.values() if e.domain_randomization}<={'geom_friction'}
    assert m.nu==8 and all(m.actuator_trnid[i,0]==m.joint(f'robot/joint{i+1}').id for i in range(7))
    origin=env.scene.env_origins.cpu().numpy();fields=['qpos','qvel','mocap_pos','mocap_quat']
    placed={f:a[f].copy() for f in fields}
    for address,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:placed['qpos'][:,address:address+3]+=origin
    placed['mocap_pos']+=origin[:,None]
    for f in fields:getattr(env.sim.data,f)[:]=torch.as_tensor(placed[f],device=env.device)
    env.sim.model.geom_friction[:]=torch.as_tensor(a['geom_friction'],device=env.device)
    env.sim.data.ctrl.zero_();env.sim.forward();env.sim.data.qacc_warmstart.zero_();env.scene.update(dt=env.physics_dt)
    models=[copy.copy(m) for _ in range(n)];cpus=[]
    for i,model in enumerate(models):
      model.geom_friction[:]=a['geom_friction'][i];d=mujoco.MjData(model)
      for f in fields:getattr(d,f)[:]=a[f][i]
      mujoco.mj_forward(model,d);d.qacc_warmstart[:]=0;cpus.append(d)
    command=env.command_manager.get_term(next(iter(cfg.commands)))
    board=m.body('board/board').id
    objects={i for i in range(m.ngeom) if m.geom(i).name.startswith('board/')}
    walls={i for i in range(m.ngeom) if m.geom(i).name.startswith('wall/')}
    z=torch.tensor([0.,0.,1.],device=env.device).expand(n,3)
    history=[];states=[];finite=True
    for tick in range(ticks):
      ctrl=controls[:,tick]
      env.sim.data.ctrl[:]=torch.as_tensor(ctrl,device=env.device)
      env.sim.step();env.scene.update(dt=env.physics_dt)
      for i,d in enumerate(cpus):d.ctrl[:]=ctrl[i];mujoco.mj_step(models[i],d)
      if tick%4==3:
        env.sim.forward();env.scene.update(dt=env.physics_dt)
        axis=quat_apply(command.object.data.root_link_quat_w,z)
        gpu_angle=torch.rad2deg(torch.acos(axis[:,2].abs().clamp(0,1))).cpu().numpy()
        gpu_wall=touching(command,command.object,command.wall).cpu().numpy()
        gpu_pos=command.object.data.root_link_pos_w.cpu().numpy()-origin
        cpu_angle=[];cpu_wall=[];cpu_pos=[]
        for i,d in enumerate(cpus):
          mujoco.mj_forward(models[i],d)
          cpu_angle.append(np.rad2deg(np.arccos(np.clip(abs(d.xmat[board].reshape(3,3)[2,2]),0,1))))
          cpu_wall.append(any(c.dist<=.001 and ((c.geom[0] in objects and c.geom[1] in walls) or (c.geom[1] in objects and c.geom[0] in walls)) for c in d.contact))
          cpu_pos.append(d.xpos[board].copy())
        finite=bool(torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all() and all(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() for d in cpus))
        states.append((env.sim.data.qpos.cpu().numpy().copy(),env.sim.data.qvel.cpu().numpy().copy(),np.stack([d.qpos.copy() for d in cpus]),np.stack([d.qvel.copy() for d in cpus])))
        history.append(np.column_stack([gpu_angle,cpu_angle,gpu_wall,cpu_wall,gpu_pos,np.asarray(cpu_pos)]))
        if not finite:break
    h=np.asarray(history);archive=args.output.with_suffix('.npz')
    # Keep raw samples outside the compact experiment records directory.
    archive=Path(__file__).resolve().parent/'runs'/archive.name
    np.savez_compressed(archive,samples=h,lanes=a['lanes'],gpu_qpos=np.stack([v[0] for v in states]),gpu_qvel=np.stack([v[1] for v in states]),cpu_qpos=np.stack([v[2] for v in states]),cpu_qvel=np.stack([v[3] for v in states]),env_origins=origin)
    rows=[dict(env_id=int(a['lanes'][i]),gpu_pivot=bool(((h[:,i,0]>20)&(h[:,i,2]>0)).any()),
      cpu_pivot=bool(((h[:,i,1]>20)&(h[:,i,3]>0)).any()),gpu_max_tilt_deg=float(h[:,i,0].max()),
      cpu_max_tilt_deg=float(h[:,i,1].max()),terminal_board_position_difference_m=float(np.linalg.norm(h[-1,i,4:7]-h[-1,i,7:10]))) for i in range(n)]
    summary=dict(episodes=n,all_finite=finite,gpu_pivot_count=sum(r['gpu_pivot'] for r in rows),cpu_pivot_count=sum(r['cpu_pivot'] for r in rows),
      median_terminal_board_difference_m=float(np.median([r['terminal_board_position_difference_m'] for r in rows])))
    report=dict(task='Mjlab-Pivot-Lift-Franka',checkpoint_sha256=str(a['checkpoint_sha256']),control_archive=str(args.controls.resolve()),
      control_archive_sha256=hashlib.sha256(args.controls.read_bytes()).hexdigest(),sample_archive=str(archive),
      sample_columns=['gpu_tilt_deg','cpu_tilt_deg','gpu_wall','cpu_wall','gpu_x','gpu_y','gpu_z','cpu_x','cpu_y','cpu_z'],
      elliptic_hessian_compat=cfg.sim.elliptic_hessian_compat,primitive_box_box_compat=cfg.sim.primitive_box_box_compat,free_body_implicitfast_compat=cfg.sim.free_body_implicitfast_compat,
      method='All exported original states/friction, same float32-quantized physical commands on native CPU and GPU; sampled every20ms. Actual opt-in compatibility flags recorded separately; native physical model unchanged. Positive20deg+wall contact is a diagnostic mechanism, not full task success or RL evidence. No demonstrations, reset data or physical model changes.',summary=summary,rows=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))
  finally:env.close()


if __name__=='__main__':main()
