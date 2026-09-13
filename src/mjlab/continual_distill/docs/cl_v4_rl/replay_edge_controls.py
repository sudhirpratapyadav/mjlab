"""Matched native CPU/GPU Edge control replay, separate from RL evaluation."""
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
from contact_grasp import opposed_grasp


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--controls', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--elliptic-hessian', action='store_true')
  parser.add_argument('--gyro', action='store_true')
  parser.add_argument('--primitive-box-box', action='store_true')
  args = parser.parse_args()
  archive = Path(__file__).resolve().parent/'runs'/args.output.with_suffix('.npz').name
  if args.output.exists() or archive.exists():
    parser.error('Output or sample archive exists')
  with np.load(args.controls) as source:
    a = {k: source[k] for k in source.files}
  assert str(a['task']) == 'Mjlab-Edge-Grasp-Franka'
  controls = a['controls'].astype(np.float32); n, ticks, nu = controls.shape
  assert nu == 8 and np.isfinite(controls).all()
  assert np.all(controls >= np.r_[LOWER[:7],0.]-1e-6)
  assert np.all(controls <= np.r_[UPPER[:7],.04]+1e-6)
  cfg = load_env_cfg(str(a['task'])); cfg.scene.num_envs = n; cfg.seed = 20260914
  assert not any((cfg.sim.free_body_implicitfast_compat,cfg.sim.elliptic_hessian_compat,cfg.sim.primitive_box_box_compat))
  cfg.sim.free_body_implicitfast_compat = args.gyro
  cfg.sim.elliptic_hessian_compat = args.elliptic_hessian
  cfg.sim.primitive_box_box_compat = args.primitive_box_box
  env = ManagerBasedRlEnv(cfg,device='cuda:0')
  try:
    env.reset(); model = env.sim.mj_model
    assert abs(env.physics_dt-float(a['timestep'])) < 1e-12
    assert {e.params.get('field') for e in cfg.events.values() if e.domain_randomization} <= {'geom_friction'}
    assert model.nu == 8 and all(model.actuator_trnid[i,0] == model.joint(f'robot/joint{i+1}').id for i in range(7))
    origin = env.scene.env_origins.cpu().numpy()
    fields = ['qpos','qvel','mocap_pos','mocap_quat','ctrl']
    placed = {f: a[f].copy() for f in fields}
    for address,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        placed['qpos'][:,address:address+3] += origin
    placed['mocap_pos'] += origin[:,None]
    for f in fields:
      getattr(env.sim.data,f)[:] = torch.as_tensor(placed[f],device=env.device)
    env.sim.model.geom_friction[:] = torch.as_tensor(a['geom_friction'],device=env.device)
    env.sim.forward(); env.sim.data.qacc_warmstart.zero_(); env.scene.update(dt=env.physics_dt)
    models = [copy.copy(model) for _ in range(n)]; cpus = []
    for i,m in enumerate(models):
      m.geom_friction[:] = a['geom_friction'][i]; d = mujoco.MjData(m)
      for f in fields:
        getattr(d,f)[:] = a[f][i]
      mujoco.mj_forward(m,d); d.qacc_warmstart[:] = 0; cpus.append(d)
    command = env.command_manager.get_term(next(iter(cfg.commands)))
    body = model.body('plate/plate').id
    pads = [model.geom('robot/'+v).id for v in ['left_finger_pad','right_finger_pad']]
    objects = {i for i in range(model.ngeom) if model.geom(i).name.startswith('plate/')}
    initial_z = np.asarray([d.xpos[body,2] for d in cpus])
    history = []; states = []; finite = True
    for tick in range(ticks):
      ctrl = controls[:,tick]
      env.sim.data.ctrl[:] = torch.as_tensor(ctrl,device=env.device)
      env.sim.step(); env.scene.update(dt=env.physics_dt)
      for i,d in enumerate(cpus):
        d.ctrl[:] = ctrl[i]; mujoco.mj_step(models[i],d)
      finite &= bool(torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
                     and all(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all() for d in cpus))
      if not finite:
        break
      if tick%4 == 3:
        env.sim.forward(); env.scene.update(dt=env.physics_dt)
        gpu_pos = command.object.data.root_link_pos_w.cpu().numpy()-origin
        gpu_held = opposed_grasp(command).cpu().numpy()
        cpu_pos = []; cpu_held = []
        for i,d in enumerate(cpus):
          mujoco.mj_forward(models[i],d); held = [False,False]
          for c in d.contact:
            if c.dist > .001:
              continue
            x,y = c.geom
            for side,pad in enumerate(pads):
              if x == pad and y in objects:
                normal = c.frame[:3]
              elif y == pad and x in objects:
                normal = -c.frame[:3]
              else:
                continue
              inward = d.geom_xpos[pads[1-side]]-d.geom_xpos[pad]
              inward /= np.linalg.norm(inward)
              held[side] |= bool(normal@inward > .5)
          cpu_pos.append(d.xpos[body].copy()); cpu_held.append(all(held))
        cpu_pos = np.asarray(cpu_pos)
        history.append(np.column_stack([gpu_pos[:,2]-initial_z,cpu_pos[:,2]-initial_z,gpu_held,cpu_held,gpu_pos,cpu_pos]))
        states.append((env.sim.data.qpos.cpu().numpy().copy(),env.sim.data.qvel.cpu().numpy().copy(),
                       np.stack([d.qpos.copy() for d in cpus]),np.stack([d.qvel.copy() for d in cpus])))
    h = np.asarray(history)
    assert len(h), 'No finite trajectory samples'
    np.savez_compressed(archive,samples=h,lanes=a['lanes'],env_origins=origin,
      **{name:np.stack([v[i] for v in states]) for i,name in enumerate(['gpu_qpos','gpu_qvel','cpu_qpos','cpu_qvel'])})
    rows = [dict(env_id=int(a['lanes'][i]),gpu_held_lift=bool(((h[:,i,0]>.02)&(h[:,i,2]>0)).any()),
      cpu_held_lift=bool(((h[:,i,1]>.02)&(h[:,i,3]>0)).any()),gpu_ever_opposed=bool(h[:,i,2].any()),
      cpu_ever_opposed=bool(h[:,i,3].any()),terminal_position_difference_m=float(np.linalg.norm(h[-1,i,4:7]-h[-1,i,7:10]))) for i in range(n)]
    summary = dict(episodes=n,completed_ticks=tick+1,requested_ticks=ticks,all_finite=finite,
      gpu_held_lift_count=sum(r['gpu_held_lift'] for r in rows),cpu_held_lift_count=sum(r['cpu_held_lift'] for r in rows),
      gpu_ever_opposed=sum(r['gpu_ever_opposed'] for r in rows),cpu_ever_opposed=sum(r['cpu_ever_opposed'] for r in rows),
      median_terminal_position_difference_m=float(np.median([r['terminal_position_difference_m'] for r in rows])),
      cpu_warning_count=sum(int(d.warning.number.sum()) for d in cpus))
    report = dict(task=str(a['task']),checkpoint_sha256=str(a['checkpoint_sha256']),control_archive=str(args.controls.resolve()),
      control_archive_sha256=hashlib.sha256(args.controls.read_bytes()).hexdigest(),sample_archive=str(archive),
      sample_columns=['gpu_lift_m','cpu_lift_m','gpu_opposed','cpu_opposed','gpu_x','gpu_y','gpu_z','cpu_x','cpu_y','cpu_z'],
      free_body_implicitfast_compat=args.gyro,elliptic_hessian_compat=args.elliptic_hessian,primitive_box_box_compat=args.primitive_box_box,
      method='All32 original diagnostic states/friction/recorded wrist and identical float32 physical8D commands. Native CPU versus GPU, full3.5s path, forward/contact samples every20ms. Initial cold caches matched. Held lift is a diagnostic mechanism, never task success, demonstrations, policy actions or reset training data.',
      summary=summary,rows=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(summary))
  finally:
    env.close()


if __name__ == '__main__':
  main()
