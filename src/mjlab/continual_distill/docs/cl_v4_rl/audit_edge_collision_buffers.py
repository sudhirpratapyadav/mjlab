"""Compare CPU/GPU contact buffers at identical recorded diagnostic states."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from contact_grasp import opposed_grasp


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--controls',type=Path,required=True)
  p.add_argument('--replay',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  with np.load(args.controls) as z:a={k:z[k] for k in z.files}
  with np.load(args.replay) as z:t={k:z[k] for k in z.files}
  assert str(a['task'])=='Mjlab-Edge-Grasp-Franka' and np.array_equal(a['lanes'],t['lanes'])
  cases=[(source,seconds,lane) for source in ['cpu','gpu'] for seconds in [1.5,2.] for lane in range(len(a['lanes']))]
  cfg=load_env_cfg(str(a['task']));cfg.scene.num_envs=len(cases);cfg.seed=20260914
  assert not any((cfg.sim.free_body_implicitfast_compat,cfg.sim.elliptic_hessian_compat,cfg.sim.primitive_box_box_compat))
  env=ManagerBasedRlEnv(cfg,device='cuda:0')
  try:
    env.reset();m=env.sim.mj_model;origin=env.scene.env_origins.cpu().numpy()
    assert abs(env.physics_dt-float(a['timestep']))<1e-12
    fields=['qpos','qvel','mocap_pos','mocap_quat','ctrl'];local=[];frictions=[]
    for source,seconds,lane in cases:
      frame=round(seconds/(4*env.physics_dt))-1;tick=round(seconds/env.physics_dt)-1
      state={f:a[f][lane].copy() for f in fields}
      state['qpos']=t[source+'_qpos'][frame,lane].copy()
      state['qvel']=t[source+'_qvel'][frame,lane].copy()
      state['ctrl']=a['controls'][lane,tick].copy()
      if source=='gpu':
        for address,kind in zip(m.jnt_qposadr,m.jnt_type):
          if kind==mujoco.mjtJoint.mjJNT_FREE:state['qpos'][address:address+3]-=t['env_origins'][lane]
      local.append(state);frictions.append(a['geom_friction'][lane])
    placed={f:np.stack([r[f] for r in local]) for f in fields}
    for address,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:placed['qpos'][:,address:address+3]+=origin
    placed['mocap_pos']+=origin[:,None]
    for f in fields:getattr(env.sim.data,f)[:]=torch.as_tensor(placed[f],device=env.device)
    env.sim.model.geom_friction[:]=torch.as_tensor(np.asarray(frictions),device=env.device)
    env.sim.data.qacc_warmstart.zero_();env.sim.forward();env.scene.update(dt=env.physics_dt)
    command=env.command_manager.get_term(next(iter(cfg.commands)))
    gpu_opposed=opposed_grasp(command).cpu().tolist()
    contact=env.sim.data.contact;count=int(env.sim.data.nacon[0])
    gpu={f:getattr(contact,f)[:count].cpu().numpy() for f in ['geom','worldid','dist','pos','frame']}
    pads=[m.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
    objects={i for i in range(m.ngeom) if m.geom(i).name.startswith('plate/')}
    robots={i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
    def pair(x,y):return (x in objects and y in robots) or (y in objects and x in robots)
    rows=[]
    for index,(source,seconds,lane) in enumerate(cases):
      model=copy.copy(m);model.geom_friction[:]=frictions[index];d=mujoco.MjData(model)
      for f in fields:getattr(d,f)[:]=local[index][f]
      mujoco.mj_forward(model,d);cpu_contacts=[];held=[False,False]
      for c in d.contact:
        x,y=c.geom
        if pair(x,y):cpu_contacts.append(dict(geoms=[m.geom(x).name,m.geom(y).name],distance_m=float(c.dist),position_m=c.pos.tolist(),normal=c.frame[:3].tolist()))
        if c.dist>.001:continue
        for side,pad in enumerate(pads):
          if x==pad and y in objects:normal=c.frame[:3]
          elif y==pad and x in objects:normal=-c.frame[:3]
          else:continue
          inward=d.geom_xpos[pads[1-side]]-d.geom_xpos[pad];inward/=np.linalg.norm(inward)
          held[side]|=bool(normal@inward>.5)
      gpu_contacts=[]
      for j in np.flatnonzero(gpu['worldid']==index):
        x,y=gpu['geom'][j]
        if pair(x,y):gpu_contacts.append(dict(geoms=[m.geom(int(x)).name,m.geom(int(y)).name],distance_m=float(gpu['dist'][j]),position_m=(gpu['pos'][j]-origin[index]).tolist(),normal=gpu['frame'][j,0].tolist()))
      rows.append(dict(env_id=int(a['lanes'][lane]),state_source=source,time_s=seconds,cpu_opposed=all(held),gpu_opposed=gpu_opposed[index],cpu_contacts=cpu_contacts,gpu_contacts=gpu_contacts))
    summary={}
    for source in ['cpu','gpu']:
      selected=[r for r in rows if r['state_source']==source]
      summary[source+'_states']=dict(cases=len(selected),cpu_opposed=sum(r['cpu_opposed'] for r in selected),gpu_opposed=sum(r['gpu_opposed'] for r in selected),label_agreement=sum(r['cpu_opposed']==r['gpu_opposed'] for r in selected))
    report=dict(task=str(a['task']),checkpoint_sha256=str(a['checkpoint_sha256']),source_controls=str(args.controls.resolve()),source_replay=str(args.replay.resolve()),
      method='All32 diagnostic cases at1.5/2s, both CPU/GPU-generated qpos/qvel states. Identical local states, original friction/mocap/controls; CPU/GPU forward queries only, no integration. Native model and all compatibility flagsOFF. Opposed grasp is training geometry, not native task success. No policy actions, demonstrations or reset data.',summary=summary,rows=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary))
  finally:env.close()


if __name__=='__main__':main()
