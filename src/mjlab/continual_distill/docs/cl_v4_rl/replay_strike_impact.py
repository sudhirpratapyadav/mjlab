"""Short native CPU substep witness using recorded GPU states and controls."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import LOWER,UPPER


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  p.add_argument('--gpu',action='store_true',help='Also compare an identically scheduled cold GPU replay')
  a=p.parse_args()
  if a.output.exists():p.error('Output exists')
  ev=json.loads(a.evaluation.read_text());assert ev['task']=='Mjlab-Strike-Slide-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as z:t={k:z[k] for k in z.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  assert abs(m.opt.timestep-.005)<1e-12
  body=m.site_bodyid[m.site('puck/object_site').id];joint=m.body_jntadr[body]
  q,v=int(m.jnt_qposadr[joint]),int(m.jnt_dofadr[joint]);grip=m.site('robot/gripper').id
  robots={i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
  objects={i for i in range(m.ngeom) if m.geom(i).name.startswith('puck/')}
  rows=[]
  for record in ev['records']:
    lane,end=record['env_id'],record['steps'];speed=np.linalg.norm(t['qvel'][:end+1,lane,v:v+2],axis=-1)
    launch=np.flatnonzero(speed>.05);start=max((int(launch[0]) if len(launch) else int(speed.argmax()))-1,0)
    mujoco.mj_resetData(m,d);m.geom_friction[:]=t['initial_model_geom_friction'][lane]
    for f in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,f)[:]=t[f][start,lane]
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=t['origins'][lane]
    d.mocap_pos[:]-=t['origins'][lane];d.ctrl[:]=t['ctrl'][start,lane]
    mujoco.mj_forward(m,d);d.qacc_warmstart[:]=0;samples=[];comparisons=[];finite=True
    for step in range(start+1,min(start+3,end)+1):
      ctrl=t['ctrl'][step,lane]
      assert np.isfinite(ctrl).all() and (ctrl>=np.r_[LOWER[:7],0.]-1e-6).all() and (ctrl<=np.r_[UPPER[:7],.04]+1e-6).all()
      d.ctrl[:]=ctrl
      for sub in range(4):
        mujoco.mj_forward(m,d);jac=np.zeros((3,m.nv));mujoco.mj_jacSite(m,d,jac,None,grip);hand=jac@d.qvel
        contact=any(c.dist<=.001 and ((c.geom[0] in robots and c.geom[1] in objects) or (c.geom[1] in robots and c.geom[0] in objects)) for c in d.contact)
        before=d.qvel[v:v+3].copy();mujoco.mj_step(m,d)
        finite &=bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())
        if not finite:break
        samples.append(dict(step=step,substep=sub,contact_before=contact,hand_velocity_before_m_s=hand.tolist(),puck_velocity_before_m_s=before.tolist(),puck_velocity_after_m_s=d.qvel[v:v+3].tolist(),puck_position_after_m=d.qpos[q:q+3].tolist(),puck_speed_gain_m_s=float(np.linalg.norm(d.qvel[v:v+2])-np.linalg.norm(before[:2]))))
      if not finite:break
      expected=t['qpos'][step,lane,q:q+3]-t['origins'][lane]
      comparisons.append(dict(step=step,position_difference_m=float(np.linalg.norm(d.qpos[q:q+3]-expected)),velocity_difference_m_s=float(np.linalg.norm(d.qvel[v:v+3]-t['qvel'][step,lane,v:v+3]))))
    rows.append(dict(env_id=lane,recorded_success=record['success'],recorded_launched=bool(len(launch)),start_step=start,all_finite=finite,warnings=int(d.warning.number.sum()),samples=samples,comparisons=comparisons))
  endpoints=[r['comparisons'][-1] for r in rows if r['comparisons']]
  summary=dict(episodes=len(rows),all_finite=all(r['all_finite'] for r in rows),warnings=sum(r['warnings'] for r in rows),
    median_final_position_difference_m=float(np.median([x['position_difference_m'] for x in endpoints])),
    median_final_velocity_difference_m_s=float(np.median([x['velocity_difference_m_s'] for x in endpoints])),
    final_position_difference_over5mm=sum(x['position_difference_m']>.005 for x in endpoints),
    final_velocity_difference_over05m_s=sum(x['velocity_difference_m_s']>.05 for x in endpoints))
  if a.gpu:
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=len(rows);cfg.seed=20260914
    assert not any((cfg.sim.free_body_implicitfast_compat,cfg.sim.elliptic_hessian_compat,cfg.sim.primitive_box_box_compat))
    env=ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
      env.reset();origin=env.scene.env_origins.cpu().numpy();lanes=np.array([r['env_id'] for r in rows]);starts=np.array([r['start_step'] for r in rows])
      assert env.sim.mj_model.nq==m.nq and env.sim.mj_model.nv==m.nv and env.physics_dt==.005
      for field in ['qpos','qvel','mocap_pos','mocap_quat','ctrl']:
        values=t[field][starts,lanes].copy()
        if field=='qpos':
          for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
            if kind==mujoco.mjtJoint.mjJNT_FREE:values[:,adr:adr+3]+=origin-t['origins'][lanes]
        if field=='mocap_pos':values+=(origin-t['origins'][lanes])[:,None]
        getattr(env.sim.data,field)[:]=torch.as_tensor(values,device=env.device)
      env.sim.model.geom_friction[:]=torch.as_tensor(t['initial_model_geom_friction'][lanes],device=env.device)
      env.sim.forward();env.sim.data.qacc_warmstart.zero_()
      for row in rows:row['gpu_comparisons']=[]
      for tick in range(12):
        steps=np.array([min(r['start_step']+tick//4+1,ev['records'][i]['steps']) for i,r in enumerate(rows)])
        env.sim.data.ctrl[:]=torch.as_tensor(t['ctrl'][steps,lanes],device=env.device)
        env.sim.forward();env.sim.step()
        assert torch.isfinite(env.sim.data.qpos).all() and torch.isfinite(env.sim.data.qvel).all()
        gp=env.sim.data.qpos[:,q:q+3].cpu().numpy()-origin;gv=env.sim.data.qvel[:,v:v+3].cpu().numpy()
        for i,row in enumerate(rows):
          if tick>=len(row['samples']):continue
          sample=row['samples'][tick]
          if tick%4==3:
            ep=t['qpos'][steps[i],lanes[i],q:q+3]-t['origins'][lanes[i]]
            evv=t['qvel'][steps[i],lanes[i],v:v+3]
            row['gpu_comparisons'].append(dict(step=int(steps[i]),cpu_gpu_position_difference_m=float(np.linalg.norm(gp[i]-sample['puck_position_after_m'])),cpu_gpu_velocity_difference_m_s=float(np.linalg.norm(gv[i]-sample['puck_velocity_after_m_s'])),cold_gpu_original_position_difference_m=float(np.linalg.norm(gp[i]-ep)),cold_gpu_original_velocity_difference_m_s=float(np.linalg.norm(gv[i]-evv))))
      final=[r['gpu_comparisons'][-1] for r in rows if r['gpu_comparisons']]
      summary['matched_gpu']={key:float(np.median([r[key] for r in final])) for key in final[0] if key!='step'}
      summary['matched_gpu'].update(episodes=len(final),all_finite=True,cold_gpu_original_velocity_over05m_s=sum(r['cold_gpu_original_velocity_difference_m_s']>.05 for r in final),cpu_gpu_velocity_over05m_s=sum(r['cpu_gpu_velocity_difference_m_s']>.05 for r in final))
    finally:env.close()
  a.output.write_text(json.dumps(dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],matched_cold_gpu=a.gpu,method='All128cases, including nonlaunch; nativeCPU cold solver replay of up to3recorded20ms intervals after prelaunch state,4substeps each. Optional GPU uses identicalcoldstates/friction/controls and per5ms forward schedule, allbackendflagsOFF. Original states/friction/mocap and bounded controls, no action intervention. Compare with originalGPU coarse endpoints before transferring any inferred impact law. No new policy evaluation/reset data/demonstrations.',summary=summary,rows=rows),indent=2,allow_nan=False)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
