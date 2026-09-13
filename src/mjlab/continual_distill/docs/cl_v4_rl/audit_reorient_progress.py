"""Recorded orientation/contact reward coverage; no dynamics or success substitution."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());assert ev['task']=='Mjlab-Reorient-Object-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:trace={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1;cmd=next(iter(cfg.commands.values()))
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  site=m.site(cmd.asset_name+'/object_site').id;body=m.site_bodyid[site]
  pads=[m.geom('robot/'+name).id for name in ['left_finger_pad','right_finger_pad']]
  objects={i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.asset_name+'/')}
  axis=np.asarray(cmd.body_axis,dtype=float);axis/=np.linalg.norm(axis)
  target=np.asarray(cmd.target_axis,dtype=float);target/=np.linalg.norm(target)
  object_joints=[j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith(cmd.asset_name+'/')]
  assert len(object_joints)==1
  qadr=int(m.jnt_qposadr[object_joints[0]])
  rows=[]
  for record in ev['records']:
    lane,end=record['env_id'],record['steps'];samples=[]
    if 'initial_model_geom_friction' in trace:
      m.geom_friction[:]=trace['initial_model_geom_friction'][lane]
    anchor=trace['qpos'][0,lane,qadr:qadr+3]-trace['origins'][lane]
    for step in sorted(set(range(0,end+1,10))|{end}):
      for field in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,field)[:]=trace[field][step,lane]
      for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
        if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=trace['origins'][lane]
      d.mocap_pos[:]-=trace['origins'][lane]
      mujoco.mj_forward(m,d)
      cosine=float(np.clip((d.xmat[body].reshape(3,3)@axis)@target,-1,1))
      contact=[False,False]
      for c in d.contact:
        if c.dist>.001:continue
        for side,pad in enumerate(pads):
          if c.geom[0]==pad and c.geom[1] in objects:normal=c.frame[:3]
          elif c.geom[1]==pad and c.geom[0] in objects:normal=-c.frame[:3]
          else:continue
          inward=d.geom_xpos[pads[1-side]]-d.geom_xpos[pad];inward/=np.linalg.norm(inward)
          contact[side]|=bool(normal@inward>.5)
      drift=float(np.linalg.norm(d.site_xpos[site,:2]-anchor[:2]));held=all(contact)
      samples.append(dict(step=step,cosine=cosine,angle_deg=float(np.rad2deg(np.arccos(abs(cosine) if cmd.symmetric_axis else cosine))),opposed_contact=held,valid_drift=drift<cmd.max_drift,drift_m=drift,old_orientation_credit=float(max(cosine,0)**2*held*(drift<cmd.max_drift)),height_m=float(d.xpos[body,2])))
    rows.append(dict(env_id=lane,terminal=samples[-1],samples=samples))
  held=[v for r in rows for v in r['samples'] if v['opposed_contact'] and v['valid_drift']]
  summary=dict(episodes=len(rows),sampled_states=sum(len(r['samples']) for r in rows),held_valid_samples=len(held),held_valid_nonpositive_cosine=sum(v['cosine']<=0 for v in held),
    terminal_opposed_contact=sum(r['terminal']['opposed_contact'] for r in rows),terminal_nonpositive_cosine=sum(r['terminal']['cosine']<=0 for r in rows),
    median_terminal_angle_deg=float(np.median([r['terminal']['angle_deg'] for r in rows])),
    episodes_ever_sampled_under45deg=sum(any(v['angle_deg']<45 for v in r['samples']) for r in rows),
    median_old_held_orientation_credit=float(np.median([v['old_orientation_credit'] for v in held])) if held else None)
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],symmetric_axis=cmd.symmetric_axis,body_axis=cmd.body_axis,target_axis=cmd.target_axis,
    method='CPU forward/collision queries every tenth recorded20ms state plus exact terminal. Opposed pad normals use cosine>0.5 and distance<=1mm. Drift anchored to original free-body spawn XY. '+('Original friction restored from trace. ' if 'initial_model_geom_friction' in trace else 'Original friction absent; contact geometry only. ')+'No integration or new native success measurement. Old orientation score is max(dot,0)^2 gated by opposing contact and valid drift.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
