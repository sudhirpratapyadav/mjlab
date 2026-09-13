"""Recorded Stack capture, support contact and clearance progress; not a success gate."""
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
  ev=json.loads(args.evaluation.read_text());assert ev['task']=='Mjlab-Stack-Cube-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1;cmd=next(iter(cfg.commands.values()))
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  site=m.site(cmd.asset_name+'/object_site').id;body=m.site_bodyid[site]
  base_site=m.site(cmd.base_asset_name+'/object_site').id;base=m.site_bodyid[base_site]
  objects={i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.asset_name+'/')}
  supports={i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.base_asset_name+'/')}
  robots={i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
  pads=[m.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
  rows=[]
  for rec in ev['records']:
    lane,end=rec['env_id'],rec['steps'];samples=[]
    for step in sorted(set(range(0,end+1,10))|{end}):
      for field in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,field)[:]=t[field][step,lane]
      for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
        if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=t['origins'][lane]
      d.mocap_pos[:]-=t['origins'][lane];mujoco.mj_forward(m,d)
      goal=d.xpos[base]+d.xmat[base].reshape(3,3)@np.array([0,0,cmd.stack_height])
      delta=d.xpos[body]-goal;held=[False,False];object_support=False;robot_support=False
      for c in d.contact:
        if c.dist>.001:continue
        a,b=c.geom
        object_support|=(a in objects and b in supports) or (b in objects and a in supports)
        robot_support|=(a in robots and b in supports) or (b in robots and a in supports)
        for i,pad in enumerate(pads):
          if a==pad and b in objects:normal=c.frame[:3]
          elif b==pad and a in objects:normal=-c.frame[:3]
          else:continue
          inward=d.geom_xpos[pads[1-i]]-d.geom_xpos[pad];inward/=np.linalg.norm(inward)
          held[i]|=bool(normal@inward>.5)
      samples.append(dict(step=step,object_height_m=float(d.xpos[body,2]),target_height_m=float(goal[2]),height_below_target_m=float(-delta[2]),xy_error_m=float(np.linalg.norm(delta[:2])),opposed_contact=all(held),object_base_contact=bool(object_support),robot_base_contact=bool(robot_support)))
    rows.append(dict(env_id=lane,terminal=samples[-1],max_sampled_height_m=max(v['object_height_m'] for v in samples),
      min_sampled_xy_error_m=min(v['xy_error_m'] for v in samples),ever_opposed_contact=any(v['opposed_contact'] for v in samples),
      ever_above_stack_height=any(v['object_height_m']>v['target_height_m'] for v in samples),
      held_below_target_samples=sum(v['opposed_contact'] and v['height_below_target_m']>.02 for v in samples),
      held_object_base_contact_samples=sum(v['opposed_contact'] and v['object_base_contact'] for v in samples)))
  summary=dict(episodes=len(rows),ever_opposed_contact=sum(r['ever_opposed_contact'] for r in rows),ever_above_stack_height=sum(r['ever_above_stack_height'] for r in rows),
    terminal_opposed_contact=sum(r['terminal']['opposed_contact'] for r in rows),terminal_object_base_contact=sum(r['terminal']['object_base_contact'] for r in rows),terminal_robot_base_contact=sum(r['terminal']['robot_base_contact'] for r in rows),
    median_terminal_height_m=float(np.median([r['terminal']['object_height_m'] for r in rows])),median_terminal_height_below_target_m=float(np.median([r['terminal']['height_below_target_m'] for r in rows])),median_terminal_xy_error_m=float(np.median([r['terminal']['xy_error_m'] for r in rows])),
    held_below_target_samples=sum(r['held_below_target_samples'] for r in rows),held_object_base_contact_samples=sum(r['held_object_base_contact_samples'] for r in rows))
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],method='CPU FK/contact queries every tenth recorded20ms state plus exact terminal. Stack target reconstructed from current base root pose and native stack_height. Contact distance<=1mm; opposed pads additionally require inward normal cosine>0.5. Original friction absent, so geometry only; no integration, intervention or success substitution.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
