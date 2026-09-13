"""Reconstruct container endpoint subconditions from recorded states for diagnosis."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.asset_zoo.objects.goal import object_support_points


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  ev=json.loads(args.evaluation.read_text());assert ev['task'] in ['Mjlab-Place-In-Container-Franka','Mjlab-Throw-To-Bin-Franka']
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:t={k:a[k] for k in a.files}
  cfg=load_env_cfg(ev['task']);cfg.scene.num_envs=1;cmd=next(iter(cfg.commands.values()))
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  site=m.site(cmd.asset_name+'/object_site').id;body=m.site_bodyid[site]
  container=m.body(cmd.container_asset_name+'/container_base').id
  support=object_support_points(cfg.scene.entities[cmd.asset_name].spec_fn())
  objects={i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.asset_name+'/')}
  containers={i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.container_asset_name+'/')}
  robots={i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
  joint=next(j for j in range(m.njnt) if m.jnt_type[j]==mujoco.mjtJoint.mjJNT_FREE and m.joint(j).name.startswith(cmd.asset_name+'/'))
  vadr=int(m.jnt_dofadr[joint]);rows=[]
  for rec in ev['records']:
    lane,end=rec['env_id'],rec['steps']
    for field in ['qpos','qvel','mocap_pos','mocap_quat']:getattr(d,field)[:]=t[field][end,lane]
    for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=t['origins'][lane]
    d.mocap_pos[:]-=t['origins'][lane]
    if 'initial_model_geom_friction' in t:m.geom_friction[:]=t['initial_model_geom_friction'][lane]
    mujoco.mj_forward(m,d)
    corners=support@d.xmat[body].reshape(3,3).T+d.site_xpos[site]
    local=(corners-d.xpos[container])@d.xmat[container].reshape(3,3)
    robot_contact=False;container_contact=False
    for c in d.contact:
      if c.dist>.001:continue
      a,b=c.geom
      robot_contact|=(a in objects and b in robots) or (b in objects and a in robots)
      container_contact|=(a in objects and b in containers) or (b in objects and a in containers)
    lin=float(np.linalg.norm(d.qvel[vadr:vadr+3]));ang=float(np.linalg.norm(d.qvel[vadr+3:vadr+6]))
    flags=dict(inside_xy=bool((np.abs(local[:,:2])<cmd.inner_half_width+.001).all()),below_rim=bool(local[:,2].max()<cmd.inner_rim_z),above_floor=bool(local[:,2].min()>cmd.inner_floor_z-.002),linear_settled=lin<cmd.settle_speed,angular_settled=ang<.3,released=not bool(robot_contact),container_contact=bool(container_contact))
    rows.append(dict(env_id=lane,native_success=rec['success'],reconstructed_success=all(flags.values()),conditions=flags,linear_speed_m_s=lin,angular_speed_rad_s=ang,min_corner_height_local_m=float(local[:,2].min()),max_corner_xy_extent_m=float(np.abs(local[:,:2]).max())))
  failures=[r for r in rows if not r['native_success']]
  summary=dict(episodes=len(rows),native_successes=ev['successes'],reconstructed_successes=sum(r['reconstructed_success'] for r in rows),disagreements=sum(r['native_success']!=r['reconstructed_success'] for r in rows),failed_native_episodes=len(failures),failure_condition_counts={k:sum(not r['conditions'][k] for r in failures) for k in rows[0]['conditions']})
  report=dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],method='Exact recorded terminal poses/velocities with CPU FK/collision and native support corners/container dimensions. Contact distance<=1mm; native linear settle threshold and angular0.3. Reconstructed subconditions are diagnostic only; disagreement with authoritative GPU evaluation is reported, never reclassified. No integration or new success measurement.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
