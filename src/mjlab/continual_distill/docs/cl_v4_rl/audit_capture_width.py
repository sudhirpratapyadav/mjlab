"""Explain capture-width shaping at recorded poses; no integration or RL claims."""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.asset_zoo.objects.goal import object_support_points
from contact_grasp import box_ray_width


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  args=parser.parse_args()
  e=json.loads(args.evaluation.read_text())
  cfg=load_env_cfg(e['task']);cfg.scene.num_envs=1
  model=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(model)
  data=mujoco.MjData(model)
  command=next(iter(cfg.commands.values()));asset=command.asset_name
  site=model.site(asset+'/object_site').id;body=model.site_bodyid[site]
  grip=model.site('robot/gripper').id
  pads=[model.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
  support=object_support_points(cfg.scene.entities[asset].spec_fn())
  lower,upper=torch.tensor(support.min(0)),torch.tensor(support.max(0))
  with np.load(Path(e['trace_dir'])/'trace.npz') as archive:
    trace={k:archive[k] for k in archive.files}
  rows=[]
  for record in e['records']:
    i,n=record['env_id'],record['steps']
    for field in ['qpos','qvel','mocap_pos','mocap_quat']:
      getattr(data,field)[:]=trace[field][n,i]
    for adr,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:data.qpos[adr:adr+3]-=trace['origins'][i]
    data.mocap_pos[:]-=trace['origins'][i]
    mujoco.mj_kinematics(model,data)
    pos=data.site_xpos[site];rot=data.xmat[body].reshape(3,3)
    world=support@rot.T+pos
    target=pos.copy()
    if getattr(command,'insertion',False):target[2]=world[:,2].min()+.70*np.ptp(world[:,2])
    else:target[2]+=.006
    delta=data.geom_xpos[pads[1]]-data.geom_xpos[pads[0]]
    direction=delta/np.linalg.norm(delta)
    origin=(data.geom_xpos[pads].mean(0)-pos)@rot
    local_direction=torch.tensor(direction@rot)
    width,valid=box_ray_width(torch.tensor(origin),local_direction,lower,upper)
    central,central_valid=box_ray_width((lower+upper)/2,local_direction,lower,upper)
    assert central_valid
    projected=float(np.ptp(world@direction))
    rows.append(dict(env_id=i,ray_intersects=bool(valid),ray_width_m=float(width),
      old_effective_width_m=float(width) if valid else projected,central_ray_width_m=float(central),
      projected_width_m=projected,inner_pad_gap_m=float(np.linalg.norm(delta)-.0152),
      target_error_m=float(np.linalg.norm(data.site_xpos[grip]-target)),
      target_delta_m=(data.site_xpos[grip]-target).tolist(),pad_center_local_m=origin.tolist()))
  summary=dict(poses=len(rows),ray_intersections=sum(r['ray_intersects'] for r in rows))
  for key in ['old_effective_width_m','central_ray_width_m','projected_width_m','inner_pad_gap_m','target_error_m']:
    summary['median_'+key]=float(np.median([r[key] for r in rows]))
  report=dict(task=e['task'],checkpoint_sha256=e['checkpoint_sha256'],method='CPU FK at128 recorded first-episode terminal poses. Reproduce current pad-centerline width and full-projection miss fallback; compare a central box chord as a candidate approach-width estimate. No integration, changed actions or success reclassification.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
