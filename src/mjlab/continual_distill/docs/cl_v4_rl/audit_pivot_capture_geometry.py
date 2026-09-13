"""Check pad/board geometry after the recorded bounded diagnostic tipping path."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--controls',type=Path,required=True)
  p.add_argument('--replay',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  args=p.parse_args()
  if args.output.exists():p.error('Output exists')
  with np.load(args.controls) as z:a={k:z[k] for k in z.files}
  with np.load(args.replay) as z:t={k:z[k] for k in z.files}
  assert np.array_equal(a['lanes'],t['lanes'])
  cfg=load_env_cfg('Mjlab-Pivot-Lift-Franka');cfg.scene.num_envs=1
  m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m);d=mujoco.MjData(m)
  geom=m.geom('board/board_geom').id;body=m.geom_bodyid[geom]
  assert m.geom_type[geom]==mujoco.mjtGeom.mjGEOM_BOX
  half=m.geom_size[geom].copy();pads=[m.geom('robot/'+name).id for name in ['left_finger_pad','right_finger_pad']]
  fingers=[m.jnt_qposadr[m.joint('robot/'+name).id] for name in ['finger_joint1','finger_joint2']]
  rows=[]
  for lane in range(len(a['lanes'])):
    mujoco.mj_resetData(m,d);m.geom_friction[:]=a['geom_friction'][lane]
    for f in ['mocap_pos','mocap_quat']:getattr(d,f)[:]=a[f][lane]
    d.qpos[:]=t['cpu_qpos'][-1,lane];d.qvel[:]=t['cpu_qvel'][-1,lane]
    d.ctrl[:]=a['controls'][lane,-1];mujoco.mj_forward(m,d)
    rotation=d.geom_xmat[geom].reshape(3,3);positions=d.geom_xpos[pads].copy()
    direction=positions[1]-positions[0];direction/=np.linalg.norm(direction)
    origin=rotation.T@(positions.mean(0)-d.geom_xpos[geom]);axis=rotation.T@direction
    entry,exit=-np.inf,np.inf;valid=True
    for j in range(3):
      if abs(axis[j])<1e-8:
        valid &=abs(origin[j])<=half[j]
      else:
        near,far=sorted([(-half[j]-origin[j])/axis[j],(half[j]-origin[j])/axis[j]])
        entry=max(entry,near);exit=min(exit,far)
    valid &= exit>=entry
    rows.append(dict(env_id=int(a['lanes'][lane]),board_tilt_deg=float(np.rad2deg(np.arccos(np.clip(abs(rotation[2,2]),0,1)))),
      closing_axis_normal_angle_deg=float(np.rad2deg(np.arccos(np.clip(abs(axis[2]),0,1)))),
      pad_midpoint_board_local_m=origin.tolist(),closing_axis_board_local=axis.tolist(),
      centerline_intersects=bool(valid),centerline_width_m=float(exit-entry) if valid else None,
      centerline_entry_m=float(entry) if valid else None,centerline_exit_m=float(exit) if valid else None,
      lower_pad_index=int(np.argmin(positions[:,2])),pad_positions_m=positions.tolist(),finger_qpos_m=d.qpos[fingers].tolist(),
      pad_inner_gap_m=float(np.linalg.norm(positions[1]-positions[0])-.0152)))
  summary=dict(cases=len(rows),centerline_intersects=sum(r['centerline_intersects'] for r in rows),
    median_tilt_deg=float(np.median([r['board_tilt_deg'] for r in rows])),
    median_axis_normal_angle_deg=float(np.median([r['closing_axis_normal_angle_deg'] for r in rows])),
    median_centerline_width_m=(float(np.median([r['centerline_width_m'] for r in rows if r['centerline_intersects']])) if any(r['centerline_intersects'] for r in rows) else None))
  report=dict(task='Mjlab-Pivot-Lift-Franka',checkpoint_sha256=str(a['checkpoint_sha256']),source_controls=str(args.controls.resolve()),source_replay=str(args.replay.resolve()),
    method='All32 native CPU terminal states of the bounded5s diagnostic path. Original friction/mocap/controls; FK and exact box/centerline slab intersection only. Centerline intersection is a geometric diagnostic, not contact, force closure, dynamic feasibility or RL success. No integration or policy/reset data.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary));print(json.dumps(rows[0]))


if __name__=='__main__':main()
