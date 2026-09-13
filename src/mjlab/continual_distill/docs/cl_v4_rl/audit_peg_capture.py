"""Inspect recorded Peg contacts and arm targets without integrating or re-querying."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from probe_pivot_geometry_dynamics import contact_wrench


def main():
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument('--evaluation', type=Path, required=True)
  p.add_argument('--output', type=Path, required=True)
  args = p.parse_args()
  if args.output.exists(): p.error('Output exists')
  ev = json.loads(args.evaluation.read_text())
  assert ev['task'] == 'Mjlab-Peg-Insertion-Franka'
  with np.load(Path(ev['trace_dir'])/'trace.npz') as a:
    t = {k:a[k] for k in a.files}
  assert 'ctrl' in t and 'initial_model_geom_friction' in t
  cfg = load_env_cfg(ev['task']); cfg.scene.num_envs = 1
  cmd = next(iter(cfg.commands.values()))
  m = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(m)
  d, target = mujoco.MjData(m), mujoco.MjData(m)
  site = m.site(cmd.asset_name+'/object_site').id
  body = m.site_bodyid[site]; grip = m.site('robot/gripper').id
  objects = {i for i in range(m.ngeom) if m.geom(i).name.startswith(cmd.asset_name+'/')}
  robots = {i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/')}
  pads = [m.geom('robot/'+n).id for n in ['left_finger_pad','right_finger_pad']]
  arm = [m.jnt_qposadr[m.joint(f'robot/joint{i}').id] for i in range(1,8)]
  fingers = [m.jnt_qposadr[m.joint('robot/'+n).id] for n in ['finger_joint1','finger_joint2']]
  assert m.nu == 8 and all(m.actuator_trnid[i,0] == m.joint(f'robot/joint{i+1}').id for i in range(7))
  joint = next(i for i in range(m.njnt) if m.jnt_bodyid[i] == body and m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
  qadr = m.jnt_qposadr[joint]
  rows = []
  for rec in ev['records']:
    lane, end = rec['env_id'], rec['steps']
    m.geom_friction[:] = t['initial_model_geom_friction'][lane]
    for field in ['qpos','qvel','mocap_pos','mocap_quat','ctrl']:
      getattr(d,field)[:] = t[field][end,lane]
    for adr, kind in zip(m.jnt_qposadr,m.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE: d.qpos[adr:adr+3] -= t['origins'][lane]
    d.mocap_pos[:] -= t['origins'][lane]
    d.qacc_warmstart[:] = 0; mujoco.mj_forward(m,d)
    held = [False,False]; wrench = np.zeros(6); contacts = []
    for index, c in enumerate(d.contact):
      a,b = c.geom
      if not ((a in objects and b in robots) or (b in objects and a in robots)): continue
      wrench += contact_wrench(m,d,index,objects,d.xpos[body])
      for i,pad in enumerate(pads):
        if a == pad and b in objects: normal = c.frame[:3]
        elif b == pad and a in objects: normal = -c.frame[:3]
        else: continue
        inward = d.geom_xpos[pads[1-i]]-d.geom_xpos[pad]; inward /= np.linalg.norm(inward)
        held[i] |= bool(c.dist <= .001 and normal@inward > .5)
      contacts.append(dict(geom0=m.geom(a).name,geom1=m.geom(b).name,distance_m=float(c.dist),height_m=float(c.pos[2])))
    target.qpos[:] = d.qpos; target.qpos[arm] = d.ctrl[:7]
    mujoco.mj_kinematics(m,target)
    root_heights = t['qpos'][:end+1,lane,qadr+2]-t['origins'][lane,2]
    rows.append(dict(env_id=lane,terminal_opposed_contact=all(held),terminal_tip_height_m=float(d.site_xpos[site,2]),
      terminal_root_height_m=float(d.xpos[body,2]),max_root_lift_m=float(root_heights.max()-root_heights[0]),
      gripper_height_m=float(d.site_xpos[grip,2]),fingers_m=d.qpos[fingers].tolist(),finger_target_m=float(d.ctrl[7]),
      arm_target_error_rad=(d.ctrl[:7]-d.qpos[arm]).tolist(),
      arm_target_fk_displacement_m=(target.site_xpos[grip]-d.site_xpos[grip]).tolist(),
      cpu_robot_wrench_on_peg=wrench.tolist(),contacts=contacts))
  captured = [r for r in rows if r['terminal_opposed_contact']]
  summary = dict(episodes=len(rows),native_successes=ev['successes'],terminal_opposed=len(captured),
    ever_root_lift_above2cm=sum(r['max_root_lift_m']>.02 for r in rows),
    median_max_root_lift_m=float(np.median([r['max_root_lift_m'] for r in rows])),
    captured_median_tip_height_m=float(np.median([r['terminal_tip_height_m'] for r in captured])) if captured else None,
    captured_median_target_fk_delta_m=np.median([r['arm_target_fk_displacement_m'] for r in captured],axis=0).tolist() if captured else None,
    captured_median_robot_force_N=np.median([r['cpu_robot_wrench_on_peg'][:3] for r in captured],axis=0).tolist() if captured else None,
    captured_target_fk_up_above1cm=sum(r['arm_target_fk_displacement_m'][2]>.01 for r in captured),
    peg_mass_kg=float(m.body_mass[body]))
  report = dict(task=ev['task'],checkpoint_sha256=ev['checkpoint_sha256'],
    method='All128 exact terminal states with recorded controls/friction. CPU forward/contact solve is a cold diagnostic, not original GPU force. Arm target FK keeps current fingers and substitutes recorded seven actuator targets, not a feasible trajectory. Root lift measured over all recorded first-episode states. No integration, policy requery or success substitution.',summary=summary,rows=rows)
  args.output.write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(summary,indent=2))


if __name__ == '__main__': main()
