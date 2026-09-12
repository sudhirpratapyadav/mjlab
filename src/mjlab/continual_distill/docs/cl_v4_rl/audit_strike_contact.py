"""Diagnose recorded Strike approaches using CPU forward geometry, no integration."""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  report = json.loads(args.evaluation.read_text())
  assert report['task'] == 'Mjlab-Strike-Slide-Franka'
  # NpzFile indexing decompresses an entire member on every access. Materialize
  # once before traversing the25,000 recorded states.
  with np.load(Path(report['trace_dir'])/'trace.npz') as archive:
    trace = {name:archive[name] for name in archive.files}
  cfg = load_env_cfg(report['task'])
  cfg.scene.num_envs = 1
  model = Scene(cfg.scene, 'cpu').compile()
  cfg.sim.mujoco.apply(model)
  data = mujoco.MjData(model)
  grip = model.site('robot/gripper').id
  puck = model.site('puck/object_site').id
  puck_body = model.site_bodyid[puck]
  free_joint = model.body_jntadr[puck_body]
  velocity_start = model.jnt_dofadr[free_joint]
  robot_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith('robot/')}
  puck_geoms = {i for i in range(model.ngeom) if model.geom(i).name.startswith('puck/')}
  pads = {model.geom('robot/'+name).id for name in ['left_finger_pad', 'right_finger_pad']}
  rows = []
  for outcome in report['records']:
    lane, steps = outcome['env_id'], outcome['steps']
    best = None
    contacts = pad_contacts = 0
    minimum_gap = float('inf')
    max_speed = 0.
    for step in range(steps+1):
      for name in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat']:
        getattr(data, name)[:] = trace[name][step, lane]
      for address, kind in zip(model.jnt_qposadr, model.jnt_type):
        if kind == mujoco.mjtJoint.mjJNT_FREE:
          data.qpos[address:address+3] -= trace['origins'][lane]
      data.mocap_pos[:] -= trace['origins'][lane]
      # Recompute geometry only. This is not a CPU rollout or a replacement of
      # the original GPU contact/success measurements between saved 20ms states.
      mujoco.mj_forward(model, data)
      pos = data.site_xpos[puck].copy()
      goal = data.mocap_pos[1]
      direction = goal[:2]-pos[:2]
      direction /= max(np.linalg.norm(direction), .001)
      target = pos.copy()
      target[:2] -= .0353*direction
      target[2] = .028
      delta = data.site_xpos[grip]-target
      distance = np.linalg.norm(delta)
      rot = data.site_xmat[grip].reshape(3,3)
      if best is None or distance < best['distance_m']:
        best = dict(step=step, distance_m=float(distance),
                    longitudinal_error_m=float(delta[:2]@direction),
                    lateral_error_m=float(delta[0]*direction[1]-delta[1]*direction[0]),
                    vertical_error_m=float(delta[2]), down_alignment=float(-rot[2,2]),
                    paddle_alignment=float(abs(rot[:2,0]@direction)),
                    hand_position_m=data.site_xpos[grip].tolist(), puck_position_m=pos.tolist())
      hit = pad_hit = False
      for contact in data.contact:
        a,b = contact.geom
        if (a in robot_geoms and b in puck_geoms) or (b in robot_geoms and a in puck_geoms):
          minimum_gap = min(minimum_gap, float(contact.dist))
          if contact.dist <= .001:
            hit = True
            pad_hit |= a in pads or b in pads
      contacts += hit
      pad_contacts += pad_hit
      max_speed = max(max_speed, float(np.linalg.norm(trace['qvel'][step,lane,velocity_start:velocity_start+3])))
    rows.append(dict(env_id=lane, success=outcome['success'], max_speed_m_s=max_speed,
                     sampled_contact_states=contacts, sampled_pad_contact_states=pad_contacts,
                     minimum_generated_contact_gap_m=minimum_gap if np.isfinite(minimum_gap) else None,
                     closest_approach=best, initial_puck_position_m=(trace['qpos'][0,lane,9:12]-trace['origins'][lane]).tolist()))
    if (lane+1)%32 == 0:
      print(f'Audited {lane+1} episodes', flush=True)
  groups = {}
  for name, selected in [('negligible', [r for r in rows if r['max_speed_m_s']<.05]),
                         ('weak', [r for r in rows if .05<=r['max_speed_m_s']<.4]),
                         ('substantial', [r for r in rows if r['max_speed_m_s']>=.4])]:
    groups[name] = dict(episodes=len(selected), successes=sum(r['success'] for r in selected),
                       episodes_with_sampled_contact=sum(r['sampled_contact_states']>0 for r in selected),
                       episodes_with_sampled_pad_contact=sum(r['sampled_pad_contact_states']>0 for r in selected),
                       median_closest_approach={key:float(np.median([r['closest_approach'][key] for r in selected]))
                                                for key in ['distance_m','longitudinal_error_m','lateral_error_m','vertical_error_m','down_alignment','paddle_alignment']})
  result = dict(task=report['task'], checkpoint_sha256=report['checkpoint_sha256'],
                source_evaluation=str(args.evaluation.resolve()),
                method='CPU forward geometry at every saved20ms state through each uninterrupted terminal episode; no integration. Contacts between recorded states may be missed; CPU contacts are recomputed diagnostics, not original GPU contact evidence or a new success evaluation.',
                groups=groups, rows=rows)
  args.output.write_text(json.dumps(result, indent=2)+'\n')
  print(json.dumps(groups, indent=2))


if __name__ == '__main__':
  main()
