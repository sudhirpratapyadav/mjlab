import json
from pathlib import Path
import mujoco
import numpy as np
from mjlab.continual_distill.classical.render_rollout import _build_env
p=Path('src/mjlab/continual_distill/docs/cl_v3/teacher_refresh/gpu_candidate128/Mjlab-Reorient-Object-Franka')
r=json.loads((p/'result.json').read_text()); s=np.load(p/'final_states.npz')
e=_build_env(r['task_id'],1,'cpu',False,0,0); e.reset(); m=e.sim.mj_model; d=mujoco.MjData(m)
for outcome in r['outcomes']:
 if outcome['success'] or not outcome['terminated']: continue
 i=outcome['env']
 for k in ('qpos','qvel','mocap_pos','mocap_quat'): getattr(d,k)[:]=s[k][i]
 for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
  if kind==mujoco.mjtJoint.mjJNT_FREE: d.qpos[adr:adr+3]-=s['origins'][i]
 d.mocap_pos-=s['origins'][i]; mujoco.mj_forward(m,d)
 contacts=[(m.geom(c.geom[0]).name,m.geom(c.geom[1]).name,round(c.dist,4)) for c in d.contact if 'ground' in str((m.geom(c.geom[0]).name,m.geom(c.geom[1]).name)) or 'terrain' in str((m.geom(c.geom[0]).name,m.geom(c.geom[1]).name))]
 print('FAIL',i,outcome['phase'],outcome['steps'],contacts)
e.close()
