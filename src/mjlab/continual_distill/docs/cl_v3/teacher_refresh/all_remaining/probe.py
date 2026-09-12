import argparse,json
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image,ImageDraw
from mjlab.continual_distill.classical.render_rollout import _build_env
p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
r=json.loads((a.run/'result.json').read_text()); t=np.load(a.run/'trace.npz'); f=np.load(a.run/'final_states.npz')
e=_build_env(r['task_id'],1,'cpu',False,0,0);e.reset();m=e.sim.mj_model;d=mujoco.MjData(m);ren=mujoco.Renderer(m,360,640)
c=mujoco.MjvCamera();c.lookat[:]=[.43,0,.20];c.distance=1.15;c.azimuth=65;c.elevation=-25
opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.sitegroup[:]=0
fails=[o for o in r['outcomes'] if not o['success']]; selected=fails[:4] or r['outcomes'][:1]; sheet=Image.new('RGB',(640*3,390*len(selected)));report=[]
for row,o in enumerate(selected):
 i=o['env']; steps=o['steps']; origin=f['origins'][i]; samples=sorted(set([max(0,steps//2),max(0,steps-10),steps])); rec=dict(outcome=o,samples=[])
 for col,s in enumerate(samples):
  for k in ['qpos','qvel','mocap_pos','mocap_quat']: getattr(d,k)[:]=f[k][i] if s==steps else t[k][s,i]
  for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
   if kind==mujoco.mjtJoint.mjJNT_FREE:d.qpos[adr:adr+3]-=origin
  d.mocap_pos[:]-=origin;mujoco.mj_forward(m,d)
  contacts=[]
  for cc in d.contact:
   if cc.dist<.001:contacts.append([mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in cc.geom]+[float(cc.dist)])
  rec['samples'].append(dict(step=s,phase=int(t['phase'][min(s,steps-1),i]),contacts=contacts,qvel_max=float(np.abs(d.qvel).max())))
  ren.update_scene(d,camera=c,scene_option=opt);im=Image.fromarray(ren.render());sheet.paste(im,(640*col,390*row));ImageDraw.Draw(sheet).text((640*col+8,390*row+364),f"env {i} step {s} phase {t['phase'][min(s,steps-1),i]}",fill='white')
 report.append(rec)
sheet.save(a.run/'failure_review.jpg');(a.run/'failure_probe.json').write_text(json.dumps(report,indent=2));ren.close();e.close()
