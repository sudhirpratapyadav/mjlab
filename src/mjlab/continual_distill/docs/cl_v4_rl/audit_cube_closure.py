"""CPU geometry counterfactuals, never policy episodes or success measurements."""
from pathlib import Path
import argparse
import json
import mujoco
import numpy as np
from mjlab.tasks.registry import load_env_cfg
from mjlab.scene import Scene
from mjlab.asset_zoo.objects.goal import object_support_points

s = Path('src/mjlab/continual_distill/docs/cl_v4_rl')
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--evaluation',type=Path,default=s/'evidence/RL-018-R3-m1400-val-20260914.json')
parser.add_argument('--output',type=Path,default=s/'evidence/cube_closure_audit.json')
parser.add_argument('--max-closure',type=float,default=.015)
args=parser.parse_args()
r = json.loads(args.evaluation.read_text())
cfg = load_env_cfg(r['task']); cfg.scene.num_envs=1
model = Scene(cfg.scene,'cpu').compile(); cfg.sim.mujoco.apply(model)
data = mujoco.MjData(model)
asset = next(iter(cfg.commands.values())).asset_name
object_ids = {i for i in range(model.ngeom) if model.geom(i).name.startswith(asset+'/')}
pads = [model.geom('robot/'+name).id for name in ['left_finger_pad','right_finger_pad']]
grip = model.site('robot/gripper').id
site = model.site(asset+'/object_site').id
body = model.site_bodyid[site]
support = object_support_points(cfg.scene.entities[asset].spec_fn())
qids = [model.jnt_qposadr[model.joint('robot/'+name).id] for name in ['finger_joint1','finger_joint2']]
with np.load(Path(r['trace_dir'])/'trace.npz') as archive:
  trace = {key:archive[key] for key in archive.files}
rows=[]
for outcome in r['records']:
  lane,steps=outcome['env_id'],outcome['steps']
  for name in ['qpos','qvel','mocap_pos','mocap_quat']:
    getattr(data,name)[:] = trace[name][steps,lane]
  for adr,kind in zip(model.jnt_qposadr,model.jnt_type):
    if kind == mujoco.mjtJoint.mjJNT_FREE:
      data.qpos[adr:adr+3] -= trace['origins'][lane]
  data.mocap_pos[:] -= trace['origins'][lane]
  original = data.qpos[qids].copy()
  cases=[]
  for closure in np.arange(0,args.max_closure+.0001,.0005):
    data.qpos[qids] = np.maximum(original-closure,0)
    mujoco.mj_forward(model,data)
    world = support@data.xmat[body].reshape(3,3).T+data.site_xpos[site]
    rot = data.site_xmat[grip].reshape(3,3)
    local = (world-data.site_xpos[grip])@rot
    local_pads = (data.geom_xpos[pads]-data.site_xpos[grip])@rot
    lo,hi,center = local.min(0),local.max(0),local_pads.mean(0)
    enclosed = (lo[1]>=local_pads[:,1].min()+.0076-.002 and hi[1]<=local_pads[:,1].max()-.0076+.002
                and lo[0]<center[0]+.0088 and hi[0]>center[0]-.0088
                and lo[2]<center[2]+.0082 and hi[2]>center[2]-.0082)
    contacts=[[],[]]
    for c in data.contact:
      if not -.003<=c.dist<=.001: continue
      for side,pad in enumerate(pads):
        if c.geom[0]==pad and c.geom[1] in object_ids: normal=c.frame[:3]
        elif c.geom[1]==pad and c.geom[0] in object_ids: normal=-c.frame[:3]
        else: continue
        inward=data.geom_xpos[pads[1-side]]-data.geom_xpos[pad]
        inward/=np.linalg.norm(inward)
        if normal@inward>.5: contacts[side].append(float(c.dist))
    if all(contacts):
      cases.append({'closure_per_finger_m':float(closure),'aperture_m':float(data.qpos[qids].sum()),
                    'between_fingers':bool(enclosed),'contact_distances':contacts,
                    'projected_object_width_m':float(hi[1]-lo[1]),
                    'inner_pad_gap_m':float(np.linalg.norm(data.geom_xpos[pads[0]]-data.geom_xpos[pads[1]])-.0152)})
  rows.append({'env_id':lane,'initial_aperture_m':float(original.sum()),'opposed_shallow_contact_cases':cases})
report={'task':r['task'],'checkpoint_sha256':r['checkpoint_sha256'],
        'method':f'Exact recorded terminal poses, with finger closure swept0..{args.max_closure*1000:g}mm per finger in0.5mm increments. CPU forward collision geometry only, no integration or policy. Each pad must contact with inward normal cosine>0.5 and distance in[-3,+1]mm.',
        'counterfactual_not_success_evaluation':True,
        'poses_with_opposed_contact':sum(bool(row['opposed_shallow_contact_cases']) for row in rows),
        'poses_with_opposed_contact_rejected_by_enclosure':sum(any(not x['between_fingers'] for x in row['opposed_shallow_contact_cases']) for row in rows),
        'poses_with_first_opposed_contact_rejected_by_enclosure':sum(bool(row['opposed_shallow_contact_cases']) and not row['opposed_shallow_contact_cases'][0]['between_fingers'] for row in rows),
        'rows':rows}
args.output.write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='rows'})
