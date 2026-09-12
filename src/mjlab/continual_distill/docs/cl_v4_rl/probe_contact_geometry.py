"""Verify production contact queries on audited closure geometry; no training."""
import json
from pathlib import Path
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.mdp.task_geometry import between_fingers, grasped
from contact_grasp import opposed_grasp, pad_centerline_width


def main():
  here=Path(__file__).resolve().parent
  output=here/'evidence/contact_geometry_witness_preflight.json'
  if output.exists():
    raise FileExistsError(output)
  audit=json.loads((here/'evidence/cube_closure_audit.json').read_text())
  evaluation=json.loads((here/'evidence/RL-018-R3-m1400-val-20260914.json').read_text())
  selected=[]
  for row in audit['rows']:
    cases=row['opposed_shallow_contact_cases']
    if cases and not cases[0]['between_fingers']:
      selected.append((row['env_id'],cases[0]['closure_per_finger_m']))
  cfg=load_env_cfg(audit['task']);cfg.scene.num_envs=len(selected);cfg.seed=20260913
  env=ManagerBasedRlEnv(cfg,device='cuda:0')
  try:
    env.reset()
    trace=np.load(Path(evaluation['trace_dir'])/'trace.npz')
    indices=[x[0] for x in selected]
    steps=[evaluation['records'][i]['steps'] for i in indices]
    for name in ['qpos','qvel','mocap_pos','mocap_quat']:
      value=trace[name][steps,indices]
      getattr(env.sim.data,name)[:]=torch.as_tensor(value,device=env.device)
    env.scene.env_origins[:]=torch.as_tensor(trace['origins'][indices],device=env.device)
    env.sim.data.qvel.zero_()
    qids=[env.sim.mj_model.jnt_qposadr[env.sim.mj_model.joint('robot/'+name).id]
          for name in ['finger_joint1','finger_joint2']]
    closure=torch.tensor([x[1] for x in selected],device=env.device)
    env.sim.data.qpos[:,qids]=(env.sim.data.qpos[:,qids]-closure[:,None]).clamp_min(0)
    env.sim.reset_solver_state()
    env.sim.forward()
    command=env.command_manager.get_term(env.command_manager.active_terms[0])
    native=grasped(command);enclosure=between_fingers(command);opposed=opposed_grasp(command)
    width=pad_centerline_width(command)
    report={'task':audit['task'],'counterfactual_not_success_evaluation':True,
            'method':'Restore audited saved arm/object poses, apply only chosen counterfactual finger closure, GPU forward contact geometry; no integration or policy updates.',
            'cases':len(selected),'native_two_pad_count':int(native.sum()),
            'old_enclosed_grasp_count':int((native&enclosure).sum()),
            'opposed_grasp_count':int(opposed.sum()),
            'opposed_rejected_by_old_gate_count':int((opposed&~enclosure).sum()),
            'width_finite':bool(torch.isfinite(width).all()),
            'rows':[{'source_env_id':i,'closure_per_finger_m':c,'native_two_pad':bool(native[j]),
                     'enclosure':bool(enclosure[j]),'opposed_grasp':bool(opposed[j]),'centerline_width_m':float(width[j])}
                    for j,(i,c) in enumerate(selected)]}
    output.write_text(json.dumps(report,indent=2)+'\n')
    assert report['width_finite'] and report['opposed_rejected_by_old_gate_count']>0
    assert not (opposed&~native).any()
    print({k:v for k,v in report.items() if k!='rows'},flush=True)
  finally:
    env.close()


if __name__=='__main__':
  main()
