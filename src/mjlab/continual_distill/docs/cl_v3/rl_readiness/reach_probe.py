import json
from pathlib import Path
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.mdp.task_geometry import tracking_position
here=Path(__file__).resolve().parent
rows=[]
for task in ['Open-Door','Open-Drawer','Open-Lid','Push-Button','Push-Flap','Flip-Switch','Rotate-Valve','Slide-Window','Axial-Extract','Turn-Lever']:
 cfg=load_env_cfg('Mjlab-'+task+'-Franka');cfg.scene.num_envs=4;cfg.seed=20260913
 env=ManagerBasedRlEnv(cfg,device='cuda:0');env.reset()
 rc=env.reward_manager.get_term_cfg('reach_object');p=rc.params;robot=env.scene['robot'];obj=env.scene[p['object_asset_name']];pos=robot.data.site_pos_w[:,p['robot_asset_cfg'].site_ids].squeeze(1)
 dist=torch.linalg.vector_norm(pos-tracking_position(obj),dim=-1)
 value=rc.func(env,**p)
 rows.append(dict(task=task,distance_m=dist.cpu().tolist(),approach_reward=value.cpu().tolist(),passed_but_unused_reaching_max_dist=p.get('reaching_max_dist')))
 env.close()
(here/'reach_probe.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
