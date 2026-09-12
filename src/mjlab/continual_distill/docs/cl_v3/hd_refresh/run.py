import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--shard',type=int);p.add_argument('--shards',type=int,default=3);p.add_argument('--tasks',nargs='*');a=p.parse_args()
old=Path('/ihub/homedirs/svs_ald/cl_v3_work/physics_refresh/videos')
root=Path('/ihub/homedirs/svs_ald/cl_v3_work/hd_refresh/videos')
tasks=a.tasks or sorted(p.parent.name for p in old.glob('*/result.json'))
for task in tasks[a.shard::a.shards]:
 out=root/task;out.mkdir(parents=True,exist_ok=True)
 if (out/'result.json').exists():continue
 cmd=[sys.executable,'-u','-m','mjlab.continual_distill.classical.render_rollout','--task',task,'--stats-from',str(old/task/'result.json'),'--out',str(out),'--max-render-episodes','512']
 print('START',task,flush=True)
 with (out/'render.log').open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 print('END',task,r.returncode,flush=True)
 if r.returncode:print((out/'render.log').read_text()[-2000:],flush=True)
