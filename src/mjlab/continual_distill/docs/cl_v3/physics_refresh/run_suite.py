"""Sequential task evaluations on one assigned GPU; safe to shard across workers."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--shard',type=int,default=0)
p.add_argument('--shards',type=int,default=3)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--num-envs',type=int,default=16)
p.add_argument('--num-episodes',type=int,default=2)
p.add_argument('--tasks',nargs='*')
a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
names=['Flip-Switch','Rotate-Valve','Open-Lid','Push-Button','Turn-Lever','Push-Flap','Open-Drawer','Slide-Window','Axial-Extract','Lift-Cube','Stack-Cube','Place-In-Container','Reach-Target','Reorient-Object','Topple-Block','Push-Cuboid','Drag-Pull','Cage-Drag','Strike-Slide','Throw-To-Bin','Open-Door','Peg-Insertion','Tool-Pull','Edge-Grasp','Pivot-Lift']
tasks=a.tasks or ['Mjlab-'+s+'-Franka' for s in names]
for task in tasks[a.shard::a.shards]:
 result=a.out/(task+'.json')
 if result.exists():continue
 cmd=[sys.executable,'-m','mjlab.continual_distill.classical.test_classical','--task',task,'--num-envs',str(a.num_envs),'--num-episodes',str(a.num_episodes),'--legacy-window']
 started=time.time();print('START',task,flush=True)
 digest=hashlib.sha256()
 for f in sorted(Path('src/mjlab/continual_distill/classical').glob('*.py')):digest.update(f.read_bytes())
 with (a.out/(task+'.log')).open('w') as log:
  proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 logtext=(a.out/(task+'.log')).read_text()
 match=re.search(r'OVERALL success rate: ([\d.]+) over (\d+) episodes',logtext)
 record=dict(task=task,exit_code=proc.returncode,seconds=time.time()-started,teacher_hash=digest.hexdigest(),geometry_revision='2026-09-10-full-collision',evaluation_protocol='legacy_fixed_window_retries',n=a.num_envs*a.num_episodes)
 if match:record.update(sr=float(match[1]),n=int(match[2]))
 result.write_text(json.dumps(record,indent=2)+'\n');print('RESULT',json.dumps(record),flush=True)
