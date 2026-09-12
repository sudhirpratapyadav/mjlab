"""Render and evaluate the corrected-physics gallery; safe to resume by task."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
p.add_argument('--shard', type=int, default=0)
p.add_argument('--shards', type=int, default=3)
p.add_argument('--tasks', nargs='*')
a = p.parse_args()
names = ['Flip-Switch', 'Push-Flap', 'Open-Lid', 'Push-Button', 'Axial-Extract',
         'Lift-Cube', 'Stack-Cube', 'Reach-Target', 'Place-In-Container',
         'Topple-Block', 'Strike-Slide', 'Reorient-Object', 'Push-Cuboid',
         'Drag-Pull', 'Cage-Drag', 'Throw-To-Bin', 'Open-Door', 'Edge-Grasp',
         'Peg-Insertion', 'Tool-Pull', 'Pivot-Lift']
tasks = a.tasks or [f'Mjlab-{n}-Franka' for n in names]
for task in tasks[a.shard::a.shards]:
  out = a.out / task
  out.mkdir(parents=True, exist_ok=True)
  if (out / 'result.json').exists():
    continue
  print('START', task, flush=True)
  start = time.time()
  cmd = [sys.executable, '-m', 'mjlab.continual_distill.classical.render_rollout',
         '--task', task, '--num-episodes', '128', '--batch-size', '128',
         '--max-render-episodes', '3', '--out', str(out)]
  with (out / 'render.log').open('w') as log:
    result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
  print('END', task, result.returncode, round(time.time() - start), flush=True)
  if result.returncode:
    print((out / 'render.log').read_text()[-3000:], flush=True)
  else:
    print((out / 'result.json').read_text(), flush=True)
