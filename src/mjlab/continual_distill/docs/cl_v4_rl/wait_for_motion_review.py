"""Audit and render a Lift trial after its identified strict evaluator finishes."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--plan', type=Path, required=True)
  parser.add_argument('--evaluation-watcher-pid', type=int, required=True)
  args = parser.parse_args()
  plan = json.loads(args.plan.read_text())
  assert plan['task'] == 'Mjlab-Lift-Cube-Franka'
  run = HERE / 'runs' / plan['run_id']
  receipt = run / 'motion-review-dispatch.json'
  assert not receipt.exists(), 'Existing dispatcher receipt requires inspection'
  manifest = json.loads((run / 'manifest.json').read_text())
  assert os.environ['CUDA_VISIBLE_DEVICES'] == manifest['gpu_uuid']
  assert os.environ['PYTHONPATH'] == str(Path(manifest['worktree']) / 'src')
  process = Path('/proc') / str(args.evaluation_watcher_pid)
  command = (process / 'cmdline').read_bytes().split(b'\0')
  assert any(c.endswith(b'/wait_for_final_evaluation.py') for c in command)
  assert any(c.endswith(args.plan.name.encode()) for c in command)
  identity = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
  def record(state, **extra):
    receipt.write_text(json.dumps(dict(state=state, pid=os.getpid(),
      evaluation_watcher_pid=args.evaluation_watcher_pid,
      updated_utc=datetime.now(timezone.utc).isoformat(), **extra), indent=2) + '\n')
    print(state, flush=True)
  record('waiting_for_identified_evaluation_watcher')
  while True:
    try:
      stat = (process / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
      break
    if stat[19] != identity or stat[0] == 'Z':
      break
    time.sleep(15)
  try:
    evaluation = json.loads((run / 'evaluation-dispatch.json').read_text())
    assert evaluation['state'] == 'evaluation_finished_video_review_required'
    label = evaluation['label']
    suffix = plan['run_id'].split('-')[2].lower()
    baseline = HERE / 'evidence/RL-002-R16-m21990-val-20260914.json'
    candidate = HERE / 'evidence' / f'{label}-val-20260914.json'
    confirmation = HERE / 'evidence' / f'{label}-confirm-{plan["confirmation_seed"]}.json'
    audit = HERE / 'evidence' / f'lift_{suffix}_motion_audit.json'
    videos = HERE / 'runs' / f'lift-{suffix}-motion-review'
    def call(script, *arguments):
      subprocess.run([sys.executable, str(HERE / script), *map(str, arguments)], check=True)
    record('auditing_complete_episodes')
    call('audit_motion_quality.py', '--evaluation', candidate, '--baseline', baseline, '--output', audit)
    record('rendering_paired_real_time_videos')
    call('render_motion_review.py', '--baseline', baseline, '--candidate', candidate, '--output', videos)
    call('build_motion_review.py', '--audit', audit, '--videos', videos,
         *(['--confirmation', confirmation] if confirmation.exists() else []))
    record('local_comparison_ready_for_review')
  except Exception as error:
    record('failed_requires_inspection', error=repr(error))
    raise


if __name__ == '__main__':
  main()
