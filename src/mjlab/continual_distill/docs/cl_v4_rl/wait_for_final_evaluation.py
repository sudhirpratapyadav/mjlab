"""Wait for an identified trainer to exit, then evaluate only its planned final model."""
import argparse
from datetime import datetime, timezone
import fcntl
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
  parser.add_argument('--trainer-pid', type=int, required=True)
  args = parser.parse_args()
  plan = json.loads(args.plan.read_text())
  run = HERE / 'runs' / plan['run_id']
  manifest = json.loads((run / 'manifest.json').read_text())
  assert manifest['task'] == plan['task']
  assert manifest['requested_updates'] == plan['updates']
  assert manifest['resume_sha256'] == plan['source_checkpoint_sha256']
  for key in ('recipe', 'max_update_kl', 'update_learning_rate_ceiling'):
    if key in plan:
      assert manifest.get(key) == plan[key], f'Training manifest differs from plan: {key}'
  assert os.environ['CUDA_VISIBLE_DEVICES'] == manifest['gpu_uuid']
  assert os.environ['PYTHONPATH'] == str(Path(manifest['worktree']) / 'src')
  assert plan['validation_seed'] == 20260914 and plan['episodes_per_batch'] == 128
  checkpoint = run / plan['expected_final_checkpoint']
  label = '-'.join(plan['run_id'].split('-')[:3]) + '-m' + checkpoint.stem.removeprefix('model_')
  result = HERE / 'evidence' / f'{label}-val-20260914.json'
  lock = (run / 'final-evaluation.lock').open('a')
  fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  assert not result.exists(), f'Already evaluated: {result}'
  process = Path('/proc') / str(args.trainer_pid)
  command = (process / 'cmdline').read_bytes().split(b'\0')
  assert plan['run_id'].encode() in command and any(v.endswith(b'/train_teacher.py') for v in command)
  identity = (process / 'stat').read_text().split(') ', 1)[1].split()[19]
  receipt = run / 'evaluation-dispatch.json'
  def record(state, **extra):
    receipt.write_text(json.dumps(dict(updated_utc=datetime.now(timezone.utc).isoformat(), state=state,
      trainer_pid=args.trainer_pid, dispatcher_pid=os.getpid(), plan=str(args.plan.resolve()),
      checkpoint=str(checkpoint), label=label, **extra), indent=2) + '\n')
    print(state, label, flush=True)
  record('waiting_for_identified_trainer_exit')
  while True:
    try:
      stat = (process / 'stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
      break
    if stat[19] != identity or stat[0] == 'Z':
      break
    time.sleep(15)
  import torch
  try:
    assert checkpoint.is_file(), 'Trainer exited without the planned final checkpoint'
    assert not (run / 'numerical_failure.pt').exists(), 'Trainer archived a numerical failure'
    model = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert int(model['iter']) == int(checkpoint.stem.removeprefix('model_'))
    def tensors(value):
      if isinstance(value, torch.Tensor):
        yield value
      elif isinstance(value, dict):
        for child in value.values():
          yield from tensors(child)
      elif isinstance(value, (list, tuple)):
        for child in value:
          yield from tensors(child)
    values = list(tensors(model))
    assert values and all(torch.isfinite(v).all() for v in values)
    del values, model
    record('evaluating_verified_final_checkpoint')
    subprocess.run([sys.executable, str(HERE / 'evaluate_candidate.py'), '--task', plan['task'],
      '--checkpoint', str(checkpoint), '--label', label, '--diagnostics',
      '--confirmation-seed', str(plan['confirmation_seed'])], check=True)
    record('evaluation_finished_video_review_required')
  except Exception as error:
    record('failed_requires_inspection', error=repr(error))
    raise


if __name__ == '__main__':
  main()
