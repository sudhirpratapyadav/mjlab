"""Retain one measured motion trial in scoped W&B without replacing a teacher."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from wandb_config import ENTITY, PROJECT, configure

HERE = Path(__file__).resolve().parent


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--validation', type=Path, required=True)
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--confirmation', type=Path)
  parser.add_argument('--audit', type=Path, required=True)
  parser.add_argument('--plan', type=Path, required=True)
  parser.add_argument('--preflight', type=Path, required=True)
  parser.add_argument('--review-dir', type=Path, required=True)
  parser.add_argument('--public-url', required=True)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Publication receipt exists; inspect the existing upload before retrying')
  validation = json.loads(args.validation.read_text())
  baseline = json.loads(args.baseline.read_text())
  audit = json.loads(args.audit.read_text())
  videos = json.loads((args.review_dir / 'videos.json').read_text())
  summary = json.loads((args.review_dir / 'summary.json').read_text())
  checkpoint = Path(validation['checkpoint'])
  plan = json.loads(args.plan.read_text())
  assert plan['run_id'] == checkpoint.parent.name and plan['expected_final_checkpoint'] == checkpoint.name
  sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
  assert sha == validation['checkpoint_sha256'] == audit['checkpoint_sha256']
  assert baseline['checkpoint_sha256'] == audit['baseline']['checkpoint_sha256']
  assert baseline['seed'] == validation['seed'] and baseline['task'] == validation['task']
  assert videos['paired_initial_states_verified'] and len(videos['clips']) == 6
  assert audit['episodes'] == validation['episodes'] == 128
  assert audit['successes'] == validation['successes'] == summary['validation_successes']
  confirmation = json.loads(args.confirmation.read_text()) if args.confirmation else None
  if confirmation:
    assert confirmation['checkpoint_sha256'] == sha and confirmation['seed'] != validation['seed']
  configure()
  import wandb
  with wandb.init(entity=ENTITY, project=PROJECT, name=checkpoint.parent.name + '-motion-review',
                  job_type='motion-review', config={
                    'task': validation['task'], 'checkpoint_sha256': sha,
                    'public_url': args.public_url, 'review_status': summary['review_status'],
                    'baseline_preserved': True, 'other_teachers_paused': True,
                    'paired_env_ids': [0, 1, 2], 'video_fps': 50}) as run:
    metrics = {'strict/validation_success_rate': validation['success_rate']}
    if confirmation:
      metrics['strict/confirmation_success_rate'] = confirmation['success_rate']
    for label, source in [('before', audit['baseline']), ('after', audit)]:
      for metric, values in source['summary'].items():
        for statistic, value in values.items():
          metrics[f'motion/{label}/{metric}/{statistic}'] = value
    for gate, passed in audit['provisional_motion_gates'].items():
      metrics['motion_gates/' + gate] = int(passed)
    run.log(metrics)
    for clip in videos['clips']:
      run.log({f'paired_videos/episode_{clip["env_id"]}/{clip["label"]}':
        wandb.Video(str(args.review_dir / clip['video']), format='mp4',
                    caption=f'{clip["label"]}; 1x real time, 50fps; success={clip["success"]}')})
    artifact = wandb.Artifact(checkpoint.parent.name + '-motion-review', type='motion-trial',
                              metadata={'checkpoint_sha256': sha, 'review_status': summary['review_status']})
    artifact.add_file(str(checkpoint), name='candidate/model.pt')
    for name in ['manifest.json', 'source.patch']:
      artifact.add_file(str(checkpoint.parent / name), name='candidate/' + name)
    artifact.add_dir(str(checkpoint.parent / 'params'), name='candidate/params')
    for path in [args.validation, args.baseline, args.audit] + ([args.confirmation] if args.confirmation else []):
      artifact.add_file(str(path), name='evidence/' + path.name)
    for label, evaluation in [('before', baseline), ('after', validation)]:
      artifact.add_file(str(Path(evaluation['trace_dir']) / 'trace.npz'), name=f'traces/{label}.npz')
    artifact.add_file(str(checkpoint.parent / 'diagnostics.jsonl'), name='candidate/diagnostics.jsonl')
    artifact.add_dir(str(args.review_dir), name='review')
    for record in [args.plan, args.preflight, HERE / 'evidence/lift_smooth_baseline_full_audit.json']:
      artifact.add_file(str(record), name='evidence/' + record.name)
    retained = run.log_artifact(artifact)
    retained.wait()
    receipt = {'published_utc': datetime.now(timezone.utc).isoformat(),
               'wandb_run_url': run.url, 'wandb_artifact': retained.qualified_name,
               'checkpoint_sha256': sha, 'public_url': args.public_url,
               'review_status': summary['review_status'], 'baseline_preserved': True}
  # Read the remote manifest back; upload completion alone is not a retention audit.
  remote = wandb.Api().artifact(receipt['wandb_artifact'])
  paths = set(remote.manifest.entries)
  assert 'candidate/model.pt' in paths and 'candidate/params/env.yaml' in paths
  for clip in videos['clips']:
    assert 'review/' + clip['video'] in paths
  receipt['remote_files_verified'] = sorted(paths)
  args.output.write_text(json.dumps(receipt, indent=2) + '\n')
  print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
  main()
