"""Publish the authoritative certificate inventory and measured validation results."""
import json
from datetime import datetime, timezone
from pathlib import Path
from mjlab.tasks.manipulation.benchmark import active_cl_tasks
from wandb_config import configure, ENTITY, PROJECT

HERE = Path(__file__).resolve().parent

def main():
  rows = []
  evidence = []
  for path in (HERE / 'evidence').glob('*.json'):
    value = json.loads(path.read_text())
    if isinstance(value, dict) and value.get('first_episode_only') and value.get('seed') == 20260914 and value.get('episodes', 0) >= 128:
      evidence.append(value)
  manifests = [json.loads(path.read_text()) for path in (HERE / 'runs').glob('*/manifest.json')]
  for task in active_cl_tasks():
    certificate_path = HERE / 'evidence' / f'{task}-certificate.json'
    certificate = json.loads(certificate_path.read_text()) if certificate_path.exists() else None
    validations = [value for value in evidence if value['task'] == task]
    best = max(validations, key=lambda value: value['success_rate']) if validations else None
    training = sorted([value for value in manifests if value['task'] == task and value['run_id'].startswith('RL-')], key=lambda value: value['started_utc'])
    preflight = sorted([value for value in manifests if value['task'] == task and value['run_id'].startswith('PREFLIGHT-')], key=lambda value: value['started_utc'])
    rows.append({'task': task, 'certified': certificate is not None,
                 'validation_successes': certificate['validation']['successes'] if certificate else None,
                 'validation_episodes': certificate['validation']['episodes'] if certificate else None,
                 'confirmation_successes': certificate['confirmation']['successes'] if certificate else None,
                 'confirmation_episodes': certificate['confirmation']['episodes'] if certificate else None,
                 'retained_run_url': certificate.get('wandb_run_url') if certificate else None,
                 'best_measured_validation_rate': best['success_rate'] if best else None,
                 'latest_training_run': training[-1]['run_id'] if training else (Path(best['checkpoint']).parent.name if best else None),
                 'latest_preflight_run': preflight[-1]['run_id'] if preflight else None})
  report = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'certified_count': sum(row['certified'] for row in rows),
            'active_tasks': len(rows), 'interface': 'franka_shared_60_v2',
            'criterion': 'Two independent terminal-first-episode batches, each strictly >90%, plus retained checkpoint/normalizers and video review',
            'rows': rows}
  path = HERE / 'evidence/scoreboard.json'
  path.write_text(json.dumps(report, indent=2) + '\n')
  configure()
  import wandb
  with wandb.init(entity=ENTITY, project=PROJECT, name='CL24-certification-scoreboard', job_type='scoreboard') as run:
    columns = list(rows[0])
    run.log({'certified_teachers': report['certified_count'], 'active_tasks': len(rows),
             'teachers': wandb.Table(columns=columns, data=[[row[column] for column in columns] for row in rows])})
    artifact = wandb.Artifact('CL24-certification-scoreboard', type='scoreboard')
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    records = wandb.Artifact('CL24-experiment-records', type='experiment-records')
    for name in ('CONTEXT', 'GOAL', 'STATUS', 'PLAN', 'EXPERIMENTS', 'LOGS', 'PUBLICATION'):
      records.add_file(str(HERE / f'{name}.md'), name=f'{name}.md')
    for pattern in ('*-review.json', 'training_wave*.json', '*preflight*.json', '*replay.json', '*audit.json',
                    'cold_solver_reset.json', 'rl_training_coverage.json', 'public_gallery_verification.json'):
      for record in sorted((HERE / 'evidence').glob(pattern)):
        records.add_file(str(record), name=f'evidence/{record.name}')
    run.log_artifact(records)
    # Keep the full simulator failure diagnostic with the experiment results.
    for failure in sorted((HERE/'runs').glob('*/numerical_failure.pt')):
      diagnostic = wandb.Artifact(failure.parent.name+'-numerical-failure', type='diagnostics')
      diagnostic.add_file(str(failure))
      diagnostic.add_file(str(failure.with_suffix('.json')))
      run.log_artifact(diagnostic)
    url = run.url
  print(f"{report['certified_count']}/{len(rows)} certified: {url}")

if __name__ == '__main__':
  main()
