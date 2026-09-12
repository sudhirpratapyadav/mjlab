"""Build the public summary from certificates and recorded RL evaluations.

Only public summary fields and policy media are copied. Run artifacts, credentials,
private paths and detailed experiment records stay outside the web directory.
"""
import argparse
import hashlib
import html
import json
import shutil
from mjlab.scripts.train import TrainConfig
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WANDB = 'https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-cl24-rl-teachers-20260912'


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=Path, required=True)
  args = parser.parse_args()
  args.output.mkdir(parents=True, exist_ok=True)
  tasks = json.loads((HERE/'active_tasks.json').read_text())['active_tasks']
  evaluations = []
  for path in (HERE/'evidence').glob('*.json'):
    data = json.loads(path.read_text())
    if isinstance(data, dict) and data.get('first_episode_only') and data.get('episodes') == 128 and data.get('seed') == 20260914:
      evaluations.append((path.stat().st_mtime_ns, path.name, data))
  rows = []
  for task in tasks:
    certificate = HERE/'evidence'/f'{task}-certificate.json'
    certified = certificate.exists()
    if certified:
      cert = json.loads(certificate.read_text())
      evaluation = json.loads(Path(cert['validation']['path']).read_text())
      clip = cert['videos']['success']
      confirmation = cert['confirmation']['successes']
    else:
      _, _, evaluation = max(item for item in evaluations if item[2]['task'] == task
                              and item[2].get('trace_dir')
                              and (Path(item[2]['trace_dir'])/'videos.json').is_file())
      videos = json.loads((Path(evaluation['trace_dir'])/'videos.json').read_text())
      # Show the failure example for an unfinished policy, even when some trials pass.
      clip = videos['clips'].get('failure') or videos['clips']['success']
      confirmation = None
    assert clip['source'] == 'exact first-episode states; rendering only, no resimulation'
    assert clip.get('seed', 20260914) == 20260914
    video = Path(clip['path'])
    poster = video.with_name(video.stem+'-0000.png')
    assert video.is_file() and poster.is_file(), task
    dest = args.output/'media'/task
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(video, dest/'policy.mp4')
    shutil.copy2(poster, dest/'poster.png')
    checkpoint = Path(evaluation['checkpoint'])
    rows.append(dict(task=task, name=task.removeprefix('Mjlab-').removesuffix('-Franka').replace('-', ' '),
                     certified=certified, validation=evaluation['successes'], confirmation=confirmation,
                     episodes=128, duration=evaluation.get('episode_length_s') or TrainConfig.from_task(task).env.episode_length_s,
                     checkpoint=checkpoint.parent.name+'/'+checkpoint.name,
                     checkpoint_sha256=evaluation['checkpoint_sha256'],
                     video=f'media/{task}/policy.mp4', poster=f'media/{task}/poster.png',
                     video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
                     clip_outcome='Success example' if clip['outcome']['success'] else 'Failure example'))
  rows.sort(key=lambda row: (not row['certified'], row['name']))
  count = sum(row['certified'] for row in rows)
  updated = datetime.now(timezone.utc).strftime('%d %b %Y · %H:%M UTC')
  cards = []
  for row in rows:
    state = 'certified' if row['certified'] else 'progress'
    status = 'Certified' if row['certified'] else 'In progress'
    rate = f"{100*row['validation']/128:.1f}%"
    if row['certified']:
      rate += f" / {100*row['confirmation']/128:.1f}%"
      detail = 'Validation / confirmation · 128 episodes each'
    else:
      detail = f"{row['validation']}/128 passed · latest evaluated checkpoint"
    cards.append(f'''<article class="task" data-status="{state}" data-name="{html.escape(row['name'].lower())}">
<div class="task-head"><h2>{html.escape(row['name'])}</h2><span class="badge {state}">{status}</span></div>
<div class="media"><video controls playsinline preload="none" poster="{row['poster']}" aria-label="{html.escape(row['name'])} RL policy, {row['clip_outcome'].lower()}"><source src="{row['video']}" type="video/mp4">Your browser cannot play this video. <a href="{row['video']}">Download policy video</a>.</video></div>
<div class="task-body"><div class="result"><strong>{rate}</strong><span>{row['clip_outcome']}</span></div><p>{detail}</p><div class="basic">PPO · {row['duration']:g}s episode · Franka</div><div class="checkpoint" title="Evaluated checkpoint">{html.escape(row['checkpoint'])}</div></div></article>''')
  template = (HERE/'gallery.html').read_text()
  for key, value in {'COUNT':str(count), 'TOTAL':str(len(rows)), 'REMAINING':str(len(rows)-count), 'UPDATED':updated, 'WANDB':WANDB, 'CARDS':'\n'.join(cards)}.items():
    template = template.replace('{{'+key+'}}', value)
  (args.output/'index.html').write_text(template)
  report = dict(updated_utc=datetime.now(timezone.utc).isoformat(), certified_count=count,
                active_tasks=len(rows), observation_dim=60, action_dim=8, rows=rows)
  (args.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
  print(f'Built {len(rows)} policy video cards; {count} certified in {args.output}')


if __name__ == '__main__':
  main()
