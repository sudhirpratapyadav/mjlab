"""Publish completed refresh artifacts, recording exactly which sources were sent."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent.parent
strategies = {
  'Slide-Window': 'Approach the exposed handle from the front, pinch across its width, and translate along the mount slide axis.',
  'Turn-Lever': 'Tilt the wrist away from the fixture and push tangentially along the observed handle arc.',
  'Rotate-Valve': 'Push a spoke with a closed fingertip on an outboard circular path; estimate angle from the exact target pose.',
  'Open-Drawer': 'Seat a closed fingertip at the handle edge, pull with a shorter lead, and reseat after losing contact.',
}
for result_path in sorted((a.root / 'videos').glob('*/result.json')):
  folder = result_path.parent
  r = json.loads(result_path.read_text())
  task = r['task_id']
  name = task.removeprefix('Mjlab-').removesuffix('-Franka')
  if folder.name != task or r['n'] != 128 or r['evaluation_protocol'] != 'single_episode_no_reset_retries':
    raise ValueError(f'Invalid refresh metadata: {result_path}')
  for flag, filename in [('has_teacher_clip', 'teacher.mp4'), ('has_failure_clip', 'failure.mp4')]:
    if bool(r[flag]) != (folder / filename).is_file():
      raise ValueError(f'Clip metadata mismatch: {folder / filename}')
  if not r['has_teacher_clip'] and not r['has_failure_clip']:
    raise ValueError(f'No video: {folder}')
  stamp = folder / 'published.json'
  if stamp.exists() and json.loads(stamp.read_text()) == r:
    continue
  old_path = a.root / 'previous_gallery' / task / 'task.json'
  old = json.loads(old_path.read_text()) if old_path.exists() else {}
  metadata = {'init_spec': old.get('init_spec', 'Randomized training reset distribution; unchanged during this teacher refresh.'),
              'strategy': strategies.get(name, 'Observation-only scripted teacher on the corrected task geometry.'),
              'notes': 'Meets the 90% evaluation bar.' if r['pass_bar'] else 'Below the 90% evaluation bar; included for failure review.'}
  (folder / 'task.json').write_text(json.dumps(metadata, indent=2) + '\n')
  subprocess.run(['bash', str(here / 'publish_v3.sh'), task, str(folder)], check=True)
  subprocess.run([sys.executable, str(here / 'update_status.py'), '--task', name,
                  '--owner', 'physics-refresh', '--gate', 'T=' + ('ok' if r['pass_bar'] else '~'),
                  '--gate', 'V=ok', '--set', f"Current SR (n, HEAD)={r['sr']:.4f} (128, {r['source_sha256'][:12]})",
                  '--notes', f"2026-09-10 corrected physics; {r['num_success']}/128 strict first episodes; new videos published. Historical retry-based rates are not comparable."], check=True)
  stamp.write_text(json.dumps(r, indent=2) + '\n')
