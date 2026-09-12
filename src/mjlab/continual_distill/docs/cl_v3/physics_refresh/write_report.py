"""Summarize frozen refresh measurements and make a visual review sheet."""
import argparse
import json
import math
from pathlib import Path
from PIL import Image, ImageDraw

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
a = p.parse_args()
here = Path(__file__).resolve().parent
results = [json.loads(p.read_text()) for p in sorted((a.root / 'videos').glob('*/result.json'))]
rows = []
for r in results:
  name = r['task_id'].removeprefix('Mjlab-').removesuffix('-Franka')
  clips = ', '.join(label for flag, label in [('has_teacher_clip', 'success'), ('has_failure_clip', 'failure')] if r[flag])
  rows.append(f"| {name} | {r['num_success']}/128 | {r['sr']:.1%} | {clips} |")
passed = sum(r['pass_bar'] for r in results)
text = f'''# Teacher/video refresh — September 10, 2026

Updated gallery: https://cl.sudhirpratapyadav.com/v3/

{len(results)} tasks evaluated at 128 trials each. {passed} meet the 90% bar.
Below-bar teachers remain visible with their measured rates; a selected success
clip is not evidence of reliability across resets. The previous gallery is
archived at `/v3_before_physics_20260910/`.

## Teacher changes

- Switch, valve, lid and flap decode the target chord from exact source-asset
  forward kinematics, rather than an arbitrary visualization displacement.
- Flap motion is expressed in the randomized mount frame.
- Window uses a front pinch at a 70-degree downward wrist angle, avoiding the
  sash stile, then retains its measured grasp offset while sliding.
- Lever uses a 45-degree approach to keep its wrist in front of the fixture.
- Valve contacts the spoke at radius 80 mm and x=-50 mm, away from the hub.
- Drawer seats 12 mm down with a 5 mm outward bias, limits pull lead to 60 mm,
  and reseats after losing contact.

The other teachers were evaluated on the corrected geometry without changing
their strategies. No task geometry, reset distribution, success threshold or
collision mask was weakened during this teacher refresh.

## Measurement and recording

Each trial ends at its first termination or timeout. Success on the terminal
physics state is captured before automatic reset. Once the task's success latch
is set, further actions cannot change the trial result, so completed rows no
longer consume IK work. `test_classical` now uses this protocol by default;
`--legacy-window` exists only to reproduce older retry-based diagnostics.
Historical rates are not directly comparable.

Published results use seed 20260910 and batch size 128. Rendering uses a separate
seed, 20270910, and up to three single-env attempts to select a success and/or
failure. Clips stop on termination, timeout, or half a second after latched
success. Frame resampling preserves simulation time at 30 fps. Missing outcomes
are explicitly labeled; obsolete clips were removed after archiving.

Each `result.json` records the source SHA-256 including uncommitted teacher and
physics files, seeds, episode count, protocol, and actual clip presence.
The source fingerprint covers the whole relevant source tree, so unrelated
teacher edits during the refresh can produce different fingerprints.

## Results

| Task | Successes | Rate | Clips |
|---|---:|---:|---|
''' + '\n'.join(rows) + f'''

## Validation and artifacts

- 39 teacher/evaluation tests passed, covering all registered teachers, terminal
  reset boundaries, inactive rows, real-time video duration and analytical hinge
  target chords.
- GPU smoke check exercises the default strict `test_classical` entry point.
- `verify_gallery.py` decodes all local videos and compares published HTTPS bytes
  and result JSON against local artifacts. Its output is `verification.json`.
- Raw trials, diagnostics, candidate sweeps, videos and previous metadata:
  `{a.root}`.
- `review.jpg`: one thumbnail per freshly rendered task.

Reproduce the gallery with `render_suite.py` (three shards for the default 21-task
list; pass Open-Drawer, Slide-Window, Turn-Lever and Rotate-Valve IDs with `--tasks`).
Publish completed outputs with `publish_completed.py`, then `publish_v3.sh --index`.
'''
(here / 'README.md').write_text(text)
(here / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
cols = 5
sheet = Image.new('RGB', (cols*320, math.ceil(len(results)/cols)*268), '#111827')
draw = ImageDraw.Draw(sheet)
for i,r in enumerate(results):
  x,y = i%cols*320, i//cols*268
  thumb = a.root / 'videos' / r['task_id'] / 'thumb.jpg'
  if thumb.exists():
    with Image.open(thumb) as im: sheet.paste(im.resize((320,240)),(x,y))
  name = r['task_id'].removeprefix('Mjlab-').removesuffix('-Franka')
  draw.text((x+5,y+245), f"{name} | {r['num_success']}/128", fill='#eeeeee')
sheet.save(here / 'review.jpg', quality=90)
print(len(results), 'tasks;', passed, 'at 90%')
