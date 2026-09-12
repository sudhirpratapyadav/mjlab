# Teacher/video refresh — September 10, 2026

**Historical measurement:** task semantics were corrected on September 11. See
[the task verification report](../../../../../../docs/task_geometry_review/TASK_VERIFICATION_RESULTS.md)
before using these rates with the current source. No replacement n=128 teacher
scoreboard is claimed by that audit.

Updated gallery: https://cl.sudhirpratapyadav.com/v3/

25 tasks evaluated at 128 trials each. 8 meet the 90% bar.
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
| Axial-Extract | 127/128 | 99.2% | success |
| Cage-Drag | 36/128 | 28.1% | failure |
| Drag-Pull | 24/128 | 18.8% | success, failure |
| Edge-Grasp | 34/128 | 26.6% | failure |
| Flip-Switch | 128/128 | 100.0% | success |
| Lift-Cube | 103/128 | 80.5% | success |
| Open-Door | 8/128 | 6.2% | failure |
| Open-Drawer | 79/128 | 61.7% | success, failure |
| Open-Lid | 124/128 | 96.9% | success |
| Peg-Insertion | 40/128 | 31.2% | success, failure |
| Pivot-Lift | 0/128 | 0.0% | failure |
| Place-In-Container | 121/128 | 94.5% | success |
| Push-Button | 120/128 | 93.8% | success |
| Push-Cuboid | 37/128 | 28.9% | success, failure |
| Push-Flap | 128/128 | 100.0% | success |
| Reach-Target | 128/128 | 100.0% | success |
| Reorient-Object | 95/128 | 74.2% | success |
| Rotate-Valve | 68/128 | 53.1% | success, failure |
| Slide-Window | 112/128 | 87.5% | success |
| Stack-Cube | 100/128 | 78.1% | success |
| Strike-Slide | 111/128 | 86.7% | success, failure |
| Throw-To-Bin | 65/128 | 50.8% | success, failure |
| Tool-Pull | 0/128 | 0.0% | failure |
| Topple-Block | 123/128 | 96.1% | success |
| Turn-Lever | 108/128 | 84.4% | success, failure |

## Validation and artifacts

- 39 teacher/evaluation tests passed, covering all registered teachers, terminal
  reset boundaries, inactive rows, real-time video duration and analytical hinge
  target chords.
- GPU smoke check exercises the default strict `test_classical` entry point.
- `verify_gallery.py` decodes all local videos and compares published HTTPS bytes
  and result JSON against local artifacts. Its output is `verification.json`.
- Raw trials, diagnostics, candidate sweeps, videos and previous metadata:
  `/ihub/homedirs/svs_ald/cl_v3_work/physics_refresh`.
- `review.jpg`: one thumbnail per freshly rendered task.

Reproduce the gallery with `render_suite.py` (three shards for the default 21-task
list; pass Open-Drawer, Slide-Window, Turn-Lever and Rotate-Valve IDs with `--tasks`).
Publish completed outputs with `publish_completed.py`, then `publish_v3.sh --index`.
