# Experiments — CL-V5 SI-15

Format follows `docs/P1_EXPERIMENTS.md` (run-name pattern, wandb project,
headline table).

- **wandb project:** `mjlab-cl15-si-20260918` (entity
  `sudhirpratapyadav-indian-institute-of-technology-jodhpur`)
- **Cluster:** dgx1, holder `20277` (`hold_dgx_amit`), GPUs 1-6 in use, GPU0/7 free

## Block 1 — dataset extraction (DONE, 2026-09-18)

15/15 succeeded. `extract_teacher_dataset.py`, 150k samples / 512 envs / seed 0,
from the exact certified CL-V4 checkpoint per task (`active_tasks.json`). One
CPU-pinning bug hit and fixed (arbitrary `taskset` ranges outside the cgroup's
actual `0-3,128-131` allocation crashed 13/15 jobs instantly; fix: drop
`taskset`, let the 8-cpu cgroup schedule freely). Logs: `extract_logs/`.

Note: the extraction script's own "TEACHER SUCCESS RATE" printout (stochastic
action sampling, 1-2 episodes x 512 envs) reads lower than CL-V4's certified
deterministic rate for several tasks (e.g. OpenDrawer 0.745 vs certified 1.000,
ThrowToBin 0.670 vs 0.977). This is expected — extraction samples from the
policy distribution for BC diversity rather than taking the deterministic mean
action — and is not a re-certification of the teachers; CL-V4's `scoreboard.json`
remains the authoritative teacher-competence record.

## Block 2 — pipeline smoke test (DONE, 2026-09-18)

2 tasks (ReachTarget, AxialExtract), 5 epochs each, SI, width 4096, lr 3e-5.
Wandb run `smoke-si-2task-5ep`. Hit and fixed one bug: `wandb_config.configure()`
run as a separate `python -c` subprocess doesn't export env vars back to the
parent shell, so the real training process fell back to an ambient, wrong wandb
login (`domimagi`) and got a 403 on upload. Fix: export `WANDB_API_KEY` /
`WANDB_ENTITY` / `WANDB_MODE` / `WANDB_CONFIG_DIR` / `NETRC` directly in the
same shell invocation as `continual_distill.py`. After the fix: checkpoints
saved, eval rollouts ran (student success 1.00 on ReachTarget, 0.75 on
AxialExtract at epoch 5 — expected, undertrained), wandb upload succeeded.

Used this to validate correctness, not to re-sweep capacity/LR/SI-coeff — see
STATUS.md for why the N=4/N=6 optimum (4096/3e-5/c=1.0) was reused directly for
the full launch rather than re-bracketed at N=15 first.

## Block 3 — full sequences (RUNNING, started 2026-09-18T10:33 UTC)

6 runs = 2 orderings x 3 seeds, SI only, width `[4096,2048,1024]`, lr 3e-5,
si-coeff 1.0, 500 epochs/task, default eval/checkpoint cadence
(`eval-every=20`, `env-eval-every=100`, `checkpoint-every=100`).

**Fragile-first ordering** (hardest CL-V4 teacher first, by validation SR,
extraction-time stochastic SR as tiebreak for the ten 100%-val tasks):
DragPull → ToppleBlock → ThrowToBin → PushCuboid → RotateValve → OpenDrawer →
FlipSwitch → AxialExtract → OpenDoor → PushButton → OpenLid → SlideWindow →
TurnLever → PushFlap → ReachTarget

**Random ordering** (Python `random.seed(20260918)` shuffle, fixed/reproducible):
ToppleBlock → RotateValve → OpenDrawer → FlipSwitch → ThrowToBin → ReachTarget →
SlideWindow → PushButton → PushFlap → PushCuboid → AxialExtract → TurnLever →
OpenDoor → OpenLid → DragPull

| run | seed | GPU | wandb run id |
|---|---|---|---|
| cl15-si-ff-s0 | 0 | 1 | euly61e5 |
| cl15-si-ff-s1 | 1 | 2 | dmc23doh |
| cl15-si-ff-s2 | 2 | 3 | 2fy1qf8v |
| cl15-si-rnd-s0 | 0 | 4 | xvgrgkdx |
| cl15-si-rnd-s1 | 1 | 5 | zsj0maz9 |
| cl15-si-rnd-s2 | 2 | 6 | y9h9lrd8 |

## Planned next

| block | purpose |
|---|---|
| Block 4 — final offline+env eval per run | already built into `continual_distill.py`'s end-of-sequence pass; read off final per-task retention + average from each of the 6 wandb runs once done |
| Block 5 — scalability point | add N=15 final-average point to the existing N=4 (0.960) -> N=6 (0.792) curve |
| Block 6 — video rendering | render the final retained student on each of the 15 tasks (reuse/adapt CL-V4's `render_evaluation.py`/`render_rollout.py` pattern for the JAX student policy format) |
| Block 7 — publish | build gallery, rsync to `cl.sudhirpratapyadav.com/v5-cl-si/`, same authorization pattern as CL-V4's `PUBLICATION.md` |
| (conditional) re-bracket | only if Block 3's results look anomalous (e.g. much worse than N=6 extrapolation, or a width/LR instability) — capacity/LR/SI-coeff sweep at N=15 per PLAN.md P1, not run pre-emptively |

## Block 3.5 — quick hyperparameter bracket (RUNNING, started 2026-09-18T12:10 UTC)

User authorized (2026-09-18) tuning LR, SI coefficient, batch size, and student
width up to 16384 — architecture changes are out of scope for this phase.
Running on the otherwise-idle GPU7, in parallel with Block 3's 6 main runs, so
there's signal ready for a possible follow-up launch once those finish.

2 tasks (DragPull, ToppleBlock — the two hardest by CL-V4 val SR), 100
epochs/task each, sequential, 8 points against the width-4096/lr-3e-5/coeff-1.0
default: `width8192`, `width16384`, `lr1e-4`, `lr1e-5`, `si0.3`, `si3.0`,
`batch256`, `batch1024`. Script: `bracket_gpu7.sh`, logs in `bracket_logs/`,
wandb run names `bracket-<name>` in the same project.

This is a cheap short-horizon signal (100 epochs, 2 tasks), not a replacement
for validating a chosen setting on the full 15-task sequence — any change this
suggests still needs a full confirmation run before being called final.

**Results so far** (final DragPull / ToppleBlock student_success at 100 epochs,
2-task sequence; baseline = width 4096, lr 3e-5, si-coeff 1.0, batch 512 — see
main runs, which hit ~93-98% / ~98-100% on these two tasks at equivalent points):

| point | DragPull | ToppleBlock | verdict |
|---|---|---|---|
| width8192 (lr unchanged, 3e-5) | 0.969-0.984 | 0.984 | on par with baseline, no clear win |
| width16384 (lr unchanged, 3e-5) | **0.000** | 0.672-0.734 | **collapsed** — same LR-width coupling problem as the original P0/P1 finding (8192 needed a lower LR than 3e-5); 16384 needs an even lower LR to be usable at all |

Reading so far: going wider than 4096 buys nothing at the current LR and risks
outright failure without also lowering LR — consistent with prior N=4/N=6
findings. No reason yet to move off width 4096 for the main runs. Still
waiting on lr1e-4/lr1e-5/si0.3/si3.0/batch256/batch1024.
