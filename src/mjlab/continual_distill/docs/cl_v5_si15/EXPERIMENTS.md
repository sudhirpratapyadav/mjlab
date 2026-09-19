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

## Block 3 results (ALL 6 RUNS COMPLETE, 2026-09-18 ~15:11 UTC)

No tracebacks; wall time 232-274 min/run. Final average success (15-task mean),
and per-run minimum (weakest task):

| run | final avg | weakest task (rate) |
|---|---|---|
| ff-s0 | 0.686 | AxialExtract (0.000) |
| ff-s1 | 0.725 | AxialExtract (0.047) |
| ff-s2 | 0.803 | PushCuboid (0.438) |
| rnd-s0 | 0.810 | PushCuboid (0.156) |
| rnd-s1 | 0.740 | AxialExtract (0.047) |
| rnd-s2 | 0.837 | PushCuboid (0.266) |

**fragile-first: mean 0.738 ± 0.049. random: mean 0.795 ± 0.041.**

**N=15 scalability point: ~0.74-0.80**, essentially flat vs. the N=6 point
(0.792) rather than continuing the steep N=4→N=6 drop (0.960→0.792). Better
retention at scale than a naive extrapolation would predict.

**Task-specific fragility, not just an ordering effect:** PushCuboid is the
single most consistently forgotten task — low in every one of the 6 runs
(0.156-0.547) regardless of its position in the sequence (4th in fragile-first,
10th in random). ThrowToBin and RotateValve are also consistently weak
(0.266-0.641 and 0.438-0.719 respectively). This echoes the P1-6 finding that
grasp/contact-heavy tasks (there, LiftCube) are intrinsically harder for the
student to retain, independent of when they're trained.

**Ordering signal that contradicts the N=4/N=6 "fragile-first wins" pattern:**
DragPull — the single hardest CL-V4 teacher (0.953 val SR) — was trained FIRST
in the fragile-first ordering and LAST in the random ordering. It actually
retained fine either way (0.906-0.953 first vs 0.828-0.938 last), so it's not
itself the deciding factor. But the *overall average* is higher for random
(0.795) than fragile-first (0.738), and AxialExtract's catastrophic failures
(0.000, 0.047) both happened in fragile-first runs, where it sits mid-sequence
(position 8 of 15) — i.e. trained neither first (protected the least by
recency) nor last (never protected by SI's recency bias). This suggests that
at N=15, being trained in the *middle* of a long sequence may be worse than
either extreme, unlike the simple "earlier is safer" story that held at N=6.
Worth testing directly with a **fragile-last** ordering (the reverse of
fragile-first) — see Block 4 below.

## Block 4 results — fragile-last follow-up (COMPLETE, 2026-09-19 ~12:53 UTC)

Ordering: ReachTarget → PushFlap → TurnLever → SlideWindow → OpenLid →
PushButton → OpenDoor → AxialExtract → FlipSwitch → OpenDrawer → RotateValve →
PushCuboid → ThrowToBin → ToppleBlock → DragPull (exact reverse of
fragile-first). 3 seeds, no tracebacks, ~175 min/run.

| run | final avg | weakest task (rate) |
|---|---|---|
| fl-s0 | 0.652 | PushButton (0.062) |
| fl-s1 | 0.718 | PushCuboid (0.109) |
| fl-s2 | 0.702 | PushCuboid (0.078) |

**fragile-last: mean 0.691 ± 0.028** — the *worst* of the three orderings,
not the best.

| ordering | mean | std |
|---|---|---|
| random | **0.795** | 0.041 |
| fragile-first | 0.738 | 0.049 |
| fragile-last | 0.691 | 0.028 |

**The mid-sequence hypothesis from Block 3 is not confirmed — it's
contradicted.** Putting the 5 fragile tasks last didn't help; it hurt overall,
because it pushed the 10 *easy* tasks (100% standalone teacher SR) to the
front of the sequence, where they now absorb the most forgetting pressure
instead. And several of those "easy" tasks turned out not to be easy to
*retain*: PushButton collapsed to 0.062-0.219 and FlipSwitch to 0.219-0.766
across the fl seeds, despite both having a perfect standalone teacher.
**Standalone teacher difficulty and retention difficulty are different
properties** — a task can be trivial to learn but fragile to retain against
future interference, and vice versa (DragPull, the single hardest teacher,
retained fine as either the first or the last task trained, in both ff and
rnd). Fragile-last's lowest std (0.028) also suggests it's a more
*consistent* ordering, just consistently worse.

**Best single run remains rnd-s2 (0.837)** — no change to the published
gallery's videos. Random ordering (mean 0.795, tightest spread among the two
better orderings) is the best default ordering strategy found so far at
N=15, ahead of the naive "train hard tasks first" intuition that held at
N=4/N=6.

## Planned next

| block | purpose |
|---|---|
| Block 4 — final offline+env eval per run | already built into `continual_distill.py`'s end-of-sequence pass; read off final per-task retention + average from each of the 6 wandb runs once done |
| Block 5 — scalability point | add N=15 final-average point to the existing N=4 (0.960) -> N=6 (0.792) curve |
| Block 6 — video rendering | **script ready** (`render_student.py`): loads a student checkpoint, deterministic rollout, headless EGL render, picks a success clip if any env succeeded. Test-rendered against an in-flight ff-s2 checkpoint (task 9/PushButton at epoch 100/500) — worked first try, 100% success over 4 envs, valid 646KB mp4. Ready to run once final per-run checkpoints exist. |
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

**All 8 points complete (2026-09-18T13:04 UTC), no tracebacks.** Full table
(final DragPull / ToppleBlock student_success):

| point | DragPull | ToppleBlock | vs. baseline |
|---|---|---|---|
| baseline (4096/3e-5/1.0/512) | 0.93-0.98 | 0.98-1.00 | — |
| width8192 | 0.97-0.98 | 0.98 | on par |
| width16384 | **0.00** | 0.67-0.73 | **collapsed** |
| lr1e-4 | 0.95-0.98 | 0.98 | on par |
| lr1e-5 | 0.97-0.98 | 0.98 | on par |
| si0.3 | 0.95-0.98 | 0.98 | on par |
| si3.0 | 0.94-1.00 | 0.98 | on par |
| batch256 | 0.97-0.97 | 0.98 | on par |
| batch1024 | 0.97-1.00 | 0.97-0.98 | on par |

**Conclusion: the baseline (width 4096, lr 3e-5, si-coeff 1.0, batch 512) is
confirmed as a solid, robust choice.** Every point tested within a sane range
lands in the same ~95-100% band on this short 2-task/100-epoch signal — nothing
tested clearly beats it. The only real finding is negative: width 16384 fails
outright at this LR (needs its own, lower LR to even train, same coupling
pattern as the original 8192-needs-1e-5 finding). No hyperparameter change is
being made to the 6 in-flight main runs on the strength of this bracket; it
serves as confirmation, not a reason to relaunch.
