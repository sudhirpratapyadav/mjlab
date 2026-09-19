# Status — CL-V5 SI-15

**6 full training runs in progress** (started 2026-09-18T10:33 UTC). No results
finalized yet.

## Readiness

| Check | Status | Evidence / action |
|---|---|---|
| Worktree | DONE | `../mjlab-cl15-si-20260918`, branch `exp/cl15-si-20260918`, from `checkpoint/cl24-rl-handoff-20260912` |
| Task selection (15, >95% val SR) | DONE | `active_tasks.json` |
| Method decision | DONE | SI only (user decision) |
| W&B destination | LIVE | https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-cl15-si-20260918 |
| Fresh teacher datasets | DONE | all 15 extracted 2026-09-18 from CL-V4 certified checkpoints; `teacher_datasets/`, config at `config/tasks_cl_v5_si15.yaml` |
| Capacity/LR/SI-coefficient bracket at N=15 | SKIPPED for this launch | reused N=4/N=6 optimum (width 4096, lr 3e-5, si-coeff 1.0) directly; validated the *pipeline* with a 2-task/5-epoch smoke run instead (`results_smoke/`, wandb run `smoke-si-2task-5ep`) rather than a full new parameter sweep — a re-bracket can still be run later if the full results look off |
| Orderings chosen | DONE | fragile-first (hardest CL-V4 teacher first) and one fixed random shuffle (seed 20260918); see EXPERIMENTS.md |
| Full runs launched | RUNNING | 6 runs = 2 orderings x 3 seeds (0,1,2), GPUs 1-6, `launch_full_runs.sh`, logs in `run_logs/` |
| Website page | LIVE | https://cl.sudhirpratapyadav.com/v5-cl-si/ — Wave 1 published, all 15 tasks, see PUBLICATION.md |

## Current runs

| run | ordering | seed | GPU | wandb |
|---|---|---|---|---|
| cl15-si-ff-s0 | fragile-first | 0 | 1 | euly61e5 |
| cl15-si-ff-s1 | fragile-first | 1 | 2 | dmc23doh |
| cl15-si-ff-s2 | fragile-first | 2 | 3 | 2fy1qf8v |
| cl15-si-rnd-s0 | random | 0 | 4 | xvgrgkdx |
| cl15-si-rnd-s1 | random | 1 | 5 | zsj0maz9 |
| cl15-si-rnd-s2 | random | 2 | 6 | y9h9lrd8 |

Project: https://wandb.ai/sudhirpratapyadav-indian-institute-of-technology-jodhpur/mjlab-cl15-si-20260918

Throughput check at 10:37 UTC (4 min in): fragile-first runs already at epoch
100/500 on task 0 (DragPull); no tracebacks on any of the 6. Rough estimate
~1.5s/epoch early on → ~3-5h per run for the full 15x500-epoch sequence,
comfortably inside the holder's 2026-09-28 expiry. Will refine as more tasks
complete (later tasks have longer episodes, e.g. Reach-Target at 1000 steps,
so per-task eval cost varies).

## Public summary

None yet. Will link here once `build_gallery.py`-equivalent output is published.
