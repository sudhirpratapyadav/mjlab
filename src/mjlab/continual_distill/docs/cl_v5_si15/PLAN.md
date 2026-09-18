# Plan — CL-V5: SI across 15 tasks

Method fixed by user decision: **SI only**, coefficient starting near c=1.0 (the
N=4/N=6 optimum from `P1_EXPERIMENTS.md`), re-bracketed at this scale before a
full launch. No EWC/L2 runs in this phase.

## P0: fresh teacher datasets

The old `teacher_datasets/` folder predates the CL-V4 placement fix and the newly
certified checkpoints for most of these 15 tasks — do not reuse it blindly.
Re-run `extract_teacher_dataset.py` against the actual retained checkpoint for
each task in `active_tasks.json` (`teacher_run` / `wandb_run` columns point at
the source). Gate each extracted dataset on a fresh competence check (>0.95,
128 episodes) before it's usable, same pattern as `verify_p1b_teachers.sh`.

## P1: re-bracket capacity, LR, and SI coefficient at N=15

Prior findings (N=4/N=6 only, not validated at N=15):
- Student trunk `[4096, 2048, 1024]` saturates capacity — 8192 no better.
- LR 3e-5 optimal for widths ≤4096, 1e-5 for 8192.
- SI c=1.0 was the coefficient used; no explicit sweep was published for it
  (unlike EWC's λ sweep) — worth a small bracket (e.g. 0.3 / 1.0 / 3.0) here
  given 15 tasks accumulate far more anchor terms than 4-6.

Run this bracket short (small epoch budget) on 2-3 tasks before committing to
full 15-task sequences.

## P2: orderings

Don't brute-force all 15! orderings. Run:
1. **Fragile-first** — rank the 15 by standalone teacher difficulty (lower val
   SR = more fragile) and train hardest-first, matching the "primacy" finding
   that fragile tasks trained early survive better.
2. **Random** — one arbitrary shuffle, as an unbiased baseline.
3. (Optional, if budget allows) **Fragile-last** — reverse of (1), as the worst-
   case stress test, matching the best/worst pattern from P0/P1.

3 seeds each.

## P3: full launches, tracked against the existing scalability curve

N=4 → 0.960 (SI, best ordering). N=6 → 0.792 (fragile-first). Expect further
decay at N=15; log the point on the same curve. Report per-task retention, not
just the average (P1-6's LiftCube collapse to 0.016 was invisible in the
average alone).

## P4: publish

Once real runs complete:
- Curves/metrics to the `mjlab-cl15-si-20260918` W&B project (see STATUS.md).
- A new gallery page at `https://cl.sudhirpratapyadav.com/v5-cl-si/`, built the
  same way as `docs/cl_v4_rl/build_gallery.py` (per-task video + retention
  numbers), authorized for publication the same way CL-V4's was
  (`docs/cl_v4_rl/PUBLICATION.md`). No placeholder/fake content goes on the
  live site before real results exist.

## Compute

Same shared holder **20277** (expires **2026-09-28**), GPUs 1-7 authorized,
GPU0 unused. Recheck actual placement before each launch; never cancel another
user's job. This is a separate worktree from the paused RL-teacher work
(`exp/rl-teachers-24-20260918`) — do not write into that worktree's `runs/`.
