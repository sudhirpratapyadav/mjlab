# Plan — certify all24 independent RL teachers

## Current state

**13/24 independent PPO RL teachers certified.** Stack GPU3, Peg GPU4, Cage R2 GPU5, fresh bounded Lid V3 GPU6, Place GPU7 and Reorient R5 GPU2 are training. GPU1 handles remaining-task PPO preflights/diagnostics before the first Strike pilot. Lift R4 stopped on a simulator numerical failure; a30-update replay completed without reproducing it and retained model2028. Lift continuation awaits a slot with preceding-state capture enabled. Edge/Pivot/Throw passed physical readiness and await full pilots. GPU0 remains unused.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: finish remaining-task PPO checks, inspect Stack/Place/Peg debug checkpoints, launch Strike first pilot.
- GPU2: Reorient R5, reorient_v4,2048 ×2000 additional updates from1900, gripper std0.25.
- GPU3: Stack completion_v2,2048 ×2500 updates.
- GPU4: Peg completion_v2,2048 ×3000 updates.
- GPU5: Cage cage_v2,1024 ×2000 additional updates from1499.
- GPU6: fresh Lid lid_v2,2048 ×2000 updates. Prior unbounded R2 stopped after verified full saturation.
- GPU7: Place completion_v2,2048 ×2500 updates.

Lift R4 numerical failure did not recur in a30-update diagnostic replay. Preserve finite replaymodel2028; continue with pre-step capture when a slot is free. Do not mask invalid simulator state or change native physics silently.

Inspect meaningful learning progress and numerical diagnostics; evaluate saved candidates when evidence warrants it. Check actual checkpoints and process state before invoking evaluation or reusing a GPU. Stop only this experiment's exact Slurm step after checkpoint retention, or when diagnosing a verified failure. Never cancel holder20277 or another person's jobs.

## Remaining readiness work

Edge/Pivot/Strike/Throw now have explicit recipes;70 CPU tests and all four32-env ×100-step physical preflights passed. Complete short guarded PPO preflights before full pilots. Edge shapes near-rim exposure then side grasp; Pivot shapes ramp approach and genuine wall-contact tilt; Strike predicts sliding endpoint using registered friction; Throw predicts the descending rim-plane crossing. These are training rewards only: native predicates, geometry, initialization, horizons and60D/8D stay fixed.

Monitor actual grasp/contact diagnostics for the completion tasks. Reward or return increases alone do not demonstrate released/settled completion. Finite source checks do not prove learnability; use pilots and strict saved-checkpoint evaluation.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
