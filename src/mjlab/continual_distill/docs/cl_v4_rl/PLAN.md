# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Current assignments: Strike GPU1, Throw GPU2, Stack R3 GPU3, Peg V2 GPU4, Pivot GPU5, Edge GPU6 and Place V2 GPU7. GPU0 remains unused. All24 active tasks now have RL training evidence; this is coverage, not certification.

Lift, Reorient and Cage need successful retries and await slots. Cage R2 final0/128: valid open enclosure without contact; cage_v3 contact-target preflight passed. Reorient R5 final0/128: saturated closed fingers without capture; width-matched rewards and gripper-output reset are available. Stack V2 stopped on a numerical failure in one lane; R3 retains finite arm control, resets only the gripper output/exploration and enables preceding-state capture. The numerical cause remains unresolved. Full24-teacher goal active.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: Strike strike_v1,2048 ×3000 updates.
- GPU2: Throw throw_v3,2048 ×3000 updates; pre-step capture enabled.
- GPU3: Stack R3 completion_v4,2048 ×1800 additional updates from ownV2/model700; gripper mean0.5/std0.15 reset; pre-step capture enabled.
- GPU4: Peg V2 completion_v3,2048 ×3000 updates.
- GPU5: Pivot pivot_v2,2048 ×3000 updates.
- GPU6: Edge edge_v2,2048 ×3000 updates.
- GPU7: Place V2 completion_v3,2048 ×2500 updates.

Lift/Reorient/Cage retries await slots. Prepared lift_v5/reorient_v6 width-matched capture and cage_v3 trailing-pad contact target. Explicit --resume-gripper-mean now resets only the eighth actor output and its Adam rows while preserving seven arm outputs; std reset is separate. Use when supported by observed saturation/capture failure and record it. Enable --capture-pre-step for new numerical-failure investigations. Do not mask invalid state or silently change native physics.

Stack V2 debug confirms open enclosure without contact; numerical failure at777 affected one lane and left policy parameters finite. R3 retains approach while improving capture.88 focused tests passed, with a finite resumed PPO preflight and verified preservation of arm outputs and unrelated state.

Inspect meaningful learning progress and numerical diagnostics; evaluate saved candidates when evidence warrants it. Check actual checkpoints and process state before invoking evaluation or reusing a GPU. Stop only this experiment's exact Slurm step after checkpoint retention, or when diagnosing a verified failure. Never cancel holder20277 or another person's jobs.

## Remaining readiness work

Edge/Pivot/Strike/Throw now have explicit recipes;70 CPU tests and all four32-env ×100-step physical preflights passed. Complete short guarded PPO preflights before full pilots. Edge shapes near-rim exposure then side grasp; Pivot shapes ramp approach and genuine wall-contact tilt; Strike predicts sliding endpoint using registered friction; Throw predicts the descending rim-plane crossing. These are training rewards only: native predicates, geometry, initialization, horizons and60D/8D stay fixed.

Monitor actual grasp/contact diagnostics for the completion tasks. Reward or return increases alone do not demonstrate released/settled completion. Finite source checks do not prove learnability; use pilots and strict saved-checkpoint evaluation.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
