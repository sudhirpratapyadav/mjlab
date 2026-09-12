# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Current assignments: Cage R3 GPU1, Throw GPU2, Stack R3 GPU3, Lift R5 GPU4, Pivot GPU5, Edge GPU6 and Reorient R6 GPU7. GPU0 remains unused. All24 active tasks have RL training evidence; this is coverage, not certification.

Strike finished3000 updates: final2999 strict9/128, reviewed actual strike/slide trajectories. Place/Peg V2 model1500 each strict0/128: Place encloses without opposed contact, while Peg topples the shaft and presses it down. Their own steps were stopped and checkpoints retained; these three tasks await targeted retries. Cage contact-target retry and Lift/Reorient capture-width retries are now launched. The latter preserve learned arm outputs and reset only gripper mean0.5/std0.15. Preceding-state capture is enabled on all three and on Stack/Throw; numerical causes from earlier failures remain unresolved. Full24-teacher goal active.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: Cage R3 cage_v3,1024 ×2000 additional updates from3498; pre-step capture.
- GPU2: Throw throw_v3,2048 ×3000 updates; pre-step capture.
- GPU3: Stack R3 completion_v4,2048 ×1800 additional from700; gripper mean0.5/std0.15; pre-step capture.
- GPU4: Lift R5 lift_v5,2048 ×2000 additional from finite diagnosticreplay2028; gripper mean0.5/std0.15; pre-step capture.
- GPU5: Pivot pivot_v2,2048 ×3000 updates.
- GPU6: Edge edge_v2,2048 ×3000 updates.
- GPU7: Reorient R6 reorient_v6,2048 ×2000 additional from3899; gripper mean0.5/std0.15; pre-step capture.

Strike/Place/Peg need targeted retries after strict final/candidate failures. Strike achieved9/128 with110/128 undershooting the goal by more than5cm longitudinally; inspect missed contact and launch consistency before simply extending. Place is now geometrically enclosed without contact; width shaping is a candidate, but monitor Stack's ongoing learning first. Peg topples before capture; preserve native geometry/predicate and improve training grasp alignment/clearance. Do not mask invalid simulator state or silently change physics.

Two resumed64-env3-update Lift/Reorient preflights passed and synced online (capture_resume_preflight.json). Width shaping and gripper-only resets preserve the seven learned arm outputs. Current runs remain subject to strict deterministic first-episode validation and actual-state video review.

Stack V2 debug confirms open enclosure without contact; numerical failure at777 affected one lane and left policy parameters finite. R3 retains approach while improving capture.88 focused tests passed, with a finite resumed PPO preflight and verified preservation of arm outputs and unrelated state.

Inspect meaningful learning progress and numerical diagnostics; evaluate saved candidates when evidence warrants it. Check actual checkpoints and process state before invoking evaluation or reusing a GPU. Stop only this experiment's exact Slurm step after checkpoint retention, or when diagnosing a verified failure. Never cancel holder20277 or another person's jobs.

## Remaining readiness work

Edge/Pivot/Strike/Throw now have explicit recipes;70 CPU tests and all four32-env ×100-step physical preflights passed. Their short guarded PPO preflights also passed; full first pilots have all launched. Edge shapes near-rim exposure then side grasp; Pivot shapes ramp approach and genuine wall-contact tilt; Strike predicts sliding endpoint using registered friction; Throw predicts the descending rim-plane crossing. These are training rewards only: native predicates, geometry, initialization, horizons and60D/8D stay fixed.

Monitor actual grasp/contact diagnostics for the completion tasks. Reward or return increases alone do not demonstrate released/settled completion. Finite source checks do not prove learnability; use pilots and strict saved-checkpoint evaluation.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
