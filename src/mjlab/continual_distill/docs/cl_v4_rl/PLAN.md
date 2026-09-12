# Plan — certify all24 independent RL teachers

## Current state

Ten teachers are certified; seven are training with pre-recorded budgets on GPUs1–7. Seven tasks still need first-pilot readiness. See STATUS.md and EXPERIMENTS.md for the authoritative per-task state. GPU0 must remain unused. The full24-task target is unchanged.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: fresh bounded-policy Lift V3,2,048 environments ×2,000 updates.
- GPU2: fresh bounded-policy Reorient V3,2,048 ×2,000, end-face/grasp/lift/orientation reward.
- GPU3: Axial-Extract mechanism pilot,1,024 ×1,500.
- GPU4: Open-Lid mechanism pilot,1,024 ×1,500.
- GPU5: Cage-Drag first pilot,1,024 ×1,500, open-gripper initialization and caged-transport reward.
- GPU6: Rotate-Valve mechanism pilot,1,024 ×2,000.
- GPU7: Slide-Window mechanism pilot,1,024 ×1,500.

Inspect meaningful learning progress and numerical diagnostics; evaluate saved candidates when evidence warrants it. Check actual checkpoints and process state before invoking evaluation or reusing a GPU. Stop only this experiment's exact Slurm step after checkpoint retention, or when diagnosing a verified failure. Never cancel holder20277 or another person's jobs.

## Remaining readiness work

- Stack/Place/Peg: existing dense goal return does not sufficiently favor release and settled support/containment/seating. Add a grasp/transport stage and native-completion incentive. Peg's approach must target a safe upper grasp point rather than its bottom insertion tip. Preserve the full square-bore fit and uprightness checks.
- Edge/Pivot: provide useful shaping for edge exposure or actual wall-assisted pivot, followed by grasp/lift. Pivot's required contact/tilt history remains part of the true success rule.
- Strike/Throw: tune velocity penalties for intentional fast motion and reward strike or launch/release progress, with final native success unchanged. Test compact-state limitations; do not add hidden observations or restore143D.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
