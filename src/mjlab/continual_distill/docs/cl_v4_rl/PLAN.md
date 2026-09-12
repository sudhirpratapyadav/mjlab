# Plan — certify all24 independent RL teachers

## Current state

**13/24 independent PPO RL teachers certified:** Reach, Topple, Button, Drawer, Drag, Flip, Door, Push-Cuboid, Flap, Lever, Window, Axial and Valve. Valve model1999 measured127/128 validation +128/128 confirmation; actual rotation and its ground-collision failure were reviewed; retained artifact uploaded.

Active: Stack GPU3 step1704, Peg GPU4 step1714, Cage retry GPU5 step1716, Lid retry GPU6 step1717, Place GPU7 step1705. Reorient R4 stopped on nonfinite simulator state at iteration1970; policy parameters finite, model1900 under diagnostic evaluation onGPU2. Lift final V3 strict0/128 with no sampled two-pad grasp; closure/exploration retry is next. Edge/Pivot/Strike/Throw still need first-pilot readiness. GPU0 remains unused. Full24-task goal is active.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: prepare Lift closure/exploration retry after final V3 strict0/128.
- GPU2: diagnose finite Reorient model1900 following simulator numerical failure.
- GPU3: Stack completion_v2,2048 environments ×2500 updates.
- GPU4: Peg completion_v2,2048 ×3000 updates.
- GPU5: Cage cage_v2,1024 ×2000 additional updates from own1499.
- GPU6: Lid lid_v1,1024 ×1500 additional updates from own1499.
- GPU7: Place completion_v2,2048 ×2500 updates.

Inspect meaningful learning progress and numerical diagnostics; evaluate saved candidates when evidence warrants it. Check actual checkpoints and process state before invoking evaluation or reusing a GPU. Stop only this experiment's exact Slurm step after checkpoint retention, or when diagnosing a verified failure. Never cancel holder20277 or another person's jobs.

## Remaining readiness work

- Stack/Place/Peg: existing dense goal return does not sufficiently favor release and settled support/containment/seating. Add a grasp/transport stage and native-completion incentive. Peg's approach must target a safe upper grasp point rather than its bottom insertion tip. Preserve the full square-bore fit and uprightness checks.
- Edge/Pivot: provide useful shaping for edge exposure or actual wall-assisted pivot, followed by grasp/lift. Pivot's required contact/tilt history remains part of the true success rule.
- Strike/Throw: tune velocity penalties for intentional fast motion and reward strike or launch/release progress, with final native success unchanged. Test compact-state limitations; do not add hidden observations or restore143D.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
