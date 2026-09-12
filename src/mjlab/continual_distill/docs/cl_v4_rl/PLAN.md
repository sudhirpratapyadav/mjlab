# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Current assignments: Cage R4 GPU1, Throw GPU2, Stack R4 GPU3, Lift R5 GPU4, Place R3 GPU5, Edge GPU6 and Reorient R7 GPU7. GPU0 remains unused. All24 active tasks have RL training evidence; coverage is not certification.

A captured Stack failure reproduced in1 and2048 GPU worlds and was fixed by clearing stale solver acceleration at episode reset. The production fix passed the captured transition and100 PPO updates; all24 tasks already start with zero cache, so fresh initialization is unchanged. Cage showed the same reset signature. Reorient has a distinct instability with extreme object velocities; its replay still fails after cache clearing. New captures retain8 preceding states and explicit nonfinite reward/observation lane IDs. Stack/Place use contact-closure shaping after measured open-enclosure failures. Strike/Peg/Pivot await targeted retries; full24-teacher goal active.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: Cage R4 cage_v3,1024 ×1000 additional updates from4600; reset fix and8-state capture.
- GPU2: Throw throw_v3,2048 ×3000 updates; older process, pre-step capture.
- GPU3: Stack R4 completion_v5,2048 ×1800 additional from1400, gripper std0.15; reset fix and8-state capture.
- GPU4: Lift R5 lift_v5,2048 ×2000 additional from diagnosticreplay2028; older process, pre-step capture.
- GPU5: Place R3 completion_v5,2048 ×2000 additional from1500, gripper std0.15; reset fix, one-state capture (launched before history extension).
- GPU6: Edge edge_v2,2048 ×3000 updates; older process.
- GPU7: Reorient R7 reorient_v6,2048 ×1500 additional from4600; reset fix and8-state capture, preserve output/std.

Strike/Peg/Pivot need targeted retries. Strike final9/128 mostly undershoots; Peg/Pivot press objects down. Actual CPU contact-normal audit rejects the observed top pressing; no evidence yet warrants replacing the training grasp classifier. Inspect wrist/ramp alignment and precursor rewards before the next Pivot trial. Stack/Place now test contact closure: desired gap shifts from width+3mm to width-4mm at zero approach distance, with positive clearance retained far away. This is reward shaping; contact stiffness and native success remain fixed.

The stale reset acceleration issue is reproduced and fixed. All24 fresh environments have zero initial cache. Existing long-running processes keep their loaded old source; use current reset fix when they next resume. Reorient's captured lane853 already has extreme object velocities and fails even with zero warm start;8-state history is enabled to locate the earlier onset. Do not label that distinct instability resolved, discard invalid lanes, or change physics silently.

Strict candidate evaluations this wave: Pivot1500 and StackR3/1400 each0/128, actual failure videos reviewed. The latter has100% final enclosure but0% contact. Captured-transition regression and100 PPO updates validate reset bookkeeping, not teacher success.

## Remaining readiness work

Edge/Pivot/Strike/Throw now have explicit recipes;70 CPU tests and all four32-env ×100-step physical preflights passed. Their short guarded PPO preflights also passed; full first pilots have all launched. Edge shapes near-rim exposure then side grasp; Pivot shapes ramp approach and genuine wall-contact tilt; Strike predicts sliding endpoint using registered friction; Throw predicts the descending rim-plane crossing. These are training rewards only: native predicates, geometry, initialization, horizons and60D/8D stay fixed.

Monitor actual grasp/contact diagnostics for the completion tasks. Reward or return increases alone do not demonstrate released/settled completion. Finite source checks do not prove learnability; use pilots and strict saved-checkpoint evaluation.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.
