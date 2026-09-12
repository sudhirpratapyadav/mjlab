# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Training: Throw GPU2, Stack R5 GPU3, Lift R6 GPU4, Place R4 GPU5 and Peg R3 GPU6. Cage R4 and Reorient R7 completed and were evaluated; GPUs1/7 are available after their evaluation steps exit. GPU0 remains unused. All24 active tasks have RL training evidence; coverage is not certification.

Edge final2999 and LiftR5/model3500 each scored0/128 with actual failure videos reviewed. A cube closure audit found64 opposing-contact cases rejected by whole-object enclosure; the production GPU query confirms64/64 opposing contacts and0/64 old enclosed grasps on those counterfactual states. New lift_v6/completion_v6 use opposing pad normals and a local centerline width for training; native success/physics/60D/8D stay unchanged. Lift/Peg/Stack/Place retries passed PPO preflights and are running. Other training recipes retain their versioned behavior. The reset-cache fix remains verified; Reorient's separate high-velocity instability remains unresolved, with8-state capture enabled. Strike/Edge/Pivot await targeted retries. Full24-teacher goal active.

## Fixed evaluation and retention gate

Use deterministic means under the approved60D observations and normalized8D absolute joint actions. Preserve registered geometry, initialization, physics, horizon and success predicates. Evaluate only each environment's first uninterrupted episode and capture its actual terminal predicate before automatic reset. Validation seed20260914 selects a checkpoint; independent confirmation seed20260915 is used only after validation passes. Both128-episode batches must exceed90% (at least116 successes each). Inspect success and highest-return failure videos, then retain the checkpoint, optimizer, normalizers, effective configs, provenance and results in W&B. Training success logs are not certification.

The evaluator has adversarial reset/retry tests. evaluate_candidate.py runs both batches conditionally and renders validation states; certification still requires an actual visual review. If only confirmation has failures, obtain a recorded failure clip before certification rather than silently omitting it.

## Active training and next evaluations

- GPU1: free after Cage/Lift/Throw strict evaluations; prepare a targeted Cage transport retry.
- GPU2: Throw throw_v3,2048 ×3000; model2500 strict0/128, evaluate final after exit.
- GPU3: Stack R5 completion_v6,2048 ×2000 additional from2200/std0.15;8-state capture.
- GPU4: Lift R6 lift_v6,2048 ×2000 additional from3500/std0.15; real grasping acquired, model4100 strict0/128, monitor height/goal progress.
- GPU5: Place R4 completion_v6,2048 ×2000 additional from2300/std0.15;8-state capture.
- GPU6: Peg R3 completion_v6,2048 ×2500 additional from1500, mean0.5/std0.15;8-state capture.
- GPU7: Reorient R7 final6099 strict0/128, reviewed; prepare opposing-contact/local-width retry after task-specific preflight.

Strike/Edge/Pivot need targeted retries. Strike final9/128 mostly undershoots; Peg/Pivot press objects down. Actual CPU contact-normal audit rejects the observed top pressing. A separate closure counterfactual now proves that the full-box enclosure also rejects valid opposing contacts; Lift/Peg trials replace that training gate with real opposing pad normals. Inspect wrist/ramp alignment and precursor rewards before the next Pivot trial. Stack/Place now test contact closure: desired gap shifts from width+3mm to width-4mm at zero approach distance, with positive clearance retained far away. This is reward shaping; contact stiffness and native success remain fixed.

New contact_geometry_witness_preflight.json confirms64/64 GPU opposing contacts and0/64 old enclosed grasps for the audited closure states. These are geometry witnesses, not successful RL episodes.95 tests plus Lift/Peg3-update PPO checks passed. Monitor actual opposed-contact and lift fractions before strict evaluation.

The stale reset acceleration issue is reproduced and fixed. All24 fresh environments have zero initial cache. Existing long-running processes keep their loaded old source; use current reset fix when they next resume. Reorient's captured lane853 already has extreme object velocities and fails even with zero warm start;8-state history is enabled to locate the earlier onset. Do not label that distinct instability resolved, discard invalid lanes, or change physics silently.

Strict candidate evaluations this wave: Pivot1500 and StackR3/1400 each0/128, actual failure videos reviewed. The latter has100% final enclosure but0% contact. Captured-transition regression and100 PPO updates validate reset bookkeeping, not teacher success.

## Remaining readiness work

Edge/Pivot/Strike/Throw now have explicit recipes;70 CPU tests and all four32-env ×100-step physical preflights passed. Their short guarded PPO preflights also passed; full first pilots have all launched. Edge shapes near-rim exposure then side grasp; Pivot shapes ramp approach and genuine wall-contact tilt; Strike predicts sliding endpoint using registered friction; Throw predicts the descending rim-plane crossing. These are training rewards only: native predicates, geometry, initialization, horizons and60D/8D stay fixed.

Monitor actual grasp/contact diagnostics for the completion tasks. Reward or return increases alone do not demonstrate released/settled completion. Finite source checks do not prove learnability; use pilots and strict saved-checkpoint evaluation.

Record each new recipe and budget before launch. Check recipe invariants and short finite-state rollouts, then real PPO/checkpoint reload where architecture changes. All policy outputs remain eight learnable normalized action dimensions; Cage initialization is a prior, not a gripper action mask.

## Environment and launch

Work only in /ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex, branch exp/rl-teachers-24-codex, based on4149157. Reuse the shared interpreter without reinstalling packages. Inside every Slurm step explicitly cd to this worktree, export PYTHONPATH="$PWD/src", export the assigned GPU UUID from allocation.py, and set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FORCE_CPU=0. Invoke /ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python. Verify imports resolve into this worktree.

The stage launcher uses the verified experiment-scoped W&B entity/project. Never print credentials or call global wandb login. Holder20277 expires2026-09-28T08:58:54 scheduler time; recheck holder, GPU process placement and disk before long runs. Runs are local to this worktree; committed JSON evidence and W&B artifacts carry reviewable results.

Keep the public /v4-rl/ summary current after reviewed evaluations/certifications; use PUBLICATION.md. Do not publish a training diagnostic as a success rate.
