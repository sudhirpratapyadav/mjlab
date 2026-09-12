# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Latest strict results: Cage R7/final11096 **97/128 (75.8%)**, Place R6/model6300 **96/128 (75.0%)**, Edge R2/final4998 and Peg R5/model3400 **0/128**. None crosses the116/128 validation gate; no confirmation batches were run for these candidates. GPU0 remains unused.

Current full PPO runs: Pivot R2 GPU1, Throw R3 GPU2, Lift R9 GPU4, Peg R6 GPU6 and Cage R8 GPU7. Edge/Cage completed their budgets and final evaluation. Place R6 stopped after iteration6329 with a captured nonfinite simulator state in lane995; latest retained6300 evaluated successfully. Preserve the failure trace and diagnose before resuming Place. Model6000 previously measured90/128. Peg R5 was deliberately stopped after the aperture fallback defect audit and a finite peg_v1 preflight; R6 restores original3098 with the preregistered gripper reset and inherited gyro compatibility. Lift R9 uses the preregistered gentler gripper initialization after matched-friction hold probes and finite PPO preflight.

## Acceptance

Deterministic policy means,60D observations and normalized8D absolute joint targets. Use native terminal success in the first uninterrupted episode. Validation seed20260914 and confirmation seed20260915,128 episodes each, both strictly>90% (116+). Review actual recorded success/failure videos, then retain policy/optimizer/normalizers/configs, hashes and W&B artifacts. Restore each checkpoint's recorded backend compatibility setting. Never count a training contact fraction or a selected successful clip as a measured success rate.

## Active work

- GPU1: Pivot R2 pivot_v3,2048 ×2000 from1500 after finite3-update preflight. Learn open-ramp tipping and opposed capture with floor-safe reward targets.
- GPU2: Throw R3 throw_v4,2048 ×2000 from5498 unchanged. Convert new capture/lift into launch; diagnose reward if holding persists at budget.
- GPU3: Edge R2 edge_v3,2048 ×2000 from2999 after finite3-update preflight. Learn side pinch and settled lift after exposure.
- GPU4: Lift R8 lift_v7,2048 ×1500 from7498 after finite3-update preflight. Learn gentle loaded grip and native settling; monitor new actual-speed diagnostics.
- GPU5: Place R6 completion_v6 from5399,2048 ×2000, inherited Adam LR5e-5. Improve native released containment from35/128.
- GPU6: Peg R4 completed3000 additional updates normally at3098; strict evaluation is running with its recorded optional gyro correction.
- GPU7: Cage R7 cage_v4 from9097,2048 ×2000 after finite10-update full-batch preflight. Improve37/128 without changing native no-pinch history.

Stack, Reorient and Strike remain targeted retries. Stack has acquired opposing capture but stays low beside its base. Reorient remains near the floor after its gyro-enabled budget. Strike improved9→20/128; quantify remaining launch coverage and speed/direction error before its next retry. Pivot ramp geometry preparation and PPO preflight passed; the retry is now live. Use actual trajectory evidence before changing shaping; do not loosen predicates or import scripted actions.

## Numerical readiness

Selective warm-start reset is retained. The separate captured high-spin failure now has an opt-in CPU-compatible free-body correction, sourcef821aa0, with recorded-state and controlled-spin regressions.100 full-size PPO updates passed for both affected tasks. Existing certified configurations remain unchanged. No shared-venv edits, state clipping, dropped invalid lanes, new observation fields or altered success tests. Continue8-state captures; a new failure requires diagnosis, not an assumption that the correction covers it.

## Records and resources

Keep STATUS/EXPERIMENTS/LOGS, current training_wave inventory and scoped W&B records current. Publish reviewed policy summaries/videos to /v4-rl/ using PUBLICATION.md. GPU0 stays free; verify real placement, holder20277 expiry2026-09-28T08:58:54 and available disk before resource reuse. Stop only verified own steps. Full24-teacher goal remains active until every teacher is independently certified.
