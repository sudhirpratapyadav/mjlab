# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Current training: Pivot R2 GPU1, Throw R3 GPU2, Edge R2 GPU3, Lift R8 GPU4, Place R6 GPU5, Cage R7 GPU7. Peg R4 completed its budget; strict final3098 evaluation is running onGPU6. GPU0 remains unused. All24 tasks have RL training evidence; coverage is not certification.

Latest strict validation: Cage R6/final9097 improved16→37/128; Lift R7/final7498, Stack R6/final5198, Reorient R9/final8598 and Throw R2/final5498 each0/128. All completed budgets normally and their actual videos were reviewed. Lift places111 terminal objects inside5cm but none both inside and settled; lift_v7 adds mild loaded-grip/settling credit after a finite PPO preflight. Throw now captures/lifts (94.49% final opposing contacts) but holds without release; its bounded unchanged continuation targets launch. Stack remains low beside the base; Reorient remains near the floor.

Strike's contact-height and Edge's full side-wrist reward changes passed finite online PPO preflights before full continuations. Cage's2048-world10-update preflight passed before doubling its rollout batch. Place improved0→35/128 at final5399 and continues unchanged with the previously verified5e-5 Adam learning rate. Optional free-body gyroscopic compatibility remains enabled only for the recorded Peg/Reorient lineages; all14 certified configurations retain their saved backend. Native predicates, model parameters and60D/8D remain fixed. Matched Lift hold interventions support a gentler-grip hypothesis but are not RL success measurements; original evaluation friction is not captured in old traces.

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
