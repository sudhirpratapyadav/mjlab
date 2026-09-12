# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Best current unfinished validations include Cage97/128, Place96/128 and Strike20/128. CageR8/model12600 latest90/128; full budget continues. Latest completed LiftR9/final10496, ThrowR3/final7497 and PivotR2/final3499 each0/128; actual failure clips reviewed. No confirmation for failed validations. GPU0 remains unused.

Current full PPO runs: EdgeR3 GPU1, ThrowR4 GPU2, StrikeR3 GPU3, LiftR10 GPU4, PlaceR7 GPU5, PegR6 GPU6 and CageR8 GPU7. All seven process placements verified; GPU0 remains0MiB. Lift/Throw preflights passed before launch. Peg3900 strict0/128 with narrower opening but no capture; full R6 budget continues. Place6329/lane995 failure was traced to an incorrect small-denominator guard in the installed dense elliptic-contact Hessian. An opt-in normalized outer-product implementation passes the original2048-world warmstart replay,24 focused checks,64×3 PPO preflight and2048×100 stress. PlaceR7 restores original6300 with this recorded correction and inherited Adam5e-5. Native model, forces, predicates,60D/8D and integration settings remain fixed; every new compatibility lineage requires its own strict evaluation.

## Acceptance

Deterministic policy means,60D observations and normalized8D absolute joint targets. Use native terminal success in the first uninterrupted episode. Validation seed20260914 and confirmation seed20260915,128 episodes each, both strictly>90% (116+). Review actual recorded success/failure videos, then retain policy/optimizer/normalizers/configs, hashes and W&B artifacts. Restore each checkpoint's recorded backend compatibility setting. Never count a training contact fraction or a selected successful clip as a measured success rate.

## Active work

- GPU1: Edge R3 edge_v3,2048 ×2000 from4998 unchanged. Continue measured side-approach improvement; strict evaluation at budget.
- GPU2: Throw R4 throw_v4,2048 ×2000 from7497, gripper mean-0.4/std0.3 initialization after finite preflight. Test release exploration with retained transport policy.
- GPU3: Strike R3 strike_v2,2048 ×2000 from4998, verified Adam5e-5. Refine launch endpoint while retaining learned contact.
- GPU4: Lift R10 lift_v7,2048 ×1500 from10496, cone compatibility after finite preflight. Test stable contact arithmetic with retained policy and no new initialization reset.
- GPU5: Place R7 completion_v6,2048 ×2000 from6300, cone compatibility and inherited Adam5e-5. Improve96/128 containment after diagnosed numerical failure.
- GPU6: Peg R6 peg_v1,2048 ×2000 from3098, recorded gripper mean-0.25/std0.15 initialization and gyro compatibility.3900 strict0/128, smaller opening but no capture; finish bounded budget before next width/approach diagnosis.
- GPU7: Cage R8 cage_v4,2048 ×2000 from11096 unchanged. Latest12600 strict90/128; best11096 remains97/128. Finish budget with native no-pinch history intact.

Stack, Reorient and Pivot need further trajectory-based diagnosis. Stack has opposing capture but stays low beside its base. Reorient remains near the floor after its gyro-enabled budget. Pivot R2 learns wide opening/contact but no tipping or opposed capture. Prioritize the next completed checkpoints, confirm only qualifying validation rates, and use recorded state evidence before shaping changes. Never loosen predicates or import scripted policy actions.

## Numerical readiness

Selective episode-reset warmstart clearing and optional standalone free-body gyro compatibility are retained. New dense elliptic Hessian compatibility corrects a small-T denominator defect; the original2048-world Place failure passes with its warmstart retained,24 focused checks pass, and100 full-size PPO stress updates are finite. Only PlaceR7/LiftR10 opt in; all14 certified configurations preserve their saved backend. Persist and restore both compatibility fields in training/evaluation/certification. No shared-venv edits, physical state clipping, dropped invalid lanes, new observations, or altered success tests. Keep8-state captures and diagnose any new failure independently.

## Records and resources

Keep STATUS/EXPERIMENTS/LOGS, current training_wave inventory and scoped W&B records current. Publish reviewed policy summaries/videos to /v4-rl/ using PUBLICATION.md. GPU0 stays free; verify real placement, holder20277 expiry2026-09-28T08:58:54 and available disk before resource reuse. Stop only verified own steps. Full24-teacher goal remains active until every teacher is independently certified.
