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


### Wave22

**14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Latest completed CageR8/final13095 strict99/128 and StrikeR3/final6997 strict21/128, both small improvements and below the116/128 validation gate. Place retained6300 remains96/128. New Cage/Strike actual success and failure frames reviewed; no confirmation or new certification. GPU0 remains unused.

Current full PPO runs: EdgeR3 GPU1, ThrowR4 GPU2, LiftR10 GPU4, PlaceR7 GPU5 and PegR6 GPU6. GPUs3/7 are free after completed Strike/Cage runs and evaluations. Five process placements verified; GPU0 remains0MiB. PlaceR7 and LiftR10 use the recorded opt-in dense elliptic Hessian correction; PegR6 retains free-body gyro compatibility. All14 certified teachers retain their saved original backend. Holder20277 expires2026-09-28T08:58:54 scheduler time.

Pivot R2 contact audit: median closest reward waypoint6.45mm, wrist error8.74deg, no sampled tilt above20deg. All363 sampled closest robot/board contacts use right pad with mean normal[0.333,-0.004,-0.943], mostly downward. A bounded32-pose IK shift audit finds no inward/upward-normal solution at0–40mm; a separately stated tipping-moment criterion passes25/32 at5mm,1/32 at10mm,0otherwise. This is collision geometry only, without integration or RL success. No new reward, physical model, success predicate or action change. Preserve this distinction before a dynamics probe/new preflight.


### Wave23

Previous turn made progress: final Cage99/128 and Strike21/128 strict results and reviewed videos, Pivot geometric evidence and verified public/W&B publication. Five full trainers revalidated live; GPU0/3/7 at0MiB.

Preregister PREFLIGHT-CAGE-REFINE-037: ownR8/model13095, unchanged cage_v4,64 environments×3 PPO updates, explicit Adam5e-5, preserve optimizer moments/normalizers/policy/gripper and legacy backend. Cage improved97→99/128 over2000 updates but remains below116; a bounded smaller-step refinement tests endpoint consistency. If finite with correct saved LR, launch RL-013-R9-cage-refine from original13095,2048×2000 onGPU7, same settings. Native gate/60D/8D/model unchanged. No confirmation unless strict validation passes116/128.


StrikeR3 endpoint audit:113/128 substantial launches versus97 before; median final error15.82cm,57 undershoots>8cm,41 overshoots>8cm and20 lateral misses>8cm. Peak-speed Coulomb prediction differs from measured endpoint by1.84mm median among substantial launches; peak may still contact hand, so this is empirical support for shaping, not a model proof. The bottleneck is now endpoint calibration. Preregister strike_v3: preserve strike_v2 approach/broad rewards/native20 bonus and add4×(exp(-predicted_error/.05)+exp(-actual_error/.05)/(1+XYspeed/.05)). This rewards precise flight and a settled near-goal endpoint; no gate/model/action changes. PREFLIGHT-STRIKE-PRECISION-038: ownR3/6997,64×3, inherited5e-5 Adam/normalizers/means/std/backend. After finite preflight, RL-023-R4-strike-precision from original6997,2048×2000 onGPU3 after Place evaluator exits.

Cage preflight13097 finite with5e-5 confirmed; fullR9 launched original13095 onGPU7. PlaceR7/7400 strict105/128, below116; full budget continues. PegR6 completed normally at5097; strict evaluation started onGPU6. Pivot constant-control CPU probes preserve actual source states/friction; four initial modes and20 small XY/Z variations all32cases show no tilt>20deg, no warnings or deep obstacle penetration. No Pivot reward change is justified by these negative dynamic probes.


PegR6/final5097 strict0/128, actual low capture reviewed. At last diagnostic123 active episodes, opposing-contact fraction50.4% versus0% at3900, aperture21.82mm, tip height4.71mm. Preregister RL-020-R7-peg-lift: original5097, unchanged peg_v1,2048×2000, preserve saved Adam/normalizers/gripper distributions/LR and gyroON/coneOFF, GPU6 after evaluator exit. The full R6 budget was finite and now acquired physical capture; no new reward or initialization reset and no new preflight needed for this unchanged continuation.

Strike precision tests65 pass (plus10 affected tests after preserving the disabled default branch), and3-update PPO preflight6999 is finite with inherited5e-5. FullR4 launched from original6997 onGPU3.


LiftR10/final11995 strict0/128 (123timeouts,4ground,1bounds), actual near-goal hold reviewed. Exact terminal audit115 inside5cm,0settled; median error11.70mm, linear0.3187m/s, angular2.0946rad/s. The opt-in cone correction alone did not solve policy settling. Existing lift_v7 has only5×continuous quiet credit and position/grasp shaping, no explicit native completion bonus. Preregister lift_v8: retain all existing terms, increase near-goal quiet weight5→15 and add25×the unchanged refreshed native Lift predicate. No success/model/action/backend change. PREFLIGHT-LIFT-COMPLETION-039 from original11995,64×3, preserve Adam1e-4/normalizers/gripper/arm/noise, coneON/gyroOFF. After finite PPO gate launch RL-002-R11-lift-completion, original11995,2048×2000 onGPU4 after evaluator exits.


### Wave24

Previous turn made progress: Place105/128, actual Peg capture acquisition and Lift settling diagnosis, finite-preflighted Cage/Strike/Lift retries and unchanged Peg continuation; public/W&B records verified. Seven full trainers revalidated live, GPU0 at0MiB.

ReorientR9 actual trajectory/contact audit:12643 sampled states,6723 opposing-contact states within drift bound;6562 (97.6%) have nonpositive axis cosine and zero old orientation credit.124/128 terminal axes remain beyond90deg, median92.54deg; no sampled episode reaches45deg. This is a reward plateau, not a success-rate replacement. Preregister reorient_v8: preserve contact/approach/lift terms, replace max(dot,0)^2 with(1+clamp(dot,-1,1))/2 for the current directed axis (absolute cosine for symmetric variants), still gated by held+valid drift; add25×refreshed unchanged native success. No model/60D/8D/predicate change. PREFLIGHT-REORIENT-AXIS-040: originalR9/8598,64×3, preserve arm/gripper/Adam1e-4/normalizers/noise and gyroON/coneOFF. Tiny preflight may co-locate onGPU1 with its sole2048-world Edge trainer (about4.6GB on80GB). After finite preflight and EdgeR3 final evaluation, launch RL-005-R10-reorient-axis from original8598,2048×2000 onGPU1. No second full trainer per GPU.


Reorient axis preflight completed at8600 with finite tensors, Adam1e-4 and gyroON/coneOFF;63 focused checks passed. FullR10 remains queued until EdgeR3 final budget and strict evaluation exitGPU1.

StackR6 trace audit:118/128 ever show opposing contact, only4 sampled episodes rise above the stack target;104 terminal opposing contacts, median height26.37mm,26.11mm below target and86.73mm XY error. There are zero terminal object/base or robot/base contacts; this does not support a persistent physical obstruction at the endpoint.10229 held samples remain more than20mm below target. R6 was a finite1000-update continuation; arm noise remains nonzero. Preregister RL-018-R7-stack-lift: originalR6/5198, unchanged completion_v6,2048×3000, inherited Adam1e-4/normalizers/gripper and coneOFF/gyroOFF. UseGPU5 after PlaceR7 completes and its strict evaluation/conditional confirmation finish; no second full trainer. This budget tests transport after acquired capture, without a new reward or initialization reset. Failed Place validation remains queued for subsequent refinement; certification criteria unchanged.


ThrowR4/final9496 improves0→116/128 validation, but confirmation20260915 scores107/128; no certification. Actual validation success/failure clips reviewed; individual confirmation failures not inspected for tuning. Preregister PREFLIGHT-THROW-REFINE-041: original9496, unchanged throw_v4,64×3, Adam5e-5 override with moments/normalizers/learned arm+gripper/noise preserved, coneOFF/gyroOFF. After finite preflight, RL-024-R5-throw-refine original9496,2048×2000 onGPU2. Finish the full budget before validation; this improves the barely-passing validation margin. Retire failed confirmation seed20260915 for this next Throw candidate and freeze20260916 now. Do not retry9496 with a new seed. New candidate must pass both128 batches at116+; Throw-R5-evaluation-plan.json records the prospective schedule. evaluate_candidate accepts an explicit distinct confirmation seed, leaving other task defaults unchanged.
