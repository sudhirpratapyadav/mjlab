# Plan — certify all24 independent RL teachers

## Current state

**14/24 independent PPO RL teachers certified.** Training: Cage R6 GPU1, Throw R2 GPU2, Stack R6 GPU3, Lift R7 GPU4, Peg R4 GPU6 and Reorient R9 GPU7. Place R5 is training onGPU5 from retained3400 with verified Adam LR5e-5 after its later policy regressed. GPU0 remains unused. All24 tasks have RL training evidence; coverage is not certification.

Lift R6 final5499 strict0/128, but actual grasp/lift/transport now occurs: mean final goal error9.50cm and height19.65cm, improved from26.06cm/2.78cm at4100. Continue its unchanged policy from5499. Throw final2999 and Peg R3/2300 each strict0/128; actual failure clips reviewed. Place learned grasps then regressed after3400 into open-hand joint-limit saturation; retained finite3400, strict0/128, reviewed grasp/lift beside the container. R5 continues3400 at half the learning rate after a finite3-update PPO check.

Captured Peg/Reorient spin failures identify a missing native MuJoCo free-body gyroscopic correction in the installed Warp implicitfast path. Optional backend compatibility sourcef821aa0 passes CPU comparisons,64-world sparse/dense CUDA-graph regressions, two2048-world captured-state regressions and both100-update2048-world PPO stress runs. Peg's new fresh PPO lineage and Reorient's own resumed policy enable the recorded correction. It defaults off and is restored from the run manifest in evaluation/certification; all existing certified teachers retain their configuration. Model parameters, timestep, success predicates and60D/8D remain unchanged. Contact trajectories may differ from CPU; no claim that all simulator instability is resolved.

## Acceptance

Deterministic policy means,60D observations and normalized8D absolute joint targets. Use native terminal success in the first uninterrupted episode. Validation seed20260914 and confirmation seed20260915,128 episodes each, both strictly>90% (116+). Review actual recorded success/failure videos, then retain policy/optimizer/normalizers/configs, hashes and W&B artifacts. Restore each checkpoint's recorded backend compatibility setting. Never count a training contact fraction or a selected successful clip as a measured success rate.

## Active work

- GPU1: Cage R6 cage_v4,1024 ×1500 additional from7598, preserve all settings. Prior strict16/128 versus2/128; improve goal accuracy and native no-pinch history.
- GPU2: Throw R2 throw_v4,2048 ×2500 from2999/std0.15; learn actual opposing capture then launch.
- GPU3: Stack R6 completion_v6,2048 ×1000 from4199, preserve all settings. Prior strict0/128, but65.08% final opposed contacts; evaluate lift/stack/release at budget.
- GPU4: Lift R7 lift_v6,2048 ×2000 from5499, no reset/reward/backend changes. Goal error is improving; evaluate a retained checkpoint when native success rises or budget ends.
- GPU5: Place R5 completion_v6 from own3400,2048 ×2000 additional, Adam LR5e-5. The3-update PPO preflight saved finite tensors and effective LR5e-5. Preserve means/std/moments/normalizers/backend; evaluate released containment at budget or a justified checkpoint.
- GPU6: Peg R4 completion_v6 from fresh stressmodel99 ×3000 additional, optional gyro correction inherited. Previous resumed lineage toppled/pressed; this is fresh PPO, not distillation.
- GPU7: Reorient R9 reorient_v7 from own stress6599 ×2000 additional, optional gyro correction inherited; output/std preserved.

Edge/Pivot/Strike remain targeted retries, using the next suitable free slot. Edge exposes the plate then misses the pinch; Pivot presses the board flat; Strike9/128:58 negligible launches remain a median8.2cm above contact height. strike_v2 precision/drive reward is prepared and CPU-tested; next free GPU1/3 after current final evaluation runs a3-update PPO preflight before its2000-update retry. Diagnose geometry/approach alignment and contact launch rather than blindly extending failed recipes.

## Numerical readiness

Selective warm-start reset is retained. The separate captured high-spin failure now has an opt-in CPU-compatible free-body correction, sourcef821aa0, with recorded-state and controlled-spin regressions.100 full-size PPO updates passed for both affected tasks. Existing certified configurations remain unchanged. No shared-venv edits, state clipping, dropped invalid lanes, new observation fields or altered success tests. Continue8-state captures; a new failure requires diagnosis, not an assumption that the correction covers it.

## Records and resources

Keep STATUS/EXPERIMENTS/LOGS, current training_wave inventory and scoped W&B records current. Publish reviewed policy summaries/videos to /v4-rl/ using PUBLICATION.md. GPU0 stays free; verify real placement, holder20277 expiry2026-09-28T08:58:54 and available disk before resource reuse. Stop only verified own steps. Full24-teacher goal remains active until every teacher is independently certified.

Wave17 pending readiness: strike_v2 is implemented and57 focused tests pass; runPREFLIGHT-STRIKE-CONTACT-024 (own2999,64×3, online, no resets) on the first suitable freeGPU1/3 after evaluating its current final checkpoint, thenRL-023-R2 (2048×2000) only if finite. Edge FK evidence shows the exposed plate is still20.3cm from the closest pinch target with poor wrist alignment; prepare a reachable full side-approach orientation before its retry. Current seven runs remain on their loaded configurations; no new teacher certification.
