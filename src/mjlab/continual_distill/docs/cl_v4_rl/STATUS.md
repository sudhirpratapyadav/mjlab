# Status — RL teachers

**17/24 independent PPO RL teachers certified.** Throw14098 is newest at125/128 validation and126/128 reserved confirmation20260916. Actual videos reviewed; native CPU endpoint reconstruction matches all128 validation labels, all76 checkpoint tensors finite,13 W&B teacher files retained and verified. Cage19092 remains119/128+120/128 and Place7500 remains117/128+119/128. Seven tasks remain unfinished.

Current full runs: ReorientR14 GPU1 final18593, EdgeR6 GPU2 final12994, StrikeR10 GPU3 final18991, LiftR16 GPU4 final21990, StackR10 GPU5 final14194, PegR10 GPU6 final10198 and PivotR3 GPU7 final6498. StrikeR9 finished its full budget and strict59/128; reviewed and continued unchanged from16992 with saved Adam1.25e-6 and max_update_kl.03, without another noise reset. GPU0 stays unused; holder20277 expires2026-09-28T08:58:54 scheduler time.

Latest strict results: ReorientR13/16594 1/128 and PegR9/10198 0/128, with actual recorded frames reviewed and all76 tensors finite in each. StackR9/12195 remains0/128 before the new run. LiftR15/19991 improves retained90→109/128, all76 tensors finite and actual videos reviewed; unchanged guarded continuation retains saved5e-6 LR. StrikeR9/16992 improves39→59/128, median endpoint error123.71→86.21mm; all76 tensors finite and actual videos reviewed. Previous measurements stay historical. Matched training probes exposed large post-update Gaussian KL despite tiny pre-update normalizer drift. Opt-in fixed-schedule PPO backtracking restores policy/normalizers/Adam/RNG and halves LR when actual-rollout mean KL exceeds.03; tested rollback and50-update full-batch preflights passed. Lift/Strike/Peg restart from retained pre-collapse sources with this guard. Reorient's118/128 orientation/drift passes still fail settling, while fixed recorded controls settle30/31 diagnostic GPU cases: reorient_v10 rewards steady actions. Stack_v1 rewards release near physical support. Pivot_v4 uses actual near-edge geometry and early wrist/tilt progress. Edge_v5 rewards wider arrival and makes capture dominate the precursor. All new recipes passed focused invariants and finite preflights. Diagnostics never become demonstrations, resets or policy actions. Native success, physical models and60D/8D stay fixed.

## Readiness

| Check | Status | Evidence / action |
|---|---|---|
| Active suite and deferred task | PASS | 24 tasks; Tool-Pull remains registered; `active_tasks.json` |
| Common observations/actions | PASS | 60D/8D; 49 focused tests and prior all-task GPU checks |
| Registered PPO configs | PASS with launch override | All 24 load; defaults have one environment, override explicitly |
| PPO optimization/checkpoint round trip | PASS | `evidence/ppo_smoke.json`, Reach only |
| GPU allocation | User expanded | GPUs1–7 authorized; leave GPU0 unused; recheck placement before launch |
| Generic CLI with UUID pinning | BLOCKED route; workaround prepared | Integer-only GPU parsing; stage-local launcher bypasses reselection |
| Stage-local launcher | PASS on Reach; checkpoint resume verified | Full optimization/config/checkpoint path exercised |
| W&B destination | PASS | Sudhir IIT Jodhpur entity; successful write/read; shared login unchanged |
| Strict RL checkpoint evaluation | PASS; adversarial loop tests complete | Captures predicate before internal reset, masks later episodes; 44 focused tests including reset-boundary capture, subset resets and retry exclusion |
| Reward learnability across all tasks | NOT READY for blanket launch | Issues below; smoke success is not learning success |
| Solver reset bookkeeping | FIXED for captured Stack case | Selective qacc_warmstart reset;1/2048-lane reproduction, cold-cache regression,100 PPO updates; Reorient contact instability remains separate |
| Physics presets | Explicit baseline frozen | Preserve registered physics, including eight zero-gravity tasks; no physics or success changes |
| Old checkpoints/datasets | Incompatible action units | Start fresh or explicitly convert; never infer compatibility from 60D width |

## Per-task tracking

Every row has 60D observations and 8D actions. Rates below are deterministic first-episode terminal measurements; an incomplete pilot is not a certified teacher.

| Task | Episode s | Gravity | Next stage | RL success | Main issue |
|---|---:|---|---|---|---|
| Axial-Extract | 4 | off | Certified | 128/128 +128/128 | model200 retained; plug visibly extracted from socket. |
| Cage-Drag | 4 | on | Certified | 119/128 +120/128 | model19092 retained; open-gripper transport reviewed and no-pinch audit passed. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R6 | 0/128 R5 final10995 | Wide-arrival/capture-dominance recipe passed2048x50 guarded preflight. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R16 | 109/128 R15 final19991 | Reviewed improvement90→109; unchanged guarded continuation with saved5e-6 Adam. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R10 | 0/128 R9 final10198 | Capture collapsed; guarded existing smooth recipe from retained8199. |
| Pivot-Lift | 6 | on | Training R3 | 0/128 R2 final3499 | Tested geometry/early wrist reward; diagnostic controls never used as policy data. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R14 | 1/128 R13 final16594 |118 orientation/drift passes; constant-control diagnostic settles30/31, steady-action reward preflight passed. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R10 | 0/128 R9 final12195 |108 terminal opposing grasps; support-release shaping, learned transport retained. |
| Strike-Slide | 4 | on | Training R10 | 59/128 R9 final16992 | Reviewed endpoint improvement; unchanged guarded continuation, saved1.25e-6 LR/no repeat noise reset. |
| Throw-To-Bin | 5 | on | Certified | 125/128 +126/128 | model14098, reserved confirmation16; actual videos and native endpoint audit passed. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.

Wave34 Pivot readiness: corrected all8D-bounded32-state replay shows CPU28/32 versus baselineGPU14/32 tilt+wall mechanism; existing primitive-box compatibility restoresGPU28/32 and median endpoint parity0.0894mm. Diagnostic only. Complete currentR3 fullbudget/evaluation, then select/preregister retained-checkpoint primitive-only PPO preflight; no changes to active training or native physical model/interface. Evidence: pivot_r2_backend_comparison_audit.json.

Wave34 evaluation scheduling: wait_for_final_evaluation.py is watching each of the seven identified trainer processes. It starts the existing strict evaluator only after process exit and verification of that plan's finite final checkpoint; no intermediate evaluation. Each run/evaluation-dispatch.json records progress, and per-run locks prevent duplicate dispatch. Actual video review and both independent>90% batches remain mandatory.
