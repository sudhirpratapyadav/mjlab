# Status — RL teachers

**18/24 independent PPO RL teachers certified.** Lift R16/model21990 is newest:118/128 validation and117/128 independent confirmation. Actual success/failure frames reviewed; all76 checkpoint tensors finite and13 W&B teacher files remotely verified (28m72bwp). The optional post-integration terminal recomputation agrees126/128 labels and passes116/128; its cached-velocity limitation is explicit in lift_r16_terminal_audit.json. The native measured rates remain authoritative. Six tasks remain unfinished.

Current full budgets: ReorientR15 GPU1 final20592, StackR12 GPU5 final16193, PegR11 GPU6 final12197 and PivotR4 GPU7 final8497. Their identified-process watchers start strict evaluation only after verified final checkpoint and trainer exit. Peg073 and Pivot074 passed50-update2048-world finite preflights with acceptedKL<=.03; full runs start from original retained parents. GPUs2/3 are available for Edge/Strike diagnosis; GPU4 is free after Lift certification. GPU0 stays unused; holder20277 expires2026-09-28T08:58:54 scheduler time.

Latest reviewed strict results: ReorientR14 4/128, StrikeR11 50/128, StackR11 85/128; EdgeR6/PegR10/PivotR3 all0/128. Reorient now reaches orientation/drift in123/128 but fails settling. Strike has123 substantial launches but poor endpoint calibration; longer96-step returns/lambda.99 completed a matched sample budget but decreased57→50/128. Stack improved0→85/128 with primitive-only contact compatibility and protected optimizer steps; all128 native CPU/GPU terminal labels agree. Unchanged R12 continues from reviewed14194. PegR10 preserves100/128 opposed grasps but presses down; stronger lifting/native-completion weights passed preflight and are training. PivotR3 tips108/128 boards beyond20deg but fails grasp/lift; matched replay supports primitive contacts, while earlier closing failed0/32. Edge's canonical-wrist diagnostic did not improve7/32 held lifts (6/32 with correction), so no tighter-wrist recipe is justified. All diagnostics remain separate from policy data, actions and resets. Native physical models/success and60D/8D remain fixed.

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
| Edge-Grasp | 6 | on | Diagnosis after R6 | 0/128 R6 final12994 | Reviewed; wider arrival worsened vertical miss; canonical-wrist diagnostic negative. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Certified | 118/128 +117/128 | R16/model21990, actual videos reviewed;13 W&B retention files verified. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R11 | 0/128 R10 final10198 |100 terminal opposed captures, no2cm lift; stronger lifting credit, saved5e-6 Adam/guard. |
| Pivot-Lift | 6 | on | Training R4 | 0/128 R3 final6498 |108 sampled tilts>20deg; primitive-only parity correction and optimizer guard; no aperture change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R15 | 4/128 R14 final18593 |123 orientation/drift passes; stronger quiet/steady/native completion weights. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R12 | 85/128 R11 final14194 |124 ever captured;100 supported/104 released terminal cubes. Native CPU/GPU labels agree128/128; unchanged guarded continuation. |
| Strike-Slide | 4 | on | Diagnosis after R11 | 50/128 R11 final19490 | Longer returns did not improve matched-budget result;125 launches, median endpoint103.11mm. |
| Throw-To-Bin | 5 | on | Certified | 125/128 +126/128 | model14098, reserved confirmation16; actual videos and native endpoint audit passed. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.

Wave35 evaluation scheduling: each full trainer has a frozen final-checkpoint plan and an identified-process dispatcher. Do not duplicate dispatch or certify until actual video review and W&B retention checks pass.
