# Status — RL teachers

Checked2026-09-13. **15/24 independent PPO RL teachers certified.** All24 have RL training evidence. PlaceR8/model7500 is newly certified at117/128 validation and119/128 independent confirmation, with reviewed actual videos and local/remote retained artifacts. CageR10 passes117/128 validation but fails confirmation115/128; ThrowR4 passes116/128 validation but fails confirmation107/128. Neither is certified. StrikeR5 improves to42/128; ReorientR10, LiftR11 and PegR7 remain0/128.

Current full runs: ReorientR11 onGPU1, StrikeR6 onGPU3, LiftR12 onGPU4, StackR7 onGPU5, EdgeR4 onGPU6 and CageR11 onGPU7. GPU2 is reserved for Throw failure diagnosis; GPU0 remains unused. Holder20277 expires2026-09-28T08:58:54 scheduler time. ThrowR5 failed at iteration11136 before its planned final11495; do not evaluate a substitute or consume confirmation20260916. CageR11 must finish its full2000-update budget at19092 and use prospectively frozen confirmation20260916 only after116+ validation.

Place certification follows the verified primitive box-contact correction; prior14 certified backends remain unchanged. LiftR12 tests a reward for arm targets close to actual joints near the held goal: exact-control replay achieved25/30 successful endpoints when holding current arm joints with recorded grip, versus0/30 freezing original targets. These are diagnostic interventions, not RL rates. The new reward passed69 focused checks and a finite64-world3-update PPO preflight. Native physical models, success predicates and60D/8D interface remain unchanged.

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
| Cage-Drag | 4 | on | Training R11 | 117/128 +115/128 R10 | Confirmation fails; new full-budget refinement with fresh prospective seed16. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R4 | 0/128 R3 final6997 | Closest pinch miss39.84mm; unchanged bounded continuation onGPU6. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R12 | 0/128 R11 final13994 | Arm-target equilibrium reward; finite preflight, exact-control diagnosis recorded. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Capture regression diagnosis | 0/128 R7 final7096 | Lost acquired capture; retainR6/5097, do not continue7096 unchanged. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Small side/height target shifts did not produce tipping in CPU dynamics; no reward change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R11 | 0/128 R10 final10597 | Improved held orientation credit; unchanged bounded continuation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R7 | 0/128 R6 final5198 | Capture acquired, transport low; unchanged3000-update continuation onGPU5. |
| Strike-Slide | 4 | on | Training R6 | 42/128 R5 final10995 | Unchanged precision refinement onGPU3. |
| Throw-To-Bin | 5 | on | Failure diagnosis | 116/128 +107/128 R4 | R5 stopped at11136 with simulator lane224 nonfinite; preserve capture, no premature evaluation. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
