# Status — RL teachers

**17/24 independent PPO RL teachers certified.** Throw14098 is newest at125/128 validation and126/128 reserved confirmation20260916. Actual videos reviewed; native CPU endpoint reconstruction matches all128 validation labels, all76 checkpoint tensors finite,13 W&B teacher files retained and verified. Cage19092 remains119/128+120/128 and Place7500 remains117/128+119/128. Seven tasks remain unfinished.

Current full runs: ReorientR13 GPU1 final16594, StrikeR8 GPU3 final16992, LiftR14 GPU4 final19991, StackR9 GPU5 final12195 and PegR9 GPU6 final10198. Five actual PID/UUID/worktree/PYTHONPATH/eight-CPU affinities verified in training_wave31.json. GPUs2/7 free after Throw certification and Edge evaluation; GPU0 unused. Holder20277 expires2026-09-28T08:58:54 scheduler time.

Latest strict results: Lift90/128, Reorient0/128 and Edge0/128. Lift continues unchanged after56→90 progress. Reorient median axis error39.63→10.68deg;82/128 pass orientation/drift but settling fails, so reorient_v9 adds orientation-weighted quiet credit after64 focused checks/finite64x3. Peg native CPU lifts30/32 captured states and matched CPU/GPU frozen controls agree; actual policy targets alternate1.677rad RMS. Peg_v3 strengthens only action-change regularization after69 checks/finite64x3. Strike scales learned noise0.5 once after state-preservation tests/finite64x3; deterministic means retained. Edge wrist error61.97→11.80deg but side-pinch hand remains31.10mm high; approach-path diagnosis next. Pivot remains in contact-path diagnosis. Native success, physical models and60D/8D stay fixed.

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
| Edge-Grasp | 6 | on | Diagnosis | 0/128 R5 final10995 | Wrist error11.80deg but hand remains31.10mm high; approach-path diagnosis. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R14 | 90/128 R13 final17992 | Continued56→90; actual sustained lift/hold reviewed. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R9 | 0/128 R8 final8199 | Matched solver replay agrees; smoother action recipe targets control oscillation. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Small side/height target shifts did not produce tipping in CPU dynamics; no reward change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R13 | 0/128 R12 final14595 | Median axis10.68deg;82 orientation/drift pass but settling fails; quiet reward preflight passed. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R9 | 2/128 R8 final10196 | Better transport; mean-only gripper reset passed preflight, std retained. |
| Strike-Slide | 4 | on | Training R8 | 39/128 R7 final14993 | Precision continuation scales learned noise0.5 once; means retained. |
| Throw-To-Bin | 5 | on | Certified | 125/128 +126/128 | model14098, reserved confirmation16; actual videos and native endpoint audit passed. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
