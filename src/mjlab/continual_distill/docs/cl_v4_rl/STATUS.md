# Status — RL teachers

Checked2026-09-13. **16/24 independent PPO RL teachers certified.** Cage19092 is newest at119/128 validation and120/128 prospectively frozen confirmation20260916. Its actual videos, all-episode no-pinch aperture audit, finite checkpoint and13-file W&B teacher artifact are verified. Place7500 remains117/128+119/128. Throw9496 failed confirmation107/128 and remains uncertified.

Current full runs: ReorientR11-cpu GPU1 final12596, ThrowR6 GPU2 final13099, StrikeR7 GPU3 final14993, StackR8 GPU5 final10196 and PegR8 GPU6 final8199. LiftR12 completed15993 and is evaluating onGPU4. GPU7 is free after Cage certification; GPU0 remains unused. Every new Slurm step requests8 CPUs, numerical threads1; holder20277 expires2026-09-28T08:58:54 scheduler time.

StrikeR6 final12994 scored37/128 after prior42/128. Endpoint errors dominate despite121/128 substantial launches. New strike_v4 doubles fine endpoint and native-success weights while preserving the native gate and reward hierarchy;72 focused checks and finite64x3 PPO preflight passed before original12994x2000 full training. ThrowR6 and all other full runs retain their frozen budgets; Throw conditional confirmation16 remains reserved. Edge and Pivot remain in diagnosis. Native success, physical models and approved60D/8D interface stay fixed.

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
| Edge-Grasp | 6 | on | Approach plateau diagnosis | 0/128 R4 final8996 | Closest pinch39.53mm, mostly37.75mm vertical; retain checkpoint and diagnose. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Evaluating R12 final15993 | 0/128 R11 final13994 | Arm-target equilibrium reward; finite preflight, exact-control diagnosis recorded. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R8 | 0/128 R7 pre-collapse6200 |74.2% opposed capture; lift shaping4 andAdam5e-5 after finite gate. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Small side/height target shifts did not produce tipping in CPU dynamics; no reward change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R11 | 0/128 R10 final10597 | Improved held orientation credit; unchanged bounded continuation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R8 | 0/128 R7 final8197 |117/128 sampled lifts above base; test gripper release exploration with learned means retained. |
| Strike-Slide | 4 | on | Training R7 | 37/128 R6 final12994 | Prior42/128; stronger endpoint/native shaping passed preflight. |
| Throw-To-Bin | 5 | on | Training R6 | 116/128 +107/128 R4 | Cone correction passes failure replay and full-world stress; evaluate new13099 only after budget. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
