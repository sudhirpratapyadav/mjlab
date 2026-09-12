# Status — RL teachers

Checked 2026-09-12. **5/24 certified: Reach, Topple, Button, Drawer and Drag.** All retained checkpoints, normalizers, configurations and reviewed videos are in the verified W&B project. Push V2 measured 111/128 and is receiving 1,000 further updates. Fresh Flip/Door/Flap/Lever teachers are training with the validated mechanism recipe. Lift V2 continues; Reorient V2 completed with zero training success and measured 0/128 strict validation and needs diagnosis. GPU0 remains unused. See evidence/scoreboard.json and training_wave3.json.

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
| Physics presets | Explicit baseline frozen | Preserve registered physics, including eight zero-gravity tasks; no physics or success changes |
| Old checkpoints/datasets | Incompatible action units | Start fresh or explicitly convert; never infer compatibility from 60D width |

## Per-task tracking

Every row has 60D observations and 8D actions. Rates below are deterministic first-episode terminal measurements; an incomplete pilot is not a certified teacher.

| Task | Episode s | Gravity | Next stage | RL success | Main issue |
|---|---:|---|---|---|---|
| Axial-Extract | 4 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Cage-Drag | 4 | on | Needs preparation | Not measured | Centered gripper policy closes the cage and invalidates episode; initialize open without changing action mapping. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Training mechanism variant | Not measured | RL-009, GPU2; finite rollout preflight passed. |
| Lift-Cube | 20 | on | Training reward variant | 0/128 baseline | Fresh lift_v1: actual grasp/lift/goal shaping; 2,048 environments onGPU1. |
| Open-Door | 3 | off | Training mechanism variant | Not measured | RL-010, GPU7; baseline zero gravity retained. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Peg-Insertion | 20 | on | Needs preparation | Not measured | Separate safe grasp approach from bottom-tip insertion; completion shaping. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | Needs preparation | Not measured | Align final incentive with released, settled containment. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Training continuation | 111/128 model999 | 1,000 further stable_v1 updates on GPU3. |
| Push-Flap | 3 | off | Training mechanism variant | Not measured | RL-011, GPU4; baseline zero gravity retained. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Strict evaluation and diagnosis | 0/128 baseline and V2 | V2 completed1,500 updates with zero training success and full action saturation; recorded trace retained for physical diagnosis. |
| Rotate-Valve | 8 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Slide-Window | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Stack-Cube | 20 | on | Needs preparation | Not measured | Goal reward must favor released, supported, settled placement. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Training mechanism variant | Not measured | RL-012, GPU6; baseline zero gravity retained. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
