# Status — RL teachers

Checked 2026-09-12. **4/24 certified: Reach, Topple, Button and Drawer.** All have retained checkpoint/normalizers/configs and reviewed videos in the verified W&B project. Push V2 completed with strict validation111/128; a further continuation is prepared. Drag stopped on a nonfinite simulator state after iteration1380; its finite model1300 scored122/128 validation and is undergoing independent confirmation. Lift/Reorient V2 continue; remaining mechanism preflight passed for all eight tasks. GPU0 remains unused.

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
| Drag-Pull | 3 | on | Independent confirmation | 122/128 model1300 | Later training hit one nonfinite simulator environment; diagnostics retained. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Lift-Cube | 20 | on | Training reward variant | 0/128 baseline | Fresh lift_v1: actual grasp/lift/goal shaping; 2,048 environments onGPU1. |
| Open-Door | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. Door regularizers are disabled. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Peg-Insertion | 20 | on | Needs preparation | Not measured | Separate safe grasp approach from bottom-tip insertion; completion shaping. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | Needs preparation | Not measured | Align final incentive with released, settled containment. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Prepared continuation | 111/128 model999 | Stable V2 completed1,000 updates; extend own finite checkpoint. |
| Push-Flap | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training reward variant | 0/128 baseline | Fresh reorient_v1 grasp/broad-angle shaping;2,048 environments onGPU5. |
| Rotate-Valve | 8 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Slide-Window | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Stack-Cube | 20 | on | Needs preparation | Not measured | Goal reward must favor released, supported, settled placement. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
