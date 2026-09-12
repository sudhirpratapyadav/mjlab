# Status — RL teachers

Checked 2026-09-12. **Reach passed the measured rate gate (128/128 validation and 128/128 confirmation); 0/24 fully certified while evaluator edge-case checks and video review remain pending.** Six pilots were launched. Four were interrupted to fix W&B routing; their checkpoints are retained. Push-Cuboid failed at iteration 268. Correct W&B destination is now verified; Lift, Drag-Pull, Reorient and Topple have resumed on GPUs 1, 4, 5 and 6; see EXPERIMENTS.md.

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
| Strict RL checkpoint evaluation | End-to-end exercised; edge-case tests pending | Captures predicate before internal reset, masks later episodes; Reach 128/128 twice, Push-Cuboid 1/128 |
| Reward learnability across all tasks | NOT READY for blanket launch | Issues below; smoke success is not learning success |
| Physics presets | Decision pending | Eight tasks have zero gravity; preserve baseline until deliberately resolved |
| Old checkpoints/datasets | Incompatible action units | Start fresh or explicitly convert; never infer compatibility from 60D width |

## Per-task tracking

Every row has 60D observations and 8D actions. Rates below are deterministic first-episode terminal measurements; an incomplete pilot is not a certified teacher.

| Task | Episode s | Gravity | Next stage | RL success | Main issue |
|---|---:|---|---|---|---|
| Axial-Extract | 4 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Cage-Drag | 4 | on | Needs preparation | Not measured | Centered gripper policy closes the cage and invalidates episode; initialize open without changing action mapping. |
| Drag-Pull | 3 | on | Resume partial pilot | Not measured | model_300 retained after W&B pause. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Lift-Cube | 20 | on | Resume partial pilot | Not measured | model_300 retained after W&B pause. |
| Open-Door | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. Door regularizers are disabled. |
| Open-Drawer | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Open-Lid | 5 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Peg-Insertion | 20 | on | Needs preparation | Not measured | Separate safe grasp approach from bottom-tip insertion; completion shaping. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | Needs preparation | Not measured | Align final incentive with released, settled containment. |
| Push-Button | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Push-Cuboid | 3 | on | Diagnose PPO failure | 1/128 at model_200 | Invalid action std at iteration 268; diagnosis required before replacement run. |
| Push-Flap | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Reach-Target | 20 | on | Certification review | 128/128 + 128/128 | model_499; evaluator edge-case tests and videos pending. |
| Reorient-Object | 20 | on | Resume partial pilot | Not measured | model_300 retained after W&B pause. |
| Rotate-Valve | 8 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Slide-Window | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Stack-Cube | 20 | on | Needs preparation | Not measured | Goal reward must favor released, supported, settled placement. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Resume partial pilot | Not measured | model_200 retained after W&B pause. |
| Turn-Lever | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
