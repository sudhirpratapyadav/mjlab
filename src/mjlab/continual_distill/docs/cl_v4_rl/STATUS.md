# Status — RL teachers

Checked 2026-09-12. **0/24 teachers certified above 90%.** No full teacher-training job is running for this stage. Only PREFLIGHT-PPO-001 was executed (two optimizer updates; no success evaluation).

## Readiness

| Check | Status | Evidence / action |
|---|---|---|
| Active suite and deferred task | PASS | 24 tasks; Tool-Pull remains registered; `active_tasks.json` |
| Common observations/actions | PASS | 60D/8D; 49 focused tests and prior all-task GPU checks |
| Registered PPO configs | PASS with launch override | All 24 load; defaults have one environment, override explicitly |
| PPO optimization/checkpoint round trip | PASS | `evidence/ppo_smoke.json`, Reach only |
| GPU availability snapshot | PASS | Assigned GPUs1–3 idle at check; recheck before launch |
| Generic CLI with UUID pinning | BLOCKED route; workaround prepared | Integer-only GPU parsing; stage-local launcher bypasses reselection |
| Stage-local launcher | Dry-run checked only | Full pilot/config-saving path still to validate |
| Strict RL checkpoint evaluation | PENDING | Must capture first-episode terminal success; no retry inflation |
| Reward learnability across all tasks | NOT READY for blanket launch | Issues below; smoke success is not learning success |
| Physics presets | Decision pending | Eight tasks have zero gravity; preserve baseline until deliberately resolved |
| Old checkpoints/datasets | Incompatible action units | Start fresh or explicitly convert; never infer compatibility from 60D width |

## Per-task tracking

Every row has 60D observations and 8D actions. “Pilot” is a preparation priority, not a trained teacher. No RL success percentages are available yet.

| Task | Episode s | Gravity | Next stage | RL success | Main issue |
|---|---:|---|---|---|---|
| Axial-Extract | 4 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Cage-Drag | 4 | on | Needs preparation | Not measured | Centered gripper policy closes the cage and invalidates episode; initialize open without changing action mapping. |
| Drag-Pull | 3 | on | Pilot pending | Not measured | Approach/contact learning and short horizon. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Lift-Cube | 20 | on | Second pilot | Not measured | Grasp-gated objective; calibrate approach versus regularization. |
| Open-Door | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. Door regularizers are disabled. |
| Open-Drawer | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Open-Lid | 5 | on | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Peg-Insertion | 20 | on | Needs preparation | Not measured | Separate safe grasp approach from bottom-tip insertion; completion shaping. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | Needs preparation | Not measured | Align final incentive with released, settled containment. |
| Push-Button | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Push-Cuboid | 3 | on | Pilot pending | Not measured | Short horizon and strict goal/settling evaluation. |
| Push-Flap | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Reach-Target | 20 | on | First pilot | Not measured | Direct reach objective; verify full launcher/evaluator and learning. |
| Reorient-Object | 20 | on | Pilot pending | Not measured | Check approach signal and settled completion; compact-state dynamics risk. |
| Rotate-Valve | 8 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Slide-Window | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |
| Stack-Cube | 20 | on | Needs preparation | Not measured | Goal reward must favor released, supported, settled placement. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Pilot pending | Not measured | Check approach signal and settled completion. |
| Turn-Lever | 3 | off | Needs preparation | Not measured | Approach reward saturates at reset; review idle bonus and gravity preset. |

## Resource limits

Three assigned A100 80GB GPUs; snapshot and UUIDs in CONTEXT/evidence. Holder expiry is September 28 per Slurm. About 1.6 TB free on the shared filesystem. Neither sustained training throughput nor optimal environment count is established by the tiny smoke. Recheck GPU use, free space and holder lifetime before a long run.
