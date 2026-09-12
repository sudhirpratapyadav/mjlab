# Status — RL teachers

Checked 2026-09-12. **12/24 certified:** Reach, Topple, Button, Drawer, Drag, Flip, Door, Push-Cuboid, Flap, Lever, Window and Axial. All have retained W&B artifacts and two passing held-out batches with video review. Lift/Reorient/Cage/Lid/Valve continue training. Stack/Place/Peg completion recipes passed53 CPU tests and three real-physics preflights; a short PPO test precedes full pilots. GPUs3/7 currently handle preflight/diagnostics; GPU0 remains unused.

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
| Axial-Extract | 4 | off | Certified | 128/128 +128/128 | model200 retained; plug visibly extracted from socket. |
| Cage-Drag | 4 | on | Training first pilot | Not measured | Open-gripper prior verified above no-pinch threshold; all8 action dimensions learnable; GPU5. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training bounded-policy variant | 0/128 baseline and V2 | Fresh lift_v2 on GPU1; configured-pose initialization, lower LR/exploration. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Training first pilot | Not measured | mechanism_v1, GPU4; registered gravity preserved. |
| Peg-Insertion | 20 | on | PPO preflight | Not measured | Safe upper-body grasp target; native square-bore fit and released/settled completion preserved. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | PPO preflight | Not measured | Native full-object containment and released support receive dominant completion incentive. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training bounded-policy variant | 0/128 baseline and V2 | V2 ground collision after8 steps diagnosed; fresh end-face reward/policy on GPU2. |
| Rotate-Valve | 8 | off | Training first pilot | Not measured | mechanism_v1, GPU6;2,000 updates, registered8-second horizon. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | PPO preflight | Not measured | completion_v1 rewards actual release/support/settling; CPU and physics checks passed. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
