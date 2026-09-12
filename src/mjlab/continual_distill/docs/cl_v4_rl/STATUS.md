# Status — RL teachers

Checked2026-09-12. **13/24 independent PPO RL teachers certified:** Reach, Topple, Button, Drawer, Drag, Flip, Door, Push-Cuboid, Flap, Lever, Window, Axial and Valve. Valve model1999 measured127/128 validation +128/128 confirmation; actual rotation and its ground-collision failure were reviewed; retained artifact uploaded.

Active: Stack GPU3 step1704, Peg GPU4 step1714, Cage retry GPU5 step1716, Lid retry GPU6 step1717, Place GPU7 step1705. Reorient R4 stopped on nonfinite simulator state at iteration1970; policy parameters finite, model1900 under diagnostic evaluation onGPU2. Lift final V3 strict0/128 with no sampled two-pad grasp; closure/exploration retry is next. Edge/Pivot/Strike/Throw still need first-pilot readiness. GPU0 remains unused. Full24-task goal is active.

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
| Cage-Drag | 4 | on | Training retry | 0/128 first pilot | cage_v2 resumes1499 onGPU5; guide valid-cage transport. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Needs preparation | Not measured | Reward useful edge exposure before grasp/lift. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Preparing retry | 0/128 final V3 | No sampled two-pad grasp; final aperture0.05514m and distance0.00728m. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Training retry | 0/128 first pilot | lid_v1 resumes1499 onGPU6; closure/contact and hinge progress. |
| Peg-Insertion | 20 | on | Training first pilot | Not measured | completion_v2 onGPU4; native square-bore fit unchanged. |
| Pivot-Lift | 6 | on | Needs preparation | Not measured | Reward wall-assisted pivot/capture; hidden history is a compact-state risk. |
| Place-In-Container | 20 | on | Training first pilot | Not measured | completion_v2 onGPU7; native released/contained/settled completion. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Diagnosing stopped R4 | 0/128 earlier variants | One simulator lane nonfinite at1970; finite model1900 under debug evaluation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training first pilot | Not measured | completion_v2 onGPU3; smooth closure and native completion incentive. |
| Strike-Slide | 4 | on | Needs preparation | Not measured | Dynamic task under compact state; tune speed penalties and horizon. |
| Throw-To-Bin | 5 | on | Needs preparation | Not measured | Place-style shaping lacks launch/release stages; tune speed penalty. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
