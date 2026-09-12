# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** Training: Throw GPU2, Stack R5 GPU3, Lift R6 GPU4, Place R4 GPU5 and Peg R3 GPU6. Cage R4 and Reorient R7 completed and were evaluated; GPUs1/7 are available after their evaluation steps exit. GPU0 remains unused. All24 active tasks have RL training evidence; coverage is not certification.

Edge final2999 and LiftR5/model3500 each scored0/128 with actual failure videos reviewed. A cube closure audit found64 opposing-contact cases rejected by whole-object enclosure; the production GPU query confirms64/64 opposing contacts and0/64 old enclosed grasps on those counterfactual states. New lift_v6/completion_v6 use opposing pad normals and a local centerline width for training; native success/physics/60D/8D stay unchanged. Lift/Peg/Stack/Place retries passed PPO preflights and are running. Other training recipes retain their versioned behavior. The reset-cache fix remains verified; Reorient's separate high-velocity instability remains unresolved, with8-state capture enabled. Strike/Edge/Pivot await targeted retries. Full24-teacher goal active.

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
| Cage-Drag | 4 | on | Retry needed | 2/128 R4 final5599 | Valid open-hand approach but almost no sustained contact transport; actual success/failure reviewed. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Rejected first pilot; retry needed | 0/128 final2999 | Slides plate to overhang then hovers behind; no contact or pinch. Actual failure reviewed. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R6 | 0/128 R6 model4100 | lift_v6 from3500/std0.15 onGPU4; actual opposed contacts and local pad-center width;8-state capture. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R3 | 0/128 V2 model1500 | completion_v6 from1500, gripper mean0.5/std0.15 onGPU6; native insertion unchanged. |
| Pivot-Lift | 6 | on | Rejected; retry needed | 0/128 model1500 | Actual board top pressing. CPU terminal contact audit:124/128 two-pad contacts,0 opposing pairs. |
| Place-In-Container | 20 | on | Training R4 | 0/128 V2 model1500 | completion_v6 from ownR3/2300/std0.15 onGPU5; opposing-contact training gate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Retry needed | 0/128 R7 final6099 | Full budget completed finite, but deterministic policy hovers without contact. Earlier numerical instability remains a separate issue. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R5 | 0/128 R3 model1400 | completion_v6 from ownR4/2200/std0.15 onGPU3;3 resumed PPO checks passed. |
| Strike-Slide | 4 | on | Finished pilot; retry needed | 9/128 final2999 | Strikes and retracts; most trials undershoot. Actual success/failure reviewed; native4s/friction/predicate unchanged. |
| Throw-To-Bin | 5 | on | Training first pilot | 0/128 model2500 | Open-hand hover without lift or throw; actual failure reviewed. Full budget still running onGPU2. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
