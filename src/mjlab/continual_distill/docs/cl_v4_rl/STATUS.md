# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** Current assignments: Cage R4 GPU1, Throw GPU2, Stack R4 GPU3, Lift R5 GPU4, Place R3 GPU5, Edge GPU6 and Reorient R7 GPU7. GPU0 remains unused. All24 active tasks have RL training evidence; coverage is not certification.

A captured Stack failure reproduced in1 and2048 GPU worlds and was fixed by clearing stale solver acceleration at episode reset. The production fix passed the captured transition and100 PPO updates; all24 tasks already start with zero cache, so fresh initialization is unchanged. Cage showed the same reset signature. Reorient has a distinct instability with extreme object velocities; its replay still fails after cache clearing. New captures retain8 preceding states and explicit nonfinite reward/observation lane IDs. Stack/Place use contact-closure shaping after measured open-enclosure failures. Strike/Peg/Pivot await targeted retries; full24-teacher goal active.

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
| Cage-Drag | 4 | on | R4 continuation | 0/128 R2 | R3 stopped at4649, reset-cache failure reproduced; continue own4600 onGPU1 with selective cache reset. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training first pilot | Not measured | edge_v2,2048 ×3000 onGPU6; enclosed grasp credit after exposure. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R5 | 0/128 final V3 | lift_v5 from finite diagnosticreplay2028; gripper mean0.5/std0.15,2048 ×2000 additional onGPU4; pre-step capture. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Rejected V2; retry needed | 0/128 V2 model1500 | Actual video: knocks shaft down and presses it; no geometric enclosure. Finite checkpoint retained. |
| Pivot-Lift | 6 | on | Rejected; retry needed | 0/128 model1500 | Actual board top pressing. CPU terminal contact audit:124/128 two-pad contacts,0 opposing pairs. |
| Place-In-Container | 20 | on | Training R3 | 0/128 V2 model1500 | completion_v5 contact-closure target, own1500/std0.15 onGPU5; geometry/native success unchanged. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R7 | 0/128 R5 final | R6 stopped at4609; extreme object velocity failure persists with cold cache. Own4600 retained;8-state capture onGPU7. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R4 | 0/128 R3 model1400 | Reset-cache regression fixed; completion_v5 from1400/std0.15 onGPU3 after100+3 PPO checks. |
| Strike-Slide | 4 | on | Finished pilot; retry needed | 9/128 final2999 | Strikes and retracts; most trials undershoot. Actual success/failure reviewed; native4s/friction/predicate unchanged. |
| Throw-To-Bin | 5 | on | Training first pilot | Not measured | throw_v3,2048 ×3000 onGPU2; capture width/ballistic shaping and pre-step capture. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
