# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** Current assignments: Strike GPU1, Throw GPU2, Stack R3 GPU3, Peg V2 GPU4, Pivot GPU5, Edge GPU6 and Place V2 GPU7. GPU0 remains unused. All24 active tasks now have RL training evidence; this is coverage, not certification.

Lift, Reorient and Cage need successful retries and await slots. Cage R2 final0/128: valid open enclosure without contact; cage_v3 contact-target preflight passed. Reorient R5 final0/128: saturated closed fingers without capture; width-matched rewards and gripper-output reset are available. Stack V2 stopped on a numerical failure in one lane; R3 retains finite arm control, resets only the gripper output/exploration and enables preceding-state capture. The numerical cause remains unresolved. Full24-teacher goal active.

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
| Cage-Drag | 4 | on | Contact retry ready; awaiting slot | 0/128 R2 | All final states valid open cages but no contact. cage_v3 passed3 resumed PPO updates; exact native no-pinch/transport preserved. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training first pilot | Not measured | edge_v2,2048 ×3000 onGPU6; enclosed grasp credit after exposure. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Waiting for slot after diagnostic replay | 0/128 final V3 | R4 failed at2009; same30-update replay finite, model2028 retained. Capture preceding state on next continuation. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training fresh V2 | 0/16 first-pilot debug | completion_v3 GPU4; upper-body grasp, actual enclosure/contact; native seating unchanged. |
| Pivot-Lift | 6 | on | Training first pilot | Not measured | pivot_v2,2048 ×3000 onGPU5; actual wall/tilt history unchanged. |
| Place-In-Container | 20 | on | Training fresh V2 | 0/16 first-pilot debug | completion_v3 GPU7; prior nearly closed top-contact policy replaced. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Retry prepared; awaiting slot | 0/128 R5 final | Closed empty fingers/no contact. reorient_v6 width shaping and gripper-output reset ready. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R3 | 0/16 V2 debug | completion_v4 from finite700, gripper mean0.5/std0.15; pre-step capture onGPU3. |
| Strike-Slide | 4 | on | Training first pilot | Not measured | strike_v1 GPU1;3000 updates, native4-second horizon/friction/predicate unchanged. |
| Throw-To-Bin | 5 | on | Training first pilot | Not measured | throw_v3,2048 ×3000 onGPU2; capture width/ballistic shaping and pre-step capture. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
