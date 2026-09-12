# Status — RL teachers

Checked2026-09-12. **13/24 independent PPO RL teachers certified.** Stack GPU3, Peg GPU4, Cage R2 GPU5, fresh bounded Lid V3 GPU6, Place GPU7 and Reorient R5 GPU2 are training. GPU1 handles remaining-task PPO preflights/diagnostics before the first Strike pilot. Lift R4 stopped on a simulator numerical failure; a30-update replay completed without reproducing it and retained model2028. Lift continuation awaits a slot with preceding-state capture enabled. Edge/Pivot/Throw passed physical readiness and await full pilots. GPU0 remains unused.

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
| Edge-Grasp | 6 | on | PPO preflight | Not measured | edge_v1 physical preflight passed; exposure and side-pinch shaping. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Waiting for slot after diagnostic replay | 0/128 final V3 | R4 failed at2009; same30-update replay finite, model2028 retained. Capture preceding state on next continuation. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Training fresh bounded V3 | 0/128 first pilot | R2 fully saturated, ownstep1717 stopped. Fresh lid_v2 onGPU6 after3 finite PPO updates. |
| Peg-Insertion | 20 | on | Training first pilot | Not measured | completion_v2 onGPU4; native square-bore fit unchanged. |
| Pivot-Lift | 6 | on | PPO preflight | Not measured | pivot_v1 physical preflight passed; actual wall/tilt history unchanged. |
| Place-In-Container | 20 | on | Training first pilot | Not measured | completion_v2 onGPU7; native released/contained/settled completion. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R5 closure retry | 0/16 R4 debug | reorient_v4/std reset0.25 from ownmodel1900; capture unchanged interface and physics. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training first pilot | Not measured | completion_v2 onGPU3; smooth closure and native completion incentive. |
| Strike-Slide | 4 | on | PPO preflight; first pilot next | Not measured | strike_v1 passed physics; GPU1 assigned after diagnostics. |
| Throw-To-Bin | 5 | on | PPO preflight | Not measured | throw_v1 passed physics; ballistic shaping, native contained/released/settled success unchanged. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.
