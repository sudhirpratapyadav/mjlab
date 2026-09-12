# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** Training: Cage R6 GPU1, Throw R2 GPU2, Stack R6 GPU3, Lift R7 GPU4, Peg R4 GPU6 and Reorient R9 GPU7. Place R5 is training onGPU5 from retained3400 with verified Adam LR5e-5 after its later policy regressed. GPU0 remains unused. All24 tasks have RL training evidence; coverage is not certification.

Lift R6 final5499 strict0/128, but actual grasp/lift/transport now occurs: mean final goal error9.50cm and height19.65cm, improved from26.06cm/2.78cm at4100. Continue its unchanged policy from5499. Throw final2999 and Peg R3/2300 each strict0/128; actual failure clips reviewed. Place learned grasps then regressed after3400 into open-hand joint-limit saturation; retained finite3400, strict0/128, reviewed grasp/lift beside the container. R5 continues3400 at half the learning rate after a finite3-update PPO check.

Captured Peg/Reorient spin failures identify a missing native MuJoCo free-body gyroscopic correction in the installed Warp implicitfast path. Optional backend compatibility sourcef821aa0 passes CPU comparisons,64-world sparse/dense CUDA-graph regressions, two2048-world captured-state regressions and both100-update2048-world PPO stress runs. Peg's new fresh PPO lineage and Reorient's own resumed policy enable the recorded correction. It defaults off and is restored from the run manifest in evaluation/certification; all existing certified teachers retain their configuration. Model parameters, timestep, success predicates and60D/8D remain unchanged. Contact trajectories may differ from CPU; no claim that all simulator instability is resolved.

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
| Cage-Drag | 4 | on | Training R6 | 16/128 R5 final7598 | Native caging improved from2/128; unchanged cage_v4 from7598 ×1500 additional, GPU1. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Rejected first pilot; retry needed | 0/128 final2999 | Slides plate to overhang; FK audit shows20.3cm closest pinch-target miss and0.40 vertical closing alignment. edge_v3 full side-wrist/capture reward prepared;32 endpoint checks and62 focused tests pass; PPO preflight pending. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R7 | 0/128 R6 final5499 | lift_v6 from5499, unchanged reward/output/std/backend; genuine lift/transport, exact goal still missed. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R4 | 0/128 R3 model2300 | Fresh completion_v6 stress lineage from99; gyro correction passed100 full-size PPO updates, GPU6. |
| Pivot-Lift | 6 | on | Rejected; retry needed | 0/128 model1500 | Actual board top pressing. CPU terminal contact audit:124/128 two-pad contacts,0 opposing pairs. |
| Place-In-Container | 20 | on | Training R5 | 0/128 R4 model3400 | Actual grasp/lift beside container without release; LR5e-5 continuation after verified PPO preflight, GPU5. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R9 | 0/128 R7 final6099 | R8 stopped6589 with spin failure; own stress6599 resumes with optional gyro correction onGPU7. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R6 | 0/128 R5 final4199 | Opposing grasps65.08% at final surviving-world sample; no stacking. Unchanged completion_v6 from4199 ×1000, GPU3. |
| Strike-Slide | 4 | on | Contact retry prepared | 9/128 final2999 | Strikes and retracts;58/128 negligible launches,25 weak launches,45 substantial launches with9 successes. Actual success/failure reviewed; native4s/friction/predicate unchanged. |
| Throw-To-Bin | 5 | on | Training R2 | 0/128 final2999 | throw_v4 from2999/std0.15, opposed contacts/local width;3-update PPO preflight passed, GPU2. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
