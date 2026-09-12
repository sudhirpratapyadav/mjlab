# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Latest strict results: Cage R7/final11096 **97/128 (75.8%)**, Place R6/model6300 **96/128 (75.0%)**, Edge R2/final4998 and Peg R5/model3400 **0/128**. None crosses the116/128 validation gate; no confirmation batches were run for these candidates. GPU0 remains unused.

Current full PPO runs: Pivot R2 GPU1, Throw R3 GPU2, Lift R9 GPU4 and Peg R6 GPU6. Edge/Cage completed their budgets and final evaluation. Place R6 stopped after iteration6329 with a captured nonfinite simulator state in lane995; latest retained6300 evaluated successfully. Preserve the failure trace and diagnose before resuming Place. Model6000 previously measured90/128. Peg R5 was deliberately stopped after the aperture fallback defect audit and a finite peg_v1 preflight; R6 restores original3098 with the preregistered gripper reset and inherited gyro compatibility. Lift R9 uses the preregistered gentler gripper initialization after matched-friction hold probes and finite PPO preflight.

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
| Cage-Drag | 4 | on | Retry prepared | 97/128 R7 final11096 | Improved37→97; actual caging/dragging reviewed; unchanged bounded R8 continuation preregistered. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Retry diagnosis | 0/128 R2 final4998 | Side-wrist shaping still ends pressing on the supported plate; budget and evaluation complete. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R9 | 0/128 R8 final8997 | Mean0/std0.05 gripper initialization after exact-friction hold probe; lift_v7/arm policy retained, GPU4. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R6 | 0/128 R5 model3400 | Corrected missed-ray aperture estimate in peg_v1, finite preflight, original3098 mean-0.25/std0.15, GPU6. |
| Pivot-Lift | 6 | on | Training R2 | 0/128 model1500 | Open-ramp/floor-safe pivot_v3 passed32 endpoint checks,66 focused tests and3 finite online PPO updates; own1500 ×2000 onGPU1. |
| Place-In-Container | 20 | on | Numerical failure diagnosis | 96/128 R6 model6300 | Improved35→74→90→96; training failed after6329/lane995. Retained checkpoint and failure trace; no blind restart. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Retry diagnosis | 0/128 R9 final8598 | Full gyro-enabled budget finite; actual low capture without goal reorientation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Retry diagnosis | 0/128 R6 final5198 | Final opposing grasp95.8%, height2.62cm; held beside base without released stack. |
| Strike-Slide | 4 | on | Retry diagnosis | 20/128 R2 final4998 | Improved9→20; actual launches and close misses reviewed. Quantify launch coverage versus endpoint error before next retry. |
| Throw-To-Bin | 5 | on | Training R3 | 0/128 R2 final5498 | Actual capture/lift,94.49% final opposing contacts; holds without release. Bounded unchanged continuation, GPU2. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
