# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Latest completed CageR8/final13095 strict99/128 and StrikeR3/final6997 strict21/128, both small improvements and below the116/128 validation gate. Place retained6300 remains96/128. New Cage/Strike actual success and failure frames reviewed; no confirmation or new certification. GPU0 remains unused.

Current full PPO runs: EdgeR3 GPU1, ThrowR4 GPU2, LiftR10 GPU4, PlaceR7 GPU5 and PegR6 GPU6. GPUs3/7 are free after completed Strike/Cage runs and evaluations. Five process placements verified; GPU0 remains0MiB. PlaceR7 and LiftR10 use the recorded opt-in dense elliptic Hessian correction; PegR6 retains free-body gyro compatibility. All14 certified teachers retain their saved original backend. Holder20277 expires2026-09-28T08:58:54 scheduler time.

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
| Cage-Drag | 4 | on | Retry diagnosis | 99/128 R8 final13095 | Improved97→99; full budget completed and actual clips reviewed; below target. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R3 | 0/128 R2 final4998 | Closest pinch miss improved20.3→14.9cm; bounded unchanged continuation onGPU1. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R10 | 0/128 R9 final10496 | Near-goal holding remains unsettled; dense cone correction passed finite preflight; retained policy continues onGPU4. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Training R6 | 0/128 R6 model3900 | Corrected missed-ray aperture estimate in peg_v1, finite preflight, original3098 mean-0.25/std0.15, GPU6. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Open fingers contact low board; no opposing grasp or completed pivot. Actual frames reviewed. |
| Place-In-Container | 20 | on | Training R7 | 96/128 R6 model6300 | Small-T Hessian defect reproduced/fixed; full-size100-update stress passed; resumed original6300 with recorded cone compatibility, GPU5. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Retry diagnosis | 0/128 R9 final8598 | Full gyro-enabled budget finite; actual low capture without goal reorientation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Retry diagnosis | 0/128 R6 final5198 | Final opposing grasp95.8%, height2.62cm; held beside base without released stack. |
| Strike-Slide | 4 | on | Retry diagnosis | 21/128 R3 final6997 | Lower-LR refinement completed; endpoint calibration remains weak; actual clips reviewed. |
| Throw-To-Bin | 5 | on | Training R4 | 0/128 R3 final7497 | Reliable elevated capture, no release; gripper mean-0.4/std0.3 initialization passed preflight; full run onGPU2. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
