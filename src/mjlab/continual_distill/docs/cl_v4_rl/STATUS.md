# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** Current training: Strike R2 GPU1, Throw R3 GPU2, Edge R2 GPU3, Lift R8 GPU4, Place R5 GPU5, Peg R4 GPU6 and Cage R7 GPU7. GPU0 remains unused. All24 tasks have RL training evidence; coverage is not certification.

Latest strict validation: Cage R6/final9097 improved16→37/128; Lift R7/final7498, Stack R6/final5198, Reorient R9/final8598 and Throw R2/final5498 each0/128. All completed budgets normally and their actual videos were reviewed. Lift places111 terminal objects inside5cm but none both inside and settled; lift_v7 adds mild loaded-grip/settling credit after a finite PPO preflight. Throw now captures/lifts (94.49% final opposing contacts) but holds without release; its bounded unchanged continuation targets launch. Stack remains low beside the base; Reorient remains near the floor.

Strike's contact-height and Edge's full side-wrist reward changes passed finite online PPO preflights before full continuations. Cage's2048-world10-update preflight passed before doubling its rollout batch. Place retains the previously verified5e-5 Adam learning rate. Optional free-body gyroscopic compatibility remains enabled only for the recorded Peg/Reorient lineages; all14 certified configurations retain their saved backend. Native predicates, model parameters and60D/8D remain fixed. Matched Lift hold interventions support a gentler-grip hypothesis but are not RL success measurements; original evaluation friction is not captured in old traces.

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
| Cage-Drag | 4 | on | Training R7 | 37/128 R6 final9097 | Improved16→37; unchanged cage_v4 from9097,2048 ×2000 after finite10-update batch preflight, GPU7. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R2 | 0/128 final2999 | Full side-wrist edge_v3 passed endpoint checks,62 focused tests and3 PPO updates; own2999 ×2000, GPU3. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R8 | 0/128 R7 final7498 |111 inside5cm but0 inside and settled; lift_v7 gentler loaded closure/settling bonus, preflight passed, GPU4. |
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
| Reorient-Object | 20 | on | Retry diagnosis | 0/128 R9 final8598 | Full gyro-enabled budget finite; actual low capture without goal reorientation. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Retry diagnosis | 0/128 R6 final5198 | Final opposing grasp95.8%, height2.62cm; held beside base without released stack. |
| Strike-Slide | 4 | on | Training R2 | 9/128 final2999 | strike_v2 precise contact approach; finite3-update preflight then own2999 ×2000 onGPU1. |
| Throw-To-Bin | 5 | on | Training R3 | 0/128 R2 final5498 | Actual capture/lift,94.49% final opposing contacts; holds without release. Bounded unchanged continuation, GPU2. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
