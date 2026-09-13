# Status — RL teachers

Checked2026-09-13. **14/24 independent PPO RL teachers certified.** All24 have RL training evidence. ThrowR4=116/128 validation but107/128 confirmation, so not certified. Best other unfinished validations: Cage108/128, Place105/128, Strike34/128. Latest LiftR11/final13994 and PegR7/final7096 remain0/128, with actual video reviews. Peg lost capture; retainR6/5097 for diagnosis. GPU0 remains unused.

Current full continuations, one per GPU1–7: ReorientR10, ThrowR5, StrikeR5, PlaceR8, StackR7, EdgeR4 and CageR10. Place passed its2048-world100-update stability gate and started from original best7400 with coneON/primitiveON/gyroOFF. First prospective candidate7500; full maximum9399. Holder20277 expiry2026-09-28T08:58:54 scheduler time. ThrowR5 must finish all2000 updates before validation and use fresh confirmation20260916 only if it passes; do not retest9496 on that seed.

Place contact diagnosis:17/30 failed replay states rock on one GPU floor contact, with no robot contact; native CPU at identical poses finds four contacts. Opt-in primitive_box_box_compat settles30/30 versus13/30 baseline under matched controls/friction, with32 synthetic GPU support cases and16 native CPU support comparisons passing. The13 focused checks include strict evaluation/backend restoration. Native model/60D/8D/success predicates are unchanged, and all14 certificates retain their original backends. Short Place64×3 preflight is finite atAdam5e-5. No new RL rate has been claimed from these interventions.

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
| Cage-Drag | 4 | on | Continuing R10 | 108/128 R9 final15094 | Unchanged lower-LR refinement onGPU7. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R4 | 0/128 R3 final6997 | Closest pinch miss39.84mm; unchanged bounded continuation onGPU6. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Grip/control diagnosis | 0/128 R11 final13994 | Still near-goal hold without settling; primitive-contact replay adds little. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Capture regression diagnosis | 0/128 R7 final7096 | Lost acquired capture; retainR6/5097, do not continue7096 unchanged. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Small side/height target shifts did not produce tipping in CPU dynamics; no reward change. |
| Place-In-Container | 20 | on | Training R8 | Best105/128 at7400; final8299=97/128 | Primitive contacts passed full-world PPO gate; first candidate7500 onGPU4. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R10 | 0/128 R9 final8598 | Axis-credit plateau audited; preflighted reorient_v8 onGPU1. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R7 | 0/128 R6 final5198 | Capture acquired, transport low; unchanged3000-update continuation onGPU5. |
| Strike-Slide | 4 | on | Continuing R5 | 34/128 R4 final8996 | Unchanged precision refinement onGPU3. |
| Throw-To-Bin | 5 | on | Training R5 | 116/128 validation +107/128 confirmation | Not certified; lower-LR refinement, fresh prospective confirmation seed16. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
