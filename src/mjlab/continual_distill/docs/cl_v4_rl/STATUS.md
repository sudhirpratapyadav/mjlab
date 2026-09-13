# Status — RL teachers

Checked2026-09-13. **16/24 independent PPO RL teachers certified.** Cage19092 is newest at119/128 validation and120/128 prospectively frozen confirmation20260916. Its actual videos, all-episode no-pinch aperture audit, finite checkpoint and13-file W&B teacher artifact are verified. Place7500 remains117/128+119/128. Throw9496 failed confirmation107/128 and remains uncertified.

Current full runs: ReorientR12 GPU1 final14595, ThrowR7 GPU2 final14098, LiftR13 GPU4 final17992, StackR9 GPU5 final12195 and EdgeR5 GPU7 final10995. Five actual process placements and eight-CPU affinities verified in training_wave30.json; GPUs3/6 are free after completed Strike/Peg evaluations. GPU0 remains unused. Holder20277 expires2026-09-28T08:58:54 scheduler time.

Latest completed strict validations: ThrowR6 110/128, StackR8 2/128, StrikeR7 39/128, PegR8 0/128. Actual clips reviewed; none pass the116 gate, so no confirmation or new certificate. Throw native/GPU recorded-control replay isolates box-floor angular settling; existing opt-in primitive compatibility passes18-case replay, native parity tests and64x3/2048x100 finite preflights before fullR7. Stack transport improves but gripper mean saturates; mean-only initialization reset passes64x3 and bounded-reset tests before fullR9, retaining learned std. Pivot staged diagnostic arrival develops upward force and median10.56deg tilt, below20deg; no new Pivot training recipe. Native success, physical models and60D/8D remain fixed.

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
| Cage-Drag | 4 | on | Certified | 119/128 +120/128 | model19092 retained; open-gripper transport reviewed and no-pinch audit passed. |
| Drag-Pull | 3 | on | Certified | 122/128 + 120/128 | model1300 retained; later training simulator failure archived separately. |
| Edge-Grasp | 6 | on | Training R5 | 0/128 R4 final8996 | Endpoint clearance passes32/32 at both openings; stronger side-pinch reward passed preflight. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | Training R13 | 56/128 R12 final15993 | Actual sustained lift/hold reviewed;122 endpoints inside5cm, settling remains incomplete. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Diagnosis | 0/128 R8 final8199 | Reviewed policy remains near fixture; next change awaits diagnosis. |
| Pivot-Lift | 6 | on | Retry diagnosis | 0/128 R2 final3499 | Small side/height target shifts did not produce tipping in CPU dynamics; no reward change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Training R12 | 1/128 R11 final12596 | Median angle39.63deg versus89.87; unchanged full continuation after finite-checkpoint gate. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Training R9 | 2/128 R8 final10196 | Better transport; mean-only gripper reset passed preflight, std retained. |
| Strike-Slide | 4 | on | Diagnosis | 39/128 R7 final14993 | Reviewed overshoot failure; endpoint precision remains incomplete. |
| Throw-To-Bin | 5 | on | Training R7 | 110/128 R6 final13099 | Matched replay and finite stress gate support existing box-contact compatibility; confirmation16 reserved. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.
