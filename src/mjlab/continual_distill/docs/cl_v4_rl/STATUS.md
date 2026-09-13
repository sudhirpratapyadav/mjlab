# Status — RL teachers

**Current: S11–S13 finished and failed success/motion review (11/128,3/128,59/128). All training is currently complete. S14/S15 preflights are prepared from S13: wider arm exploration and a10× bracket of continuous goal reward, with existing motion costs retained. Other23 teachers paused; baseline preserved.**

**Current authorization: motion-quality work on ONE teacher, Lift-Cube, only. All other teachers remain paused. Preserve the certified baseline and await user video review before applying changes elsewhere. See STATUS.md and the current Lift evaluation plans. This supersedes the blanket pause only for Lift.**

**PAUSED BY USER. All training and evaluation watchers stopped on 2026-09-13T09:49:28.168542+00:00. Do not resume automatically. Latest checkpoint/resume inventory: `evidence/user_pause_20260913.json`. Shared holder20277 remains intact.**

Latest completed results:18/24certified; Reorient109/128, Stack101/128, Strike57/128; Edge/Peg/Pivot0/128. Earlier running descriptions below are historical snapshots.

**18/24 independent PPO RL teachers certified.** Lift R16/model21990 is newest:118/128 validation and117/128 independent confirmation. Actual success/failure frames reviewed; all76 checkpoint tensors finite and13 W&B teacher files remotely verified (28m72bwp). The optional post-integration terminal recomputation agrees126/128 labels and passes116/128; its cached-velocity limitation is explicit in lift_r16_terminal_audit.json. The native measured rates remain authoritative. Six tasks remain unfinished.

Lift S3 completed all 3000 updates (final26488, SHA07a779f768d4838f54302f0beb6605fdfef224549ca79e4d8c71dd2b70f82c05). Strict validation: 117/128 (91.4%); independent confirmation seed20260919: 114/128 (89.1%), below the required116. All three provisional motion targets fail: p95 episode peak speed8.067→7.503rad/s, acceleration RMS proxy58.917→57.971rad/s², largest consecutive target jump4.557→4.609rad. Reject S3 as a replacement; original certified R16 remains intact. Paired real-time50fps videos and full metrics are archived at https://cl.sudhirpratapyadav.com/v4-rl/lift-smooth-review/s3/ and W&Bc1wirmb5. GPU0–7 each show0MiB/0% after completion; shared holder20277 remains intact, expires2026-09-28T08:58:54 scheduler time. Other23 teachers remain paused. No S4 launched or approved as a recipe; next Lift-only work must address the diagnosed command/actuator saturation with a tested, preregistered reward hypothesis, not merely repeat stronger motion weights. Confirmation19 is consumed and must not become the next confirmation seed.

S2 completed but is rejected:39/128 strictvalidation, no confirmation18. Allmotiongatesfail (p95peak7.431rad/s,acceleration55.988rad/s²,max-target-jump5.064rad). S2/model24988 is not a training parent. Its videos/results are archived at https://cl.sudhirpratapyadav.com/v4-rl/lift-smooth-review/s2/ and W&B4j9o7w5o. Sixplaybacks/mobile and8HTTPScontenthashes verified. S1 preserved successes while its motion remained insufficient; S3 returns to that successfulsource.

S1 finished but **did not fix motion quality**:117/128 validation and119/128 confirmation; all three motion gates fail (p95peak7.785rad/s,p95accelerationproxy58.423rad/s²,p95max-target-step4.624rad). Original certifiedR16 checkpoint remains preserved. Three fixed paired real-time50fps video pairs published at https://cl.sudhirpratapyadav.com/v4-rl/lift-smooth-review/s1/; browser6playbacks/mobile and HTTPS8contenthashes verified. W&B motion-review pbyye5xz retains candidate,full traces,configs,metrics,videos; remote candidate digest verified. User visual approval has not been requested for this insufficient result.

ReorientR16, EdgeR7, StackR13 and PivotR5 were stopped before their planned budgets; their partial checkpoints are recorded in user_pause_20260913.json and are not final evaluation candidates. The other teachers remain paused until the user reviews Lift motion quality. GPU0 is unused; the shared holder20277 remains intact and expires2026-09-28T08:58:54 scheduler time.

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
| Edge-Grasp | 6 | on | Paused | 0/128 R6 final12994 | Matched CPU7/GPU3 diagnostic held lifts; clearance waypoint reward, no backend change. |
| Flip-Switch | 3 | on | Certified | 128/128 +128/128 | model800 retained; physical switch review passed. |
| Lift-Cube | 20 | on | S9/S10 training; motion unresolved | Baseline118/128 +117/128 | R16 retained; S4 111/128, S5 116/128 +118/128, both fail motion targets. |
| Open-Door | 3 | off | Certified | 128/128 +128/128 | model900 retained; physical door review passed. |
| Open-Drawer | 3 | off | Certified | 128/128 + 128/128 | model1000 retained; stopped only own training step20277.1635 after certification. |
| Open-Lid | 5 | on | Certified | 128/128 +128/128 | Bounded lid_v2 model700; actual lid opening reviewed; artifact retained. |
| Peg-Insertion | 20 | on | Paused | 0/128 R11 final12197 | Stronger lifting credit did not solve lift;92 terminal opposed captures, no2cm lift. |
| Pivot-Lift | 6 | on | Paused | 0/128 R4 final8497 |97 sampled tilts>20deg; correct tipping-to-grasp reward inversion, finite50-update preflight. No aperture change. |
| Place-In-Container | 20 | on | Certified | 117/128 +119/128 | model7500 retained; placement/release physically reviewed,15th certificate. |
| Push-Button | 3 | off | Certified | 128/128 + 128/128 | model700 retained; stopped only own training step20277.1632 after certification. |
| Push-Cuboid | 3 | on | Certified | 126/128 +125/128 | model1998 retained; moving endpoint is valid under unchanged position-only predicate. |
| Push-Flap | 3 | off | Certified | 128/128 +128/128 | model900 retained with physical video review. |
| Reach-Target | 20 | on | Certified | 128/128 + 128/128 | model499 retained with normalizers, reviewed video and W&B artifact. |
| Reorient-Object | 20 | on | Paused | 109/128 R15 final20592 | Reviewed settling improvement4→109; unchanged guarded continuation from20592. |
| Rotate-Valve | 8 | off | Certified | 127/128 +128/128 | model1999 retained; sole validation failure was ground collision. |
| Slide-Window | 3 | off | Certified | 128/128 +128/128 | model300 retained; physical slide review passed. |
| Stack-Cube | 20 | on | Paused | 101/128 R12 final16193 |126 ever captured;111 supported/119 released terminal cubes. Native CPU/GPU labels agree128/128. R13 partial budget stopped. |
| Strike-Slide | 4 | on | Paused | 57/128 R12 final18991 | Half-noise refinement remains below R9 best59/128; no confirmation. |
| Throw-To-Bin | 5 | on | Certified | 125/128 +126/128 | model14098, reserved confirmation16; actual videos and native endpoint audit passed. |
| Topple-Block | 4 | on | Certified | 123/128 + 118/128 | model499 retained; highest-return failure correctly fails settling predicate. |
| Turn-Lever | 3 | off | Certified | 128/128 +128/128 | model1000 retained with physical video review. |

## Resource limits

Seven A100 80GB GPUs authorized by the user (indices 1–7); never use GPU0. Holder expiry is September 28 per Slurm. Last filesystem snapshot: about 1.6 TB free. Pilot processes used about 1 GB (Reach) or 2.5 GB (contact tasks) per GPU at 1,024 environments. Recheck GPU use, free space and holder lifetime before a long run.

## Public summary

https://cl.sudhirpratapyadav.com/v4-rl/ — new current-experiment home-page card, concise24-task gallery, actual RL policy videos and clear certified/in-progress status. Detailed records remain in W&B.

Wave35 evaluation scheduling: each full trainer has a frozen final-checkpoint plan and an identified-process dispatcher. Do not duplicate dispatch or certify until actual video review and W&B retention checks pass.
