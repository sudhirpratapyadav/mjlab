# Experiments — RL teachers

Use a unique run ID, record the hypothesis and config change before launch, then append results and promotion decisions. Never overwrite failed or weaker trials. Planned budgets below are not running jobs.

| ID | Task/scope | Purpose | Budget | Status | Result |
|---|---|---|---|---|---|
| PREFLIGHT-CFG-001 | All 24 | Check registered interface/PPO/physics | Read-only configs | Complete | 60D/8D; eight zero-gravity presets; all default num_envs=1 |
| PREFLIGHT-GPU-001 | dgx1/holder20277 | Availability and assignment | Snapshot | Complete | GPUs1–3 assigned and idle; no experiment steps at snapshot |
| PREFLIGHT-PPO-001 | Reach | Real optimization + checkpoint round trip | 64 envs × 24 steps × 2 updates | Complete | 3,072 transitions; finite losses; parameters changed; reload and optimizer restore passed; no success evaluation |
| PREFLIGHT-CLI-001 | Launcher | UUID pinning compatibility | No training | Complete | Generic selector fails; direct runner works; stage-local launcher dry-run validated |
| EVAL-001 | Evaluator | Strict first-episode checkpoint evaluation | Reach smoke checkpoint, 8 episodes seed20260913 | Complete | End-to-end smoke0/8; adversarial actual-loop tests passed including subset resets, terminal capture and retry exclusion |
| RL-001 | Reach | First full training/evaluation pilot; baseline physics and reward | 1,024 envs, 500 iterations, seed20260912, GPU2 | Complete | model_499.pt: 128/128 validation seed20260914, 128/128 confirmation seed20260915; strict first-episode evaluator; both timeout at full horizon; history, evaluation and checkpoint imported into the verified Sudhir project |
| RL-002 | Lift | Learn grasp + lift with common interface; baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU1 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-003 | Push-Cuboid | Test short-horizon push learning with baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU3 | Failed | PPO action std became invalid at iteration 268; model_200.pt strict validation 1/128 seed20260914 |
| RL-004 | Drag-Pull | Pilot contact learning under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU4 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-005 | Reorient-Object | Pilot settled orientation learning under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU5 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-006 | Topple-Block | Pilot contact and settled completion under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU6 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_200.pt |
| WANDB-002 | Logging | Verify Sudhir credential/destination and import all six pilot histories | One test write/read, six history/checkpoint imports | Complete | Correct entity is sudhirpratapyadav-indian-institute-of-technology-jodhpur; global settings unchanged; evidence/wandb_destination.json and wandb_pilot_imports.json |
| RL-002-R1 | Lift | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU1 | Complete | model499 strict validation0/128; baseline rejected, fresh shaped variant follows |
| RL-004-R1 | Drag-Pull | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU4 | Complete | model499 strict validation100/128; longer training follows |
| RL-005-R1 | Reorient | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU5 | Complete | model499 strict validation0/128; baseline rejected, fresh shaped variant follows |
| RL-006-R1 | Topple | Complete interrupted baseline pilot from its own model_200 | 1,024 envs, 300 additional updates, seed20260912, GPU6 | Certified | model499 validation123/128 and confirmation118/128; settled success and high-return moving failure reviewed; retained W&B artifact |
| REWARD-001 | Mechanisms | Recover useful reset-to-contact reward signal | Define after curve review | Planned | Not launched |
| INIT-001 | Cage | Keep valid open cage during initial exploration | Define open-gripper prior/curriculum | Planned | Not launched |
| COMPLETE-001 | Stack/Peg/Place | Align dense return and strict completion | Define after reward review | Planned | Not launched |
| PREFLIGHT-GUARD-001 | Push-Button | Exercise finite-state/update guard and mechanism_v1 recipe | 64 envs, 3 updates, seed20260912, GPU2 | Complete | Finite updates, positive log std and W&B diagnostics verified; no success claim |
| RL-007 | Push-Button | Learn contact/manipulation with non-saturated approach | mechanism_v1, 1,024 envs, 1,500 updates, seed20260912, GPU2 | Certified | model700:128/128 validation +128/128 confirmation; physically depressed button reviewed; stopped training step1632 after artifact retention. |
| RL-008 | Open-Drawer | Learn approach and pull with the same mechanism recipe | mechanism_v1, 1,024 envs, 1,500 updates, seed20260912, GPU7 | Certified | model1000:128/128 +128/128; physically open drawer reviewed; stopped training step1635 after retention. |
| RL-002-V2 | Lift | Reward alignment/closure, actual two-pad grasp, lift and target progress | lift_v1, 2,048 envs, 1,500 updates, seed20260912, GPU1 | Complete; rejected | model1499 strict0/128; fresh bounded-policy variant follows. |
| RL-005-V2 | Reorient | Replace saturated reach/tiny angular shaping with grasp and broad orientation progress | reorient_v1, 2,048 envs, 1,500 updates, seed20260912, GPU5 | Complete; rejected | model1499 strict validation0/128; action saturation1.0. Recorded trace retained; diagnose before next recipe. |
| RL-003-V2 | Push-Cuboid | Diagnose/prevent original invalid std failure | stable_v1, 1,024 envs, 1,000 updates, seed20260912, GPU3 | Complete | model999 strict111/128; no numerical failure; prepared further training. |
| RL-004-R2 | Drag-Pull | Extend 100/128 baseline teacher | baseline_long, 1,024 envs, 1,000 additional updates, seed20260912, GPU4 | Stopped by numerical guard; model1300 certified | model1300 strict122/128 +120/128; success/failure videos reviewed and artifact retained. Later iteration1380 simulator failure remains archived. |

| PREFLIGHT-MECH-002 | Remaining mechanisms | Finite rollout and interface checks for the validated mechanism recipe | 16 environments × 32 steps each, GPU2 | Complete | All eight passed finite rollout and interface/benchmark invariants; evidence/mechanism_preflight2.json |
| RL-009 | Flip-Switch | Transfer successful Button/Drawer learning recipe to switch geometry | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU2 | Certified | model800:128/128 +128/128; switch physically flips to its target side; retained artifact and stopped own step1655. |
| RL-010 | Open-Door | Learn approach and door rotation | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU7 | Certified | model900:128/128 +128/128; physical door opening reviewed; retained artifact and stopped own step1656. |
| RL-011 | Push-Flap | Learn flap contact and rotation | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU4 | Certified | model900 strict128/128 +128/128; physical flap rotation reviewed; stopped own step1659 after retention. |

| RL-003-R3 | Push-Cuboid | Extend the stable policy after strict validation 111/128 | stable_v1, 1,024 environments × 1,000 additional updates, seed20260912, GPU3 | Certified | model1998 strict126/128 +125/128; reviewed actual success/failure. Native criterion requires position within0.02m, not settling; moving success explicitly recorded. |

| RL-012 | Turn-Lever | Apply validated mechanism approach/goal recipe | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU6 | Certified | model1000 strict128/128 +128/128; physical downward lever rotation reviewed; stopped own step1662 after retention. |

| PREFLIGHT-BOUNDED-001 | Cage/Lift/Reorient | Validate bounded Gaussian means and open-gripper initialization | 64 environments × 200 steps each; then 3 guarded PPO updates | Complete | 48 tests; three200-step physical cases finite; Cage min aperture0.076m >0.06949m; three guarded PPO updates and strict checkpoint reload passed. |
| RL-005-V3 | Reorient | Recover from saturated, ground-colliding V2 policy | reorient_v2, 2,048 environments × 2,000 updates, seed20260912, GPU2 | Paused for revised reward | Stopped own step1675; model1500 resumed inR4. |
| RL-002-V3 | Lift | Test bounded-mean policy and gentler exploration | lift_v2, 2,048 environments × 2,000 updates, seed20260912, GPU1 | Complete; rejected | model1999 strict0/128; no sampled two-pad grasp. |
| RL-013 | Cage-Drag | Keep initial exploration valid and reward actual open-cage transport | cage_v1, 1,024 environments × 1,500 updates, seed20260912, GPU5 | Complete; rejected | model1499 strict0/128;97.7% final caging, negligible transport. |

| RL-014 | Slide-Window | Learn the remaining slide mechanism | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU7 | Certified | 128/128 validation and128/128 independent confirmation; actual recorded motion reviewed; checkpoint/normalizers/video retained. |
| RL-015 | Open-Lid | Learn lid approach and rotation | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU4 | Complete; rejected | model1499 strict0/128, alltimeouts. |
| RL-016 | Rotate-Valve | Learn large-angle valve rotation | mechanism_v1, 1,024 environments × 2,000 updates, seed20260912, GPU6 | Certified | model1999:127/128 +128/128; videos reviewed and artifact retained. |
| RL-017 | Axial-Extract | Learn plug approach and extraction | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU3 | Certified | 128/128 validation and128/128 independent confirmation; actual recorded motion reviewed; checkpoint/normalizers/video retained. |

| PREFLIGHT-COMPLETE-001 | Stack/Place/Peg | Check safe grasp geometry, release incentive and finite physics | 53 focused CPU tests;32 environments ×100 steps each;3 PPO updates | Complete | Physics/PPO passed; smooth closure revision separately checked before full pilots. |
| RL-018 | Stack-Cube | Learn grasp/transport/released supported completion | completion_v2,2,048 environments ×2,500 updates, seed20260912, GPU3 | Stopped for reward loophole | Exactstep1704 stopped; finite model1800 retained. Model1500 debug0/16; top pressing with two-pad contact/zero enclosure verified in video. |
| RL-019 | Place-In-Container | Learn grasp/transport/released contained completion | completion_v2,2,048 environments ×2,500 updates, seed20260912, GPU7 | Stopped for reward loophole | Exactstep1705 stopped; finite model1500 retained. Model1200 debug0/16, nearly closed contact/no lift. |
| RL-020 | Peg-Insertion | Grasp upper peg body, align, seat and release | completion_v2,2,048 environments ×3,000 updates, seed20260912, GPU4 | Stopped for reward loophole | Exactstep1714 stopped; finite model1100 retained. Model800 debug0/16, nearly closed contact/no seating. |

| PREFLIGHT-CLOSURE-002 | Stack/Place/Peg/Reorient | Remove the hard35mm closure-reward discontinuity | Monotonicity test;32 environments ×100 steps;3 guarded PPO updates | Complete | 58 tests; revised physical and PPO preflights finite. |
| RL-005-R4 | Reorient | Escape the verified approach/closure reward cliff | reorient_v3,2,048 environments ×1,500 additional updates, seed20260912, GPU2 | Stopped by numerical guard | One lane nonfinite qpos/qvel at1970; model1900 debug evaluation pending. |

| PREFLIGHT-TRANSPORT-003 | Cage/Lid | Verify targeted transport and hinge-progress retries | Three resumed guarded PPO updates on64 environments each | Complete | 61 tests; both resumed64-env PPO preflights3 finite updates. |
| RL-013-R2 | Cage-Drag | Escape stationary valid caging | cage_v2,1,024 environments ×2,000 additional updates, seed20260912, GPU5 | Complete; rejected | Finalmodel3498 strict0/128;100% valid open cage but0% final contact. Actual video reviewed; contact-target retry prepared. |
| RL-015-R2 | Open-Lid | Learn a grasp and useful opening before the narrow goal reward | lid_v1,1,024 environments ×1,500 additional updates, seed20260912 | Stopped after verified full saturation | Stopped exactstep1717; mean12644, saturation1.0, no training success. Evidence/Lid-R2-saturation.json. |

## Run record template

- ID, task, hypothesis, status, start/end UTC, Slurm step, GPU UUID.
- Training seed, environment count, iterations/transitions, interface version, source/config hashes, reward and physics changes.
- Checkpoint and normalizer path/hash, losses, reward components, action saturation, termination counts, throughput and measured GPU memory.
- Strict successes/trials, evaluation seeds, deterministic/stochastic policy mode, episode budget, first-episode accounting, videos.
- Decision: retain/reject/continue, why, and next experiment. A teacher is certified only by the GOAL.md criterion.

| PREFLIGHT-LIFT-CLOSURE-004 | Lift | Validate stronger closure and gripper-only exploration reset | lift_v3, resume ownV3/model1999, gripper std0.2,64 environments ×3 updates, GPU1 | Complete | 3 finite PPO updates; isolated std/momentum reset checks passed. |
| RL-002-R4 | Lift | Learn actual contact from successful approach | lift_v3, ownV3/model1999, gripper std0.2,2048 environments ×2000 additional updates, seed20260912, GPU1 | Stopped by numerical guard | Nonfinite simulator state at2009; policy parameters finite.30-update diagnostic replay did not reproduce. |

| PREFLIGHT-REORIENT-CLOSURE-005 | Reorient | Check stronger smooth closure and gripper std reset | reorient_v4, ownR4/model1900, gripper std0.25,64 environments ×3 updates, GPU2 | Complete | 3 finite PPO updates and checkpoint; std0.25 verified. |
| RL-005-R5 | Reorient | Learn end-face closure from finite own approach policy | reorient_v4, ownR4/model1900, gripper std0.25,2048 environments ×2000 additional updates, seed20260912, GPU2 | Complete; rejected | Final3899 strict0/128; closed-finger hover, no contact/enclosure; video reviewed. |

| PREFLIGHT-LIFT-FAILURE-006 | Lift | Capture preceding physics state for numerical failure diagnosis | Same ownV3/model1999, lift_v3/std0.2,2048 environments ×30 updates, GPU1 | Complete; no recurrence | 30 updates finite; model2028 retained. Capture preceding state on next full continuation. |
| PREFLIGHT-LID-BOUNDED-007 | Lid | Check fresh bounded policy after verified R2 saturation | lid_v2,64 environments ×3 updates, GPU6 | Complete | 3 finite PPO updates and checkpoint verified. |
| RL-015-V3 | Lid | Learn hinge progress with bounded action means | lid_v2,2048 environments ×2000 updates, seed20260912, GPU6 | Certified | model700:128/128 +128/128; actual opening reviewed; certification runp5lo2knx. |

| PREFLIGHT-REMAINING-008 | Edge/Pivot/Strike/Throw | Validate precursor/dynamic shaping with native completion | Each32 environments ×100 steps, GPU1;70 focused CPU tests | Complete | 70 CPU tests; four real-physics cases finite; evidence/remaining_preflight_v1.json and W&B5g2qp9h4. |
| RL-021 | Edge-Grasp | Learn edge exposure, rim grasp and lift | edge_v2,2048 environments ×3000 updates, seed20260912, GPU6 | Running | Fresh edge_v2 onGPU6 after Lid certification; enclosed grasp credit, native benchmark unchanged. |
| RL-022 | Pivot-Lift | Learn wall-assisted pivot, capture and lift | pivot_v2,2048 environments ×3000 updates, seed20260912, GPU5 | Stopped; rejected | Finite1500 strict0/128; actual board pressing reviewed; no pivot/lift. |
| RL-023 | Strike-Slide | Learn accurate contact launch to unreachable goal | strike_v1,2048 environments ×3000 updates, seed20260912 | Complete; rejected | Final2999 strict9/128; actual strike/slide reviewed, mostly undershoot. |
| RL-024 | Throw-To-Bin | Learn grasp, launch and release into distant bin | throw_v3,2048 environments ×3000 updates, seed20260912, GPU2 | Running | Fresh throw_v3 GPU2; pre-step capture enabled; training_wave11.json. |

| PREFLIGHT-REMAINING-PPO-009 | Edge/Pivot/Strike/Throw | Check real guarded PPO updates before full pilots | Each64 environments ×3 updates, seed20260912, sequential onGPU1 | Complete | All four cases3 finite PPO updates and finite model2 checkpoints; remaining_ppo_preflight.json. |

RL-023 first full Strike pilot assignedGPU1 after remaining-task PPO preflight. Lift R4 numerical failure did not recur in the30-update diagnostic replay; retained finite checkpoint2028. Its continuation waits for a free slot while remaining tasks receive first pilots; preceding-state capture is prepared for a recurrence. No finite-state recovery or altered solver/physics has been introduced.

| PREFLIGHT-ENCLOSURE-010 | Stack/Place/Peg | Reject top pressing as a training grasp while preserving native success | completion_v3, fresh64 environments ×3 updates each, GPUs3/7/4 | Complete | 74 tests; three3-update PPO preflights/checkpoints finite; enclosed_grasp_preflight.json. |
| RL-018-V2 | Stack | Learn opposed grasp and released stacking | completion_v3, fresh2048 environments ×2500 updates, seed20260912, GPU3 | Stopped by numerical guard | Iteration777, nonfinite lane261; finite700 retained; debug0/16 open enclosure/no contact. |
| RL-019-V2 | Place | Learn opposed grasp and released containment | completion_v3, fresh2048 environments ×2500 updates, seed20260912, GPU7 | Stopped; rejected | Finite1500 strict0/128, open enclosure/no opposed contact; Place-V2-review.json. |
| RL-020-V2 | Peg | Learn opposed upper-body grasp and released seating | completion_v3, fresh2048 environments ×3000 updates, seed20260912, GPU4 | Stopped; rejected | Finite1500 strict0/128, topples shaft then presses down; Peg-V2-review.json. |

| PREFLIGHT-REMAINING-ENCLOSURE-011 | Edge/Pivot/Throw | Exclude top pressing before first full pilots | edge_v2/pivot_v2/throw_v2, each64 environments ×3 updates sequential onGPU6, seed20260912 | Complete | 79 tests; Edge/Pivot/Throw each3 finite PPO updates/checkpoint; remaining_enclosure_preflight.json. |

| PREFLIGHT-CAGE-CONTACT-012 | Cage | Verify contact-point target after0/128 hover failure | cage_v3, resume ownR2/model3498,64 environments ×3 updates, GPU5 | Complete | 81 tests;3 resumed PPO updates and finite model3500. cage_contact_preflight.json. |
| RL-013-R3 | Cage | Learn open-finger contact transport from valid caging policy | cage_v3, ownR2/model3498,1024 environments ×2000 additional updates, seed20260912 | Stopped by numerical guard |4649, lane447; recorded reset-cache failure reproduced, finite4600 retained. |

| PREFLIGHT-CAPTURE-WIDTH-013 | Throw/Lift/Reorient | Avoid closing empty fingers before object capture | throw_v3/lift_v5/reorient_v6, fresh64 environments ×3 updates each, sequential GPU2, seed20260912 | Complete | Three3-update PPO cases and checkpoints finite; capture_width_preflight.json. |

Reorient R5 finalmodel3899 strict0/128. Recorded failure shows closed-finger hovering; final mean action-0.999663, aperture0.0000375m, no contact/enclosure. The gripper-only output reset is now implemented and tested; use explicit --resume-gripper-mean and --resume-gripper-std when supported by measured saturation. Seven arm outputs are preserved. Fresh geometry-aware variants use a half-open initial gripper.

| PREFLIGHT-GRIPPER-OUTPUT-014 | Stack | Validate gripper-output reset preserving learned arm approach | completion_v4, ownV2/model700, gripper mean0.5/std0.15,64 environments ×3 updates, GPU3 | Complete | 88 tests;3 resumed PPO updates and finite702; isolated gripper reset verified. |
| RL-018-R3 | Stack | Learn width-matched capture while preserving valid approach policy | completion_v4, ownV2/model700, mean0.5/std0.15,2048 environments ×1800 additional updates, seed20260912, GPU3 | Stopped by numerical guard |1478, lane1322; reproduced reset-cache failure. Finite1400 strict0/128, open enclosure. |

### Wave12 scheduling — 2026-09-13

Pre-launch decision (retained as recorded): after Strike completes3000 PPO updates and syncs finalmodel2999, run strict128-episode validation and actual-state video onGPU1 after process exit. Assign Cage R3 the same slot after that review: cage_v3, ownR2/model3498,1024 environments ×2000 additional updates, seed20260912, preceding-state capture enabled. Its contact-target PPO preflight already passed. No gripper reset is planned because Cage must preserve the successfully learned open/no-pinch behavior. All registered physics, success predicates and60D/8D interfaces stay fixed.

Place/Peg V2 paused exactsteps1738/1739 after verifying task/PID identity and finite model1500. Strict validation0/128 each with actual failure videos reviewed. Place holds an open geometric cage without opposed contact; Peg knocks the upright shaft over and presses the horizontal peg. These are rejected candidates. Slots4/7 now serve prepared Lift/Reorient retries while completion-task interventions are reassessed.

| PREFLIGHT-CAPTURE-RESUME-015 | Lift/Reorient | Check width-shaped rewards with explicit gripper-only output/exploration reset from own finite policies | lift_v5 from diagnosticreplay2028 onGPU4; reorient_v6 fromR5/3899 onGPU7; each64 environments ×3 updates, mean0.5/std0.15, capture enabled | Complete | Both3 updates/checkpoints finite; online sync verified; capture_resume_preflight.json. |
| RL-002-R5 | Lift | Learn actual capture from own finite approach policy | lift_v5, ownPREFLIGHT-LIFT-FAILURE-REPLAY/model2028, mean0.5/std0.15,2048 environments ×2000 additional updates, seed20260912, GPU4 | Running | Preserve learned arm; preceding-state capture enabled. |
| RL-005-R6 | Reorient | Escape verified closed empty-finger policy with width-shaped capture | reorient_v6, ownR5/model3899, mean0.5/std0.15,2048 environments ×2000 additional updates, seed20260912, GPU7 | Stopped by numerical guard |4609; replay finds exploding lane853. Finite4600 retained; distinct from reset-cache issue. |

### Captured Stack failure and reset-cache readiness fix

Stack R3 guard stopped at1478 onlane1322, policy finite. Captured pre-step qpos/qvel show a freshly reset episode, while qacc_warmstart still contains the previous episode's acceleration. Replaying those physical states/actions reproduces first-substep failure in both1 and2048 worlds; CPU MuJoCo is finite with no warnings. Clearing only qacc_warmstart makes all4 substeps finite. Added selective solver-cache reset to the actual environment reset path; no solver parameters, geometry, success or60D/8D changes.85 focused tests passed before the additional reward variant below.

| PREFLIGHT-RESET-PPO-016 | Stack | Exercise the reset-cache fix under real PPO after captured-state regression | completion_v4, ownR3/model1400,2048 environments ×100 additional updates, seed20260912, GPU3, capture enabled | Complete |100 updates/model1499 finite and online; stack_reset_preflight.json. |
| RL-018-R4 | Stack | Learn contact closure with stale solver acceleration removed at reset | completion_v5, ownR3/model1400, gripper std0.15,2048 environments ×1800 additional updates, seed20260912, GPU3, capture enabled | Running | Strict R3 model1400 is0/128,100% enclosure but0% contact. Use contact-closure shaping after reset and task-specific PPO preflights. |
| PREFLIGHT-SQUEEZE-017 | Place | Validate width-aware contact pressure after open-cage failure | completion_v5, ownV2/model1500, gripper std0.15,64 environments ×3 updates, GPU5 | Complete |3updates/model1502 finite and online; new reward target clearance shifts from+3mm to-4mm at zero distance, smoothly retaining positive clearance far away; actual object width prevents empty closure. Physics/contact stiffness unchanged. |
| RL-019-R3 | Place | Learn bilateral contact from successful geometric enclosure | completion_v5, ownV2/model1500, gripper std0.15,2048 environments ×2000 additional updates, seed20260912, GPU5, capture enabled | Running | Preserve learned arm and gripper means; reward-only squeeze target plus solver-cache reset fix. |

Pivot model1500 strict0/128; actual failure presses board flat. CPU contact-normal audit found124/128 two-pad terminal contacts but0 opposing inward-normal pairs. Peg likewise18/128 two-pad contacts but0 opposed. Original GPU contact buffers were not saved, so these are recomputed CPU geometry checks on exact recorded states. The current whole-object enclosure gate correctly rejects the observed presses; the possible corner-grasp limitation remains an unproven hypothesis and no classifier change is made.

Fresh-start probe across all24 registered tasks found qacc_warmstart already zero before the first reset; the new clearing operation is a no-op there (cold_solver_reset.json). It prevents stale cross-episode acceleration during ongoing training. Stack R3 model1400 strict0/128, actual frames reviewed:100% final enclosure but0% contact, aperture0.06151m. Its next full retry therefore also uses completion_v5 after a3-update task-specific preflight, with gripper std0.15 and preserved mean. The100-update reset regression remains completion_v4 to isolate the numerical fix.

| PREFLIGHT-STACK-SQUEEZE-018 | Stack | Check contact-closure shaping after finite reset regression | completion_v5, ownR3/model1400, gripper std0.15,64 environments ×3 updates, GPU3 | Complete |3updates/model1402 finite and online; stack_squeeze_preflight.json. |

Reorient R6 stopped at4609 after a nonfinite rollout result; its post-reset qpos/qvel were finite, so the old guard did not identify a lane. Captured-state physics replay finds lane853 nonfinite on substep4, even when warm starts are cleared. Before that transition, object linear speed~78m/s and angular speed~9100rad/s are already extreme. This is distinct from the resolved Stack reset-cache failure. Finite4600 retained. Future capture keeps8 preceding control states and reports nonfinite reward/observation lane IDs, including failures whose native resets already replaced qpos/qvel. No invalid-lane recovery or physics changes.

| RL-005-R7 | Reorient | Continue finite capture policy while preserving enough history to locate the remaining instability | reorient_v6, ownR6/model4600,2048 environments ×1500 additional updates, seed20260912, GPU7; no output/std reset | Running | Reset-cache fix applied;8-state history and explicit nonfinite reward/observation lane IDs. Numerical cause remains unresolved for this case. |

Cage R3 stopped at4649 onlane447 with finite policy and a freshly reset qpos/qvel. Its captured previous qvel is zero but warm-start acceleration is nonzero. The recorded-state GPU replay reproduces first-substep failure; CPU is finite. Test zero-cache replay before launching the remaining-budget continuation.

| RL-013-R4 | Cage | Finish contact-target budget with correct episode cache reset | cage_v3, ownR3/model4600,1024 environments ×1000 additional updates, seed20260912, GPU1,8-state capture; no output/std reset | Running | All1024 lanes finite for4substeps with cold cache; no physics/reward changes. |
