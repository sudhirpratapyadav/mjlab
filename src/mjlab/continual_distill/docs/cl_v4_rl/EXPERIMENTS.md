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
| RL-021 | Edge-Grasp | Learn edge exposure, rim grasp and lift | edge_v2,2048 environments ×3000 updates, seed20260912, GPU6 | Complete; rejected | Final2999 strict0/128; exposes plate but hovers behind with no pinch. |
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
| RL-002-R5 | Lift | Learn actual capture from own finite approach policy | lift_v5, ownPREFLIGHT-LIFT-FAILURE-REPLAY/model2028, mean0.5/std0.15,2048 environments ×2000 additional updates, seed20260912, GPU4 | Stopped; rejected | Finite3500 strict0/128; approaches without lifting. |
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

### Wave14 — contact geometry correction

Edge first pilot completed3000 updates, final2999 strict0/128. Actual failure frames show plate exposure followed by open-hand hovering behind it; no pinch/lift. Native success correctly rejects.

A CPU geometry counterfactual on128 recorded Stack terminal poses swept finger closure0..15mm per finger in0.5mm steps.74 poses admit opposing inward pad normals with shallow contacts[-3,+1]mm. The old whole-object enclosure rejects the first such contact in64/74, and at least one such state in74/74. These are collision-geometry witnesses, not integrated grasps or RL success episodes. Median added closure8mm per finger. Full projected collider width overestimates the local cross-section near the pads. Contact-normal audits of actual Peg/Pivot top presses still reject every opposing grasp.

Added training-only opposed_grasp (real pad-object contacts, inward normal cosine>0.5 for each pad, native1mm distance tolerance and valid contact-buffer/world masks) plus ray/box span through actual pad center. Full projection remains a miss fallback; box bounds approximate curved mesh cross-sections. lift_v6/completion_v6 preserve all native scene/physics/success/60D/8D;95 focused geometry/reward/benchmark/interface tests passed. Old recipe versions preserve their original behavior. New runs log opposing-contact fraction explicitly.

| PREFLIGHT-CONTACT-GEOMETRY-019 | Lift/Peg | Verify local-width shaping and actual opposing-contact reward in PPO | Lift lift_v6 fromownR5/model3500, std0.15; Peg completion_v6 fromownV2/model1500, mean0.5/std0.15; each64 environments ×3 updates, GPU6 sequentially | Complete | Both3updates/checkpoints finite and online; contact_geometry_preflight.json. |
| RL-002-R6 | Lift | Learn a real grasp despite full-box enclosure rejecting contact | lift_v6, ownR5/model3500, gripper std0.15,2048 environments ×2000 additional updates, seed20260912, GPU4 | Running | Native lift criterion and physics unchanged; preserve learned means;8-state capture. |
| RL-020-R3 | Peg | Learn actual opposed upper-body capture and insertion | completion_v6, ownV2/model1500, gripper mean0.5/std0.15,2048 environments ×2500 additional updates, seed20260912, GPU6 | Running | Existing arm policy retained, gripper reopened after topple/press failure; native insertion unchanged. |

Production GPU contact witness: restored the64 audited closure configurations without integration or policy updates. Native two-pad contact64/64, new opposing-contact grasp64/64, old enclosed grasp0/64; ray widths finite. This verifies real contact-buffer indexing/normals and the rejected-closure mechanism. It is not a grasp stability demonstration or success-rate evaluation. Lift R6 is beginning to receive occasional actual opposing-contact reward; strict success is unmeasured.

At LiftR6 iteration3677 (177 additional updates), sampled opposing-contact fraction6.59%, while the old enclosure gate still misses most contacts. This is a training contact diagnostic, not a success-rate claim. The production64-case witness and this live behavior justify applying completion_v6 to Stack/Place after their remaining whole-enclosure recipes continue to report0 native success. Verified own PIDs and finite checkpoints2200/2300; stop onlysteps1785/1781 before task-specific PPO preflights.

| PREFLIGHT-COMPLETION-CONTACT-020 | Stack/Place | Verify contact geometry on each retained task policy | completion_v6, Stack fromownR4/model2200 onGPU3, Place fromownR3/model2300 onGPU5, gripper std0.15, each64 environments ×3 updates | Complete | Both3-update PPO checkpoints finite and online; completion_contact_preflight.json. |
| RL-018-R5 | Stack | Learn grasp and released stack with correct local contact credit | completion_v6, ownR4/model2200, gripper std0.15,2048 environments ×2000 additional updates, seed20260912, GPU3 | Running | Reset fix and8-state capture; native success unchanged. |
| RL-019-R4 | Place | Learn grasp and released containment with correct local contact credit | completion_v6, ownR3/model2300, gripper std0.15,2048 environments ×2000 additional updates, seed20260912, GPU5 | Running | Reset fix and8-state capture; native success unchanged. |

Wave14 follow-up: Cage R4 final5599 strict2/128; Lift R6/model4100 strict0/128 despite97.6% final opposing contacts; Throw/model2500 strict0/128; Reorient R7 final6099 strict0/128. Each actual video reviewed; no confirmation launched for a failed validation. Stack R5/Place R4 full runs launched after their3-update PPO checks. New public RL gallery authorized and published via untu_vps with24 task videos and concise summaries.

### Wave15 — preregistered contact and transport retries

Previous goal turn made progress: four strict evaluations, new Stack/Place training runs and the authorized public24-policy gallery. Cage R4 final5599 scores2/128 with only1.57% terminal contact; the policy stalls around the cube. cage_v4 increases the reward-only forward contact drive from1mm to12mm to encourage sustained transport. Actual poses, contacts, no-pinch history and native caging predicate remain unchanged. Reorient R7 final6099 and Throw/model2500 score0/128 with open enclosure/no grasp. reorient_v7 and throw_v4 apply the already-tested opposing-normal contact gate and local ray-width closure used by Lift/Place, replacing whole-box enclosure. Bottle ray-box width is a shaping approximation; actual contact normals remain required.

| PREFLIGHT-CONTACT-TRANSPORT-021 | Cage/Reorient/Throw | Validate each new recipe on its own retained policy |64 environments ×3 PPO updates, seed20260912; Cage fromR4/5599 onGPU1, Reorient fromR7/6099 onGPU7, Throw fromfinal2999 onGPU2 after full-run exit | Prepared | No gripper mean resets; Reorient/Throw std0.15;8-state capture. |
| RL-013-R5 | Cage | Learn sustained open-hand transport | cage_v4, ownR4/model5599,1024 environments ×2000 additional updates, seed20260912, GPU1 | Prepared | Preserve all policy outputs/std. |
| RL-005-R8 | Reorient | Learn actual opposed grasp before rotation | reorient_v7, ownR7/model6099,2048 environments ×2000 additional updates, seed20260912, GPU7 | Prepared | Preserve means; gripper std0.15; native orientation/drift fixed. |
| RL-024-R2 | Throw | Learn actual opposed capture before launch | throw_v4, ownfirst/model2999 after completion,2048 environments ×2500 additional updates, seed20260912, GPU2 | Prepared | Preserve means; gripper std0.15; flight/containment criteria fixed. |

### Wave15 numerical readiness — optional native free-body integration

Peg R3 stopped at2329, lane184, finite policy; own2300 strict0/128 (topples and presses the shaft). Eight-state history starts at145rad/s and reaches~772krad/s. GPU replay reproduces the growth; CPU MuJoCo3.11.1 implicitfast from the same oldest state decays to38.53rad/s without warnings. Independent CPU implicit/RK4/timestep ablations remain finite. The installed Warp implicitfast omits the native leaf-free-body gyroscopic solve. A first prototype copied the installed full-implicit inverse-bias sign and worsened growth; that rejected diagnostic is retained. Correct M+h*d(bias)/dv restores the CPU trajectory: GPU38.52rad/s, maximum final qpos difference~7e-5. No timestep/inertia/contact/success changes.

Added opt-in SimulationCfg.free_body_implicitfast_compat, defaultFalse. It reuses Warp smooth-force solve, adds the RNE bias derivative with positive sign, and substitutes only six acceleration entries of each leaf free joint before the unchanged integration routine. Articulated accelerations retain the original fast solution. Scope/provenance must be recorded with --free-body-gyro; resume inherits it, deterministic evaluation restores it from the manifest. Existing teachers/runs remain on their saved configuration. This is a backend compatibility correction, not a reward change or invalid-state recovery.

Reorient R8 subsequently stopped at6589 with nonfinite reward lane35 after native reset; the8-state history shows the same growth171→10847rad/s. Corrected replay stays~84.79rad/s versus CPU83.29; contact trajectories differ, so do not claim bitwise equivalence. Both64-world synthetic100-step sparse/dense CUDA-graph checks match CPU within1e-3qpos/1e-2qvel and keep an independent hinge within1e-5qvel. Production2048-world recorded-state replays and task-specific PPO checks are required before training.

| PREFLIGHT-GYRO-022 | Peg/Reorient | Validate optional native free-body correction in training | After2048-world regressions, Peg fresh completion_v6 and Reorient ownR8/6500 reorient_v7, each64 environments ×3 updates; then2048 ×100-update stress continuations | Prepared | --free-body-gyro; native model parameters, horizons, predicates and60D/8D unchanged. |
| RL-020-R4 | Peg | Fresh opposed-grasp learner with corrected free-body integration | Fresh completion_v6,2048 ×3000 updates, seed20260912, GPU6 | Prepared | Discard failed-policy initialization after persistent topple/press; no pretrained/scripted policy. |
| RL-005-R9 | Reorient | Continue own finite approach policy with corrected integration | ownR8/model6500, reorient_v7,2048 ×2000 additional updates, seed20260912, GPU7 | Prepared | Preserve policy output/std; correct only backend free-body integration. |

| RL-002-R7 | Lift | Continue genuine grasp/lift/transport progress toward exact goal | lift_v6, ownR6/model5499,2048 ×2000 additional updates, seed20260912, GPU4 | Prepared | No output/std/reward/backend changes; R6 final0/128 but mean goal error0.26057→0.09496m and height0.02777→0.19646m. Actual video reviewed. |

After successful100-update gyro stress runs, preserve their finite PPO work: Peg R4 continues the fresh stress lineage frommodel99 for3000 additional updates; Reorient R9 continues its stressmodel6599 for2000 additional updates. Both inherit the recorded optional integration correction. No borrowed or scripted teacher checkpoint is introduced.

Wave15 launch update: all three contact/transport preflights passed. CageR5 and ThrowR2 are running; ReorientR8 stopped at6589 as recorded above. Both gyro PPO preflights and100-update stress checks passed; PegR4 fromstress99 and ReorientR9 fromstress6599 launched onGPUs6/7, inheriting the recorded correction. LiftR7 launched from5499 unchanged. PlaceR4 stopped only exactstep1808 after verified policy regression beyond3400; strict retained3400 evaluation active.

### Place R5 — conservative optimizer continuation

Place R4/model3400 strict0/128; reviewed actual frames show a grasped cube lifted beside the container without released containment. Later3500 policy lost grasp/approach while model and normalizer tensors remained finite. The cause is not established. Preregister a bounded continuation from own3400 with completion_v6 and Adam learning rate5e-5 (previous1e-4), preserving policy means/std, Adam moments, normalizers and legacy backend. First64 environments ×3 PPO updates must save the intended effective optimizer LR and finite state; then2048 ×2000 additional updates, seed20260912, GPU5. This changes step size only; native success and60D/8D remain fixed.

| PREFLIGHT-PLACE-LR-023 | Place | Verify restored optimizer uses5e-5 | ownR4/model3400, completion_v6,64 ×3 updates | Prepared | Real Adam checkpoint-load regression tests passed; no moment reset. |
| RL-019-R5 | Place | Preserve grasp progress while learning released containment | ownR4/model3400, completion_v6, LR5e-5,2048 ×2000 additional, GPU5 | Prepared | Full run conditional on finite PPO preflight with saved effective LR5e-5. |

Place LR preflight p0ncrxmz completed3 updates, all saved checkpoint tensors finite and actual Adam param-group LR5e-5. Full RL-019-R5-place-lr launched from original3400 onGPU5, source8982f35,2048 ×2000 additional; means/std/moments/normalizers/backend preserved. AllGPUs1–7 have one full PPO process; GPU0 remains unused.

### Wave16 — retained-policy progress

Cage R5 final7598 strict16/128 (previous2/128), actual success/failure reviewed; still incomplete, no confirmation. Stack R5 final4199 strict0/128 but65.08% final surviving-world opposing grasps, object height0.02463m; actual cube capture beside base without stacking. These measurements justify bounded unchanged continuations. No reward, model, backend, mean/std or LR changes; preserve optimizer/normalizers, seed20260912 and8-state failure capture.

| RL-013-R6 | Cage | Continue improved native caging transport | cage_v4, ownR5/model7598,1024 ×1500 additional updates, GPU1 | Prepared |16/128 prior, unchanged settings; strict eval at budget. |
| RL-018-R6 | Stack | Continue newly acquired opposing grasp toward lift/stack/release | completion_v6, ownR5/model4199,2048 ×1000 additional updates, GPU3 | Prepared |0/128; actual grasp precursor improved, unchanged settings; strict eval at budget. |

Wave16 launch update: CageR6 runningstep1852/GPU1, StackR6 runningstep1853/GPU3, inherited saved actual Adam LR1e-4; no optimizer/output/std/normalizer/reward/backend resets. All current seven runs are finite at the placement check.

### Wave17 — Strike approach/contact readiness

Previous goal turn made progress: Cage16/128 andStack0/128 strict evaluations, unchanged bounded continuations, current public gallery/W&B and launch-speed diagnosis. The new128-episode CPU forward audit uses every saved20ms state with no integration. The58 negligible-launch episodes have0 sampled CPU contacts and median closest-target vertical error81.85mm;25 weak launches have61.19mm median vertical error,45 substantial launches24.00mm. Between-frame contacts may be missed, and recomputed CPU contacts do not replace original GPU evidence. Thus precise approach coverage is the main measured issue, not simply launch speed.

The old35.3mm reward standoff is a seating waypoint about3mm outside the60mm wedge contact geometry. strike_v2 moves only the reward target12mm toward the puck and adds2*exp(-distance/0.05) to the existing exp(-distance/0.20) approach term. Existing orientation/aperture factors, sliding prediction, native success, gravity, friction,4s horizon,60D/8D, agent architecture and optimizer settings are unchanged.57 focused tests passed, including unchanged benchmark configuration and a geometry-based gap check. This is a hypothesis for RL contact acquisition; no policy or success improvement is claimed yet.

| PREFLIGHT-STRIKE-CONTACT-024 | Strike | Check new precise approach/contact reward with retained policy | strike_v2, ownRL-023/model2999,64 ×3 PPO updates, seed20260912, next available GPU1 or3 after current final evaluation | Prepared | Must finish finite and online before full continuation. |
| RL-023-R2 | Strike | Improve approach/contact coverage and subsequent native launch | strike_v2, ownRL-023/model2999,2048 ×2000 additional updates, seed20260912 | Prepared | Preserve means/std/optimizer/normalizers; no borrowed policy, no scripted actions; contingent on PPO preflight. |

### Wave18 — Edge side-wrist readiness

Previous turn completed new Strike contact diagnosis and CPU-tested reward preparation; all seven full runs were revalidated live. Edge endpoint IK on32 evenly indexed recorded exposed-plate poses finds all32 pass for each heading0,+70,-70degrees at20degree downward pitch, with35mm/finger opening. Gate: position<2cm, orientation<15degrees, no recomputed robot/ledge/floor penetration deeper than3mm. This is endpoint geometry, not collision-free trajectories, stable pinches or RL success. Solver branch seeds are diagnostic only and never initialize/supervise/command the RL policy. Historical controller posture traps do not prove these endpoints unreachable.

edge_v3 scores the full side-wrist frame against those three orientations, targets the existing near-rim pinch5mm higher, uses broader/fine approach terms and the established opposing-normal/local-ray-width capture credit. The previous reward scored only abs(vertical closing axis), which cannot distinguish a hand facing toward or away from the ledge. Maximum non-grasp precursor remains7, below the8-point grasp bonus. Native grasped+lift+settled success, ledge/plate geometry, initialization, physics,6s horizon and60D/8D stay fixed.62 focused tests passed, including wrong-heading/wrong-roll rejection and unchanged benchmark/agent configuration.

| PREFLIGHT-EDGE-WRIST-025 | Edge | Verify full side-wrist/capture shaping in PPO | edge_v3, ownRL-021/model2999,64 ×3 updates, seed20260912, next suitable freed GPU after Strike preflight/full-run placement | Prepared | No policy/std/optimizer/normalizer reset; finite online preflight required. |
| RL-021-R2 | Edge | Learn side pinch and settled lift after exposing plate | edge_v3, ownRL-021/model2999,2048 ×2000 additional updates, seed20260912 | Prepared | No scripted trajectories, endpoint-IK supervision or borrowed checkpoints. |

### Lift R8 — settled grasp control

LiftR7/model7100 strict0/128 despite110 terminal objects inside the native5cm positional threshold. No episode passes the native settling limits; median terminal speed0.2663m/s and angular speed1.866rad/s. Native grasp/contact diagnostics remain high. Final7498 also0/128. Constant-control interventions on31 recorded timeout states compare GPU with CPU under matched freshly sampled fingertip friction. Strong saved closure with frozen policy or current-arm targets drops all31 objects on both backends. Current-arm targets plus only2mm commanded finger closure retain12 GPU grasps and14 CPU grasps; refreshed GPU endpoint predicate passes8/31. These are interventions under resampled registered friction, not policy episodes or RL success. The preliminary unmatched CPU-friction report is explicitly limited; matched complete results are lift_hold_complete_audit.json.

lift_v7 retains lift_v6 and adds held-object credit for mild loaded finger closure around2mm plus quiet motion near the goal:2*exp(-abs((finger_q-ctrl)-.002)/.01) and5*exp(-goal_error/.05)/(1+linear_speed/.10+angular_speed/.5), both gated by actual opposing grasp. No action override, actuator/friction/model/backend change, or success relaxation.64 focused tests pass, including grip-load/motion discrimination and unchanged native configuration/agent.

| PREFLIGHT-LIFT-SETTLE-026 | Lift | Verify gentle loaded grasp/settling shaping on own policy | lift_v7, ownR7/model7498,64 ×3 updates, GPU4, seed20260912 | Prepared | Finite online PPO checkpoint required before full continuation. |
| RL-002-R8 | Lift | Learn settled grasp inside the native goal | lift_v7, ownR7/model7498,2048 ×1500 additional updates, GPU4, seed20260912 | Prepared | Preserve learned means/std, optimizer/normalizers and legacy backend. |

### Cage R7 — increase rollout batch after native progress

CageR6 final9097 strict37/128 versus16/128 at7598, actual success/failure frames reviewed. Continue its unchanged cage_v4 policy onGPU7 after Reorient final evaluation. Increase rollout worlds1024→2048 to use more of the A100 capacity while preserving the registered initialization distribution, timestep, rewards, means/std, LR and native gates. First10-update2048-world PPO preflight must be finite and online; then2000 additional updates from original9097. This is a training batch-size change, not a benchmark change.

| PREFLIGHT-CAGE-2048-027 | Cage | Validate doubled rollout batch and throughput | cage_v4, ownR6/model9097,2048 ×10 updates, GPU7, seed20260912 | Prepared | Preserve all policy/optimizer settings; monitor numerical capture. |
| RL-013-R7 | Cage | Continue improving native open-hand transport | cage_v4, ownR6/model9097,2048 ×2000 additional, GPU7 | Prepared | Conditional on finite10-update preflight, strict validation at budget. |

Wave18 launch update: Lift settle preflight passed finite and online (zxswj059); R8 launched from original7498 with lift_v7 onGPU4. Cage2048-world10-update preflight passed finite and online (kb7bz1ex), actual Adam LR1e-4; R7 launched from original9097 onGPU7 for2000 additional updates. No policy/std/optimizer/normalizer/backend resets.

### Throw R3 — continue acquired grasp/lift toward launch

Throw R2 final5498 completed normally, strict0/128 (127timeouts,1ground). Actual recorded frames now show cube capture and lift, followed by holding without release; final opposing contacts94.49%, height0.26976m, goal error0.4610m. This is a newly learned precursor compared with final2999 hovering. Preregister a bounded unchanged throw_v4 continuation from own5498,2048 ×2000 additional, seed20260912, GPU2. Preserve means/std, Adam state/LR, normalizers, physics and native5s success. Strict evaluation at budget; no further blind continuation if holding persists.

| RL-024-R3 | Throw | Convert acquired grasp/lift into native release/launch | throw_v4, ownR2/model5498,2048 ×2000 additional, GPU2 | Prepared | Unchanged recipe/configuration; no additional PPO preflight required for an unchanged resume. |

### Wave19 — Pivot ramp geometry and Place progress

Pivot original1500 strict0/128;128 recorded-pose FK audit finds6.94mm median closest target error,22.42degree full wrist error,29.08mm aperture and board already against the wall. No sampled episode exceeds20degree tilt. Eight of32 nominal open-ramp endpoints require3.18–4.36mm finger/floor penetration. At40mm/finger with an XML-derived1mm floor margin,32/32 endpoints pass the unchanged IK/contact gate. These are endpoint diagnostics, not trajectories, demonstrations or RL successes.

pivot_v3 preserves pivot_v2 physics/native pivot-history+grasp+goal+settled predicate and60D/8D. It adds floor-safe ramp reward waypoints, full wrist alignment and opening preference80→20mm as board tilt grows toward55degrees. Existing opposing-contact classifier replaces whole-box enclosure for actual grasp credit; same-face top pressing remains rejected. Non-held precursor is bounded7.5, below held bonus8. No scripted actions, pose writes, borrowed initialization or grasp-history shortcuts.66 focused recipe/geometry/strict-evaluation checks pass.

| PREFLIGHT-PIVOT-RAMP-028 | Pivot | Verify ramp reward in PPO | pivot_v3, ownRL022/model1500,64 ×3 updates, GPU5 temporarily after Place evaluation | Prepared | Preserve means/std/optimizer/normalizers; finite online preflight before full retry. |
| RL-022-R2 | Pivot | Learn open-ramp tipping then opposed capture and lift | pivot_v3, own1500,2048 ×2000 additional, next freeGPU1 after Strike final evaluation | Prepared | No output/std/backend changes; contingent on finite preflight. |

PlaceR5 final5399 strict35/128, improved from0/128. Actual success/failure frames reviewed: capture/transport/lowering at container, incomplete release/settling in selected failure. Preregister unchanged completion_v6 continuation from own5399,2048 ×2000 additional, GPU5 after short Pivot preflight. Inherit saved Adam LR5e-5/moments/normalizers/means/std; strict evaluation at budget.

| RL-019-R6 | Place | Improve newly acquired native container completion | completion_v6, ownR5/model5399,2048 ×2000 additional, GPU5 | Prepared |35/128 prior; unchanged configuration and inherited LR5e-5. |

Wave19 launch update: Pivot preflight map1pnz1 passed3 updates with finite1502 and Adam LR1e-4. PlaceR6 from5399 launched onGPU5 with inherited LR5e-5. StrikeR2 completed its budget normally; final4998 strict20/128, improved9/128, all timeouts, actual success/failure clips reviewed. PivotR2 now launched onGPU1 from original1500,2048 ×2000, source0b8a931, preserving all policy/optimizer/normalizer settings.

### Wave20 — Peg capture exploration

PegR4 final3098 completed3000 updates normally with recorded gyro compatibility. Strict0/128, all timeouts; actual frames show upright peg and hovering open hand, final aperture58.84mm and no robot/object contacts. Learned gripper action mean0.4879/std0.0328 in the final training sample. CPU closure sweep on128 terminal poses finds60 valid opposing-contact configurations after13–19mm closure per finger (median15.5mm); first-contact aperture23.19–32.68mm (median29.43mm). These pose modifications are geometric counterfactuals, not integrated grasps or RL success.

Preregister gripper-only PPO initialization reset: own3098, completion_v6, reset output mean to-0.25 (normalized action corresponds to15mm/finger target) and exploration std0.15. Preserve all learned arm outputs, saved optimizer/normalizers and gyro setting; no scripted action sequence, geometry, success or60D/8D change. This changes training initialization only. First64 ×3 updates finite and online; then2048 ×2000 additional updates onGPU6 after Place5600 evaluation. The output row reset uses the existing explicit runner mechanism and must be logged in the manifest.

| PREFLIGHT-PEG-CLOSURE-029 | Peg | Verify restored arm policy with capture-range gripper initialization | ownR4/model3098, completion_v6, mean-0.25/std0.15,64 ×3 updates, GPU6 | Prepared | Finite online checkpoint and inherited gyro compatibility required. |
| RL-020-R5 | Peg | Acquire opposing capture from the stable upright approach | ownR4/model3098, same recorded gripper reset,2048 ×2000 additional, GPU6 | Prepared | Conditional on preflight; strict evaluation at budget. |

StrikeR2 launch audit: substantial launches45→97/128, negligible58→21, weak25→10. All20 successes are substantial launches; median final error0.2341m versus0.5546m previously,85 undershoot by>8cm,15 overshoot by>8cm,10 lateral misses>8cm. Current reward improved acquisition; next retry should refine distance/direction while retaining contact. PlaceR6/model5600 strict74/128, improved35; full training continues unchanged.

### Lift R9 — initialize the gripper in a demonstrated stable control range

LiftR8 final8997 strict0/128;105 positions inside5cm,0 inside and settled, median linear/angular speeds0.3160m/s/2.0847rad/s, final grip action-0.913. Actual failure frames reviewed. New hold probe restores the original sampled fingertip friction from the trace for both GPU and CPU, with identical states/controls and cold solver cache. On31 evenly sampled timeouts, frozen policy targets and current-arm/strong-grip targets each drop31/31 on both backends. Current-arm plus actual finger position minus2mm retains16 GPU/17 CPU grasps and10 GPU native endpoint passes. Current-arm plus fixed20mm/finger retains23 GPU/22 CPU grasps and14 GPU native endpoint passes. These are controlled interventions, not RL episodes or proof of closed-loop success.

Preregister a gripper-only PPO initialization reset to mean0 (20mm/finger target), std0.05, preserving the learned arm policy, saved optimizer/normalizers, lift_v7 reward and legacy backend. This tests the stable control range directly without overriding policy actions during training/evaluation. First64 ×3 updates finite and online, then2048 ×1500 additional from original8997 onGPU4; strict evaluation at budget. Native physics, target distribution, first-episode gate and60D/8D remain fixed.

| PREFLIGHT-LIFT-GENTLE-030 | Lift | Verify retained transport policy with gentler gripper initialization | ownR8/model8997, lift_v7, mean0/std0.05,64 ×3 updates, GPU4 | Prepared | No learned arm reset; finite online PPO check required. |
| RL-002-R9 | Lift | Learn settled near-goal holding from stable grip range | own8997, same recorded initialization reset,2048 ×1500 additional, GPU4 | Prepared | Conditional on preflight; no control clamp or scripted policy. |

### Peg capture-width fallback correction

PegR5 begins reopening its reset fingers and has no contacts in sampled training states. Audit original R4 terminal poses: pad centerline intersects the peg in only10/128; the miss fallback uses full projected length, median55.0mm versus central cross-section26.3mm.52 of60 valid finite-pad closure counterfactuals have a missed center ray. Old estimated width exceeds their first-contact gap by median27.1mm; central-section estimate is3.1mm below the first-contact gap. This is a training-shaping defect for a slender tilted object, not evidence that native contact or success should change.

peg_v1 inherits completion_v6 and changes only the missed-ray aperture estimate to the chord through the collider-box center along the closing direction. Actual ray hits retain their local width; actual opposing contacts still determine held credit. Defaults/other recipes retain full-projection fallback.74 focused tests pass, including tilted slender geometry, unchanged actual-hit estimates and unchanged Peg model/agent/native60D8D configuration.

| PREFLIGHT-PEG-WIDTH-031 | Peg | Validate slender-object aperture fallback in PPO | peg_v1, ownR4/model3098, mean-0.25/std0.15,64 ×3, GPU6 alongside its sole full trainer | Prepared | Small preflight only; no second full trainer. Preserve original comparison initialization and gyroON. |
| RL-020-R6 | Peg | Acquire capture with corrected width incentive | peg_v1, original3098, same mean-0.25/std0.15,2048 ×2000 additional, GPU6 | Prepared | After finite preflight, stop only verified R5 step, retain/evaluate latest saved R5 checkpoint, then replace with R6. |

PlaceR6/model6000 strict evaluation launched onGPU5 alongside its sole full trainer after74/128 at5600. No policy/reward changes; confirmation only if validation exceeds90%.

Wave20 final checks: Peg width preflight17npttwg passed; R5 exact step1898 stopped, retained3400 strict0/128 and actual failure frames reviewed. R6 launched originalR4/3098 with peg_v1 and recorded gripper reset onGPU6, W&B rec0k5q3. Initial command had an incorrect directory suffix and exited at argument validation before simulation; corrected to RL-020-R4-peg-gyro. CageR7 completed, strict97/128; EdgeR2 completed, strict0/128; actual clips reviewed. PlaceR6 failed after6329 in lane995; diagnostics retained, model6300 strict96/128. No new certification.

### Cage R8 bounded continuation

CageR7 improved37→97/128 with unchanged cage_v4 and2048 environments. Preregister RL-013-R8-cage-refine: originalR7/model11096, same reward/backend/Adam/normalizers and batch,2000 additional updates onGPU7 after evaluator exit. Strict evaluation at budget; no changes to native gate or60D/8D. This extends a stable improving PPO lineage, not a new recipe.

### Wave21 — Place solver diagnosis and Strike refinement

Previous wave made progress: Cage97/128 and Place96/128, actual-video reviews, Peg/Cage launches, refreshed public gallery and150 W&B records. PlaceR6 failure after6329/lane995 is reproduced on the first substep with2048 worlds; cold warmstart is finite for all4 substeps, maximum CPU/GPU qpos difference0.000492. State history shows an ongoing episode, not a reset. Test the installed incremental Newton Hessian optimization against full Hessian rebuilding with the same saved warmstart; diagnostic only, no shared environment modification.

StrikeR2/model4998 acquired97 substantial launches but only20 native passes;85 undershoots exceed8cm. Preregister a conservative PPO refinement: keep strike_v2 reward,2048 worlds, inherited arm/gripper/Adam moments/normalizers/backend; override Adam LR to5e-5 (from1e-4). First PREFLIGHT-STRIKE-REFINE-032,64×3 from original4998, require finite checkpoint and recorded effective LR; then RL-023-R3-strike-refine,2000 additional updates from original4998 onGPU3. Strict evaluation at budget. No direct policy actions or demonstration targets.

### Stable elliptic cone Hessian compatibility

Place replay uses dense Newton/elliptic contacts. Full-Hessian rebuilding still fails; eager instrumentation finds the third Newton search nonfinite with finite gradients/Hessian and a physical-block negative eigenvalue around-0.000312. One middle-cone contact has T=4.15146e-6 and positive N; the installed GPU kernel floors T^3=7.1549e-17 to1e-15, altering the Hessian. Its unscaled cone matrix has eigenvalue-0.0781, while the native-equivalent expression is positive semidefinite. Native reference: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_constraint.c (mj_constraintUpdate_impl).

Added opt-in dense elliptic_hessian_compat: normalize tangential residuals and assemble the identical cone Hessian as outer products, avoiding T^3 and cancellation. Default remainsFalse; sparse Jacobians explicitly reject this option. Dispatch is scoped to graph capture/eager calls; shared .venv is untouched. Training manifests, resume, strict evaluation and certificates persist/check the option. No forces, constraint states, geometry, solver tolerances, integration, rewards, or native success predicates change. Captured2048-world replay with original warmstart passes4/4 substeps; maximum CPU/GPU qpos difference0.000492.

Preregister PREFLIGHT-PLACE-CONE-033: ownR6/model6300, completion_v6, inherited Adam5e-5/normalizers/means/std, elliptic_hessian_compat=True and gyroOFF,64×3. After finite checkpoint and focused kernel/evaluation checks, run STRESS-PLACE-CONE-034 from original6300,2048×100. If finite, launch RL-019-R7-place-cone from original6300,2048×2000 onGPU5. Strict gate remains116/128 on each held-out batch; this correction requires new evaluation, not retroactive certification.
