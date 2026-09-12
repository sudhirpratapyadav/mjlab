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
| RL-005-V3 | Reorient | Recover from saturated, ground-colliding V2 policy | reorient_v2, 2,048 environments × 2,000 updates, seed20260912, GPU2 | Running | Fresh reorient_v2 on GPU2; actual grasp-gated end-face/lift/upright reward; fixed benchmark. |
| RL-002-V3 | Lift | Test bounded-mean policy and gentler exploration | lift_v2, 2,048 environments × 2,000 updates, seed20260912, GPU1 | Running | Fresh lift_v2 on GPU1; same grasp reward as previous variant, bounded mean and gentler PPO. |
| RL-013 | Cage-Drag | Keep initial exploration valid and reward actual open-cage transport | cage_v1, 1,024 environments × 1,500 updates, seed20260912, GPU5 | Running | Fresh cage_v1 on GPU5; normalizer/mean/std/config recorded in manifest. |

| RL-014 | Slide-Window | Learn the remaining slide mechanism | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU7 | Certified | 128/128 validation and128/128 independent confirmation; actual recorded motion reviewed; checkpoint/normalizers/video retained. |
| RL-015 | Open-Lid | Learn lid approach and rotation | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU4 | Running | Fresh policy after Flap certification; registered gravity preserved |
| RL-016 | Rotate-Valve | Learn large-angle valve rotation | mechanism_v1, 1,024 environments × 2,000 updates, seed20260912, GPU6 | Running | Fresh policy after Lever certification;8-second horizon and zero gravity preserved |
| RL-017 | Axial-Extract | Learn plug approach and extraction | mechanism_v1, 1,024 environments × 1,500 updates, seed20260912, GPU3 | Certified | 128/128 validation and128/128 independent confirmation; actual recorded motion reviewed; checkpoint/normalizers/video retained. |

| PREFLIGHT-COMPLETE-001 | Stack/Place/Peg | Check safe grasp geometry, release incentive and finite physics | 53 focused CPU tests;32 environments ×100 steps each;3 PPO updates | Physics passed; PPO next | 53 CPU tests passed; Stack/Place/Peg each32 environments ×100 steps finite. Peg initial grasp height0.070m (upper body). |
| RL-018 | Stack-Cube | Learn grasp/transport/released supported completion | completion_v2,2,048 environments ×2,500 updates, seed20260912, GPU3 | Prepared | Fresh bounded policy, gripper mean0.5/std0.1; physical and PPO preflight before launch |
| RL-019 | Place-In-Container | Learn grasp/transport/released contained completion | completion_v2,2,048 environments ×2,500 updates, seed20260912, GPU7 | Prepared | Strict completion uses native full-object containment, support contact and settling |
| RL-020 | Peg-Insertion | Grasp upper peg body, align, seat and release | completion_v2,2,048 environments ×3,000 updates, seed20260912 | Prepared | Native upright square-bore fit and release/settling requirements remain; GPU assigned when available |

| PREFLIGHT-CLOSURE-002 | Stack/Place/Peg/Reorient | Remove the hard35mm closure-reward discontinuity | Monotonicity test;32 environments ×100 steps;3 guarded PPO updates | Prepared | Old recipe names preserve previous behavior; new completion_v2/reorient_v3 use a continuous closure incentive |
| RL-005-R4 | Reorient | Escape the verified approach/closure reward cliff | reorient_v3,2,048 environments ×1,500 additional updates, seed20260912, GPU2 | Prepared | Resume own finite V3 checkpoint after smooth-reward preflight; policy architecture and benchmark unchanged |

## Run record template

- ID, task, hypothesis, status, start/end UTC, Slurm step, GPU UUID.
- Training seed, environment count, iterations/transitions, interface version, source/config hashes, reward and physics changes.
- Checkpoint and normalizer path/hash, losses, reward components, action saturation, termination counts, throughput and measured GPU memory.
- Strict successes/trials, evaluation seeds, deterministic/stochastic policy mode, episode budget, first-episode accounting, videos.
- Decision: retain/reject/continue, why, and next experiment. A teacher is certified only by the GOAL.md criterion.
