# Experiments — RL teachers

Use a unique run ID, record the hypothesis and config change before launch, then append results and promotion decisions. Never overwrite failed or weaker trials. Planned budgets below are not running jobs.

| ID | Task/scope | Purpose | Budget | Status | Result |
|---|---|---|---|---|---|
| PREFLIGHT-CFG-001 | All 24 | Check registered interface/PPO/physics | Read-only configs | Complete | 60D/8D; eight zero-gravity presets; all default num_envs=1 |
| PREFLIGHT-GPU-001 | dgx1/holder20277 | Availability and assignment | Snapshot | Complete | GPUs1–3 assigned and idle; no experiment steps at snapshot |
| PREFLIGHT-PPO-001 | Reach | Real optimization + checkpoint round trip | 64 envs × 24 steps × 2 updates | Complete | 3,072 transitions; finite losses; parameters changed; reload and optimizer restore passed; no success evaluation |
| PREFLIGHT-CLI-001 | Launcher | UUID pinning compatibility | No training | Complete | Generic selector fails; direct runner works; stage-local launcher dry-run validated |
| EVAL-001 | Evaluator | Strict first-episode checkpoint evaluation | Reach smoke checkpoint, 8 episodes seed20260913 | In progress | End-to-end 0/8; terminal capture works, targeted subset/terminal tests pending |
| RL-001 | Reach | First full training/evaluation pilot; baseline physics and reward | 1,024 envs, 500 iterations, seed20260912, GPU2 | Complete | model_499.pt: 128/128 validation seed20260914, 128/128 confirmation seed20260915; strict first-episode evaluator; both timeout at full horizon; history, evaluation and checkpoint imported into the verified Sudhir project |
| RL-002 | Lift | Learn grasp + lift with common interface; baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU1 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-003 | Push-Cuboid | Test short-horizon push learning with baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU3 | Failed | PPO action std became invalid at iteration 268; model_200.pt strict validation 1/128 seed20260914 |
| RL-004 | Drag-Pull | Pilot contact learning under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU4 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-005 | Reorient-Object | Pilot settled orientation learning under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU5 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_300.pt |
| RL-006 | Topple-Block | Pilot contact and settled completion under baseline reward/physics | 1,024 envs, planned 500 iterations, seed20260912, GPU6 | Stopped | Stopped at user request pending correct W&B destination; last checkpoint model_200.pt |
| WANDB-002 | Logging | Verify Sudhir credential/destination and import all six pilot histories | One test write/read, six history/checkpoint imports | Complete | Correct entity is sudhirpratapyadav-indian-institute-of-technology-jodhpur; global settings unchanged; evidence/wandb_destination.json and wandb_pilot_imports.json |
| RL-002-R1 | Lift | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU1 | Running | Policy/optimizer/normalizers restored; live online W&B metrics verified |
| RL-004-R1 | Drag-Pull | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU4 | Running | Policy/optimizer/normalizers restored; live online W&B metrics verified |
| RL-005-R1 | Reorient | Complete interrupted baseline pilot from its own model_300 | 1,024 envs, 200 additional updates, seed20260912, GPU5 | Running | Policy/optimizer/normalizers restored; live online W&B metrics verified |
| RL-006-R1 | Topple | Complete interrupted baseline pilot from its own model_200 | 1,024 envs, 300 additional updates, seed20260912, GPU6 | Running | Policy/optimizer/normalizers restored; live online W&B metrics verified |
| REWARD-001 | Mechanisms | Recover useful reset-to-contact reward signal | Define after curve review | Planned | Not launched |
| INIT-001 | Cage | Keep valid open cage during initial exploration | Define open-gripper prior/curriculum | Planned | Not launched |
| COMPLETE-001 | Stack/Peg/Place | Align dense return and strict completion | Define after reward review | Planned | Not launched |
| PREFLIGHT-GUARD-001 | Push-Button | Exercise finite-state/update guard and mechanism_v1 recipe | 64 envs, 3 updates, seed20260912, GPU2 | Complete | Finite updates, positive log std and W&B diagnostics verified; no success claim |
| RL-007 | Push-Button | Learn contact/manipulation with non-saturated approach | mechanism_v1, 1,024 envs, 1,500 updates, seed20260912, GPU2 | Prepared | exp(-distance/0.3), unchanged precise joint goal; fixed LR0.0003/log std; no idle bonus; reduced velocity/action penalties |
| RL-008 | Open-Drawer | Learn approach and pull with the same mechanism recipe | mechanism_v1, 1,024 envs, 1,500 updates, seed20260912, GPU7 | Prepared | Baseline zero gravity retained; strict success unchanged |
| RL-002-V2 | Lift | Reward alignment/closure, actual two-pad grasp, lift and target progress | lift_v1, 2,048 envs, 1,500 updates, seed20260912, GPU1 | Prepared | Fresh policy; baseline pilot stalled at approach; source recipe records reward coefficients |
| RL-005-V2 | Reorient | Replace saturated reach/tiny angular shaping with grasp and broad orientation progress | reorient_v1, 2,048 envs, 1,500 updates, seed20260912, GPU5 | Prepared | Fresh policy; strict settled-axis predicate unchanged |
| RL-003-V2 | Push-Cuboid | Diagnose/prevent original invalid std failure | stable_v1, 1,024 envs, 1,000 updates, seed20260912, GPU3 | Prepared | Fresh policy, log std, fixed LR0.0003, gamma0.995; first-nonfinite diagnostic capture |
| RL-004-R2 | Drag-Pull | Extend 100/128 baseline teacher | baseline_long, 1,024 envs, 1,000 additional updates, seed20260912, GPU4 | Prepared | Resume own model_499; freeze initial velocity penalty instead of later10x/100x schedule |

## Run record template

- ID, task, hypothesis, status, start/end UTC, Slurm step, GPU UUID.
- Training seed, environment count, iterations/transitions, interface version, source/config hashes, reward and physics changes.
- Checkpoint and normalizer path/hash, losses, reward components, action saturation, termination counts, throughput and measured GPU memory.
- Strict successes/trials, evaluation seeds, deterministic/stochastic policy mode, episode budget, first-episode accounting, videos.
- Decision: retain/reject/continue, why, and next experiment. A teacher is certified only by the GOAL.md criterion.
