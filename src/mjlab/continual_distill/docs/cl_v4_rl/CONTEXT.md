# Context — read before resuming

**New agents:** follow [HANDOFF.md](HANDOFF.md) to create an isolated worktree and ensure Python imports that worktree.

## User decisions

1. Next phase is RL teacher training: **one teacher per task**, target **>90% success for every task**.
2. Preserve the working common **60D observation layout**, including its useful redundancy. The provisional 143D schema was rejected.
3. Keep normalized, bounded **8D absolute joint-position actions**, identical across tasks: seven arm targets span physical joint limits; gripper -1/0/+1 means closed/half-open/open.
4. Tool-Pull is excluded for now, leaving **24 active tasks**. Do not delete its environment, assets, registry entry, teacher code or historical results.
5. This setup pass checks readiness and prepares tracking files. It performed only a two-update PPO plumbing smoke, not full teacher training.

## Sources of truth

- Active suite: `mjlab.tasks.manipulation.benchmark.active_cl_tasks()`; snapshot: `active_tasks.json`. Do not enumerate the full registry or old 25-task result folders for a new suite run.
- Interface: `src/mjlab/tasks/manipulation/franka_interface.py`, version `franka_shared_60_v2`; details in `../cl_v3/shared_60/README.md`.
- Current registered configs: `evidence/preflight.json`. Registry finalization applies the shared interface; use `load_env_cfg`, not raw base builders.
- Prior reward audit: `../cl_v3/rl_readiness/README.md`. Its old action/observation findings are superseded; its reward and physics findings remain relevant.
- Prior numerical checks: `../cl_v3/shared_60/verification.json` (historically all 25, including the now-deferred task). 24 active environments passed; teacher comparisons passed for those 24.
- Source at preflight: git HEAD `b9b0563bd7b1b9cdcf9a369668c3b0c163e2fd72` plus a substantial pre-existing dirty worktree. HEAD alone is insufficient to reproduce it. Source hashes and dirty paths are saved in preflight evidence; archive the actual code/patch and assets for real experiments. Do not discard others' edits.

## Observations

Order: joint position 9, joint velocity 9, object position 3, object quaternion 4, gripper position 3, gripper rotation 6, object rotation 6, gripper-to-object 3, object-to-goal 3, goal-orientation difference 6, control error 8. Joint quantities use legacy physical/default-relative units. Absolute-position fields subtract scene origins. Actor/critic receive identical clean state; PPO running normalization remains enabled.

No extra tool/fixture/contact/history/velocity channels. Reach uses a virtual object at its target. Stack/Peg destination information is in the goal vector. Compact-state limitations are intentional; first test learning instead of expanding the schema automatically. Old neural checkpoints still require action-unit conversion; 143D checkpoints are incompatible.

## Compute and tooling

### Current continuation overrides (2026-09-12)

Worktree: `/ihub/homedirs/svs_ald/sudhir/mjlab-rl-teachers-24-codex`, branch `exp/rl-teachers-24-codex`, base `4149157`. User expanded the allocation to GPU indices **1–7**, leaving GPU0 unused. The older GPUs1–3 snapshot below is historical. Continue to check actual placement and use holder20277 only.

W&B is now verified under entity **`sudhirpratapyadav-indian-institute-of-technology-jodhpur`**, project **`mjlab-cl24-rl-teachers-20260912`**. The shorter profile name is not the team's entity namespace. `wandb_config.py` scopes credential/configuration to this experiment process. The private credential lives outside the repository at `~/.config/mjlab-cl24/wandb_api_key`. Never print it or call global `wandb login`. `evidence/wandb_destination.json` records a successful remote write/read and unchanged shared settings. Training was paused until this check passed; that W&B blocker is now resolved. Original pilot imports are recorded in `evidence/wandb_pilot_imports.json`.

Follow `~/use_instructions/README.md`. Existing holder **20277**, name **hold_dgx_amit**, node **dgx1**. Historical initial assignment was indices1–3; the current user authorization is indices1–7. Never cancel another user's holder or processes. Recheck actual process placement before each launch.

At 2026-09-12 14:30 UTC, all eight A100-SXM4-80GB GPUs showed 0 MiB use and no compute processes. The holder had only its batch step. Slurm reports expiration **2026-09-28T08:58:54** (scheduler time). The filesystem had about **1.6 TB free, 95% used**; plan checkpoint/video retention.

Assigned UUIDs:
- GPU1: `GPU-2496e2db-e138-0dec-41fc-27d2535b6e86`
- GPU2: `GPU-23a5dcb4-5248-01aa-94a3-d0f998660161`
- GPU3: `GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc`

Export the UUID **inside** the Slurm step; set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`. The generic `launch_training` GPU selector parses visible devices as integers and fails on UUIDs (`evidence/launcher_probe.json`). The stage-local `train_teacher.py` bypasses GPU reselection and calls `run_train` directly on the already pinned `cuda:0`. Its full training/checkpoint/config-saving path has been exercised by the retained teachers.

Use `.venv/bin/python`. Package versions are saved in preflight evidence. PPO smoke uses local TensorBoard logging, avoiding an external logging dependency. Stage-local runs default to online W&B in the verified project.

## Current outcome and next action

**17/24 independent PPO RL teachers certified.** Throw14098 is newest at125/128 validation and126/128 reserved confirmation20260916. Actual videos reviewed; native CPU endpoint reconstruction matches all128 validation labels, all76 checkpoint tensors finite,13 W&B teacher files retained and verified. Cage19092 remains119/128+120/128 and Place7500 remains117/128+119/128. Seven tasks remain unfinished.

Current full runs: ReorientR14 GPU1 final18593, EdgeR6 GPU2 final12994, StrikeR10 GPU3 final18991, LiftR15 GPU4 final19991, StackR10 GPU5 final14194, PegR10 GPU6 final10198 and PivotR3 GPU7 final6498. StrikeR9 finished its full budget and strict59/128; reviewed and continued unchanged from16992 with saved Adam1.25e-6 and max_update_kl.03, without another noise reset. GPU0 stays unused; holder20277 expires2026-09-28T08:58:54 scheduler time.

Latest strict results: ReorientR13/16594 1/128 and PegR9/10198 0/128, with actual recorded frames reviewed and all76 tensors finite in each. StackR9/12195 and LiftR14/19991 remain0/128 before the new runs. StrikeR9/16992 improves39→59/128, median endpoint error123.71→86.21mm; all76 tensors finite and actual videos reviewed. Previous measurements stay historical. Matched training probes exposed large post-update Gaussian KL despite tiny pre-update normalizer drift. Opt-in fixed-schedule PPO backtracking restores policy/normalizers/Adam/RNG and halves LR when actual-rollout mean KL exceeds.03; tested rollback and50-update full-batch preflights passed. Lift/Strike/Peg restart from retained pre-collapse sources with this guard. Reorient's118/128 orientation/drift passes still fail settling, while fixed recorded controls settle30/31 diagnostic GPU cases: reorient_v10 rewards steady actions. Stack_v1 rewards release near physical support. Pivot_v4 uses actual near-edge geometry and early wrist/tilt progress. Edge_v5 rewards wider arrival and makes capture dominate the precursor. All new recipes passed focused invariants and finite preflights. Diagnostics never become demonstrations, resets or policy actions. Native success, physical models and60D/8D stay fixed.

Wave34 Pivot readiness: corrected all8D-bounded32-state replay shows CPU28/32 versus baselineGPU14/32 tilt+wall mechanism; existing primitive-box compatibility restoresGPU28/32 and median endpoint parity0.0894mm. Diagnostic only. Complete currentR3 fullbudget/evaluation, then select/preregister retained-checkpoint primitive-only PPO preflight; no changes to active training or native physical model/interface. Evidence: pivot_r2_backend_comparison_audit.json.

Wave34 evaluation scheduling: wait_for_final_evaluation.py is watching each of the seven identified trainer processes. It starts the existing strict evaluator only after process exit and verification of that plan's finite final checkpoint; no intermediate evaluation. Each run/evaluation-dispatch.json records progress, and per-run locks prevent duplicate dispatch. Actual video review and both independent>90% batches remain mandatory.
