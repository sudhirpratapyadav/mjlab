# Logs — RL teacher stage

Append-only record. Keep planned actions separate from completed work and measured outcomes.

## 2026-09-12 — Stage creation and readiness check

User set the scope to 24 tasks, deferring Tool-Pull without deleting it, and specified **one RL teacher per task with success >90%**. Created this stage folder with goal, context, plan, task/status inventory and experiment ledger.

Completed a read-only inventory of all current active registered configs. Confirmed the selected 60D observation/8D normalized action contract and current PPO settings. Saved package versions, source hashes, git HEAD and dirty paths. Retained prior historical 25-task results without rewriting them as 24-task measurements.

At 14:30:05 UTC, holder20277 on dgx1 was running; all eight GPUs showed 0 MiB / 0% with no compute processes, but only assigned GPUs1–3 are available for this work. Slurm expiry: 2026-09-28T08:58:54. Shared disk: approximately 1.6 TB free, 95% used. Saved raw snapshots.

Ran PREFLIGHT-PPO-001 on assigned GPU2 via UUID pinning inside the holder. Reach, 64 envs, two PPO updates (3,072 transitions). Network parameters changed; losses stayed finite. Saved a checkpoint and reloaded matching deterministic inference outputs and optimizer state. This is a plumbing test, not a trained teacher or success measurement. Evidence: `evidence/ppo_smoke.json` and `.log`; checkpoint under `runs/PREFLIGHT-PPO-001/`.

Found a launcher issue: the generic GPU selector parses `CUDA_VISIBLE_DEVICES` entries as integers and rejects GPU UUIDs. Recorded the failure in `evidence/launcher_probe.json`; created a stage-local already-pinned launcher using `run_train` directly and checked its dry-run. Its full pilot/config-saving path remains pending.

Readiness conclusion: infrastructure supports a small Reach pilot after evaluator setup. A blanket 24-task training launch is premature: articulation approach rewards saturate, completion shaping has gaps, Cage's initial gripper policy tends to invalidate episodes, and eight physics presets have zero gravity. No reward/success/physics changes or full teacher jobs were made/launched in this setup pass. **0/24 certified RL teachers.**

## 2026-09-12 — Checkpoint/worktree handoff

User requested saving current work before handing off to a new agent in an isolated worktree. Preparing branch `checkpoint/cl24-rl-handoff-20260912` with source, changed assets, tests, plans and compact evidence. Large generated outputs remain in the original checkout; LOCAL_ARTIFACTS.json records the non-ignored exclusions. HANDOFF.md specifies a new worktree and explicit PYTHONPATH to prevent the shared editable environment from importing the wrong checkout.

## 2026-09-12 — Isolated RL continuation

Created worktree `mjlab-rl-teachers-24-codex` on branch `exp/rl-teachers-24-codex` at 4149157; verified shared `.venv` imports this worktree under explicit PYTHONPATH. Holder20277 and process placement were rechecked. User subsequently authorized GPUs1–7, leaving GPU0 unused.

Added strict checkpoint evaluator that captures the task command's actual success predicate at the environment's internal reset boundary, then counts only each environment's first episode. Reach smoke checkpoint scored 0/8 seed20260913; this was evaluator plumbing, not teacher evaluation. A fresh Reach pilot (RL-001) trained 1,024 environments for 500 PPO iterations on GPU2. `model_499.pt` scored 128/128 seed20260914 and 128/128 seed20260915; raw per-episode results and checkpoint hash are in evidence. Both batches ended by timeout. This is the first measured teacher passing the proposed rate gate; targeted evaluator edge-case tests remain pending.

Created W&B project `mjlab-cl24-rl-teachers-20260912` under the currently authenticated `domimagi-iitj` entity. Reach TensorBoard history was synced, and its strict evaluation plus checkpoint were uploaded. Fresh Lift, Push-Cuboid, Drag-Pull, Reorient and Topple pilots launched on GPUs1,3,4,5,6 respectively with W&B logging. User requested a Sudhir W&B destination instead. The exact target entity/project and current account's write access remain unverified; no global login or settings were changed. Subsequent launcher invocations now require an explicit per-run `--wandb-entity` to avoid further uploads to the shared default account.

User provided target profile `sudhirpratapyadav`. The currently authenticated API identity cannot create the experiment project in that entity (HTTP 404). User questioned continued running; immediately stopped only this experiment's Slurm steps 20277.1601/1604/1605/1606. Verified holder batch remains and all GPUs have zero compute processes. The stage launcher now defaults only this project's W&B entity to `sudhirpratapyadav` and supports offline logging; no global W&B change. Do not resume training or upload until destination access is resolved. Push-Cuboid independently failed at PPO iteration 268 (invalid action std); its model_200 checkpoint scored 1/128 strict validation. Lift/Drag-Pull/Reorient/Topple retain partial checkpoints.

### W&B resolved with the supplied experiment credential

Authenticated as `sudhirpratapyadav`; API reports the default entity as `sudhirpratapyadav-indian-institute-of-technology-jodhpur`. The earlier short profile namespace was not the actual team destination. Created `mjlab-cl24-rl-teachers-20260912` there via SDK initialization and verified a logged scalar through the API (probe c72lpymk). Snapshots confirm the shared `.netrc` and W&B settings did not change. Credential is stored privately outside the repository; process-local configuration selects that key/entity/project and clears the inherited `WANDB_USERNAME` override used by RSL-RL.

Copied all six original pilot histories, retained checkpoints, effective YAMLs, saved source diffs, and available strict evaluation JSONs into Sudhir's project. Remote summaries verified; `evidence/wandb_pilot_imports.json` maps all imported run IDs and checkpoint hashes. Original records remain intact. The pause condition is resolved. Prepared continuations of the four interrupted pilots from their own finite checkpoints, restoring optimizer and normalizers while restarting simulator/RNG with the recorded seed. No observation/action/reward/success/physics changes. Push-Cuboid remains under diagnosis and is not automatically restarted.
