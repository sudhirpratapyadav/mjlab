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
