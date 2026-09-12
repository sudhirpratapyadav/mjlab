# Worktree handoff

Start from the checkpoint branch `checkpoint/cl24-rl-handoff-20260912`, which saves the task/teacher/physics work, the selected 60D/8D interface, active 24-task scope, and RL experiment setup. This is an integration checkpoint, not a claim that all tasks are ready or that RL teachers have reached the target.

## Create your own worktree

From the original checkout:

```bash
git worktree add -b exp/rl-teachers-24 ../mjlab-rl-teachers-24 checkpoint/cl24-rl-handoff-20260912
cd ../mjlab-rl-teachers-24
export PYTHONPATH="$PWD/src"
/ihub/homedirs/svs_ald/sudhir/mjlab/.venv/bin/python -c 'import mjlab; print(mjlab.__file__)'
```

Use a unique branch/directory suffix if either name already exists; do not reset or overwrite another worktree. Confirm that the import path is inside **your new worktree**, not the original checkout. The existing `.venv` has shared dependencies and may use an editable installation pointing at the old checkout. Setting `PYTHONPATH` explicitly is required when reusing it; do not reinstall into or modify that shared environment. If dependencies need changing, create a separate environment.

For Slurm runs, explicitly `cd` to your worktree and export its `src` as `PYTHONPATH` inside the step before invoking the original `.venv/bin/python`. Also export the assigned GPU UUID inside the step. The documented `.venv/bin/python` commands assume a local environment; adapt the interpreter path when using the shared environment.

## Continue this experiment

Read `CONTEXT.md`, `GOAL.md`, `STATUS.md`, `PLAN.md`, `EXPERIMENTS.md`, and `LOGS.md` in this folder. Target: **one independent RL teacher per active task, strictly >90% success on each of 24 tasks**. Keep the approved common 60D observations and normalized 8D absolute joint actions. Tool-Pull stays deferred but implemented/registered. Use `benchmark.active_cl_tasks()`.

The stage has only a two-update PPO plumbing smoke, not trained teachers: 0/24 certified. Validate strict first-episode checkpoint evaluation, resolve readiness issues, run Reach/Lift pilots and continue through the suite. Preserve the success predicates; record any intentional reward or physics changes and revalidate. Log effective configs, seeds, source versions, checkpoints, normalizers and evaluation results.

Follow `~/use_instructions/README.md`; recheck holder20277 and assigned GPUs1–3. Never cancel another person's jobs. The stage-local launcher avoids the generic GPU selector's UUID parsing issue.

## Saved code versus local artifacts

The commit includes source, changed task assets, tests, plans and compact JSON results. Large traces, rendered images/videos, model checkpoints, event files and runtime logs remain in:

`/ihub/homedirs/svs_ald/sudhir/mjlab`

They were not deleted. `LOCAL_ARTIFACTS.json` inventories non-ignored, excluded evidence files; ignored videos and other outputs also remain at their original paths. Historical reports may link to these local-only artifacts. Resolve those links against the original checkout when reviewing evidence, or copy an artifact explicitly when needed. Do not share/symlink writable run directories between worktrees: write all new experiments into your own worktree or uniquely named external run directories.

Unrelated untracked portal/sweep/config-backup files remain in the original checkout and are outside this checkpoint. The original worktree is preserved; continue development in your own branch/worktree.
