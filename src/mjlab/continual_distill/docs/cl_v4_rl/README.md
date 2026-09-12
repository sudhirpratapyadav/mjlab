# CL-V4 RL teachers

**New agents:** follow [HANDOFF.md](HANDOFF.md) to create an isolated worktree and ensure Python imports that worktree.

Train **one separate RL teacher per active task**, with **strict success >90% for each of the 24 tasks**. Tool-Pull is deferred; its implementation remains available.

Start with [CONTEXT.md](CONTEXT.md), then [GOAL.md](GOAL.md), [STATUS.md](STATUS.md) and [PLAN.md](PLAN.md). Record proposed/running/completed trials in [EXPERIMENTS.md](EXPERIMENTS.md) and append decisions and outcomes to [LOGS.md](LOGS.md).

Current result: shared 60D/8D environment checks and a two-update PPO/checkpoint smoke pass. **0/24 RL teachers certified**; full teacher training has not started. Reach is the first pilot candidate. Reward and training-preset risks are documented before scaling to all tasks.

[Active tasks](active_tasks.json) · [effective configs and source hashes](evidence/preflight.json) · [GPU snapshot](evidence/gpu_snapshot.txt) · [PPO smoke result](evidence/ppo_smoke.json)
