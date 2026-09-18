# Experiments — CL-V5 SI-15

Proposed/running/completed trials go here as they happen. Format follows
`docs/P1_EXPERIMENTS.md` (run-name pattern, wandb project, headline table).

No trials yet.

## Planned blocks (not yet launched)

| block | purpose | runs (est.) |
|---|---|---|
| dataset extraction | fresh rollouts from 15 CL-V4 checkpoints | 15 |
| capacity/LR bracket | re-validate 4096-saturates at N=15, on 2-3 tasks | ~6-9 |
| SI coefficient bracket | re-validate c=1.0 at N=15, on 2-3 tasks | ~6-9 |
| full sequence: fragile-first | 3 seeds | 3 |
| full sequence: random | 3 seeds | 3 |
| full sequence: fragile-last (optional) | 3 seeds | 3 |
