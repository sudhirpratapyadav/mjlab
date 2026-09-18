# Status — CL-V5 SI-15

**Not started. Scaffolding only** — worktree, docs, task selection and W&B
destination are set up; no dataset extraction, sweep, or training run has been
launched yet.

## Readiness

| Check | Status | Evidence / action |
|---|---|---|
| Worktree | DONE | `../mjlab-cl15-si-20260918`, branch `exp/cl15-si-20260918`, from `checkpoint/cl24-rl-handoff-20260912` |
| Task selection (15, >95% val SR) | DONE | `active_tasks.json` |
| Method decision | DONE | SI only (user decision) |
| W&B destination | NOT YET CREATED | entity `sudhirpratapyadav-indian-institute-of-technology-jodhpur`, project `mjlab-cl15-si-20260918` — project is auto-created by W&B on first `wandb.init()`; nothing exists there yet |
| Fresh teacher datasets | NOT STARTED | must extract from the CL-V4 certified checkpoints listed in `active_tasks.json`, not the stale `teacher_datasets/` folder |
| Capacity/LR/SI-coefficient bracket at N=15 | NOT STARTED | see PLAN.md P1 |
| Orderings chosen | NOT STARTED | see PLAN.md P2 |
| Website page | NOT CREATED | planned at `https://cl.sudhirpratapyadav.com/v5-cl-si/`, to be built once real runs exist |

## Public summary

None yet. Will link here once `build_gallery.py`-equivalent output is published.
