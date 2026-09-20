# Plan — CL-V6 architecture

## P0: shared-network implementation (DONE)

`SharedResidualStudentMLP` + `SharedStudentPolicy` added to
`continual_distill/utils.py`; `continual_distill.py` takes
`--architecture {heads,shared_resnet}` (default `heads`, preserving CL-V5
reproducibility) plus `--residual-width`/`--num-residual-blocks`/
`--task-embed-dim`. Checkpoint format extended with `architecture` +
`architecture_kwargs` (old checkpoints read as `heads` by default).
`render_student.py` updated to reconstruct either architecture. Smoke-tested
(2 tasks, 5 epochs, width 1024 / 4 blocks) — trains, checkpoints, evaluates
correctly, no errors.

Design: only `compute_student_distribution`'s `apply_fn` differs by
architecture (a closure resolving task-specific logits, either by slicing a
per-task head or by an embedding-conditioned forward pass); every other
function in the training loop is architecture-agnostic. SI's importance
computation operates on a flattened parameter pytree
(`jax.flatten_util.ravel_pytree`) and needs no architecture-specific code.

## P1: fold in modern training-engineering findings

A research pass (parallel agent, launched 2026-09-20) is scanning
~2014-era fundamentals through 2025/2026 developments (Muon optimizer and
comparable) for a curated, non-exhaustive shortlist: well-established
low-risk wins, promising newer techniques worth a controlled try, things to
explicitly skip, and specifically how standard tricks interact with
SI/EWC-family regularizers (e.g. weight decay fighting the SI anchor,
dropout corrupting the importance estimate). Findings land here once back.

Only adopt something after a cheap 2-task smoke check confirms it doesn't
break training or SI's bookkeeping — don't take a paper's claim on faith for
our specific setting (small residual MLP, supervised regression, SI-anchored
sequential training).

## P2: capacity/depth probes (cheap, single/two-task only)

Given the mid-stage validation step is explicitly skipped this phase, keep
this step minimal: a handful of curated (width, depth) points on the
existing 2-task smoke harness (`tasks_cl_v5_si15_smoke.yaml`, DragPull +
AxialExtract or similar), not a grid search. Pick one config to carry into
P3 based on this + whatever P1 findings apply directly to init/stability at
depth.

## P3: full 15-task validation

Straight from P2 to the full 15-task sequence (skipping CL-V5's rejected
mid-scale 5-6 task stage). Same protocol as CL-V5 Block 3: SI, random
ordering (established best at N=15), 3 seeds. Compare against CL-V5's
baseline (rnd-s2: 0.837 best single run; random-ordering mean: 0.795±0.041).

## P4: publish

Extend the existing https://cl.sudhirpratapyadav.com/v5-cl-si/ page (or spin
a new page if the shared-vs-heads comparison reads better standalone) with
the architecture-comparison results and videos, once real numbers exist.

## Compute

Same shared holder **20277** (`hold_dgx_amit`, expires 2026-09-28), GPUs 1-7
authorized, GPU0 unused. Separate worktree from CL-V5
(`exp/cl15-si-20260918`) and the paused RL-teacher work — this phase is
`exp/cl-arch-20260920`.
