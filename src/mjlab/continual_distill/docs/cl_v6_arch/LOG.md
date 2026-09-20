# Log — CL-V6 architecture

- 2026-09-20 — User decisions: single shared network only (no progressive
  nets/expert routing), simple building blocks (residual connections +
  deeper nets), SI only. Maximum parameter sharing requested explicitly —
  replace per-task output heads with a single shared head conditioned on a
  task embedding. Mid-scale validation stage rejected; go smoke/single-task
  -> full 15 directly.
- 2026-09-20 — Implemented `SharedResidualStudentMLP` (task-embedding
  concat, pre-norm residual MLP blocks, single shared output head) and
  `SharedStudentPolicy` wrapper in `continual_distill/utils.py`. Wired into
  `continual_distill.py` via `--architecture {heads,shared_resnet}`
  (default `heads`, so CL-V5 remains reproducible unchanged). Design choice:
  isolated the architecture difference entirely inside `apply_fn` (a closure
  resolving task-specific logits), so `compute_student_distribution` and
  every other call site in the 1900-line training script needed no changes
  — confirmed by grep that all 5 call sites and the SI flatten/importance
  code path are architecture-agnostic.
- Extended checkpoint format with `architecture`/`architecture_kwargs`
  (backward compatible: missing key reads as `heads`). Updated
  `render_student.py` to reconstruct either architecture from checkpoint
  metadata.
- Smoke-tested end to end (2 tasks, 5 epochs, width 1024/4 blocks): trains,
  checkpoints, evaluates correctly, no errors. See EXPERIMENTS.md Block 0.
- Launched a parallel research agent for a curated (not exhaustive) modern
  training-technique shortlist: established low-risk wins, promising newer
  techniques (incl. Muon optimizer), what to explicitly skip for our
  small-residual-MLP/supervised-regression/SI-regularized setting, and
  specific SI/EWC-family interaction warnings (e.g. weight decay vs the SI
  anchor penalty, dropout corrupting the importance estimate). User
  explicitly asked to curate rather than survey exhaustively.
- Implemented P1's adopted items (grad clipping, per-task LR warmup+decay,
  lower Adam beta2, zero-init residual branches). Re-smoke-tested and found
  a real regression (ReachTarget retention 0.547->0.016). Isolated via 5
  targeted A/B smoke runs rather than accepting the research report's
  recommendation on faith: zero-init residual branches was the actual
  cause (a naive full-zero init without Fixup's accompanying depth-scaled
  init on other layers gives a slow cold-start that eats a large fraction
  of a short budget); grad-clip=1.0 was a smaller independent problem (too
  tight for this network's real gradient scale). Dropped zero-init, raised
  the clip default to 10.0, kept warmup+beta2 (neither hurt once zero-init
  was removed). Re-validated clean. Full trace in EXPERIMENTS.md Block 1.
  This is exactly the kind of "verify before trusting a paper's claim for
  our specific setting" the research report itself asked for.
