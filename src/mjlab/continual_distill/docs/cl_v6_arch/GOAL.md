# Goal — CL-V6: architecture experimentation

Extend CL-V5 (SI-only continual learning, 15 tasks) by allowing the student
network's **size and type** to vary — previously fixed to a plain MLP with
per-task output heads. User decisions (2026-09-20):

- **Single shared network only.** No progressive nets, no modular/expert
  routing, no per-task sub-networks. SI remains the only regularizer.
- **Simple, well-understood building blocks**: residual connections + deeper
  nets. Not a from-scratch architecture search.
- **Maximum parameter sharing.** The old per-task output heads (one linear
  slice per task, only the last layer differing) are replaced by a **single
  shared output head conditioned on a learned task embedding**, concatenated
  to the observation. The only task-specific parameters left are the (tiny)
  embedding table.
- **Also in scope this phase**: modern training engineering — optimizer
  choice (e.g. Muon), initialization, normalization, LR schedules, gradient
  clipping, etc. — anything well-established or credibly promising, not an
  exhaustive literature dump. A research pass covering ~2014-era fundamentals
  through 2025/2026 developments is running in parallel; findings get folded
  into PLAN.md once back.
- **Validation staging**: smoke tests / single-task or two-task checks only,
  then straight to full 15-task runs. No intermediate 5-6 task validation
  stage (explicitly rejected — that was my proposal, user overrode it).

## What "done" looks like

- A working `SharedResidualStudentMLP` (task-embedding-conditioned, residual
  MLP trunk, single output head) trained via SI across all 15 CL-V5 tasks,
  compared against CL-V5's baseline (best run rnd-s2: 0.837; best ordering
  mean, random: 0.795 ± 0.041).
- Any adopted modern-training-technique folded in only after it passes a
  cheap smoke check that it doesn't break training or SI's importance
  bookkeeping.
- Results and videos published as a further wave on the existing
  https://cl.sudhirpratapyadav.com/v5-cl-si/ page (or a new page if the
  comparison is clearer that way — decide once results exist).
