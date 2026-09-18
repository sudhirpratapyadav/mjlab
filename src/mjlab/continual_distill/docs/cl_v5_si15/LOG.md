# Log — CL-V5 SI-15

Append-only decision/outcome log, newest at bottom. Format: `YYYY-MM-DD — what
was decided/found — why`.

- 2026-09-18 — Scoped this phase to the 15 CL-V4 tasks with >95% standalone
  validation success rate (Axial-Extract, Flip-Switch, Open-Door, Open-Drawer,
  Open-Lid, Push-Button, Push-Flap, Reach-Target, Slide-Window, Turn-Lever,
  Rotate-Valve, Push-Cuboid, Throw-To-Bin, Topple-Block, Drag-Pull). User
  decision. Reasoning: reuse only teachers competent enough that any CL
  retention loss is attributable to forgetting, not to a weak teacher (same
  gating principle P1-6 used).
- 2026-09-18 — User decision: SI only for this phase, no EWC/L2 comparison run.
  P1_EXPERIMENTS.md already settled that comparison at N=4/N=6; SI had the
  higher peak retention (0.960 vs EWC's 0.923 on the best ordering).
- 2026-09-18 — Created worktree `../mjlab-cl15-si-20260918`
  (`exp/cl15-si-20260918`, from `checkpoint/cl24-rl-handoff-20260912`) and this
  docs folder. New W&B project `mjlab-cl15-si-20260918` chosen (same entity as
  CL-V4); not yet created on W&B — will auto-create on first `wandb.init()`.
  New public page planned at `https://cl.sudhirpratapyadav.com/v5-cl-si/`, not
  yet built — no runs exist to show yet.
