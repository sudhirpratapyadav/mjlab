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
- 2026-09-18 — Extracted fresh teacher datasets for all 15 tasks. Hit and fixed
  a `taskset` CPU-pinning bug (13/15 crashed instantly on the first attempt);
  fix was to drop manual pinning entirely. Verified against the cgroup's actual
  allocation (`0-3,128-131`) before retrying. Noted the extraction script's own
  success-rate printout is stochastic-action and reads lower than CL-V4's
  certified deterministic rate — expected, not a re-certification, documented
  in EXPERIMENTS.md so it isn't mistaken for a regression later.
- 2026-09-18 — Ran a 2-task/5-epoch smoke test of `continual_distill.py` before
  committing to the full launch. Hit and fixed a wandb-auth bug: configuring
  wandb in a separate `python -c` subprocess doesn't propagate env vars to the
  next command in the same bash script; the real run fell back to a stale
  ambient wandb login and got a 403. Fixed by exporting the wandb env vars
  directly in the same shell as the training invocation.
- 2026-09-18 — Decided to reuse the N=4/N=6 optimum (width 4096, lr 3e-5,
  si-coeff 1.0) for the first full launch rather than re-bracketing at N=15
  first, since PLAN.md's P1 bracket is meant to catch instability and the smoke
  test already validated the pipeline runs correctly. Judgment call to get real
  N=15 signal sooner; will re-bracket only if the full results look anomalous.
- 2026-09-18T10:33 UTC — Launched 6 full runs (fragile-first x3 seeds, random
  x3 seeds), SI only, GPUs 1-6 of holder 20277. All 6 healthy at first check
  (no tracebacks, epoch counters advancing). See EXPERIMENTS.md Block 3.
