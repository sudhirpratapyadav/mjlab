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
- 2026-09-18T11:09 UTC — Progress check: 4/15 tasks done on all 6 runs, ~9
  min/task observed so far -> revised ETA ~2-2.5h total (~12:45-13:00 UTC
  finish), faster than the original 3-5h guess. Early retention signal is
  strong: DragPull (task 0, trained first in the fragile-first ordering) is
  still at 95-100% student success after 4 more tasks were trained on top of
  it in ff-s0 — SI appears to be working as intended so far. No intervention
  needed; continuing to monitor rather than changing epochs/coefficients.
- 2026-09-18 ~15:11 UTC — Block 3 (all 6 full runs) complete, no tracebacks.
  Fragile-first mean 0.738+/-0.049, random mean 0.795+/-0.041. N=15 point
  (~0.74-0.80) roughly flat vs N=6 (0.792). PushCuboid/ThrowToBin/OpenDrawer
  consistently fragile regardless of ordering/position. AxialExtract's two
  worst failures both occurred at a mid-sequence position in fragile-first,
  suggesting mid-sequence placement may be worse than either extreme at
  N=15 (unlike the simple "earlier is safer" story that held at N=6). Full
  breakdown in EXPERIMENTS.md.
- Launched a fragile-last ordering (3 seeds, GPUs 1-3) to test that
  mid-sequence hypothesis directly, and rendered+published all 15 task
  videos from the best run (rnd-s2, avg 0.837) to
  https://cl.sudhirpratapyadav.com/v5-cl-si/ as Wave 1 (homepage updated,
  previous version backed up on the remote first). render_student.py written
  and verified against an in-flight checkpoint before real finals existed.
- 2026-09-19T09:54 UTC — Noted the fragile-last run only actually started at
  this time, notably later than expected from when it was dispatched;
  possible Slurm/holder contention delay on the shared account. Not acted on
  further since the runs are healthy now (epoch 200/500 on task 0 at the
  10:02 UTC check) — flagging for awareness, not a bug in our own code.
- 2026-09-18T11:40 UTC — Progress check: ff runs 6-7/15 tasks done (~10
  min/task), rnd runs 5-6/15 (~12 min/task, slightly slower ordering). Revised
  ETA ~13:00-13:40 UTC. Retention still strong on both orderings' first task
  (DragPull 93-98%, ToppleBlock 100%) after 6 more tasks trained on top. No
  intervention needed.
