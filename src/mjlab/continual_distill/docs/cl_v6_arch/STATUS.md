# Status — CL-V6 architecture

**P0 (shared-network implementation) and P1 (training-technique adoption)
both done and validated. No full runs launched yet.**

## Readiness

| Check | Status | Evidence / action |
|---|---|---|
| Worktree | DONE | `../mjlab-cl-arch-20260920`, branch `exp/cl-arch-20260920`, from `exp/cl15-si-20260918` |
| Shared-network implementation | DONE | `SharedResidualStudentMLP`/`SharedStudentPolicy` in `utils.py`; `--architecture shared_resnet` flag in `continual_distill.py` |
| Checkpoint format extended | DONE | `architecture`/`architecture_kwargs` fields, backward-compatible (defaults to `heads`) |
| render_student.py updated | DONE | reconstructs either architecture from checkpoint metadata |
| Modern-training-technique research | DONE | RESEARCH.md — curated shortlist, adopted/queued/skipped items with rationale |
| P1 training-technique adoption | DONE, validated | grad clipping (default 10.0, applied consistently to optimizer + SI), per-task LR warmup+decay, beta2=0.97. Zero-init residual branches tried and DROPPED after a smoke-test A/B showed it collapsed retention — see EXPERIMENTS.md Block 1 |
| Smoke test (final config) | PASS | 2 tasks, 5 epochs, width 1024/4 blocks: ReachTarget 0.375/KL 50.6, AxialExtract 0.969, no errors |
| Capacity/depth probes (P2) | SKIPPED, folded into P3 | user directive: start full runs now instead of a separate probe stage |
| Full 15-task validation (P3) | RUNNING | Wave 1: 3 seeds width2048/6blocks (primary) + width4096/6blocks and width2048/10blocks (capacity/depth probes), all random ordering, GPUs 1-5. Started 2026-09-20T16:23 UTC. Goal (user, 2026-09-20): keep iterating until >95% avg (vs CL-V5's 0.795±0.041 mean / 0.837 best) |
| Website update (P4) | NOT STARTED | will publish once a wave beats CL-V5's published result |
