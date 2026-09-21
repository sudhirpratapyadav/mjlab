# Status — CL-V6 architecture

**INVESTIGATION CONCLUDED. Six independent fix mechanisms tried and
exhausted (FiLM x3 levels, SI coefficient 0.1-10.0, capacity, block LoRA,
out_head LoRA); none closes the gap to CL-V5. Best result: Wave 2,
0.342±0.078, vs. CL-V5's 0.795±0.041. Root cause understood and
documented (EXPERIMENTS.md Block 12): SI's monotonic penalty accumulation
under 100% weight sharing causes the shared backbone to drift across a long
task sequence, and no per-task correction mechanism tried (conditioning or
additive) can compensate for a moving target. CL-V5's per-task-head result
stands as the best validated approach. Further progress would require a new
research direction (non-accumulating regularizer, or substantial dedicated
per-task capacity), not incremental tweaks.**

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
| Full 15-task validation (P3) | Wave 1 + Wave 2 both DONE, both below baseline | Wave 1 (0.22-0.30 avg): catastrophic forgetting from a too-weak conditioning mechanism, diagnosed across Blocks 2-7 (added FiLM at 3 levels, fixed a real episode-length eval bug). Wave 2 (0.342±0.078 avg, all fixes applied): STILL far below CL-V5's 0.795±0.041 -- fixing the conditioning mechanism and the eval bug was necessary but not sufficient; severe forgetting persists at full 15-task scale for a fully-shared network. Neither published. See EXPERIMENTS.md Block 8 for the full honest assessment and what's left to try (per-task adapters -- a real design choice for the user, not self-authorized). |
| Website update (P4) | NOT STARTED | will publish once a wave beats CL-V5's published result |
