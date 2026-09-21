# Pivot: delta_action distillation weighting on the ORIGINAL architecture

After CL-V6's architecture family (shared_resnet, any combination of FiLM
conditioning and per-task LoRA) was exhausted at 0.342 avg (see
EXPERIMENTS.md Block 12), reconsidered the actual goal (>95% avg) against
what's achievable even with the KNOWN-GOOD architecture: CL-V5's per-task
heads only reached 0.795 mean -- so architecture-sharing was never the only
gap. The remaining gap there is concentrated in 3 known-fragile tasks
(PushCuboid, ThrowToBin, OpenDrawer): fixing just those (rough math on
rnd-s2's per-task numbers) would put the average within reach of ~0.94.

**Found prior, already-validated work**: `docs/GRASP_FIX_EXPERIMENTS.md`
(predates CL-V5/V6, from the LiftCube-forgetting investigation) established
that plain Gaussian-KL distillation is blind to WHICH actions decide
success -- a contact-rich/precision task can match the teacher's action
distribution well overall (KL ~0.6, same as tasks that succeed) while still
collapsing on success rate, because the KL loss doesn't distinguish the
~30 success-critical steps from the ~70% of an episode that's a trivial
hold/idle phase. Fix: weight each per-sample KL by an action-change signal
(`delta_action`, needs no critic) or a teacher-value-change signal
(`delta_value`, needs a precomputed critic pass), normalized to mean 1.0 so
the overall SI/LR budget is unchanged.

**Their end-to-end validated result** (5-task sequence with a fragile task
present): delta_value weighting gave **+0.17** average vs. plain-KL SI
baseline (0.475 -> 0.644), described as a "safe default": real wins when a
fragile task is present, never a loss otherwise. delta_action (simpler, no
critic needed) gave ~3x retention on the specific fragile task in their
2-task tests (0.204 -> 0.596), slightly weaker than delta_value's 0.643 but
much cheaper to set up (no precompute step against the original .pt
checkpoints).

**Applying now**: reverted to `--architecture heads` (CL-V5's original,
validated design -- NOT shared_resnet), added
`--distill-weight-mode delta_action --distill-weight-floor 0.05
--distill-weight-clip 20 --distill-weight-tasks all --si-coeff 3.0` (their
found-optimal settings). Smoke-tested first (2 tasks, clean, no format
issues with the current teacher_datasets). Launched the real test directly
at full scale (3 seeds, 15 tasks, random ordering -- the same protocol as
CL-V5 Block 3 and CL-V6 Wave 2) since this technique was already validated
end-to-end at comparable scale in the prior work; skipped a redundant cheap
stress-harness pre-check given that existing evidence. `launch_deltaaction.sh`,
logs in `run_logs/da-s*.log`, results in `results_deltaaction/`.

Comparison target: CL-V5 plain-KL baseline 0.795 ± 0.041 mean, 0.837 best.
Goal: >0.95 avg.
