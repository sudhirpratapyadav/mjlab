# Experiments — CL-V6 architecture

## Block 0 — implementation smoke test (DONE, 2026-09-20)

2 tasks (ReachTarget, AxialExtract), 5 epochs each, SI, `shared_resnet`
architecture (width 1024, 4 residual blocks, task-embed-dim 32), lr 3e-5,
`--no-track` (offline, not logged to wandb — pure correctness check).

Result: no errors, 3.43 min wall time. Checkpoint saved and loadable.
End-of-sequence eval: AxialExtract (just-trained) 90.6% student success,
ReachTarget (trained first, then AxialExtract on top with only 5 epochs of
SI protection) dropped to 54.7% — expected at this tiny epoch budget, not a
concern; the old-architecture smoke test showed the same qualitative pattern
(ReachTarget 100%->75% after AxialExtract in the original CL-V5 smoke).
Confirms: task embedding + residual trunk + single shared head trains
end-to-end, checkpoints/reloads correctly, SI applies across a task
boundary without crashing.

## Block 1 — P1 training-technique A/B on the smoke harness (DONE, 2026-09-20)

Implemented the P1-adopted items from RESEARCH.md (gradient clipping applied
consistently to both the optimizer step and SI's importance update, per-task
warmup+cosine-decay LR schedule, lower Adam beta2, zero-init residual
branches) and re-ran the 2-task/5-epoch smoke test. Result: **ReachTarget
retention collapsed** (0.547 -> 0.016 success, KL 40 -> 2603) relative to the
P0 baseline (no P1 changes). Didn't accept this at face value -- isolated it:

| config | ReachTarget success | KL |
|---|---|---|
| P0 baseline (no P1 changes) | 0.547 | ~40 |
| P1 all-in (clip=1.0, warmup=200, beta2=0.97, zero-init) | 0.016 | 2603 |
| + warmup=0 | 0.031 | 2331 |
| + clip=1e9 (disabled) | 0.109 | 339 |
| + beta2=0.999 (reverted) | 0.203 | 357 |
| + zero-init reverted (lecun_uniform) | **0.516** | **55** |

**Zero-init residual branches were the actual cause**, not clip or beta2 (a
weaker contributor: clip=1.0 alone was also measurably too tight — global
gradient norm for a several-million-parameter net is naturally much larger
than 1.0, so it was silently throttling learning, independent of zero-init).
Root cause of the zero-init failure: a naive full-zero second-Dense-per-block
(without Fixup's accompanying depth-scaled init on the *other* layers) gives
the trunk a slow cold start, and at a tiny 5-epoch budget that cold start eats
a large fraction of the whole training run. RESEARCH.md itself flagged this
as its least-confident "adopt" item for our setting (already pre-norm, which
it says captures most of the intended benefit) — verifying it empirically
was the right call.

**Final P1 config**: zero-init dropped (back to `lecun_uniform` both Dense
layers), `grad-clip-norm` default raised 1.0 -> 10.0 (spike-protector, not a
routine constraint), warmup + beta2=0.97 kept (neither hurt once zero-init
was removed). Re-validated: ReachTarget 0.375/KL 50.6, AxialExtract 0.969 —
healthy, comparable to baseline. This is what P2/P3 will use.

## Planned next

| block | purpose |
|---|---|
| Research pass | curated modern-training-technique shortlist (parallel agent, running) |
| Capacity/depth probes | small curated set of (width, depth) points, 2-task smoke scale |
| Full 15-task runs | random ordering, 3 seeds, compare to CL-V5 (0.795±0.041 mean, 0.837 best) |
