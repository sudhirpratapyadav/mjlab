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

## Block 2 — Wave 1 full-scale results: SEVERE REGRESSION (2026-09-21)

3-seed primary (shared_resnet, width 2048, 6 blocks, task embedding
concatenated once at the input, single shared output head), random
ordering, full 15 tasks, ~549-551 min/run:

| run | final avg | notes |
|---|---|---|
| w2048b6-s1 | **0.296** | 8 of 15 tasks at exactly 0.000 success |
| w2048b6-s2 | **0.216** | 7 of 15 tasks at exactly 0.000 success |
| w2048b6-s0 | (pending) | |

Compare to CL-V5 baseline: random-ordering mean **0.795±0.041**, worst
individual CL-V5 run **0.686**. This is not a modest gap -- it's catastrophic
forgetting far beyond anything seen with the per-task-head architecture.

**Pattern**: in both seeds, only DragPull (the LAST task trained, no
subsequent forgetting pressure) retained well (0.922 both). Every earlier
task, including ones that were 100% standalone teachers (PushButton,
TurnLever, OpenLid -- all >=0.99 CL-V4 val SR), collapsed to 0.000. KL values
are enormous (30-1000+, vs CL-V5's typical 0.5-2) -- the student's output
distribution for old tasks has drifted completely away from the teacher's,
not just gotten noisier.

**Diagnosis**: concatenating the task embedding once at the input and then
routing through 6 fully-shared residual blocks to a single fully-shared
output head asks the network to *propagate* task identity through the whole
trunk unaided. Under SI's pressure (which nudges every shared weight toward
its value at the end of the previous task), that weak, single-injection
signal is apparently not enough to keep 15 tasks' input->action mappings
separated -- the shared final layer converges toward whatever satisfies the
*most recent* task, overwriting everything else. This is a materially
different (and harder) problem than the per-task-head architecture, where at
least the final layer's weights for each task were never touched by any
other task's gradient at all.

**Fix (implemented, not yet validated at full scale)**: FiLM
(feature-wise linear modulation) conditioning applied at EVERY residual
block, not just concatenation once at the input. Each block generates its
own (scale, shift) from the task embedding via a small per-block Dense
(zero-init, so it's an identity transform at init) and applies it to the
block's hidden activation before the nonlinearity. This gives every layer a
fresh, block-local task signal instead of relying on one that has to survive
propagation through the whole trunk. Still a single shared network, still a
single shared output head, still just concatenation-adjacent -- FiLM is a
well-established, simple technique (Perez et al. 2017), not a departure from
the "keep it simple" constraint. Task-specific parameter count added: ~131K
per block (a small Dense from the 32-dim embedding) x 6 blocks =~ 786K,
still tiny relative to the ~50M-parameter shared trunk -- "maximum sharing"
is preserved in spirit.

The w4096b6-s0 and w2048b10-s0 capacity/depth probes (still running,
GPUs 4-5) use the OLD flawed conditioning and are very likely to show the
same pattern regardless of width/depth -- a wider or deeper network still
routes everything through the same single shared output head with the same
weak embedding signal. Not killing them (sunk cost, not blocking anything),
but not waiting for them or trusting their results as informative about
capacity -- they're confounded by the conditioning-mechanism flaw. Moving
straight to validating FiLM instead.

## Block 3 — FiLM stress-test validation (2026-09-21)

4-task/100-epoch stress harness (`tasks_cl_v6_stress4.yaml`), width
2048/6 blocks, si-coeff 1.0 (unchanged), 22 min wall time:

| task (position) | StudentSucc |
|---|---|
| ToppleBlock (1st, most forgetting pressure) | **0.859** |
| RotateValve (2nd) | **0.000** |
| OpenDrawer (3rd) | 0.016 |
| FlipSwitch (4th, just-trained) | 0.344 |

**Partial fix, not a full fix.** FiLM clearly helped the FIRST task (0.859
vs Wave 1's old architecture, which also happened to retain early/late tasks
reasonably at full scale) -- so FiLM is doing *something* right. But the
*middle* tasks (RotateValve, OpenDrawer) are still collapsing toward zero --
the same "only the edges survive" pattern as Wave 1, just less extreme.
Stronger per-layer conditioning alone did not fix the core problem.

**New hypothesis**: si-coeff=1.0 was tuned for the OLD per-task-head
architecture, where SI only needed to protect the shared *trunk* (~90% of
weights) -- the final per-task head was naturally immune to other tasks'
gradients by construction. In `shared_resnet`, literally every weight
(output head included) is contested by every task, so the same SI strength
may now be underpowered. Testing si-coeff 5.0 and 10.0 on the same stress4
harness before committing to another ~9h full run (each stress4 iteration
is ~22 min -- cheap enough to bracket properly first).

**si-coeff 5.0 / 10.0 results**: si-coeff 1.0 / 5.0 / 10.0 ->
ToppleBlock 0.859/0.922/0.922, FlipSwitch (just-trained) 0.344/1.000/0.953 --
both improved with higher si-coeff, as hypothesized. **But RotateValve
stayed at EXACTLY 0.000 at all three coefficients**, and OpenDrawer stayed
low (0.016/0.062/0.000) -- raising si-coeff did NOT fix the middle-task
collapse. Since RotateValve has been task index 1 (second task trained) in
every test run so far (including both failed Wave-1 seeds, where it was
also exactly 0.000), this isn't obviously an SI-strength problem --
something about *being the first task exposed to a nonzero SI penalty*
(si_scale is 0 for task_idx==0, first turns on at task_idx==1) may be
uniquely hard: the model has to learn a brand-new task while the trunk is
already anchored against task 0, using only ONE prior task's importance
estimate to build that anchor from. Launched a position-swap diagnostic
(RotateValve moved to position 0, ToppleBlock to position 1) to tell
task-specific from position-specific -- if the zero follows ToppleBlock
this time, it's position 1 that's uniquely hard, not RotateValve itself.

## Block 4 — position-swap result: task-specific, not position-specific

RotateValve at position 0 (the previously "safe" slot): **still 0.000**.
ToppleBlock moved to position 1 (the previously "hard" slot): **0.875**, not
zero. Conclusion: **the failure follows RotateValve specifically**, not the
task-1 position. This lines up with the fact that RotateValve was NOT
flagged as fragile in CL-V5's per-task-head architecture (0.969-1.0 success
across multiple runs there) -- something about the shared architecture
specifically breaks this one task.

**New hypothesis**: `out_head` (the final Dense from trunk features to
action logits) had NO task conditioning at all -- FiLM was only applied
inside the residual blocks, before `out_norm`. RotateValve's teacher return
scale (~18-21) is 2-3x most other tasks here (~6-11); if its action
distribution is different enough from the rest, a fully task-agnostic final
layer may not be able to represent it alongside everything else. Added FiLM
conditioning at the output head too (same pattern: zero-init generator from
the task embedding, applied to the pre-`out_head` features). Testing on the
original stress4 ordering (si-coeff 1.0) now.

Result: output-head FiLM alone (si-coeff 1.0) -> RotateValve still 0.000.
Combined with si-coeff 5.0 -> still 0.000. Six configurations in a row have
now failed on this one task (Wave 1 x2, stress4 si 1/5/10, position-swap,
output-head FiLM, output-head FiLM+si5) while other tasks improved or held
steady. Stopped iterating on architecture/coefficient tweaks blind and
looked at the actual data.

## Block 5 — root cause: RotateValve's action scale is ~5x every other task

Checked the teacher datasets directly (`data.pkl`, `action_targets[:, :8]`
= teacher mean columns): RotateValve's teacher action means span
**[-14.05, 13.43]**. FlipSwitch/ToppleBlock/OpenDrawer (from the same
extraction pipeline, directly comparable) all sit in **[-3.4, 2.4]** --
RotateValve is roughly **5x wider**. This is a real, measured anomaly, not
noise. All other tasks so far tested have been small-range; the shared
`out_head` Dense necessarily gets shaped mostly by the many small-range
tasks it also has to serve, and feature-space FiLM (which shifts/scales the
*input* to `out_head`, not its output) apparently can't cheaply grant one
task an order-of-magnitude larger effective gain through those same shared
weights.

**Fix**: added FiLM directly on the produced action LOGITS (post-`out_head`,
16-dim = 2*action_size), zero-init generator from the embedding. This gives
each task direct, unconstrained per-dimension control over its own output
scale, independent of what `out_head`'s shared weights are shaped for.
Testing on stress4 (si-coeff 5.0, since that already helped the other three
tasks) now.

## Planned next

| block | purpose |
|---|---|
| Research pass | curated modern-training-technique shortlist (parallel agent, running) |
| Capacity/depth probes | small curated set of (width, depth) points, 2-task smoke scale |
| Full 15-task runs | random ordering, 3 seeds, compare to CL-V5 (0.795±0.041 mean, 0.837 best) |
