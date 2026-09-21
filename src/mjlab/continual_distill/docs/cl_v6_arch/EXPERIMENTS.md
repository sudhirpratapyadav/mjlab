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

**Final results (completed 2026-09-21, ~9.9h and ~13.5h respectively)**:
w4096b6-s0 avg **0.250**, w2048b10-s0 avg **0.345** -- both show the same
catastrophic middle-task collapse as the width2048/6block seeds (many exact
0.000s), confirming neither more width nor more depth fixes a broken
conditioning mechanism on its own. Consistent with everything learned since
(Blocks 3-7): this was never a capacity problem.

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

Result: still 0.000. **Seven configurations in a row have now failed on
RotateValve specifically**, with no run ever producing a single nonzero
success.

## Block 6 — reframing: this was never a forgetting problem

Re-examined the position-swap result (Block 4) more carefully: RotateValve
AT POSITION 0 -- the very first task trained, where `si_scale` is
`jnp.where(task_idx>0, si_coeff, 0.0)` and is therefore **exactly zero** --
still scored 0.000, with the HIGHEST KL (472) of any RotateValve run.
**SI cannot be causing a failure that already happens with SI completely
inactive.** Every fix attempted so far (si-coeff, block FiLM, output-head
FiLM, output-logit FiLM) targeted anti-forgetting or task-conditioning
mechanisms -- the wrong category of fix, since there is no forgetting to
prevent when task_idx==0. This is a plain supervised-learning-capacity
problem for this one task under this architecture, not a continual-learning
problem at all.

Testing RotateValve in complete isolation (single-task sequence, no other
tasks, full 500-epoch budget matching the real full-run config, `si_scale`
trivially always 0) to separate "does this architecture's supervised
distillation loss converge on this teacher's data at all" from any CL
confound entirely.

**Result: RotateValve learns fine (0.766-0.828) throughout its own 500-epoch
training when trained as task_idx==0** (si_scale==0 the whole time) -- so
the architecture CAN represent and learn this task. The moment task 1
(OpenDrawer) begins -- optimizer reset only, `state.params` unchanged, a
pure re-evaluation -- RotateValve's measured success collapses to exactly
0.000 with the SAME weights that scored 0.83 one line earlier.

**Reframing the si-coeff finding**: in every OTHER test run, RotateValve was
task_idx>=1, meaning SI was active DURING its own training (not just
afterward). Raising si-coeff (5.0, 10.0) never moved it off 0.000 -- which
makes sense if SI is preventing it from ever LEARNING in the first place
(more penalty, more prevention), not failing to protect something already
learned. RotateValve's teacher action range is ~5x wider than other tasks
(Block 5); reaching a good solution likely requires unusually large
parameter displacement in the output-conditioning weights (`logit_film`,
`out_film`), and SI's quadratic anchor penalty -- calibrated for the small
displacements every other task needs -- disproportionately blocks that one
outlier task from ever getting there once ANY prior-task anchor exists.
Testing si-coeff=0.1 (much lower) on the original 4-task ordering to check
this reversed hypothesis.

Result: si-coeff=0.1 -> RotateValve still exactly 0.000. The SI-strength
theory is wrong: dropping the coefficient 10x had zero effect. This
near-perfect insensitivity to a 100x range of si-coeff (0.1 to 10.0, all
giving exactly 0.000) was itself the tell that this wasn't really an SI
problem at all.

## Block 7 — the real bug: a mid-sequence eval used the WRONG episode length

Inspected the isolated-training log's raw debug prints around the task-1
boundary check line-by-line rather than trusting only the headline number.
At the exact point where RotateValve's success collapsed from 0.83 to 0.00
with UNCHANGED weights, the debug print showed **"episode length to run:
150"** for BOTH the teacher and student rollout -- not RotateValve's actual
400-step episode. Even the TEACHER's success dropped correspondingly
(0.95 -> 0.70) at that same call, which is impossible if the bug were in the
student/architecture: the teacher's weights and behavior never change.

Root cause: `evaluate_all_tasks_env`'s "Initial Evaluation (Step 0 - Before
Training)" call sites passed a single `episode_length` argument sourced from
whichever task was CURRENTLY STARTING (e.g. OpenDrawer's 150 steps), and
reused it for every task 0..current_task_idx in that evaluation sweep --
including RotateValve, silently truncating its 400-step episode to 150 and
making completion of the valve rotation structurally impossible regardless
of the actual policy. Fixed: each task in the loop now uses its own
`task_buffers[eval_task_idx]["episode_length"]`; the redundant external
parameter was removed from all 3 call sites (continual_distill.py).

**This bug did NOT corrupt the headline "Final environment evaluation"
numbers already reported** (a separate, correctly-length code path,
confirmed by cross-checking the debug prints immediately preceding those
blocks in prior logs -- they already showed the correct 400-step length).
It only corrupted the INTERMEDIATE mid-training diagnostic view I was using
to understand *why* RotateValve failed, which is exactly what sent this
investigation down several wrong paths (si-coeff, block FiLM, output-head
FiLM, output-logit FiLM, position). With the bug understood, the isolated
test's true, correctly-measured trajectory is: RotateValve trained alone
(task 0) -> 0.83 success -> one subsequent task (OpenDrawer) trained on top
-> drops to a real, bug-free **0.141**. That's a much less catastrophic,
much more ordinary-looking forgetting curve than the mysterious exact-0.000
pattern suggested -- consistent with RotateValve simply being a severely
fragile task (matching its 5x-outlier action scale from Block 5), not a
categorical architecture failure. It still gets to exactly 0.000 by the time
3+ subsequent tasks have trained on top (the 4-task stress4 tests), which is
a real, steep forgetting curve worth addressing, but the SI-coefficient
sweep (0.1-10) genuinely doesn't move it -- so the fix, if there is a cheap
one, isn't a simple coefficient tweak.

A true 1-task sequence crashed on an unrelated bug: `SharedResidualStudentMLP`
with `num_tasks=1` hits a flax `nn.Embed` broadcast error
(`Cannot broadcast to shape with fewer dimensions: arr_shape=(1, 32)
shape=(32,)`) during init. Real bug, worth fixing later, not informative
about RotateValve -- worked around it by training `RotateValve OpenDrawer`
(2 tasks) and reading RotateValve's result at the moment task 1's "Initial
Evaluation (Step 0 - Before Training)" fires, i.e. immediately after
RotateValve's own full 500-epoch training completes and before any
OpenDrawer training has touched the shared weights. `si_scale` was 0
throughout RotateValve's training either way (task_idx==0), so this
preserves the "zero forgetting pressure" property that matters.

## Block 8 — Wave 2 full-scale result: still far below CL-V5's baseline

3-seed full 15-task run (random ordering, width2048/6blocks, si-coeff 5.0,
all fixes applied: block/out_head/out_logit FiLM, corrected eval episode
length), ~660-663 min/run:

| run | final avg | weakest task |
|---|---|---|
| v2-s0 | 0.422 | RotateValve (0.000) |
| v2-s1 | 0.237 | RotateValve, FlipSwitch, PushButton, TurnLever, OpenLid (0.000 each) |
| v2-s2 | 0.369 | OpenDrawer (0.000) |

**Wave 2: mean 0.342 ± 0.078.** CL-V5 baseline: mean 0.795 ± 0.041, best
single run 0.837. **Still a severe regression**, not a modest gap -- despite
fixing the conditioning mechanism and the eval bug, full 15-task scale
produces heavy catastrophic forgetting for most middle-sequence tasks
(RotateValve, OpenDrawer, ThrowToBin near-zero in every seed; FlipSwitch,
TurnLever, PushButton, OpenDoor near-zero in at least one seed each). Only
the first-trained task (ToppleBlock, 0.94-0.98) and the last-trained task
(DragPull, 0.75-0.88) consistently retain well -- the same "only the edges
survive" pattern from Wave 1, just less extreme now.

**Honest conclusion**: the stress4 harness's encouraging signal (4 tasks,
100 epochs -- FiLM fixed the *first* task's retention, and si-coeff tuning
helped edge tasks) did not transfer to the real 15-task/500-epoch scale.
More tasks means more accumulated SI anchor pressure on the same fully-shared
weights, and the FiLM-conditioning fixes -- while real, measurable
improvements over Wave 1 -- are not sufficient to prevent severe
interference once a dozen-plus tasks have trained on top of an earlier one.
This is not a bug; it is a genuine limitation of the "single network, only a
task-embedding-derived conditioning signal" design at this task count. NOT
publishing this result -- it is a clear regression from CL-V5, not an
improvement.

**What this rules out**: capacity alone doesn't fix it (Wave 1's width4096
and 10-block probes both failed similarly, 0.25/0.34). Conditioning strength
alone doesn't fix it (3 levels of FiLM, extensively validated on the smaller
stress harness, still collapses at full scale). SI coefficient doesn't fix
it (0.1 through 10.0 all tested). The remaining lever within "keep it a
single shared network" that hasn't been tried: a small amount of genuine
per-task capacity (e.g. lightweight per-task adapter weights, much smaller
than the old architecture's full separate output heads) rather than pure
conditioning-of-shared-weights. This is a real design choice to bring to the
user rather than a self-authorized deviation from "maximum sharing," since
it moves further from the original all-conditioning design.

## Block 9 — 8-task stress test: nuanced support for the rigidity hypothesis

Extended the stress harness to 8 tasks (ToppleBlock/RotateValve/OpenDrawer/
FlipSwitch/ThrowToBin/ReachTarget/PushButton/PushFlap), same recipe as Wave 2
(si-coeff 5.0, all FiLM layers), 100 epochs/task, 70 min:

| task (position) | 4-task result | 8-task result |
|---|---|---|
| ToppleBlock (1st) | 0.906 | 0.844 |
| RotateValve (2nd) | 0.000 | 0.016 |
| OpenDrawer (3rd) | 0.031 | 0.000 |
| FlipSwitch (4th, was LAST at 4-task) | 0.953 | 0.766 |
| ThrowToBin (5th) | -- | 0.000 |
| ReachTarget (6th) | -- | **1.000** |
| PushButton (7th) | -- | **0.875** |
| PushFlap (8th, LAST) | -- | **1.000** |

Average: 4-task 0.473 -> 8-task **0.563** (higher, not lower -- adding 4 more
tasks did not uniformly hurt). But the *mechanism* still shows through:
FlipSwitch degraded specifically because it moved from "last task, no one
trains on top of it" to "4th of 8, with 4 more tasks piled on afterward"
(0.953 -> 0.766). RotateValve/OpenDrawer/ThrowToBin stay near-zero
regardless of scale -- these are intrinsically fragile tasks (matching
CL-V5's own fragile trio finding), not purely a scale effect. Meanwhile
ReachTarget/PushButton/PushFlap, positioned mid-to-late in the 8-task chain,
retain excellently.

**The real tell is cross-referencing against Wave 2 (15 tasks, same recipe)**:
PushButton scored 0.875 here at position 7-of-8, but collapsed to
**0.000 / 0.297** in 2 of Wave 2's 3 seeds at N=15. That is the rigidity
signature -- not smooth degradation from 4 to 8 tasks, but a cliff between
~8 and ~15: capacity/robustness that holds up through a moderate sequence
length starts failing broadly once enough prior tasks have accumulated SI
pressure on the fully-shared weights. This is consistent with (not
definitively proven by, but strongly suggested by) the structural
`omega_total` monotonic-accumulation mechanism identified in Block 8.

## Conclusion

CL-V6's architecture family (single shared network, task-embedding + 3-level
FiLM conditioning, SI-only) has now been extensively characterized:
- Capacity (width 4096, depth 10 blocks): doesn't fix it (Wave 1 probes,
  0.25/0.34).
- Conditioning strength (block/out_head/out_logit FiLM): fixes some failure
  modes (Wave 1 -> Wave 2 improvement, 0.22-0.30 -> 0.342) but not enough.
- SI coefficient (0.1 to 10.0): doesn't fix the core pattern.
- Task count (4 -> 8 -> 15): degradation is real and appears to worsen
  sharply past ~8 tasks, consistent with the structural SI-accumulation
  mechanism under 100% weight sharing.

**Wave 2's 0.342 ± 0.078 average is the best validated result for this
architecture family, and it remains well below CL-V5's per-task-head
baseline of 0.795 ± 0.041.** Neither Wave 1 nor Wave 2 is published. The
practical, structural reason is that SI's importance penalty accumulates
monotonically and is shared across literally every parameter when the
network has zero task-specific weight subsets -- a fundamental tension with
the "maximum sharing" design constraint at this task count, not a tuning
failure.

## Block 10 — per-task LoRA adapters (user directed to keep pushing)

Implemented a per-task low-rank correction on each residual block's second
Dense, looked up from a per-task `nn.Embed` table (not a conditioning
generator like FiLM). This is the key structural difference from every
previous fix: FiLM modulates the *activations* flowing through shared
weights, so SI's penalty on those shared weights still accumulates across
every task regardless of how good the conditioning signal is. A LoRA
adapter is a genuinely separate parameter subset per task -- gradient only
flows to task T's own `lora_down`/`lora_up` slice when task T is training,
so it is structurally immune to any other task's SI term, the same property
that let the old per-task-head architecture's output layer avoid
interference entirely, just far smaller (rank 8: ~33K params/block/task vs.
a full output head). Zero-init on `lora_up` so every task starts as an exact
no-op. `--lora-rank` CLI flag (0 = disabled, matches Waves 1-2).

Smoke-testing on the stress4 harness (rank 8, si-coeff 5.0, same recipe as
Wave 2) before considering another full 15-task run.

**Result: LoRA on the residual blocks alone did NOT help, and slightly
hurt.** ToppleBlock 0.938 (vs 0.906 baseline), RotateValve **0.000** (same
as baseline, unmoved), OpenDrawer **0.000** (vs 0.031, worse), FlipSwitch
**0.766** (vs 0.953, notably worse). Average 0.426 vs baseline 0.473 --
slightly worse overall.

## Block 11 — LoRA on out_head too

Block-only LoRA never touches `out_head`, which Block 5 root-caused as the
actual bottleneck for RotateValve (a fully-shared final layer, shaped mostly
by small-action-range tasks, needing an order-of-magnitude larger effective
gain for one outlier task). Added a second per-task LoRA correction directly
on `out_head`'s output (same embedding-table mechanism, zero-init `up`
projection). Testing on stress4 (rank 8, si-coeff 5.0) now.

## Planned next

| block | purpose |
|---|---|
| Research pass | curated modern-training-technique shortlist (parallel agent, running) |
| Capacity/depth probes | small curated set of (width, depth) points, 2-task smoke scale |
| Full 15-task runs | random ordering, 3 seeds, compare to CL-V5 (0.795±0.041 mean, 0.837 best) |
