# Experiments To Run

Ordered by **what a reviewer is most likely to demand**. P0 items are close to
mandatory for acceptance; P1 materially strengthens; P2 is nice-to-have.

Fixed conventions for everything below, unless a row says otherwise:
- 3 seeds, report mean ± std
- Best ordering = **PC → OW → OD → PB**; worst ordering = **PB → OD → OW → PC**
- Master trunk `[4096, 2048, 1024]`, 500 distillation epochs/task, lr `3e-5`
- Metric = final average success rate (`Final Avg SR`) + per-task final SR

---

## P0-1. Naive sequential fine-tuning (the forgetting floor) — **✅ DONE**

> **Result (3 seeds each): best ordering 0.267 ± 0.002, worst 0.499 ± 0.094.**
> Forgetting confirmed, in the predicted band. Only the last-trained task survives
> (best ordering: PushButton 1.00, all others 0.00–0.08, identical across seeds).
> Abstract sentence: **0.960 versus 0.267 for sequential fine-tuning.**
>
> Note the floor *inverts* the ordering effect (best < worst without SI, best > worst
> with it) — see `src/mjlab/continual_distill/docs/P0_EXPERIMENTS.md` for why that
> strengthens rather than weakens the primacy claim.

**Why:** The paper has *no quantitative baseline*. Table I compares properties
(✓/✗), not numbers. Right now `0.966` is unanchored — a reviewer cannot tell whether
the problem was hard. This single run makes every other number meaningful, and it is
the cheapest experiment in this file.

**Run:** identical pipeline, distill teachers sequentially into the master policy with
**SI disabled** (`c_SI = 0`), everything else unchanged.

| Config | Ordering | Seeds |
|---|---|---|
| No-SI sequential distillation | best (PC→OW→OD→PB) | 3 |
| No-SI sequential distillation | worst (PB→OD→OW→PC) | 3 |

**Expected:** catastrophic forgetting — early tasks near 0, final avg ≈ 0.25–0.4
(roughly "only the last task works"). If forgetting does NOT appear, that is a
critical finding: it would mean SI is not what is protecting the skills, and the
paper's mechanism claim needs revisiting.

**Reports into:** a new baseline row in the four-task results table + one sentence in
the abstract ("versus X for sequential fine-tuning").

---

## P0-2. Multitask joint-distillation upper bound — **✅ DONE**

> **Result: 0.958 ± 0.006 (3 seeds).** Statistically identical to sequential+SI on
> the best ordering (0.960 ± 0.014) — the anticipated "strong result": continual
> learning at **no cost** relative to joint training. Both are capped by the same
> thing: PushCuboid's teacher (student/teacher = 1.003), so ≈0.96 is the maximum this
> teacher set allows.
>
> Required new code — the trainer was sequential-only. Added `--joint-distill`.



**Why:** gives the *ceiling*. With the floor (P0-1) and ceiling (this), your 0.966
sits on a scale a reviewer can read. Also pre-empts "how much does continual cost you
versus just training on everything at once?"

**Run:** distill **all four teachers simultaneously** into one master policy (no
sequence, no SI, sample batches from all four teachers' rollouts jointly). Same
architecture, same total gradient steps as the 4-task sequential run for fairness.

| Config | Seeds |
|---|---|
| Joint distillation, all 4 teachers at once | 3 |

**Expected:** ~0.95–1.00. If the best ordering (0.966) matches this, that is a strong
result — "continual learning at no cost relative to joint training."

---

## P0-3. Capacity 8192 with a learning-rate sweep — **✅ DONE**

> **Result: the first of the two anticipated outcomes — lower LR recovers 8192, so
> the collapse is an optimizer artifact, not a capacity limit.**
>
> | 8192 @ lr | mean ± std |
> |---|---|
> | 3e-5 (original) | 0.657 ± 0.242 |
> | **1e-5** | **0.946 ± 0.010** ← matches 4096's 0.960 ± 0.014 |
> | 5e-6 | 0.757 ± 0.031 |
> | 1e-6 | 0.518 ± 0.113 |
>
> An inverted U with a locatable optimum, which is stronger than a one-sided
> recovery: 1e-6 genuinely underfits, ruling out "any LR reduction would do".
> Variance also drops 24×.
>
> **Action required: §5.G of SWEEP24_RESULTS.md must be rewritten before submission.**
> "4096 is the sweet spot; over-parameterization re-introduces instability" is
> confounded — all widths shared the LR tuned at 4096. Correct claim: *capacity is not
> the binding constraint; the LR must scale with width.*



**Why:** The paper currently asserts the 8192 collapse is an optimizer effect
("re-enters the sharp-loss regime"). That mechanism is **not tested** — capacity and
LR are confounded because all widths share the LR tuned at 4096. I have already
softened the text to an observation, but running this converts a soft spot into a
finding.

**Run:** width 8192, best ordering, sweep lr ∈ {3e-5 (current), 1e-5, 5e-6, 1e-6}, 3 seeds each.

| Width | lr | Ordering | Seeds |
|---|---|---|---|
| 8192 | 3e-5 (already have: 0.647 ± 0.210) | best | 3 ✔ done |
| 8192 | 1e-5 | best | 3 |
| 8192 | 5e-6 | best | 3 |
| 8192 | 1e-6 | best | 3 |

**Two possible outcomes, both publishable:**
- Lower LR *recovers* 8192 → the collapse is an optimizer artifact, not a capacity
  limit. Rewrite §5.G honestly as "capacity is not the binding constraint; the LR must
  scale with width."
- Lower LR does *not* recover it → genuine over-parameterization effect. The
  "sweet spot" claim becomes much stronger because the obvious confound is ruled out.

---

## P1-4. Complete the heterogeneous-teacher (PLA) grid

**Why:** This is the paper's weakest evidence relative to its claim prominence. Right
now only **2 of 4 tasks** are ever swapped (Open Drawer, Push Button), always on **one
ordering**. A reviewer will ask whether PLA holds when the *fragile* task's teacher is
swapped — which is the interesting case, since Push Cube is where all the cost lands.

### 4a. Swap the fragile task's teacher (most important)
| Config | PC teacher | OW | OD | PB | Seeds |
|---|---|---|---|---|---|
| pc-bc | **BC** | RL | RL | RL | 3 |
| pc-cl | **classical** | RL | RL | RL | 3 |

### 4b. All-four-swapped (the strongest possible PLA demonstration)
| Config | PC | OW | OD | PB | Seeds |
|---|---|---|---|---|---|
| all-bc | BC | BC | BC | BC | 3 |
| all-mixed | RL | BC | classical | BC | 3 |

### 4c. PLA on a second ordering (generality)
Repeat `crossA` (BC Drawer + classical Button) on the **worst** ordering and on one
middle ordering, to show the property is not an artifact of the best ordering.
| Config | Ordering | Seeds |
|---|---|---|
| crossA | worst (PB→OD→OW→PC) | 3 |
| crossA | Seq 3 (PC→OD→PB→OW) | 3 |

**Note:** also record each new teacher's **standalone SR** (as in Table III) — the
competence control is what makes the PLA claim fair, so it must exist for every new
teacher.

---

## P1-5. Comparison against a published replay-free CL method

**Why:** Table I says competitors are replay-based, but the paper never *runs* one.
The closest published method that is replay-free and applicable to your setup:

**Recommended: EWC** (Elastic Weight Consolidation) as a drop-in replacement for SI in
Stage 2. Same distillation pipeline, swap the regularizer. This is the fairest possible
comparison — it isolates the CL regularizer while holding everything else fixed, and it
is cheap because nothing else changes.

| Config | Ordering | Seeds |
|---|---|---|
| KL distillation + **EWC** | best | 3 |
| KL distillation + **EWC** | worst | 3 |
| KL distillation + **L2 regularization** (weakest sensible regularizer) | best | 3 |

**Optional stronger comparison — Progress & Compress** (`schwarz2018progress`, already
cited, replay-free + decoupled per Table I). Costlier to implement (needs the
active-column/knowledge-base split). Only do this if a reviewer explicitly demands a
full competing method rather than a regularizer swap.

**Do NOT attempt:** DisCoRL, PolyTask, Malagon, TAPD as full reimplementations. They are
rehearsal-based or use a different acquisition setup; reproducing them fairly is a
project in itself, and the property table already positions them.

---

## P1-6. Number of tasks beyond four (scalability)

**Why:** Every reviewer of a CL paper asks "does it scale past N=4?" With four tasks
the primacy claim rests on a short sequence. If you have (or can quickly train) a 5th
and 6th teacher, even a partial result helps.

| Config | Tasks | Orderings | Seeds |
|---|---|---|---|
| 6-task sequence | 4 existing + 2 new | 3 orderings: fragile-first, fragile-last, random | 3 |

Full factorial is 720 orderings — do **not** attempt. Three chosen orderings suffice to
show the primacy effect persists.

**If no new tasks are feasible:** state the 4-task limit explicitly in a Limitations
sentence rather than leaving it for a reviewer to raise.

---

## P2-7. Real-robot quantitative evaluation

**Why:** Current hardware evidence is one figure + one qualitative paragraph. The claim
"transfers to real hardware" is currently unfalsifiable as written.

**Run:** N = 10 trials per task on hardware with the best-ordering master policy,
report per-task success rate.

| Task | Trials |
|---|---|
| Push Cube | 10 |
| Push Button | 10 |
| Open Door | 10 |
| Open Drawer | 10 |

Even modest numbers (e.g. 7/10, 9/10) are far stronger than "it executed the tasks."
Record failure modes — they make good Limitations material.

---

## P2-8. KL vs MSE at four tasks

**Why:** The KL-vs-MSE decision is currently justified by a **two-task** control, then
applied to all four-task runs. A reviewer may note the extrapolation.

| Config | Ordering | Seeds |
|---|---|---|
| MSE distillation, 4 tasks | best | 3 |

One run closes the gap. If MSE is much worse, it also strengthens the design choice.

---

## P2-9. Seed count

Currently 3 seeds. Some numbers have large std (8192: ±0.210; Open Drawer at position
3: ±0.164). Where std is large, 3 seeds is thin. If cheap, raise the **capacity sweep**
and the **PLA configs** to 5 seeds. Not needed for the 24-ordering study (n=72 per
position already).

---

## Priority summary

| Priority | Experiment | Runs | Why |
|---|---|---|---|
| **P0** | Sequential fine-tuning floor | 6 | No baseline exists; cheapest, highest impact |
| **P0** | Joint-distillation ceiling | 3 | Anchors 0.966 on a scale |
| **P0** | 8192 + LR sweep | 9 | Removes a stated confound |
| **P1** | PLA grid completion | 18 | Headline-adjacent claim is thinnest evidence |
| **P1** | EWC / L2 regularizer comparison | 9 | Only real method comparison |
| **P1** | 6-task scalability | 9 | Standard CL reviewer question |
| **P2** | Real-robot N=10/task | 40 trials | Makes sim-to-real claim falsifiable |
| **P2** | MSE at 4 tasks | 3 | Closes an extrapolation |
| **P2** | More seeds on high-variance configs | — | Tightens error bars |

**If you run only three things: P0-1, P0-2, P0-3.** Floor, ceiling, and the confound.
Those three convert the paper from "here is our number" to "here is our number, and
here is what it means."
