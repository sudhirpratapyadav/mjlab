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

## P1-5. Comparison against a published replay-free CL method — **✅ DONE (best ordering)**

> **Result (3 seeds; λ optimum bracketed at 5000):**
>
> | method | coefficient | best ordering | worst ordering |
> |---|---|---|---|
> | SI (ours) | c=1.0 | **0.960 ± 0.014** | 0.659 |
> | EWC | λ=5000 | **0.923 ± 0.010** | **0.824 ± 0.014** |
> | L2 | c=1.0 | 0.610 ± 0.170 | — |
> | none (floor) | — | 0.267 ± 0.002 | 0.499 ± 0.094 |
>
> λ sweep (best ordering), now **bracketed** — the curve turns over, so 5000 is a
> genuine optimum and not a grid edge:
> 40→0.539, 400→0.732, **5000→0.923**, 20000→0.897, 50000→0.892.
>
> **⚠️ The headline is NOT "SI beats EWC" — it is ordering-dependent, and this is
> the most reviewer-sensitive result in the whole set:**
>
> - best ordering: SI − EWC = **+0.037** (SI wins, ~3 seed-std)
> - worst ordering: SI − EWC = **−0.165** (**EWC wins, decisively**)
>
> EWC is also far more *robust to ordering* than SI: it spans 0.923→0.824 (−0.099)
> between best and worst, where SI spans 0.960→0.659 (−0.301), a 3× larger swing.
> So SI buys peak performance when the ordering is favourable, and EWC buys
> insensitivity to ordering. That is a genuinely interesting trade-off and a stronger,
> more honest framing than a flat "ours wins" — but it does mean the paper cannot
> claim SI dominates. A reviewer running EWC on an unfavourable ordering would find
> it beats SI by 0.165.
>
> It also reframes the primacy story: SI's advantage *depends* on good ordering, which
> is consistent with P0-1 (SI's gain is +0.693 on the best ordering vs +0.160 on the
> worst) and with P0-3 (the LR failure is a consolidation failure on the first task).
>
> L2 (0.610) sits well below both and degrades at c=10 (0.222): a uniform anchor
> over-constrains. Useful ablation — *which* parameters are protected matters, not
> just that anchoring happens.
>
> **Two implementation bugs were found and fixed here** (2517a83); the first attempt
> reported EWC at 0.269, i.e. exactly the no-regulariser floor:
> 1. The Fisher was the BATCHED variant (squared batch-mean gradient) rather than the
>    canonical per-sample square-then-average. Measured 5.5× underestimate.
> 2. Dominant: EWC ran at SI's c=1.0. SI's omega is displacement-normalised so c~1 is
>    right; EWC's raw Fisher is ~6e-4, and the literature uses λ~40–5000. The penalty
>    was numerically inert — EWC was never switched on.
>
> Anyone re-running this must give each regulariser its own coefficient (`--reg-coeff`).

<details>
<summary>Original P1-5 plan</summary>

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

</details>

---

## P1-6. Number of tasks beyond four (scalability) — **✅ DONE**

> **Result (6 tasks = the 4 + LiftCube + OpenLid, 3 seeds each):**
>
> | ordering | 4096 | 8192 |
> |---|---|---|
> | fragile-first | **0.792 ± 0.016** | 0.785 ± 0.014 |
> | fragile-last | 0.724 ± 0.018 | 0.667 ± 0.115 |
> | random | 0.686 ± 0.029 | 0.679 ± 0.019 |
>
> **1. The primacy effect survives at N=6** — fragile-first still wins. The ordering
> claim is not an artifact of a short sequence.
>
> **2. Capacity is saturated at 4096.** 8192 is equal or slightly worse at every
> ordering, with each width at its OWN optimal LR (4096@3e-5, 8192@1e-5 per P0-3).
> This settles the question P0-3 left open: 4096 genuinely saturates — it is not that
> four tasks were too few to reveal a capacity difference. Two more tasks changed
> nothing.
>
> **3. Retention falls 0.960 (N=4) → 0.792 (N=6)**, giving a scalability curve rather
> than a single point.
>
> **Worth following up:** in fragile-first, LiftCube collapses to 0.016 while every
> other task is 0.80–1.00 and its own teacher scores 1.000 in the same run. It sits
> 5th of 6 there but is retained at ~0.9 when trained 2nd. So LiftCube is a SECOND
> fragile task — the primacy story may be about grasp-type tasks generally, not
> PushCuboid specifically. That is a sharper claim than the paper currently makes.
>
> Both new teachers were competence-verified on HEAD before launch (LiftCube
> 0.95–0.98 after the 97703b4 placement fix; OpenLid 1.000).

<details>
<summary>Original P1-6 plan</summary>

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

</details>

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

| Priority | Experiment | Runs | Status |
|---|---|---|---|
| **P0** | Sequential fine-tuning floor | 6 | ✅ 0.267 / 0.499 |
| **P0** | Joint-distillation ceiling | 3 | ✅ 0.958 |
| **P0** | 8192 + LR sweep | 9 | ✅ 0.946 @ 1e-5 — collapse was an LR artifact |
| **P1** | EWC / L2 regularizer comparison | 9+18 | ✅ EWC 0.923 vs SI 0.960 (λ bracketing in flight) |
| **P1** | 6-task scalability | 18 | ✅ primacy holds; 4096 saturates |
| **P1** | PLA grid completion | 18 | ⬜ **partly blocked** — see below |
| **P2** | Real-robot N=10/task | 40 trials | ⬜ hardware, not schedulable here |
| **P2** | MSE at 4 tasks | 3 | ⬜ cheap, unblocked |
| **P2** | More seeds on high-variance configs | — | ⬜ mostly moot now (see below) |

### What is actually left

**P1-4 (PLA grid) — the last substantive item, and it needs teachers that don't exist.**
Inventory against what the plan asks for:

| needed | have |
|---|---|
| PushCuboid classical | ✅ |
| PushCuboid **BC** | ❌ |
| OpenDoor BC *or* classical | ❌ **neither** |
| OpenDrawer BC / classical | ✅ both |
| PushButton BC / classical | ✅ both |

So 4a is half-runnable (`pc-cl` yes, `pc-bc` no), 4b (all-four-swapped) is blocked,
and 4c (crossA on two more orderings) is fully runnable today. Collecting the missing
teachers is now cheap — `collect_classical_dataset.py` was fixed to use the full
19-task registry (9f25bbf), and Open-Lid took ~10 min end to end.

**P2-8 (MSE vs KL at 4 tasks)** — 3 runs, no blockers, closes a stated extrapolation.

**P2-9 (more seeds)** — largely overtaken. The high-variance configs that motivated it
(8192 at ±0.210, Drawer at ±0.164) were LR artifacts; at its proper LR 8192 is
±0.010. Worth revisiting only for `fl_w8192` (±0.115) in P1-6.

**Follow-up the results themselves suggest:** LiftCube behaves as a second fragile
task (0.016 when trained 5th of 6, ~0.9 when trained 2nd). Testing whether *any*
grasp-type task is fragile would generalise the primacy claim from one task to a
task class — a stronger result than the paper currently claims.

**Classical-teacher improvement — attempted, not achieved.** Seven strategy changes
on Push-Cuboid and two on Open-Door, all measured, none reaching >99%; see the
2026-08-01 section of `continual_distill/docs/benchmark/CLASSICAL_TEACHERS.md` for
every attempt and what each ruled out. Push-Cuboid stays at ~0.10–0.25 and Open-Door
at 0.000. This matters for **P1-4**: the PLA grid needs a classical or BC teacher for
Open-Door and there is still none, so 4b remains blocked and 4a is half-runnable
(`pc-cl` only). The useful residue is diagnostic — the cuboid's failure is endgame
precision (the box reaches the goal region and is knocked around it), not transport
or contact geometry, and the door's is a ~13× throughput shortfall that wants a
longer episode or a learned teacher.

**Documentation of results:** `continual_distill/docs/P0_EXPERIMENTS.md` (18 runs,
wandb `continual_rl_mjlab_p0`) and `P1_EXPERIMENTS.md` (57 runs, wandb
`continual_rl_mjlab_p1`) carry the run inventories, per-arm numbers and readings.
