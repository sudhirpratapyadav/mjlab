# P1 experiments — method comparison, capacity de-confound, six-task scalability

Companion to [P0_EXPERIMENTS.md](P0_EXPERIMENTS.md). Covers P1-5 and P1-6 from
`docs/EXPERIMENTS_TODO.md`, plus a per-width LR sweep that finishes the capacity
de-confound P0-3 started.

- **wandb project:** `continual_rl_mjlab_p1` (57 runs)
- **Cluster:** dgx2, holder `19736`, 8× A100-80GB
- **Teachers:** the four RL teachers from SWEEP24, plus LiftCube (RL) and OpenLid
  (classical) for P1-6. All competence-verified on HEAD before launch.
- **Metric:** final average success rate across all tasks in the sequence, 3 seeds.

## Run inventory

| block | run-name pattern | runs | scheduler |
|---|---|---|---|
| P1-5 EWC | `ewc_l<λ>_{best,worst}_s<seed>` | 18 | `slurm/run_p15.sh`, `run_p15b.sh` |
| P1-5 L2 | `l2_l<c>_best_s<seed>` | 9 | `slurm/run_p15.sh` |
| LR-width sweep | `lrw<width>_lr1e5_s<seed>` | 12 | `slurm/run_p1a.sh` |
| P1-6 six-task | `t6_{ff,fl,rnd}_w<width>_s<seed>` | 18 | `slurm/run_p1b.sh` |

Results are re-derivable from the logs with `slurm/collect_p1a.py`; per-run logs are
`logs/cl_<run-name>.log`.

---

## P1-5 — regulariser comparison (EWC / L2 vs SI)

All three regularisers write the **same** quadratic penalty (`omega_total` against a
parameter snapshot) and differ only in how the per-parameter weight is estimated —
path integral (SI), diagonal Fisher (EWC), uniform (L2). That shared-penalty design
is what makes the comparison fair: objective, architecture, teachers and schedule are
all held fixed. Implemented in `continual_distill.py` behind `--regularizer` and
`--reg-coeff`.

### Headline

| method | coefficient | best ordering | worst ordering |
|---|---|---|---|
| **SI (ours)** | c=1.0 | **0.960 ± 0.014** | 0.659 |
| **EWC** | λ=5000 | **0.923 ± 0.010** | **0.824 ± 0.014** |
| L2 | c=1.0 | 0.610 ± 0.170 | — |
| none (P0-1 floor) | — | 0.267 ± 0.002 | 0.499 ± 0.094 |

### The result is ordering-dependent — do not claim SI dominates

- best ordering: SI − EWC = **+0.037** (SI wins, ≈3 seed-std)
- worst ordering: SI − EWC = **−0.165** (**EWC wins, decisively**)

EWC spans 0.923→0.824 (−0.099) between best and worst orderings; SI spans
0.960→0.659 (−0.301), a **3× larger swing**. SI buys peak performance when the
ordering is favourable; EWC buys insensitivity to ordering.

A reviewer who runs EWC on an unfavourable ordering will find it beats SI by 0.165,
so the paper should state the trade-off rather than a flat win. It is also consistent
with the rest of the results: P0-1 showed SI's gain is +0.693 on the best ordering
but only +0.160 on the worst, and P0-3 showed the low-LR failure is a consolidation
failure on the *first* task. SI's mechanism is ordering-dependent by construction.

### EWC λ sweep — optimum bracketed at 5000

| λ | mean ± std | seeds |
|---|---|---|
| 40 | 0.539 ± 0.047 | 0.586, 0.539, 0.492 |
| 400 | 0.732 ± 0.145 | 0.664, 0.633, 0.898 |
| **5000** | **0.923 ± 0.010** | 0.933, 0.914, 0.922 |
| 20000 | 0.897 ± 0.020 | 0.894, 0.879, 0.918 |
| 50000 | 0.892 ± 0.069 | 0.817, 0.953, 0.906 |

The curve turns over, so 5000 is a genuine optimum rather than a grid edge. EWC at
λ=5000 on the worst ordering: 0.820, 0.840, 0.812 → **0.824 ± 0.014**.

### L2 sweep

| c | mean ± std |
|---|---|
| 0.1 | 0.558 ± 0.084 |
| 1.0 | 0.610 ± 0.170 |
| 10 | 0.222 ± 0.053 |

Well below both SI and EWC, and it collapses when over-weighted. Useful ablation:
*which* parameters are protected matters, not merely that anchoring happens.

### ⚠️ Two implementation bugs found here — read before reusing this code

The first attempt reported EWC at **0.269**, i.e. exactly the no-regulariser floor of
0.267. That was not a result about EWC; it was two compounding bugs (fixed in
`2517a83`), and both are easy to reintroduce:

1. **Wrong Fisher variant.** The code squared the gradient of a BATCH-MEAN loss,
   giving `(E_i[g_i])²` instead of `E_i[g_i²]`. That is the "BATCHED" variant, which
   no canonical implementation uses (checked against `ariseff/overcoming-catastrophic`,
   `moskomule/ewc.pytorch`, `kuc2477/pytorch-ewc`) and which van de Ven's ICLR 2025
   blog-track paper measures as the weakest. Measured on a real end-of-task-0
   checkpoint: per-sample 5.98e-4 vs batched 1.09e-4 — a **5.5× underestimate**.
   Now uses `jax.vmap` per-sample gradients, squared before averaging, chunked at 32
   (a 512-wide vmap would materialise ~22 GB).
2. **Dominant: EWC ran at SI's coefficient of 1.0.** SI's omega is normalised by
   squared parameter displacement, so c≈1 is right for it. EWC's Fisher is a raw
   squared gradient with no normalisation (~6e-4 here), and the literature uses
   λ≈40–5000 to compensate. At 1.0 the penalty was ~6e-4 — **numerically inert**.

`--reg-coeff` now overrides `--si-coeff` so each regulariser uses its own scale. The
override is applied before the run config is built, so wandb records the effective
value.

---

## LR × width sweep — the capacity curve is LR-invariant

P0-3 de-confounded only width 8192. This sweeps the rest of the capacity curve at
lr 1e-5 (8192's optimum) to test whether those widths would also improve at a
different LR.

| width | @ lr 1e-5 | @ lr 3e-5 (SWEEP24) |
|---|---|---|
| 512 | 0.478 ± 0.082 | **0.658** |
| 1024 | 0.569 ± 0.042 | **0.792** |
| 2048 | 0.732 ± 0.077 | **0.872** |
| 4096 | 0.793 ± 0.038 | **0.960** |

**Every width is worse at 1e-5.** So 3e-5 is optimal for 512–4096 and only 8192
needed a lower rate. The monotonic 0.66 → 0.79 → 0.87 → 0.96 ranking is
LR-invariant and stands as published — which is the clean answer to a reviewer
asking whether the whole capacity curve was mistuned.

---

## P1-6 — six-task scalability × capacity

Sequence: the four SWEEP24 tasks + **LiftCube** (grasp-and-lift, RL teacher) +
**OpenLid** (hinge-up, classical teacher). Config `config/tasks_6task.yaml`.

Each width runs at **its own** best LR — 4096 @ 3e-5, 8192 @ 1e-5 — because holding
LR fixed across widths would recreate exactly the confound P0-3 removed and make any
capacity difference unreadable.

| ordering | 4096 | 8192 |
|---|---|---|
| fragile-first (PC→OW→OD→PB→LC→OL) | **0.792 ± 0.016** | 0.785 ± 0.014 |
| fragile-last (OL→LC→PB→OD→OW→PC) | 0.724 ± 0.018 | 0.667 ± 0.115 |
| random (OD→PC→OL→OW→PB→LC) | 0.686 ± 0.029 | 0.679 ± 0.019 |

**1. Primacy survives at N=6.** Fragile-first still wins. The ordering claim is not
an artifact of a short sequence.

**2. Capacity is saturated at 4096.** 8192 is equal or slightly worse at every
ordering, each at its own optimal LR. This settles what P0-3 left open: 4096 genuinely
saturates — it is not that four tasks were too few to reveal a difference. Two extra
tasks changed nothing.

**3. Retention falls 0.960 (N=4) → 0.792 (N=6)**, giving a scalability curve rather
than a single point.

### LiftCube is a second fragile task — worth following up

In fragile-first, LiftCube collapses to **0.016** while every other task is 0.80–1.00
and its own teacher scores 1.000 in the same run. It sits 5th of 6 there, but is
retained at ~0.9 when trained 2nd (fragile-last). So fragility may be a property of
**grasp-type tasks generally**, not of PushCuboid specifically — a sharper and
stronger claim than the paper currently makes, and testable with the Lift-Sphere /
Lift-Ellipsoid teachers that already exist.

### Teacher competence verified before launch

`slurm/verify_p1b_teachers.sh` measured all six on HEAD (128 episodes) and gated the
launch on every teacher clearing 0.5:

| teacher | SR |
|---|---|
| PushCuboid | 0.836 |
| OpenDrawer | 1.000 |
| OpenDoor | 1.000 |
| PushButton | 1.000 |
| LiftCube | 0.945–0.977 |
| OpenLid | 1.000 |

This gate exists because the placement regression (see P0_EXPERIMENTS.md) silently
invalidated five teachers at different points — obs_dim stays 60 and nothing errors,
so an unchecked launch produces plausible-looking garbage. It caught LiftCube at
0.023 before P1-6 ran, which is why `97703b4` exists.
