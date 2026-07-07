# Grasp-forgetting fix experiments

Tracking doc for fixing the LiftCube failure (see FINDINGS.md Finding 8): the
Gaussian-KL distillation objective is blind to success-critical actions, so
contact-rich precision tasks (LiftCube) reach only ~0.1 success while the student
matches the teacher's action distribution well (env-KL ~0.6, same as tasks that hit
1.0). Fix must be on the OBJECTIVE side, not SI/consolidation.

## Protocol (fixed for all experiments here)

- **2-task sequences only.** LiftCube is ALWAYS task 0 (the fragile task we're
  fixing); task 1 is one of the 4 others. => 4 experiments per hypothesis:
  - `LiftCube -> PushCuboid`
  - `LiftCube -> PushButton`
  - `LiftCube -> OpenDoor`
  - `LiftCube -> OpenDrawer`
- Metric: LiftCube's success at its last periodic eval (fixed reader), i.e. after
  task 1 has trained. Baseline (plain KL distill) to beat: LiftCube ~0.11.
- lr 3e-5, 4096 student, seed 0, num_epochs per task = 200 (enough to converge;
  the 5-task run used 500 but 200 already shows the collapse).
- Report LiftCube acc AND env-KL for each (KL should stay ~0.6; we want ACC up).

## Run registry (traceability)

wandb project `continual_rl_mjlab` (IITJ entity). Run name == log stem: each run is
`<group>_<pair>_<TS>`; log at `~/sudhir/mjlab/logs/cl_<runname>.log`; wandb run name
== `<runname>`; wandb id in the table below. Each group = 4 pairs
(pushcuboid/pushbutton/opendoor/opendrawer) unless noted.

| group (TS) | config | wandb ids (cuboid / button / door / drawer) |
|---|---|---|
| `graspfix_base_*_1783381305` | uniform si1 (baseline) seed0 | 10xpti8g / sf4qei7u / kboam6dz / lwebed5u |
| `graspfix_a4_*_1783381305` | A4 (floor0.3/clip8) si1 seed0 | v5yac3qt / 3jezoxip / ui8s5ykt / neaq7znu |
| `a4tune_agg_*_1783382338` | A4 (0.1/12) si1 seed0 | gb9c6zs5 / 6ck2b6xe / 5kbns469 / 12j9fect |
| `a4tune_vagg_*_1783382338` | A4 (0.05/20) si1 seed0 | mahesq6j / m0z05jim / xsls2tek / 9ufm7eoy |
| `a4si_si3_*_1783383134` | A4 vagg si3 seed0 | i4o2zoo9 / x9i4mc8l / 2t7r21qw / d408c9m9 |
| `a4si_si10_*_1783383134` | A4 vagg si10 seed0 | gvxi8ym2 / tci193z8 / yqr8j6gj / e3317911 |
| `unifsi_si3_*_1783384101` | uniform si3 seed0 | 0g8ntccn / jv6pebot / fjnag3i1 / 6v0b6mnx |
| `unifsi_si10_*_1783384101` | uniform si10 seed0 | 117qfxgq / 1atd9oi9 / sk5pl1q1 / 19w5jtf9 |
| `mseed_s1_a4si1_*_1783397368` | A4 vagg si1 **seed1** | 002mir78 / bo5gyndi / d4jloe2l / mc5y6uuw |
| `mseed_s1_unifsi1_*_1783397368` | uniform si1 **seed1** | rkb5lglf / 0ovlox6c / p4haqgsi / 2ll2ab0j |
| `mseed_s1_a4si3_*` / `mseed_s1_unifsi3_*` (_1783397368) | A4/unif si3 **seed1** | (see wandb by run name) |
| `mseed_s2_a4si1_*` / `mseed_s2_unifsi1_*` (_1783397368) | A4/unif si1 **seed2** | (see wandb by run name) |

All result tables below reference these groups by their (config) label.

## STEP 0 — Signal assessment (do BEFORE building any weighting)

Rule: never plug a signal into a weighting scheme without first confirming it
(a) varies and (b) shows the expected critical-moment structure (grasp is
mid-episode for a pick task). Findings:

- **Teacher action-std is CONSTANT across states** (rsl-rl uses a state-independent
  learned std param). Per-dim values `[.576 .505 .492 .442 .292 .281 .272 1.079]`,
  range 0.0000 across all 256k samples. => **A1 "per-state teacher sigma" is DEAD**
  as-is. Only a static PER-DIM weighting survives (dims 4-6 tight ~0.27-0.29 =
  likely grasp/orientation-critical; dim 7 loose ~1.08 = gripper). Static, cannot
  localize the grasp moment in time.
- **Teacher HAS a critic** (value head 60->512->256->128->1) => B1 candidate.
- **Rollout signal probe (32 envs, teacher):** grasp/lift happens at t~35-70
  (first-success median t=45). Timeline:
  - `|Δaction|` (per-step action change): HIGH 0.28-0.39 during approach/grasp
    (t<40), collapses to ~0.001 during the post-grasp HOLD (t>70). **Sharpest
    critical-moment signal; needs no critic.** <- best signal.
  - `V(s)`: rises 3.3->5.3 through the grasp then plateaus. Separates before/after
    grasp but is CUMULATIVE (stays high through the useless hold) -> blunt for
    up-weighting the moment.
  - reward: ~flat 0.059 dense shaping -> useless as a locator.
  - **Key insight: ~75% of the dataset is the post-grasp HOLD** where actions are
    near-constant and trivially cloned. Plain KL spends its capacity there and
    under-fits the grasp. That IS the mechanism.

## Hypotheses & options

### A — Action-sensitivity-weighted distillation loss
- **A1 per-state teacher-sigma weighting** — DEAD (std is constant). Skip.
- **A2 static per-dim weighting** — weight KL per action-dim by 1/sigma_dim^2. Cheap,
  static. Secondary.
- **A3 value weighting** — weight by teacher V(s). Blunt (value stays high through
  the useless hold). Secondary.
- **A4 |Δaction|-weighting (PRIMARY, chosen by the signal probe)** — weight each
  sample's distill loss by the teacher's per-step action change (normalized within
  each episode), so the grasp (high |Δaction|) is up-weighted and the post-grasp
  hold (near-zero |Δaction|, ~75% of data) is down-weighted. No critic needed;
  computed from the dataset's action sequence at load time.

### B — Success / return-matching auxiliary term
- **B1 value-matching regularizer** — add `lambda * (V_student_head(s) - V_teacher(s))^2`
  using the teacher critic on dataset states; student grows a small value head.
  Offline, cheap. Nudges the student toward outcome-relevant representations.
- **B2 return/success-weighted sample reweighting** — weight samples by their
  teacher episode return. Weak here (teacher is 99% success => low variance).

## Results log

### CRITICAL: dataset is STEP-MAJOR (a data-layout bug bit the analysis)

The dataset is stored STEP-MAJOR: collection stepped E=512 envs in parallel and
appended one E-row block per timestep, so `obs[i]` = (step i//E, env i%E). My first
round of signal analysis (and the first A4 impl) reshaped ENV-MAJOR ([n_ep, T]),
which SCRAMBLES timesteps across envs and made every temporal signal look flat —
leading me to (wrongly) reject A4 and the value signal, and to (wrongly) conclude
"the dataset is 80% already-lifted hold / has no grasp". ALL of that was the reshape
bug. With the CORRECT step-major layout [S=500, E=512]:
- cube starts on the ground (z=0.035 at step 0), lifts off at step ~37, held after.
- gripper approaches: g2o 0.364 -> 0.007 by step ~25.
- **|Δaction| is a SHARP grasp locator: 0.31 during grasp (t<47) vs 0.007 in the
  hold (t>67) — a 44x ratio.** ~70% of the episode is the post-grasp hold.
- (teacher V(s) is genuinely ~flat even step-major — the converged teacher's value
  is ~constant; so value-weighting A3/B1 has a weak temporal signal, deferred.)

**Lesson: verify the dataset LAYOUT before analyzing any per-timestep signal.**
A4 is well-motivated after all: up-weight the sharp, under-represented grasp steps.

### A4 delta_action RESULTS (2026-07-07, 2-task, 200 epochs, seed 0)
Runs: baseline=`graspfix_base_*_1783381305`, A4=`graspfix_a4_*_1783381305`.

LiftCube retention (last periodic eval) after task 1 trains. Weight profile:
grasp 2.4x vs hold 0.4x (FLOOR=0.3, CLIP=8, median-scaled).

| 2nd task | baseline | A4 | Δ |
|---|---|---|---|
| PushCuboid | 0.188 | 0.141 | -0.047 |
| PushButton | 0.000 | 0.000 | 0.000 |
| OpenDoor | 0.359 | 0.547 | +0.188 |
| OpenDrawer | 0.172 | 0.766 | +0.594 |
| **MEAN** | **0.180** | **0.363** | **+0.184** |

**A4 works: ~2x mean retention.** Big wins where there's headroom (OpenDrawer
0.17->0.77, OpenDoor +0.19). Two problem pairs: PushButton wipes LiftCube to 0.0
for both (floor effect — A4 can't recover from total collapse); PushCuboid -0.05
(within single-seed noise). => A4 validated but not complete. Next: tune contrast
(stronger grasp weighting: lower FLOOR / higher effective ratio) to see if the
hard pairs improve.

### A4 contrast sweep (2026-07-07) — knob exhausted
Runs: base=`graspfix_base_*_1783381305`, A4(0.3/8)=`graspfix_a4_*`, agg=`a4tune_agg_*_1783382338`, vagg=`a4tune_vagg_*_1783382338`.

| LiftCube -> | base | A4 0.3/8 | agg 0.1/12 | vagg 0.05/20 |
|---|---|---|---|---|
| PushCuboid | 0.188 | 0.141 | 0.156 | 0.141 |
| PushButton | 0.000 | 0.000 | 0.000 | 0.016 |
| OpenDoor | 0.359 | 0.547 | 0.516 | 0.562 |
| OpenDrawer | 0.172 | 0.766 | 0.906 | 0.906 |
| MEAN | 0.180 | 0.363 | 0.394 | 0.406 |

- Stronger contrast helps monotonically on winnable pairs (OpenDrawer 0.77->0.91),
  mean 0.36->0.41 (2.3x baseline). Plateaus by floor=0.05 => contrast knob EXHAUSTED.
- Hard pairs (PushButton ~0, PushCuboid ~0.14) are UNMOVED by contrast. A4 only
  reshapes WHICH actions are distilled; it adds no consolidation STRENGTH, so once
  downstream training overwrites the grasp entirely (PushButton case), reweighting
  the vanished grasp can't recover it.

**Insight:** A4 fixes the OBJECTIVE (helps where the grasp survives) but the hard
pairs need CONSOLIDATION help. Two directions left:
- **A4 + stronger SI** (si_coeff sweep on top of vagg A4): does protecting weights
  harder let A4's grasp-signal persist through PushButton? Cheap, tests the insight.
- **B1 value/return matching** — deferred (weak temporal value signal), lower prior.
Next: A4(vagg) x si_coeff {1(default),3,10} on the 2 hard pairs + 2 easy (control).

### A4(vagg) x si_coeff sweep (2026-07-07) — BREAKTHROUGH
Runs: si1=`a4tune_vagg_*_1783382338`, si3=`a4si_si3_*_1783383134`, si10=`a4si_si10_*_1783383134`.

| LiftCube -> | base | A4 si1 | A4 si3 | A4 si10 |
|---|---|---|---|---|
| PushCuboid | 0.188 | 0.141 | 0.562 | 0.969 |
| PushButton | 0.000 | 0.016 | 0.234 | 0.750 |
| OpenDoor | 0.359 | 0.562 | 0.812 | 0.938 |
| OpenDrawer | 0.172 | 0.906 | 0.969 | 0.984 |
| MEAN | 0.180 | 0.406 | 0.645 | **0.910** |

**A4 + si_coeff=10 => 0.910 mean (5x baseline), rescues ALL pairs incl. the
hopeless ones (PushButton 0->0.75, PushCuboid 0.19->0.97).** Confirms the mechanism:
the fix needs BOTH the objective reshaping (A4, learn the grasp) AND strong
consolidation (SI, protect it). Monotonic in si_coeff.

**CRITICAL CONTROL PENDING:** is this A4, or would uniform+si=10 alone do it?
(Earlier si sweeps were on the buggy eval metric AND without A4.) Running
uniform x si {3,10} on all 4 pairs to isolate A4's contribution.

### CONTROL (uniform+SI) + task-1 plasticity check (2026-07-07) — reframes it
Runs: A4+si=`a4si_si{3,10}_*_1783383134`, unif+si=`unifsi_si{3,10}_*_1783384101`.

LiftCube retention, A4+si10 vs uniform+si10 (isolating A4):
| pair | A4+si10 | unif+si10 |
|---|---|---|
| PushCuboid | 0.969 | 0.969 |
| PushButton | 0.750 | 0.688 |
| OpenDoor | 0.938 | 0.969 |
| OpenDrawer | 0.984 | 0.984 |
| MEAN | 0.910 | 0.902 |

**A4 ~= uniform at si=10 (0.910 vs 0.902): the win is STRONG SI, not A4.** A4 alone
plateaus at 0.41; si=10 alone reaches 0.90. A4's marginal contribution is within
noise. (Also: the earlier "si_coeff=3 doesn't fix task-0" was a BUGGY-EVAL-METRIC
artifact — with the fixed metric, SI clearly protects LiftCube.)

**BUT si=10 costs new-task PLASTICITY (task-1 own accuracy):**
| task1 | A4+si10 | unif+si10 | normal |
|---|---|---|---|
| PushCuboid | 0.141 | 0.297 | ~0.75 |
| PushButton | 1.000 | 1.000 | 1.000 |
| OpenDoor | 0.969 | 0.797 | ~1.0 |
| OpenDrawer | 0.984 | 0.547 | ~0.97 |

PushCuboid-as-task1 collapses 0.75->0.14 (monotone in si: 0.56/0.34/0.14 at
si=1/3/10); OpenDrawer-as-task1 to 0.55. => si=10 partially FREEZES the net. It is
a retention<->plasticity TRADE-OFF, not a free fix; the fragile/precision task
(PushCuboid) pays as task-1.

**Direction A (objective reshaping) is EXHAUSTED and largely negative:** A4 helps
modestly alone but is dominated by SI, and doesn't add over SI. The real lever is
consolidation strength, which has its own cost. Open question worth pursuing:
can we get LiftCube retention WITHOUT the plasticity hit — i.e. SELECTIVE
consolidation (protect only LiftCube's grasp-critical weights, not the whole net)?
That's a per-parameter/importance idea, distinct from a global si_coeff.

### CORRECTED FRAMING (2026-07-07): A4 at FIXED consolidation budget — IT WORKS
Runs: see si1/si3/si10 groups in the registry (A4=a4tune_vagg/a4si, unif=graspfix_base/unifsi).

Prior conclusion "win is SI not A4" was the wrong comparison (it varied the SI
budget). The right question (user's): at the SAME si (same weight-constraint), does
prioritizing success-relevant states improve retention? YES.

A4 vs uniform at MATCHED si (mean LiftCube retention):
| si | A4 | uniform | Δ |
|---|---|---|---|
| 1 | 0.406 | 0.180 | +0.227 |
| 3 | 0.645 | 0.590 | +0.055 |
| 10 | 0.910 | 0.902 | +0.008 |

Retention<->plasticity FRONTIER (mean LiftCube ret, mean task-1 acc):
| config | LiftRet | task1 |
|---|---|---|
| unif si1 | 0.180 | 0.887 |
| A4 si1   | 0.406 | 0.883 |
| unif si3 | 0.590 | 0.797 |
| A4 si3   | 0.645 | 0.813 |
| unif si10| 0.902 | 0.660 |
| A4 si10  | 0.910 | 0.773 |

**A4 DOMINATES the frontier: more retention for the same plasticity at every level.**
- A4 si1 vs unif si1: task-1 tied (0.88) but retention 0.41 vs 0.18 (+0.23) — free
  lunch at fixed budget. Exactly the A-direction hypothesis.
- A4 si10 vs unif si10: retention tied (~0.91) but task-1 0.773 vs 0.660 (+0.11) —
  A4 preserves more plasticity at matched retention.
- The benefit is largest at LOW si (room to redirect) and shrinks as si->10 clamps
  all weights (nothing left to redirect). A4 and SI are substitutable levers;
  A4 shifts the whole frontier outward.

**A-direction CONFIRMED and characterized.** Next tweaks within A: (i) A4 combined
with the modest si that best exploits the frontier (si~3), multi-seed to de-noise;
(ii) the OTHER A-signals now that layout is fixed (per-dim A2, value A3) to see if a
better state-prioritization beats |Δaction|.

### PROPER METRICS (2026-07-07): task0-KL + retention-vs-plasticity (not just si)
Runs: same groups as above (registry). seed0.

si is only a knob; the real conserved quantities are the achieved task-0 KL (the
actual weight-constraint) and the retention(task0-SR) vs plasticity(task1-SR) trade.
All 4 measured together, at end of task-1 training (seed0, mean over 4 pairs):

| config | task0-KL | task0-SR | task1-SR |
|---|---|---|---|
| unif si1 | 2.332 | 0.180 | 0.887 |
| A4   si1 | 1.911 | 0.406 | 0.883 |
| unif si3 | 0.973 | 0.590 | 0.797 |
| A4   si3 | 0.629 | 0.645 | 0.813 |
| unif si10| 0.209 | 0.902 | 0.660 |
| A4   si10| 0.122 | 0.910 | 0.773 |

**Key readings:**
- *Retention at matched KL:* A4 si1 (KL 1.91 -> SR 0.41) vs unif si1 (KL 2.33 ->
  SR 0.18): ~2.3x retention at ~the same KL constraint. Uniform must push KL to
  ~0.97 (unif si3) to match what A4 gets at KL 1.91. => A4 converts KL-reduction
  into SR far more efficiently — it spends preserved fidelity on grasp-critical
  actions, not the trivial hold.
- *Retention vs plasticity:* A4 dominates the frontier (si1: +0.23 SR free at equal
  task1; si10: +0.11 task1 at equal retention).
- A4 also yields LOWER task0-KL than uniform at every si (concentrated fidelity).

Multi-seed (seeds 1,2) at si {1,3} running to add error bars. This is the headline
result of Direction A: **at fixed consolidation (KL), prioritizing success-critical
states/actions improves retention-per-plasticity.**

### DUAL-CHECKPOINT (2026-07-07): end-of-task0 vs end-of-task1
Runs: same seed0 groups (registry).

Capturing task-0 KL/SR at END OF TASK 0 (fresh distill, before task1) and END OF
TASK 1 (after forgetting) isolates initial-learning from forgetting. seed0, mean/4:

| config | KL@end0 | SR@end0 | KL@end1 | SR@end1 | t1_SR | forget ΔSR |
|---|---|---|---|---|---|---|
| unif si1 | 0.034 | 0.996 | 2.332 | 0.180 | 0.887 | -0.82 |
| A4   si1 | 0.033 | 0.988 | 1.911 | 0.406 | 0.883 | -0.58 |
| unif si3 | 0.034 | 0.996 | 0.973 | 0.590 | 0.797 | -0.41 |
| A4   si3 | 0.033 | 0.992 | 0.629 | 0.645 | 0.813 | -0.35 |
| unif si10| 0.034 | 0.992 | 0.209 | 0.902 | 0.660 | -0.09 |
| A4   si10| 0.033 | 0.992 | 0.122 | 0.910 | 0.773 | -0.08 |

**Findings:**
1. **Identical fresh distill for ALL configs**: end-of-task0 SR ~0.99, KL ~0.033,
   invariant to si and mode. SI is off during task0 (only si_idx>0), and A4 does not
   hurt initial learning. => same starting point; all differences are task1 damage.
2. **A4 forgets LESS at matched budget**: si1 uniform -0.82 vs A4 -0.58 (30% less
   forgetting) at equal task1-SR. This is the A-direction effect stated directly.
3. **KL degradation is caused by task1, throttled by SI** (0.034->2.33 at si1;
   SI=10 holds it to 0.12-0.21). SI does NOT degrade KL — it prevents degradation.
   A4 + SI both fight forgetting (A4 by landing preserved fidelity on grasp actions,
   SI by constraining weights).

### 2-SEED consolidation (2026-07-07): seeds 0,1 — robust

Runs: seed0 groups (registry) + seed1 `mseed_s1_{a4si1,unifsi1,a4si3,unifsi3}_*_1783397368`.
Mean over seeds {0,1} x 4 pairs. KL/SR@0 = fresh distill (end task0); @1 = post-forget.

| config | KL@0 | SR@0 | KL@1 | SR@1 | t1-SR | forget ΔSR |
|---|---|---|---|---|---|---|
| unif si1 | 0.027 | 0.998 | 2.639 | 0.197 | 0.885 | -0.80 |
| A4   si1 | 0.028 | 0.994 | 3.992* | 0.447 | 0.916 | -0.55 |
| unif si3 | 0.027 | 0.998 | 1.132 | 0.588 | 0.811 | -0.41 |
| A4   si3 | 0.028 | 0.996 | 0.863 | 0.670 | 0.807 | -0.33 |

**Robust (2-seed):**
- Fresh distill identical for all (SR@0 ~0.99, KL@0 ~0.028).
- A4 forgets LESS at matched si: si1 -0.55 vs -0.80; si3 -0.33 vs -0.41.
- Retention up at both budgets: si1 0.45 vs 0.20; si3 0.67 vs 0.59. Also higher
  task1 plasticity at si1 (0.916 vs 0.885).

**HONEST caveat (*):** A4-si1 mean KL@1=3.99 > uniform 2.64 looks like "A4 keeps
worse KL", BUT it's a single-pair OUTLIER: PushCuboid A4-seed1 KL=16.77. On the
other 3 pairs A4's KL <= uniform. PushCuboid is A4's consistent FAILURE MODE (high
KL, retention 0.09-0.14, sometimes below baseline). So: on the 3 pairs A4 works, it
gives higher retention at similar/lower KL; PushCuboid is a separate problem (both
are contact tasks that likely compete for the same grasp-critical weights).
Seed2 running to firm up. Per-pair inspection (not just means) was essential.

### *** DIRECTION A HEADLINE (2026-07-07): 3-seed, si=1, CONFIRMED ***

Runs: A4=`{a4tune_vagg(s0), mseed_s1_a4si1, mseed_s2_a4si1}`,
unif=`{graspfix_base(s0), mseed_s1_unifsi1, mseed_s2_unifsi1}`. Per-seed = mean/4 pairs.

| | retention (per-seed) | mean±std | task1-SR |
|---|---|---|---|
| **A4 si1** | 0.41, 0.49, 0.44 | **0.445 ± 0.034** | 0.883 |
| unif si1 | 0.18, 0.22, 0.22 | 0.204 ± 0.018 | 0.876 |

**A4 advantage +0.241, non-overlapping error bars, task1 plasticity EQUAL.** =>
At fixed consolidation budget, prioritizing success-critical states MORE THAN
DOUBLES LiftCube retention. Per-pair (3-seed): OpenDrawer +0.70, OpenDoor +0.27
(carry the win); PushButton ~0 (needs SI); PushCuboid -0.02 (contact-competition,
A4's blind spot). **A-direction hypothesis CONFIRMED.**
