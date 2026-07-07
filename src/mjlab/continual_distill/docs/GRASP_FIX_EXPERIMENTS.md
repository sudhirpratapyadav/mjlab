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
