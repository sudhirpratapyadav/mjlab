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

(pending STEP 0 signal assessment before launching any A/B run)
