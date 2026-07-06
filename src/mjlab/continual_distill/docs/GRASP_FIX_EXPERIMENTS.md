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

### A4 |Δaction|-weighting — REJECTED at signal-assessment stage (2026-07-07)

Implemented per-sample weighting (`--distill-weight-mode delta_action`), plumbed
end-to-end (uniform reduces exactly to plain KL — regression safe). BUT assessing
the signal ON THE DATASET (not a fresh rollout) killed it:

- In the stored dataset, |Δaction| is ~UNIFORM across the episode: t<80 mean 3.10,
  t>=80 mean 3.14 (ratio 1.0x). No grasp-time concentration.
- Contradicts the earlier live-rollout probe (which showed |Δaction|->0 in the
  hold). Reason: the dataset teacher keeps making small corrective actions through
  the "hold" (holding a cube vs gravity/noise is NOT zero-action); the clean
  hold=0 pattern was an artifact of the deterministic 32-env probe.
- Also the grasp is not time-aligned across episodes (first-success t=35-71), so
  even a real per-episode signal averages out.

**Lesson: assess signals IN THE DATASET the loss actually sees, not a fresh rollout.**

### A3 / B1 value signal — also REJECTED at signal stage

Teacher V(s) on the dataset is FLAT: per-timestep mean 5.05-5.08 across the whole
episode, early-vs-hold gap 0.00, ΔV oscillates around 0. The converged teacher's
value is ~constant because it's always in a near-solved state. No temporal signal.

### *** REAL ROOT CAUSE (2026-07-07): the DATASET has almost no grasp ***

Assessing obs-space physical signals exposed it. In the LiftCube dataset:
- **80.4% of episodes START with the cube ALREADY LIFTED** (mean cube-z at t=0 =
  0.25m; gripper already on the cube).
- **80.7% of all samples** are "already-lifted-and-holding" (z>0.2, g2o<0.04).
- Only **10%** of samples show the gripper in approach (>5cm from object); only
  **15%** have the cube still on the ground.

=> The dataset is ~80% "hold a lifted cube" and barely contains the
approach-and-grasp. This is a **data-collection artifact**: episodes were collected
from the `test=True` env, which resets most envs into an already-grasped/lifted
state. The teacher then just holds.

**This explains everything cleanly and retires ALL the loss-weighting ideas:**
- low env-KL: student perfectly clones the abundant HOLD behavior.
- ~0 success: at eval the cube starts ON THE GROUND (real task); the student got
  almost no training signal for approach/grasp, so it can't grasp -> 0.
- no re-weighting can fix a skill that is barely IN the data (nothing to up-weight).

**NEW HYPOTHESIS (H2): fix the DATA, not the loss.** Re-collect LiftCube so the
dataset actually contains the approach-and-grasp. Options:
- **H2a**: collect from the `play`/train env (not `test`) if it resets with the
  cube on the ground -> full approach->grasp->lift trajectories.
- **H2b**: force cube-on-ground resets during collection (override the reset event /
  command so every episode starts pre-grasp).
- **H2c**: if the env inherently starts lifted, shorten episodes / subsample so the
  hold doesn't dominate 80%.
First: inspect the LiftCube env reset config to see WHY it starts lifted, then pick.
