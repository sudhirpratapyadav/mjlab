# Continual Distillation — Experimental Findings

Last updated: 2026-07-05 (evening — teacher comparison, probes, and size sweep complete).

## Setup recap

- **Pipeline**: per-task RL teachers (rsl-rl PPO, PyTorch, MLP 512-256-128) → offline
  datasets (1024 episodes × 150 steps = 153,600 samples per task; obs 60-d, targets
  = teacher Gaussian [mean | logstd], 16-d) → one shared JAX student (4096-2048-1024)
  with per-task heads, trained sequentially with Gaussian KL(teacher‖student) loss +
  Synaptic Intelligence (SI) regularization.
- **Sequence 5**: PushCuboid → PushButton → OpenDoor → OpenDrawer.
- **Metric**: `Accuracy/task_i` = student env success rate, evaluated after the full
  sequence; "avg" = mean over the 4 tasks.
- Runs execute on the iHub-Drishti cluster (2× DGX A100-80GB) inside a slurm holder
  job; launchers in `mjlab/slurm/`. Wandb project `continual_rl_mjlab` (IITJ entity).

## Finding 1 — Distillation is bistable at lr 1e-4 (the "seed instability")

Symptom: with default lr 1e-4, seed 0 gave avg 0.80 but seeds 1–3 collapsed
(avg 0.16 / 0.08 / 0.01).

Root cause chain:

1. **Task-0 distillation is bistable.** Bit-identical configs and seeds land in either
   a *good basin* (task-0 test KL ≈ 1) or a *bad basin* (test KL ≈ 1000). The coin
   flip comes from residual nondeterminism (XLA autotune picking different kernels),
   not from the seed itself — we reproduced both outcomes with the same seed, solo on
   a GPU (so not contention either).
2. **A bad task-0 draw poisons everything downstream.** Huge KL → huge gradients →
   SI omegas explode during task 0 → after consolidation the SI penalty (si_coeff=1)
   effectively freezes the whole network → tasks 1–3 train to ~0 accuracy.
3. **Why the loss is so sharp**: the Gaussian KL weights mean errors by 1/σ², and
   `student_min_std=1e-3` puts a floor of ~10⁶ on that weight — tiny mean errors
   become cliffs. Probe: raising `student_min_std` to 0.05 at lr 1e-4 gave a healthy
   task-0 test KL of 0.59, confirming the sharpness hypothesis.

What did NOT fix it: si_coeff 0.1 (no), si_coeff 0.01 (partial, doesn't generalize —
seed 3 still 0.13), si_epsilon tuning (partial).

**Fix: `--learning-rate 3e-5`.** Validated on seeds 0–3: avg 0.92 / 0.93 / 0.93 / 0.93
(±0.01). All pre-fix single-run comparisons at lr 1e-4 are confounded by the
bistability — compare teachers/settings only across multiple seeds at lr 3e-5.

Remaining consistent signal after the fix: **task 0 (PushCuboid) retains only
~0.72–0.81** after the full sequence (real forgetting), while tasks 1–3 end at
~0.94–1.0. This is the current main open problem.

## Finding 2 — Teacher flavors: classical and BC teachers work as drop-ins

Three teacher flavors for PushButton, swapped into sequence 5 (other 3 tasks keep
their RL teachers):

- **RL** (baseline): PPO teacher, dataset `Mjlab_Push_Button_Franka_model_1999_...`.
- **Classical**: hand-coded IK policy (`continual_distill/classical/push_button.py`),
  99.5% env success. Dataset `..._classical_20260704_013344`. The distill pipeline
  supports `teacher_type: classical` in the tasks yaml (policy runs on CPU during
  eval, non-jit).
- **BC**: DART-style — classical expert rolled out with action noise (labels = clean
  expert action), behavior-cloned into an RL-compatible TeacherActorMLP (512-256-128,
  val MSE 0.022, 100% env success), then rolled out deterministically to make the
  distill dataset `..._bc_20260704_110316`. Pipeline in `continual_distill/bc/`.

Single-seed results at lr 1e-4 (confounded, kept for the record): RL 0.80,
classical 0.86, BC 0.93. Multi-seed comparison at lr 3e-5 is in flight (see below).

## Finding 3 — Classical policy engineering (what it took / what's parked)

Architecture: `classical/base.py` — internal CPU MuJoCo Franka model for FK/Jacobian,
damped-least-squares IK, converged-IK waypoint controller (solve IK to convergence
internally, then step joints toward the goal clipped by max_dq). Drop-in replacement
for the NN teacher: consumes only the 60-d obs vector, emits joint-space actions.

Hard-won details (each cost real debugging time):

- **Use relative obs only.** World-frame obs (object_pos, gripper_pos) include a
  per-env scene-origin offset; gripper_to_object = obs[40:43] and object_to_goal =
  obs[43:46] are offset-free.
- **Per-task DEFAULT_QPOS.** Button/door/drawer envs use NEUTRAL qpos, cuboid/lift
  use HOME (joint 2 differs by 1.3 rad). Using the wrong one puts FK/Jacobian at a
  fantasy configuration → policy acts on garbage. Action = (q_des − default)/0.04.
- **Axis-only orientation targets.** Full 3-DoF orientation goals are unreachable at
  workspace edges; aligning just the approach axis (weight 0.3) is robust.
- **Per-axis position-error clipping** (max_pos_err=0.08), not norm-capping — norm
  capping starves the z-component during descent.
- **`ee_ground_collision` is a subtree sensor on link7** — it includes hand AND
  fingers, and it terminates the episode. Constrains low pushing/grasping poses.
- **Drawer**: top-down grasp is infeasible (rear finger hits panel, 2 cm gap) →
  horizontal approach closing along z, plus a fingertip offset (TIP_VEC, fingertips
  are ~6 cm beyond the EE site) and an integral term (INTEG_GAIN=0.3) to cancel the
  2–3 cm DLS steady-state bias.

Status per task (2026-07-06, after the base-controller fixes of Finding 6):

| Task | Policy | Success | Status |
|---|---|---|---|
| PushButton | `push_button.py` (2-phase hover→press, 25° tilted axis) | ~100% | done |
| OpenDrawer | `open_drawer.py` (horizontal approach, slip-retry) | 32% | frozen (uniform scaling helped: was ~22%) |
| PushCuboid | `push_cuboid.py` (shepherding, ride height 0.035) | 23% | frozen (needs legacy `step_clip_mode="per_joint"` — uniform scaling hurt it) |
| OpenDoor | `open_door.py` (cage-drag + coast) | 4% | parked, see Finding 6 |

Extra knobs the door work added to `base.py` (all opt-in): `cmd_lead_max`
(command-integration for sustained pull force — helped nothing so far),
`step_clip_mode` ("uniform" default / "per_joint" legacy).

## Finding 4 — Cluster deployment gotchas

(Details in `slurm/` scripts; summarized here because each was debugged the hard way.)

- **venv**: `uv sync` is broken on the cluster (torch pin vs CUDA 12.4 driver).
  Manual venv: torch 2.6.0+cu124 (install LAST), jax[cuda12]==0.7.2, all
  nvidia-*-cu12 libs pinned to mujoco_playground's versions (cudnn 9.10.2.21 —
  torch's cudnn 9.1 downgrade breaks JAX GPU init), tensordict 0.12.0,
  rsl-rl-lib 3.2.0, mujoco 3.3.7, mujoco-warp @46b4421. `PYTHONPATH=src`, no
  editable install.
- **GPU pinning**: slurmstepd OVERWRITES `CUDA_VISIBLE_DEVICES` passed via
  `srun --export`. Pass `RUN_GPU=<n>` instead; `_cl_run.sh` re-exports it inside the
  step. Before this fix, "parallel" seed runs all landed on GPU 0.
- **Config paths**: cluster `config/tasks*.yaml` need `/ihub/homedirs/svs_ald/...`
  dataset paths; rsyncing local `config/` clobbers them (re-sed after any sync).
- Pattern: 20-day holder job on all 8 GPUs + `srun --jobid=<holder> --overlap` steps.

## Results ledger (sequence 5, avg accuracy over 4 tasks)

| Run | lr | Teacher (PushButton) | Seed | Avg | Note |
|---|---|---|---|---|---|
| original | 1e-4 | RL | 0 | 0.80 | good-basin draw |
| seeds 1–3 | 1e-4 | RL | 1/2/3 | 0.16/0.08/0.01 | bad-basin collapses |
| mixed | 1e-4 | classical | 0 | 0.86 | single seed, confounded |
| bc | 1e-4 | BC | 0 | 0.93 | single seed, confounded |
| lrfix | 3e-5 | RL | 0/2/3 | 0.92/0.93/0.93 | fix validated |
| diag2 lr3e-5 | 3e-5 | RL | 1 | 0.93 | fix validated |

Per-task shape of the lr 3e-5 runs: task 0 ≈ 0.72–0.81, tasks 1–3 ≈ 0.94–1.0.
## Finding 5 — GPU pinning was broken all along (fixed 2026-07-05)

`continual_distill.py` had `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` at import
time, force-overwriting any external pin — so every "parallel" batch (including the
lrfix seed runs) silently stacked ALL processes on physical GPU 0. Log banners showed
correct pins (the shell set them; python discarded them) — **verify placement with
nvidia-smi, never with banners**. Fixed with `os.environ.setdefault(...)`.
Consequence: results were unaffected, but wall-clock was ~10x inflated. A full
sequence-5 run on a dedicated A100 takes **26-43 minutes**, not ~9 hours.

## Results (2026-07-05, ts 1783268576, all with true per-GPU placement)

Task order: PushCuboid / PushButton / OpenDoor / OpenDrawer.

**Teacher comparison at lr 3e-5** (PushButton teacher swapped, 3 seeds):

| Teacher | Seed avgs | Mean |
|---|---|---|
| RL (lrfix, 4 seeds) | 0.92 / 0.93 / 0.93 / 0.93 | **0.93** |
| Classical | 0.96 / 0.92 / 0.85 | 0.91 |
| BC | 0.89 / 0.90 / 0.87 | 0.89 |

All three flavors are viable drop-ins; differences are small relative to the
task-0-retention variance (0.59-0.84 across all runs), which is the dominant noise
source. Classical seed 0 was the best single run (0.96).

**Probes:**

- `student_min_std=0.05` @ lr 1e-4: avg 0.85 (0.83/1.00/0.88/0.70). Rescues the
  lr 1e-4 bistability end-to-end (confirms the 1/sigma^2 sharpness mechanism) but
  underperforms lr 3e-5 -> keep lr 3e-5 as the fix.
- `si_coeff=3` @ lr 3e-5: avg 0.90, task-0 = 0.73. Stronger SI does NOT improve
  task-0 retention and slightly hurts OpenDrawer. Forgetting of task 0 is not an
  under-consolidation problem.

**Student size sweep** (RL teachers, lr 3e-5, seeds 0/1/2; `--student-hidden-dims`
CLI flag added for this):

| Hidden dims | Seed avgs | Mean |
|---|---|---|
| 512-256-128 | 0.63 / 0.61 / 0.61 | 0.62 |
| 1024-512-256 | 0.78 / 0.74 / 0.71 | 0.74 |
| 2048-1024-512 | 0.91 / 0.87 / 0.84 | 0.88 |
| 4096-2048-1024 | 0.92 / 0.93 / 0.93 / 0.93 | **0.93** |

Clean monotonic capacity curve. 4096 remains best (2048 trails by ~5 points across
seeds — seed 0 alone had flattered it); 2048 is the efficiency trade-off at 1/4 the
parameters. Capacity loss consistently hits the sequence ends first: task 0
(forgetting pressure; collapses to 0.05-0.28 at size 512) and task 3 (least
remaining capacity, ~0.44 at sizes <=1024).

## Finding 6 — OpenDoor classical policy (2026-07-06): 0% → 4%, parked

Success requires the hinge at 84-90deg within 150 steps. What we learned:

- **Pull along the arc tangent, not the chord to the goal** — the chord presses
  into the hinge constraint. Door angle is recoverable from |o2g| (chord length);
  `object_quat` reports the static door base, NOT the swinging panel.
- **Pinch grips slip** on the smooth 2cm bar (fingertip friction is domain-
  randomized down to 0.3): reliable for only ~5-15deg. A **cage drag** (fingers
  semi-closed to a 4cm gap around the bar) is friction-free geometric containment
  and drags continuously for 45deg+.
- **You cannot push this door open** — it opens toward the robot; any push from
  the robot's side closes it. All push-phase strategies are dead ends.
- **Two base-controller bugs found here** (fixes benefit all tasks):
  (1) per-joint clipping of the outer servo step distorts the motion direction
  (EE sags off the path until grasps disengage) — use uniform scaling;
  (2) TIP engagement depth: the bar must sit BETWEEN the fingers (tips ~2cm
  past it), not at the very tips, or pulls slide straight off. The bar-to-panel
  clearance (3cm) caps cage depth — a hard containment ceiling.
- **The drag rate is joint-velocity-limited** at ~0.5-0.75 deg/step (force knobs,
  command-lead, orientation weights don't change it). Approach+engage costs
  40-80 steps, so the in-episode ceiling is ~75-80deg — right below the success
  threshold. Endgame that works when engage is clean: drag to ~26deg, release,
  and let the heavy panel (I~1.7 kg m^2, damping 0.1) COAST to the target.
- Net: 4.3% success; failures are dominated by engage churn (repeated cage
  misses under +-1.4cm obs noise). Parked — RL remains the door teacher.

## Finding 7 — Teacher-mix combinations (2026-07-06, IN FLIGHT)

Question: does distillation degrade when several teachers in the sequence are
low-quality (classical cuboid 26%, drawer 36%), and does mixing flavors matter?
OpenDoor stays RL in every mix (classical door is only 4%). Fresh classical
datasets: `Mjlab_Push_Cuboid_Franka_classical_20260706_015152` (26%),
`Mjlab_Open_Drawer_Franka_classical_20260706_015820` (36%). All at lr 3e-5,
3 seeds each (`sequence5_mix*_1783313557`):

- **mix2cl** (`tasks_mix2cl.yaml`): PushButton + OpenDrawer classical; cuboid +
  door RL.
- **mix3cl** (`tasks_mix3cl.yaml`): PushButton + PushCuboid + OpenDrawer classical;
  door RL.
- **mixcross** (`tasks_mixcross.yaml`): BC PushButton + classical OpenDrawer + RL
  cuboid/door — all three teacher flavors in one sequence.

Baselines to compare against (all lr 3e-5, avg over 4 tasks): pure RL 0.93,
single-classical-swap 0.91, single-BC-swap 0.89.

**Results** (student env success, 3 seeds; task order Cub/Btn/Door/Drw):

| Config | PushCuboid | PushButton | OpenDoor | OpenDrawer |
|---|---|---|---|---|
| mix2cl (drawer+btn classical) | 0.66 / 0.59 / 0.77 | 0.92 / 0.94 / 1.0 | 1.0 / 0.98 / 1.0 | 0.20 / 0.05 / 0.25 |
| mix3cl (+cuboid classical) | **0.02 / 0.06 / 0.08** | 1.0 / 1.0 / 1.0 | 1.0 / 0.98 / 1.0 | 0.05 / 0.02 / 0.03 |
| mixcross (BC btn) | 0.66 / 0.63 / 0.52 | 1.0 / 1.0 / 0.98 | 1.0 / 0.80 / 1.0 | 0.13 / 0.05 / 0.20 |

Teacher ceilings: cuboid RL ~0.75, cuboid classical ~0.26, drawer classical ~0.36,
button (any flavor) ~1.0, door RL ~1.0.

**Key findings:**

1. **Distillation fidelity is teacher-quality-agnostic on the strong tasks.**
   PushButton and OpenDoor finish at ~1.0 in every mix regardless of what
   flavor/quality the OTHER tasks' teachers are. A weak teacher does NOT poison
   its neighbors — the shared student + per-task heads isolate them well. Even
   PushButton's classical/BC teacher has huge action-KL (KL 30-32, its Gaussian
   is very different from the student's) yet still distills to 100% success.

2. **task-0 (PushCuboid) is uniquely fragile — but only with a WEAK task-0
   teacher.** With the RL cuboid teacher (mix2cl, mixcross) task-0 retains
   0.5-0.77 (≈ teacher ceiling 0.75, i.e. near-full retention). With the
   CLASSICAL cuboid teacher (mix3cl) task-0 collapses to 0.02-0.08 — far below
   even its own 0.26 ceiling. So a low-quality FIRST teacher doesn't just cap
   task 0 at its ceiling, it gets crushed to ~0 by downstream forgetting. The
   weak, high-entropy classical cuboid distribution gives the student nothing
   robust to hold onto across 3 later consolidations.

3. **BC vs classical vs RL for PushButton, in-sequence:** all reach ~1.0; the
   BC teacher (mixcross) additionally has LOW KL (2.0 vs 30+), i.e. the cloned
   Gaussian is much closer to the student's — cleaner to distill even though the
   end success is the same.

**Takeaway (drives the plan):** only the FIRST task's teacher quality is
load-bearing for retention; weak later-task teachers just cap their own task at
ceiling without harming others. => keep a STRONG task-0 teacher (RL), and stop
using the low-accuracy classical cuboid/drawer teachers as drop-ins.

## Open problems / next

1. Task-0 (PushCuboid) retention ~0.6-0.84 after the sequence — not fixed by SI
   strength; teacher-agnostic. Main open problem.
2. Teacher-mix results (Finding 7) — pending.
3. Classical PushCuboid (23%) / OpenDrawer (32%) / OpenDoor (4%) tuning parked;
   RL still the strongest teacher for those three.
