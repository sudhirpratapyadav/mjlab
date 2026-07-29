# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-07-29

## Current phase: **Phase 4 — all 3 embodiment classes LIVE; 20 tasks**

**Suite: 20 tasks / 5 skills / 3 embodiment classes / all 4 fragility tiers.**
ALL 20 pass benchmark-smoke — **LOCAL (A6000) and CLUSTER (A100, 20/20)**; 17 unit tests green.
Cluster: `/ihub/homedirs/svs_ald/sudhir/mjlab` (branch benchmark-manip-diversity).
sm_80 warp segfaults worked around (LEAP mesh→box colliders; cylinder/disc/ellipsoid
→capsule); use `benchmark-smoke --isolate` on cluster.
**Root cause isolated 2026-07-29** — it is a **CUDA-graph-capture** bug in the
convex/CCD narrowphase, *not* a bad collision kernel (all geoms step fine eagerly), and
it is **already fixed upstream**: warp 1.14/1.15 + mujoco-warp 3.11 pass 14/14 including
the real LEAP hand with mesh colliders re-enabled. Upgrading (pin is git rev `46b4421`
= v0.0.1) would let both workarounds be reverted, restoring true grasp geometry.
See `sm80_repro/FINDINGS.md`; guarded by `tests/test_sm80_graph_capture.py`.
- arm_gripper (Class A, 12): reach; lift ×4 (cube/cylinder/sphere/ellipsoid); stack;
  peg-insertion; push ×2 (cuboid/disc); articulation ×3 (door/drawer/button). action=8.
- floating_hand (Class C, 5): reach, lift-cube, lift-sphere, stack, peg-insertion
  (LEAP, actuated 6-DoF base, action=22).
- arm_hand (Class B, 3): reach, lift-cube, stack (Franka+LEAP, action=23).

Uniform interfaces throughout: the SAME reach/lift/stack/insertion MDP bases serve all
three embodiments (8-D / 22-D / 23-D) — only the entity, EE site, and action scale differ.

### Embodiment integration recipes (proven, reusable)
- New object: cube-template asset (xml+constants+__init__); plug into a lift/stack helper.
- New floating hand: Menagerie model → add actuated 6-DoF base + grasp_site → robot cfg
  (see leap_hand). New arm+hand: MjSpec delete gripper + attach fixed hand at link7
  (see franka_leap).

### Operating mode
Autonomous, no decision-stops. North star: as many vision-free tasks as possible,
uniform interfaces, structural-soundness (benchmark-smoke) as the acceptance gate
(train-solvability is stage two).

### Next (in priority order)
1. Breadth: more object variants across hands (lift/stack cylinder/ellipsoid on
   LEAP + Franka-LEAP), more Class-A skills (sweep/tool-use), more articulation
   (lever/valve/knob — needs new articulated assets).
2. A 2nd dexterous hand (Shadow/Allegro, available in Menagerie) for embodiment variety.
3. Procedural scaling (Phase 5): PartNet-Mobility / object libraries → 100+ instances
   (tag as instance-diversity, separate from distinct-skill count).
STAGE TWO (not now): teachers, RL/CL, train-solvability, evaluation, EE-delta action.

## Branch
`benchmark-manip-diversity`, off `continual_distill` (up to date w/ origin). Working
tree was clean of tracked mods before branching (only untracked new docs + scratch).

## Task inventory (20 tasks / 5 skill families / 3 embodiment classes)

Registered IDs (source of truth: `manifest.json`, `num_tasks: 20`); fragility tier
1=low … 4=contact-rich.

| # | Task ID | Embodiment | Skill family | Tier |
|---|---|---|---|---|
| 1 | `Mjlab-Reach-Target-Franka` | A arm_gripper | reach | 1 |
| 2 | `Mjlab-Push-Button-Franka` | A arm_gripper | articulation | 1 |
| 3 | `Mjlab-Push-Cuboid-Franka` | A arm_gripper | planar-push | 2 |
| 4 | `Mjlab-Push-Disc-Franka` | A arm_gripper | planar-push | 2 |
| 5 | `Mjlab-Open-Door-Franka` | A arm_gripper | articulation | 2 |
| 6 | `Mjlab-Open-Drawer-Franka` | A arm_gripper | articulation | 2 |
| 7 | `Mjlab-Lift-Cube-Franka` | A arm_gripper | pick-place | 3 |
| 8 | `Mjlab-Lift-Cylinder-Franka` | A arm_gripper | pick-place | 3 |
| 9 | `Mjlab-Lift-Sphere-Franka` | A arm_gripper | pick-place | 3 |
| 10 | `Mjlab-Lift-Ellipsoid-Franka` | A arm_gripper | pick-place | 3 |
| 11 | `Mjlab-Stack-Cube-Franka` | A arm_gripper | pick-place | 3 |
| 12 | `Mjlab-Peg-Insertion-Franka` | A arm_gripper | **insertion** | 4 |
| 13 | `Mjlab-Reach-Target-Leap` | C floating_hand | reach | 1 |
| 14 | `Mjlab-Lift-Cube-Leap` | C floating_hand | pick-place | 3 |
| 15 | `Mjlab-Lift-Sphere-Leap` | C floating_hand | pick-place | 3 |
| 16 | `Mjlab-Stack-Cube-Leap` | C floating_hand | pick-place | 3 |
| 17 | `Mjlab-Peg-Insertion-Leap` | C floating_hand | insertion | 4 |
| 18 | `Mjlab-Reach-Target-Franka-Leap` | B arm_hand | reach | 1 |
| 19 | `Mjlab-Lift-Cube-Franka-Leap` | B arm_hand | pick-place | 3 |
| 20 | `Mjlab-Stack-Cube-Franka-Leap` | B arm_hand | pick-place | 3 |

Distinct skills: reach, pick-place, planar-push, articulation, insertion
(**5 families, 20 tasks**). Class A: 12 (action=8). Class B: 3 (action=23).
Class C: 5 (action=22).
**All 20 pass `benchmark-smoke` on local A6000 and cluster A100 (`--isolate`)** —
the stage-one acceptance check. 17 unit tests green, plus
`tests/test_sm80_graph_capture.py` (10 passed / 4 xfail).

> Caveat on shape diversity: tasks 7/4/11 (cylinder/disc/ellipsoid) currently simulate
> **capsule** geoms, and the LEAP hand grasps with **box** phalanges, due to the sm_80
> graph-capture workaround. A warp upgrade restores the true geometry — see the header.

### On "train-validation" (stage two, informational only)
Building the suite is stage one; solving it (RL/CL) is stage two. So train-solvability
is NOT a suite-membership gate. For reference only: Reach trains to 0.66 in ~74s; Stack
did not solve in 400 PPO iters (a multi-stage task — expected to need more, stage-two
concern). Neither fact affects the task's validity for the suite.

## Phase 0 checklist — DONE (commit e4c9561)

- [x] Branch, tracking docs, framework audit, consumer audit, action-gap confirmed,
      restructure decision (additive, not reorg), committed.

## Phase 1 checklist — DONE

- [x] `taxonomy.py`: Embodiment / SkillFamily / Fragility enums + TaskTaxonomy dataclass
- [x] `registry.py`: optional `taxonomy` field + `load_taxonomy` / `list_taxonomy`
      (backward-compatible; untagged tasks return None)
- [x] Tag the 6 native Franka tasks
- [x] `benchmark.py`: filter/group query API + CL orderings (fragility-graded,
      skill-diverse) + JSON manifest export
- [x] `tests/test_benchmark_taxonomy.py` (8 tests) + existing `test_task_configs.py`
      (9 tests) all pass — 17 green
- [x] `manifest.json` snapshot checked in under docs/benchmark/
- [x] Regression: env cfgs byte-unchanged; only additive registry field → distill
      pipeline provably unaffected (full 0.93-baseline re-run needs cluster datasets,
      deferred; risk zero by construction)

## Repo-restructure decision (FINAL — additive, no reorg)

See LOG 2026-07-11 14:20. The framework is already well-factored; we do NOT move
existing tasks. New embodiment classes get sibling `config/<robot>/` dirs. Superseded
proposal kept below for reference.

### Superseded draft proposal (kept for reference)

Findings from the audit:
- The benchmark ENVS belong in `src/mjlab/tasks/manipulation/` (mjlab's home for tasks),
  NOT inside `continual_distill/`. `continual_distill/` is the CL *consumer* and should
  stay a consumer.
- `continual_distill/` root is cluttered (many top-level `.py`, logs, `train_*.log`,
  wandb/, results/). Needs tidying but is orthogonal to the benchmark envs.
- `docs/` has grown many top-level files; the `benchmark/` sub-folder is the new home
  for this effort's tracking.

Proposed structure (minimal folders, clear division) — TENTATIVE, refine in Phase 1:
```
src/mjlab/tasks/manipulation/
  mdp/                     # shared terms (exists) — extend, don't fork
  embodiments/             # NEW: robot+hand entity cfgs (franka+gripper, franka+allegro, floating shadow)
  tasks_a_gripper/         # NEW: class-A env_cfgs (migrate existing 6 here or keep config/franka)
  tasks_b_dexarm/          # NEW: class-B env_cfgs
  tasks_c_floatinghand/    # NEW: class-C env_cfgs
  taxonomy.py              # NEW: skill/fragility/embodiment enums + registry metadata
  config/                  # registration (exists) — extend per class
```
Open question: whether to physically move the existing 6 tasks or leave them and only
apply the new structure to NEW tasks (less churn, but two conventions). LEANING toward
adding taxonomy metadata to the existing registry in place first (low risk), and
introducing the class sub-folders for NEW tasks — migrate the old 6 opportunistically.
Decide at start of Phase 1.

## Known-solved bugs to preserve (audit target for Phase 1)

- Success latching: live per-env `episode_success` via `np.maximum` (not stale
  `info["log"]`). Location: continual_distill eval path — verify it's still in place.
- Reset-RNG isolation: per-env-instance start pose (LiftCube start-pose artifact).
  Status: FINDINGS says a full fix was "out of scope"; periodic eval is ground truth.
  Phase 1 must decide whether to properly fix or document-and-latch.

## Blockers / waiting-on
None. Fully autonomous. (Full CL regression re-runs need cluster teacher datasets at
`/ihub/homedirs/svs_ald/...` — deferred; not blocking env authoring.)

## === WHEN YOU'RE BACK: state of the benchmark + resume plan ===

**What exists (5 milestone commits on branch `benchmark-manip-diversity`):**
- Full taxonomy + query/ordering infra (embodiment/skill/fragility), tagged registry,
  JSON manifest, CL-ordering generators. 17 tests green.
- Two local validation tools: `benchmark-smoke` (builds/steps every task) and
  `benchmark-validate --task <id> --iters N` (short PPO run, reports success).
- AUTHORING_GUIDE.md = the exact recipe (proven). Adding a task is now fast + safe.
- 9 tasks / 4 skills (was 6/3): +Lift-Cylinder, +Reach (new skill, train-validated),
  +Stack (structural only).

**All decisions resolved (2026-07-11):** build the SUITE (not solve it); joint-space
action everywhere (no EE-delta now); dexterous hand = full joint targets; keep obs/
action/reward/success SIMILAR across tasks; keep going autonomously. See LOG 17:30.

**Resume authoring (priority order), each via AUTHORING_GUIDE, uniform interfaces:**
1. **Insertion / peg-in-hole** — biggest contact-rich skill gap. Author the peg+hole
   asset (asset_zoo `peg_in_hole/` stub is empty) + an insertion command, REUSING the
   staged reach→bring reward + latched success pattern (keep it similar to lift/stack).
   Reference: sibling `mujoco_playground` fork has `peg_in_hole.py`.
2. **Class A scale-out** — more distinct skills (sweep, tool-use, more articulation
   variants) + a few sensible grasp-object variants. Same 8-D joint action, same obs/
   reward/success shape. Keep skill vs instance diversity honest in the manifest.
3. **Embodiments B/C** — add Allegro/Shadow hand entities (MuJoCo Menagerie) under
   `asset_zoo/robots/`; author `config/<hand>/` mirroring `config/franka/`. FULL JOINT
   TARGETS. Reuse the SAME reward/success/obs patterns (just more joints). Class C
   (floating hand: Adroit/ShadowHand-style) is near-native = quick win.
4. **Procedural scaling** — PartNet-Mobility / MolmoSpaces generators for 100+ instances
   (tag as instance-diversity, separate from distinct-skill count).

Stage two (NOT now): EE-delta action, teachers, RL/CL, train-solvability, evaluation.
