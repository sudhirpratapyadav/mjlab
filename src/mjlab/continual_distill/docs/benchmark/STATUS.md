# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-09-08

## Current phase: **Phase 5 (Wave 1) — all 3 embodiment classes LIVE; 37 tasks**

**Suite: 37 tasks / 7 skills / 3 embodiment classes / all 4 fragility tiers.**
43 unit tests green (9 task-config + 8 taxonomy + 13 Class-A expansion + 13 Wave-1).
The placement pass added no tests; it corrected the pivot-lift one and re-counted —
the previous "41" undercounted the expansion suite, which is 13, not 11.
`audit_workspace` reports 0 placement problems across all 29 Class A tasks — entities
and goals.

**2026-09-08 — Taxonomy reconciled with the catalog; task videos re-recorded.**
`Mjlab-Axial-Extract-Franka` was tagged ARTICULATION because its command term
subclasses the articulation base; the family tag names the SKILL, and the skill is the
inverse of peg-insertion, so it is now INSERTION (pinned by a new test). That leaves
INSERTION a family of 2 instead of 1, and moves the task into the right bucket of every
skill-diverse CL ordering. Three catalog fragility tiers were stale relative to the
built tasks and now match the registry (strike-slide 2->3, cage-drag 1->2,
throw-to-bin 2->3 — each is one-shot, so a mis-timed impulse/release/pinch is
unrecoverable). Family counts for Class A (29 IDs): articulation 9, pick_place 10,
planar_push 4, non_prehensile 2, insertion 2, reach 1, tool_use 1. All 37 task videos
were re-recorded from post-audit HEAD and published at https://cl.untuai.com.
44 tests green across the four task suites.

**2026-08-04 — Renderer mocap desync fixed (videos lied; the sim never did).**
Task videos showed mechanisms/fixtures "overlapping the robot base" in every
mocap-mounted task (door/drawer/button/lever/valve/switch/window/lid + the new
flap/plug/ledge/wall/bin). Root cause: `OffscreenRenderer.update()` synced only
qpos/qvel — never `mocap_pos/quat` — so mocap bodies rendered at their compiled
default (the env origin) while physics had them correctly placed. Verified by a
full placement audit (4 resets x 37 tasks, plain-MuJoCo contact mirror): zero
robot<->object penetrations anywhere; entity roots all in-envelope. One REAL
placement bug found and fixed by the same audit: **Peg-Insertion-Leap spawned the
peg 30 mm inside the floor** (stack-base default z=0.02 vs peg half-height 0.05);
now rests at z=0.05. Videos re-recorded after both fixes.

**2026-08-04 — Class A Wave-1 expansion: 16 -> 25 distinct profiles** (28 -> 37
registered IDs). Added Drag-Pull, Strike-Slide, Cage-Drag, Topple-Block, Push-Flap,
Axial-Extract, Edge-Grasp, Pivot-Lift, Throw-To-Bin — the nine S-cost Class-A items
of CATALOG_100_TASKS.md Wave 1 (T17–T25). NON_PREHENSILE is a new skill family
(7th). New machinery this wave: an episode-long NEGATIVE constraint latch
(cage-drag min-aperture), deliberately-out-of-reach goals (strike, throw — exempt
from the workspace audit BY DESIGN), extrinsic-dexterity fixtures written per-env
(ledge, wall), and ungraspable-by-construction objects (block/plate/board wider
than the 0.08 m aperture). Gates passed: 13 CPU success-predicate tests with
negative controls (tests/test_class_a_wave1.py), reset+step finite-obs sanity on
all nine, zero regressions in the existing suites. Outstanding: cluster A100
`benchmark-smoke --isolate`, `benchmark_validate` learnability runs, teachers.
See CLASS_A_WAVE1.md.

**2026-07-30 — Class A motion-profile expansion: 8 -> 16 distinct profiles**
(12 -> 20 registered IDs). Added Turn-Lever, Rotate-Valve, Flip-Switch, Slide-Window,
Open-Lid, Place-In-Container, Reorient-Object, Tool-Pull. TOOL_USE is a new skill
family (6th). Counting rule: a task counts iff its MOTION PROFILE differs — object
swaps (lift-cube vs lift-sphere) do NOT count. See CLASS_A_EXPANSION.md.
Verified 23/23 benchmark-smoke on CPU; a cluster A100 re-run is the remaining check.
Cluster: `/ihub/homedirs/svs_ald/sudhir/mjlab` (branch benchmark-manip-diversity).
Use `benchmark-smoke --isolate` on cluster.
**sm_80 segfaults RESOLVED (commits 4bc5ab6, fb15756).** Root-caused to a
**CUDA-graph-capture** bug in the convex/CCD narrowphase — *not* a bad collision kernel
(all geoms step fine eagerly) — and fixed upstream. mjlab now requires
`mujoco-warp>=3.11` / `warp-lang>=1.14`, and **both workarounds are reverted**:
cylinder/disc/ellipsoid are real CYLINDER/ELLIPSOID geoms again and LEAP mesh collision
is live. 20/20 smoke + 325/325 pytest on A100.
See `sm80_repro/FINDINGS.md`; guarded by `tests/test_sm80_graph_capture.py`.
- arm_gripper (Class A, 20): reach; lift ×4 (cube/cylinder/sphere/ellipsoid); stack;
  peg-insertion; push ×2 (cuboid/disc); articulation ×7 (door/drawer/button/lever/
  valve/switch/window); lid; place-in-container; reorient; tool-pull. action=8.
  **16 distinct motion profiles** (the honest count; the 20 IDs include object variants).
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
0. **Cluster re-validation of the 8 new Class-A tasks** (A100, `--isolate`). They pass
   23/23 on CPU locally; the cluster run is outstanding.
1. Breadth: more object variants across hands (lift/stack cylinder/ellipsoid on
   LEAP + Franka-LEAP). **DONE for Class A: lever/valve/switch/window articulation +
   tool-use.** Remaining Class-A ideas: sweep (multi-object, deferred by user),
   wipe/trace-surface (needs a contact-force success predicate).
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
**All 20 pass `benchmark-smoke` on cluster A100 (`--isolate`)** — the stage-one
acceptance check — with real cylinder/ellipsoid geoms and live LEAP mesh colliders.
**325/325 pytest green**, including `tests/test_sm80_graph_capture.py` (14/14).

> Shape diversity is now genuine: since fb15756 the cylinder/disc/ellipsoid tasks
> simulate real CYLINDER/ELLIPSOID geoms (not capsule stand-ins) and the LEAP hand
> grasps with its true fingertip meshes (not box phalanges).

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
