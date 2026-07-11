# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-07-11 18:45 IST

## Current phase: **Phase 3/4 — Class-A skills done + Class-C embodiment up**

Phases 0-2 done. This session added: Lift-Cylinder, Reach (new skill), Stack (new),
Peg-Insertion (new skill; fragility axis now COMPLETE), and the floating LEAP hand
embodiment (Class C, verified as an entity). 10 Class-A tasks / 5 skills / all 4
fragility tiers. Class-C embodiment loads; first Class-C TASK blocked on one design
call (base actuation — see below).

### >>> DECISION NEEDED: floating-hand base actuation (blocks Class-C tasks) <<<
- (a) Unactuated base → in-hand reorient tasks only; needs new orientation-goal MDP
  (diverges from our position-based tasks).
- (b) Actuated 6-DoF base → floating-hand reach/pick/place reuse existing position MDP;
  keeps interfaces uniform across embodiments (my lean, matches "keep spaces similar").
See LOG 2026-07-11 18:40.

## Branch
`benchmark-manip-diversity`, off `continual_distill` (up to date w/ origin). Working
tree was clean of tracked mods before branching (only untracked new docs + scratch).

## Task inventory (distinct skills, by embodiment)

| # | Task | Embodiment | Skill family | Fragility | Status |
|---|---|---|---|---|---|
| 1 | Lift-Cube | A | pick-place | precision-grasp | native (exists) |
| 2 | Push-Cuboid | A | planar push | mild-contact | native (exists) |
| 3 | Push-Button | A | articulation | low | native (exists) |
| 4 | Push-Disc | A | planar push | mild-contact | native (exists) |
| 5 | Open-Door | A | articulation | mid | native (exists) |
| 6 | Open-Drawer | A | articulation | mid | native (exists) |
| 7 | Lift-Cylinder | A | pick-place | precision-grasp | NEW (phase 2, smoke-validated) |
| 8 | Reach-Target | A | **reach** | planar | NEW (phase 3) |
| 9 | Stack-Cube | A | pick-place | precision-grasp | NEW (phase 3) |

Distinct skills so far: reach, pick-place, planar-push, articulation (**4 families,
9 tasks**). Class B: 0. Class C: 0.
**All 9 pass `benchmark-smoke` (build/reset/step, finite obs+reward, sane shapes) —
which is the stage-one acceptance check. All 9 are valid suite tasks.**

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
