# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-07-11 16:10 IST

## Current phase: **Phase 3 — new skills (train-validated)** (in progress)

Phases 0-2 COMPLETE + committed. Phase 3 underway: authored the first NEW SKILL
(Reach) with full MDP (command+reward+obs+base) and TRAIN-VALIDATED it locally
(solvable in 74s). Key unlock this session: local training works, so new-skill
solvability CAN be validated autonomously (~1-2 min for easy tasks). This de-risks
authoring the rest of Phase 3.

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
| 8 | Reach-Target | A | **reach** | planar | NEW (phase 3, TRAIN-validated) |

Distinct skills so far: reach, pick-place, planar-push, articulation (**4 families,
8 tasks**). Class B: 0. Class C: 0.
All 8 pass `benchmark-smoke`. Reach-Target additionally TRAIN-validated: 150 PPO iters
(74s) → episode_success 0.66, goal_error 0.062m, still climbing → SOLVABLE.

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

## Next action (Phase 3 — new skills; needs training to validate)

The remaining task-authoring buys NEW SKILLS / FRAGILITY, which requires new MDP
(commands/rewards/success) that can only be VALIDATED BY TRAINING (smoke-test proves
build, not solvability). So these are best done with the user available to launch
training + judge solvability, OR carefully one-at-a-time. Priority order:

1. **Reach** (new skill: reach) — cheapest new skill; simple distance reward, low risk
   even without training. Good first Phase-3 task. Uses `reach_object_reward` +
   `gripper_to_object_vector` (already in mdp/). Needs a reach-target command.
2. **Stack** (pick_place, higher fragility) — stacking command referencing two objects
   + on-top success. Moderate new MDP.
3. **Insertion / peg-in-hole** (NEW skill: insertion; the biggest contact-rich gap) —
   must author the peg+hole asset (asset_zoo stubs are empty) + insertion command.
   Reference: sibling `mujoco_playground` fork has `peg_in_hole.py` + `pick_cartesian.py`.
4. **EE-delta action term** (open Q4) — enables Meta-World-style uniform control; build
   in `envs/mdp/actions/`. Design decision needed (IK vs mocap-weld vs impedance).
5. **Embodiments B/C** (Phase 4) — add Allegro/Shadow hand entities from Menagerie;
   Adroit/ShadowHand tasks are near-native (Class C quick win).

See AUTHORING_GUIDE.md for the exact recipe (Recipe A cheap, Recipe B new-skill).
