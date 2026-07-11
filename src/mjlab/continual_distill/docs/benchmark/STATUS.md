# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-07-11 14:35 IST

## Current phase: **Phase 2 — Class-A EE control + skill-diverse set** (starting)

Phase 0 (tracking + audit) and Phase 1 (taxonomy infra) COMPLETE + committed.

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

Distinct skills so far: pick-place, planar-push, articulation (~3-4 families, 6 tasks).
Class B: 0. Class C: 0. **This is the gap the phases fill.**

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

## Next action (Phase 2)
1. Decide EE-delta action design (IK vs mocap-weld vs impedance) — see open Q4.
2. Build the EE-delta action term in `envs/mdp/actions/`.
3. Port ~5 Meta-World tasks spanning fragility tiers onto Franka via shared bases.
4. Validate state-solvability; tag; regenerate manifest.

Note: Meta-World port needs the Meta-World repo/assets available locally — check
availability first; if absent, author equivalent task designs from the survey's
verified specs rather than blocking.
