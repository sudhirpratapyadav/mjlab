# Benchmark Build — STATUS

> Mutable snapshot of where we are. Updated as work progresses. See PLAN.md for the
> roadmap and LOG.md for the dated decision journal.

Last updated: 2026-07-11 14:12 IST

## Current phase: **Phase 0 — Setup & tracking** (in progress)

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

## Phase 0 checklist

- [x] Create branch `benchmark-manip-diversity`
- [x] Create tracking docs (PLAN / STATUS / LOG) under `docs/benchmark/`
- [x] Audit manipulation framework (manager-based, `tasks/manipulation/`, registry) — done
- [x] Audit continual_distill consumer (task-count/dim-agnostic, per-task heads) — done
- [x] Confirm action-term gap (joint-space only; no EE-delta) — confirmed
- [ ] Produce concrete repo-restructure proposal (below) — DRAFT, needs finalizing
- [ ] Commit Phase 0

## Repo-restructure proposal (DRAFT — to finalize before Phase 1)

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
None. Fully autonomous.

## Next action
Finalize restructure decision, commit Phase 0, begin Phase 1.
