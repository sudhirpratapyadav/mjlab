# Class A Wave-1 Expansion — 16 → 25 distinct motion profiles

Date: 2026-08-04. Branch: `benchmark-manip-diversity`. Companion to
`CATALOG_100_TASKS.md` (this implements its Wave 1, items T17–T25) and
`CLASS_A_EXPANSION.md` (the 2026-07-30 expansion whose counting rule and test
philosophy this wave follows).

## What was added

Nine registered tasks, each a distinct MOTION PROFILE under the counting rule
(object swaps do not count). Registry: 28 → 37 IDs; Class A: 16 → 25 profiles;
skill families: 6 → 7 (NON_PREHENSILE is new).

| Task ID | Profile | Command | New machinery |
|---|---|---|---|
| Mjlab-Drag-Pull-Franka | Engagement-inverted planar transport: object FAR, goal NEAR, arm retracts with far-face contact | `PushingCommand` (reused) | none — inverted spawn/goal geometry |
| Mjlab-Strike-Slide-Franka | Calibrated impulse; puck slides ballistically to a goal at radial 0.88–1.05, beyond the ~0.85 m stretch | `PushingCommand` (reused) | out-of-envelope goal BY DESIGN |
| Mjlab-Cage-Drag-Franka | Form-closure transport with OPEN fingers; a single pinch voids the episode | `CageDragCommand` | first min-over-time constraint latch (`min_aperture`); `gripper_closure_penalty` reward |
| Mjlab-Topple-Block-Franka | Poke an ungraspable block above its CoM, tip past the tipping point; body x-axis lands vertical (either sign) | `ReorientObjectCommand` + new `body_axis` / `symmetric_axis` cfg | generalized orientation predicate |
| Mjlab-Push-Flap-Franka | Handle-less hinge-arc face-push; contact normal rotates with the panel | `PushFlapCommand` (articulation base) | flap asset; NEGATIVE-target sign convention documented on the cfg |
| Mjlab-Axial-Extract-Franka | Pinch plug head, exceed ~4 N friction breakaway, guide straight up out of the bore | `AxialExtractCommand` (articulation base) | plug asset (frictionloss joint = tunable breakaway) |
| Mjlab-Edge-Grasp-Franka | Slide unpinchable plate over the ledge edge, pinch the exposed 16 mm at the overhang, lift clear | `EdgeGraspCommand` | ledge fixture (per-env mocap write); success/failure separated by construction (fallen plate lands BELOW the top) |
| Mjlab-Pivot-Lift-Franka | Push ungraspable flat board INTO the wall to pivot it up, then pinch and lift to an airborne goal | `PivotLiftCommand` (LiftingCommand subclass) | wall fixture (per-env mocap write) |
| Mjlab-Throw-To-Bin-Franka | Grasp, accelerate, timed release; ballistic arc into a bin at radial 0.78–0.90 | `PlaceInContainerCommand` (reused) | out-of-envelope bin BY DESIGN |

## Design decisions worth remembering

- **Ungraspable-by-construction sizing.** Block 0.10×0.14×0.18, plate 0.10×0.09×0.016,
  board 0.12×0.10×0.02. The non-prehensile strategy is forced by geometry, not by a
  reward trick a policy could route around. Two *different* mechanisms are at work,
  and an earlier version of this note wrongly claimed one of them for all four:
  the BLOCK is unspannable outright (every width > the 0.08 m aperture), while the
  PLATE and BOARD are thin enough to pinch in principle (16 mm, 20 mm) and are
  ungraspable only because they lie flat with no finger clearance underneath. That
  is exactly why edge-grasp and pivot-lift exist — both tasks are about manufacturing
  the clearance (an overhang, a pivot onto edge) that the flat pose denies.
- **Out-of-reach goals are deliberate.** Strike (goal x 0.88–1.05) and Throw (bin x
  0.78–0.90) place their goals beyond the arm's absolute stretch. `audit_workspace`
  reasoning does NOT apply to them — pulling those goals into the envelope
  degenerates both tasks into push / place. Tests pin this
  (`test_strike_slide_goal_is_beyond_the_reach_envelope`,
  `test_throw_bin_is_beyond_reach_and_requires_containment`).
- **The cage latch is a minimum, not a maximum.** Every prior latch is
  success-maximum (`torch.maximum`); cage-drag latches `min_aperture` and resets it
  on resample. Forgetting that reset would silently make every episode after the
  first pinch unwinnable — the test drives exactly this sequence.
- **Push-flap sign convention.** The face-push produces a NEGATIVE hinge rotation
  (r × F with the panel's +y lever arm); the joint range is (-1.4, 0) and the
  target -70°. A positive target would require pulling a handle that does not exist.
- **Static fixtures are written per-env.** Ledge and wall are mocap bodies; like
  the place-in-container bin, their MJCF pose only serves env 0, so the commands
  write them every resample (`test_pivot_lift_goal_is_airborne_and_wall_is_placed_per_env`
  pins the wall in env 1).
- **FK-staleness rule respected.** Spawn positions are computed from the poses just
  written, never read back through `site_pos_w` (the reorient drift-anchor bug from
  the last expansion).
- **Known inherited inconsistency:** mechanism tasks (flap, plug) run gravity-off
  like door/drawer/button/lever/valve/window; free-object tasks run gravity-on.
  Same deliberate inheritance as the 2026-07-30 expansion, same caveat: re-enabling
  gravity later invalidates teachers.

## Placement audit and the corrections it forced (2026-09-02)

The nine tasks were placed by reasoning about each asset in isolation. A full
`audit_workspace` pass over all 29 Class A tasks — extended in this pass, see below —
found that reasoning was locally right and globally wrong in five places.

| Task | Was | Now | Why |
|---|---|---|---|
| Pivot-Lift | wall x 0.57, board x 0.42–0.50 | wall x 0.50, board x 0.32–0.38 | board spawned **13.4 mm inside the wall** (measured); pivoted grasp face also sat at radial 0.545–0.57 with 0.0% approach freedom |
| Edge-Grasp | ledge x 0.44–0.50 | ledge x 0.38–0.43 | band was audited on the plate CENTRE (radial 0.547); the push CONTACT point sat at ~0.58, outside the envelope entirely |
| Topple-Block | x 0.39–0.43 | x 0.36–0.44 | symmetric 0.09 padding collapsed a 22 cm range to a **3.8 cm band**; the near-side pad was pure waste, since a topple only travels downrange |
| Drag-Pull | object x to 0.48 | object x to 0.44 | drag engages the object's FAR face, so the constraint applies to `object_x + half_extent` (radial 0.571), not to the centre |
| Lift ×4 | raw `GOAL_*` ranges | `workspace.goal_box()` | the goal box's far corner escaped the ceiling (hypot(0.55, 0.28) = 0.617) — the same corner bug `grasp_box()` was written for, never applied to goals |

**The audit's own blind spots mattered more than any single task.** Four were closed:

1. **Goals were never audited at all** — only spawned entities were. A goal outside
   the envelope is as much a design bug as an object outside it, and this is what
   caught the lift-goal corner escape, which had been live in four shipped tasks.
   New `workspace.GOAL_RADIAL_MAX` (carry) and `MECHANISM_GOAL_RADIAL_MAX` (arc end,
   any-orientation, nothing in hand).
2. **Object↔object interpenetration was never checked.** The 2026-08-04 audit covered
   robot↔object and floor burial; nothing compared two *placed* entities. That is the
   gap the board-in-wall bug lived in, and it is now a reset-time contact probe.
3. **The grasp-freedom metric hardcoded a floor-object height** (`site_z=0.10`). The
   edge-grasp plate sits on a 0.10 m ledge, so it was being scored against a slice of
   the envelope the task never enters. Now derived from each entity's actual z.
4. **Non-prehensile and fixture entities were scored by a top-down grasp metric**
   that does not describe them. Poked, straddled, and pushed-against bodies are
   classified explicitly (`_SIDE_APPROACH`, `_FIXTURES`) rather than left to report
   permanent false "sparse-reach" — and by-design out-of-reach goals are registered
   in `_GOAL_EXEMPT` with a bound, the same discipline Tool-Pull's puck already had.

Result: **0 placement problems across 29 Class A tasks**, entities and goals, with
every remaining out-of-envelope placement exempt *by name, bounded, and justified*.

Two judgement calls are recorded rather than silently applied:

- **Reach-Target's goal reaches radial 0.73**, the top-down p90 — the sparse tail this
  module otherwise warns against. Left as-is because Reach has trained teachers and
  trimming it would invalidate them. Revisit when those are retrained.
- **Open-Door's goal (radial 0.642) and Push-Flap's (0.616)** are arc ends, which is
  why the mechanism ceiling exists. If that ceiling is ever argued down, both fail.

## Validation state

- `python -m mjlab.scripts.audit_workspace` — **0 placement problems** across 29
  Class A tasks (entities AND goals), including the object-overlap probe.
- `tests/test_class_a_wave1.py` — 13 tests, all green on CPU: registration + tags,
  uniform 8-D interface, success-predicate fire for every command, and negative
  controls (near-miss, wrong orientation, resting-not-lifted, pinch-voids-cage,
  goal-beyond-reach assertions).
- Reset + 5-step rollout on all nine tasks: 60-D obs, finite obs/rewards.
- Existing suites (`test_class_a_expansion.py`, `test_task_configs.py`,
  `test_benchmark_taxonomy.py`): 30 tests, zero regressions — the
  `ReorientObjectCommand` generalization (`body_axis`, `symmetric_axis`) preserves
  the original task's behaviour by default. (43 tests across the four suites; the
  pivot-lift test now reads the wall band from the cfg instead of hardcoding 0.57,
  and asserts the board-to-wall gap that the original bug violated.)
- `manifest.json` regenerated: `num_tasks: 37`.

## Outstanding (before these tasks join CL experiments)

1. Cluster A100 `python -m mjlab.scripts.benchmark_smoke --isolate` for the nine
   new tasks (same follow-up the 2026-07-30 expansion needed).
2. `benchmark_validate` short-PPO learnability runs — per AUTHORING_GUIDE, smoke
   proves a task BUILDS; only training proves the reward shaping works. The
   likeliest reward-shaping risks, in order: strike-slide (impulse discovery),
   throw-to-bin (release timing), pivot-lift (two-phase contact), cage-drag
   (closure-penalty vs transport-reward balance).
3. Teachers: scripted candidates — drag-pull, cage-drag, topple, push-flap,
   axial-extract, edge-grasp (waypointable phase structures); RL-dense required —
   strike-slide, pivot-lift, throw-to-bin (impulse/contact-mode switching defeats
   waypoints). Mind the measured scripted-teacher failure modes (endgame precision,
   throughput) recorded in CLASSICAL_TEACHERS.md.
