# Task correction and verification — 2026-09-11

User authorization: inspect and fix all 25 tasks individually, including code,
simulation, MuJoCo state, and rendered evidence. The pre-fix findings and raw probes
are preserved in SEMANTICS_AUDIT.md and semantics_audit.json.

## Task contract decisions

- Peg-Insertion: the goal observation and replica refer to the inserted **tip**;
  center-based insertion checks are converted explicitly. Fully seated means center
  height within 3 mm, near-upright square section fitting the bore, released and slow.
- Cage-Drag: success requires at least 5 mm of progress while the cube is geometrically
  between open pads and still caged at the goal. Closing voids the whole episode,
  including an earlier success. Rewards stop paying after a violation. The aperture
  floor is now `hypot(46 mm, 45.2 mm) + 5 mm`, approximately 69.5 mm: the old 55 mm
  bound allowed a real pinch of a diagonally oriented cube.
- Lift-Cube/Pivot-Lift and Edge-Grasp: grasp success requires both finger pads in
  actual contact with the object. Edge-Grasp also rejects fast transient flicks.
- Stack-Cube: successful placement must be slow, released, and contact the base.
- Place/Throw-To-Bin: full collider extent must fit inside the basket, above its
  floor and below its rim, with actual supporting contact, release (no robot contact)
  and low linear/angular speed. The rendered
  target is a resting cube rather than a partially buried cube at the bin site.
- Tool-Pull: actual two-pad tool contact followed by tool/puck contact is required;
  direct robot/puck contact invalidates success. Rewards approach the tool first.
- Reorient/Topple: reward the same axis alignment as success, including symmetric
  faces for Topple, and reject fast transient orientation crossings.
- All ten articulated tasks use joint-progress rewards. Cartesian distance to the
  final handle position is periodic and gave misleading progress for a 270-degree
  Rotate-Valve target. The exact FK-based goal geometry remains intact.
- Test configurations preserve training task budgets and collision/out-of-bounds
  failures. Test mode only removes observation corruption and external robot pushes.

These are benchmark corrections, not teacher tuning. Historical success numbers are
not current validation for changed tasks. Existing Peg-Insertion observation datasets
use the old mixed center/tip vector and must be regenerated before reuse. The scripted
peg teacher adapts the corrected public vector back to its internal center reference,
preserving physical carry and release waypoints. No website publication is part of
this pass.

## Verification

The broad regression run passed 290 checks covering task semantics, geometry,
workspace placement, goal visualization, timing, terminal-success capture and teacher
interfaces. The original verifier also passes the focused Peg-Insertion and Cage-Drag
checks with contact-backed goal witnesses. Per-task state, contact and visual evidence
is recorded in [TASK_VERIFICATION_RESULTS.md](TASK_VERIFICATION_RESULTS.md).

All simulation commands run in holder 20277 on dgx1, one Slurm task per invocation.
CPU simulation uses FORCE_CPU=1 and CUDA_VISIBLE_DEVICES empty. Existing user changes
are preserved in place.
