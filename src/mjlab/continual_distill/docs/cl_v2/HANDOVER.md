# CL-V2 — HANDOVER (W3, lead, 2026-09-09)

All 25 Class-A tasks now use realistic textured assets and pass the seven gates
(`STATUS.md` is the scoreboard; every number below is copied from a row and has its
evidence in `runs/<Task>/verify.json`, `renders/<Task>/`, `logs/<agent>.md`). Site:
https://cl.untuai.com/v2/ — one card per task with still, turntable and the teacher
rollout (or the characterised failure).

## Teacher success, cl25 → v2 (n = 128 each, HEAD `41b36cb` + wave edits)

| Task | Asset | cl25 | v2 | Read |
|---|---|---|---|---|
| Lift-Cube | YCB 077 Rubik's cube, 46 mm mini | 1.000 | 1.000 | unchanged |
| Stack-Cube | cube on YCB 009 gelatin box | 0.328 | 0.523 | better: flat box top |
| Push-Cuboid | YCB 009 gelatin box (97 g) | 0.223 | 0.086 | real mass doubles sliding friction; teacher is transport-rate limited (cl25) |
| Drag-Pull | same carton | 0.477 | 0.391 | same cause |
| Topple-Block | YCB 003 cracker box, 100×160×210 | 0.938 | 0.898 | noise band |
| Place-In-Container | GSO Spritz plastic basket | 0.250 | 0.359 | better: wider bin |
| Throw-To-Bin | same basket, out of reach | 0.000 | 0.000 | characterised (reach-limited) |
| Reorient-Object | GSO CoQ10 packer bottle 30×53 mm | 0.547 | 0.258 | asset: end faces 40→30 mm, 25 g; both reverts measured worse |
| Cage-Drag | 46 mm cube | 0.141 | 0.016 | cl25 number was a sampling artefact (3.7 % free wins at reset, now guarded); control on cl25 sampling = 0.094 |
| Strike-Slide | built regulation hockey puck | 0.000 | 0.000 | characterised |
| Tool-Pull | built wooden reach hook + puck | 0.047 | 0.000 | characterised (squeeze ejection) |
| Peg-Insertion | built shape-sorter peg + board, 45° lead-in | 0.031 | 0.117 | better: chamfer |
| Axial-Extract | built Schuko plug in outlet box | 0.781 | 1.000 | better: real plug head geometry |
| Edge-Grasp | GSO salad plate (150 mm) on plank riser | 0.000 | 0.000 | characterised; failure moved to the side insert |
| Pivot-Lift | Poly Haven cutting board + brick wall | 0.000 | 0.000 | characterised; cl25 M4 dead-zone fixed, push now jams instead of sliding |
| Reach-Target | scene only | 1.000 | 1.000 | unchanged |
| Push-Button | built mushroom plunger button | 0.984 | 0.969 | noise band |
| Open-Drawer | built oak bedside cabinet | 0.888 | 0.938 | better: staged approach |
| Open-Door | built walnut wall cabinet, 0.3 m leaf | 0.000 | 0.297 | real door: lever arm 0.55→0.25 m, inertia ÷19 |
| Slide-Window | built aluminium slider | 1.000 | 0.969 | noise band |
| Open-Lid | built oak chest, knob | 1.000 | 0.102 | cl25 lid was winnable by doing nothing (see below) |
| Turn-Lever | built steel lever on plate | 0.941 | 0.922 | noise band |
| Rotate-Valve | built cast-iron cross handwheel | 0.473 | 0.461 | noise band |
| Flip-Switch | built industrial toggle | 0.656 | 0.609 | noise band |
| Push-Flap | built mail-chute flap | 1.000 | 1.000 | unchanged |

The site's `result.json` SR is a second, independent n=128 read from `render_rollout`'s
stats phase on the same tree; it brackets the STATUS number for every task.

## Tasks whose MECHANICS changed (not just looks) — read before comparing to cl25

1. **Open-Lid** — hinge axis reversed to `(0,-1,0)`. On the cl25 XML the −75° target was
   reached by free-fall in 0.25 s with zero actions (`verify_task` G3 fails it) and the
   lid swept through its own box. The task now matches what its cfg and command always
   documented (gravity opposes the lift). Teacher rewritten as pinch-and-lift; SR 0.102
   is real. Revert is one line in `lid.xml` if the free 1.000 is preferred.
2. **Open-Door** — real product substitution: 0.30×0.60 m / 2.7 kg cabinet door, hinge
   at y=0 (lever arm 0.551→0.253 m), damping 0.10→0.05, carcass front recessed 15 mm,
   `OpenDoorCommand.handle_to_hinge_dist` 0.55→0.25, drop 0.800→0.318. Threshold untouched.
3. **Slide-Window** — travel 0.25→0.24 m (real sash); target 0.22 untouched.
4. **Push-Button** — keeps the 50 mm stroke (the one non-real dimension) as a plunger button.
5. **Cube** (Lift, Stack, Place, Throw, Cage) — 46 mm (real mini 3×3), not the scanned
   57 mm, because Cage-Drag's 55 mm aperture latch would make it unwinnable.
6. **Topple-Block** — cracker box stretched non-uniformly in x (71.8→100 mm) so it stays
   ungraspable (the task's premise); printed faces undistorted.
7. **Cage-Drag / Push-Cuboid / Drag-Pull / Reach-Target** — success-at-reset was non-zero
   at cl25 (37/1000 for Cage-Drag). Fixed with dead bands: new opt-in
   `PushingCommandCfg.min_goal_distance` and `ReachingCommandCfg.min_gripper_clearance`
   (default 0 = inert elsewhere).
8. **Pivot-Lift** — board band moved out (0.38–0.41) so the approach leaves the
   `GRASP_RADIAL_MIN` dead zone (cl25 M4 fixed); wall follows (x 0.5605).
9. **Edge-Grasp** — 150 mm plate on a 200×260×100 riser; `ledge_top_height`,
   `plate_rest_offset`, `goal_offset`, `plate_rel_*` re-derived.
10. **Peg-Insertion** — real clearance and a 45°×6 mm lead-in; `stack_height` 0.01→0.035,
    `height_threshold` 0.03→0.015, `INSERT_DEPTH` 0.025→0.050.
11. **Place-In-Container** — `lateral_tolerance` 0.055→0.0585, `rim_height` 0.05→0.093,
    `floor_tolerance` 0.04→0.020; the old bin floated 12 mm above the floor (fixed).
12. **Reorient-Object** — `_pad` 0.05→0.031 (old value was mis-derived), spawn z 0.016.
13. `workspace.MECHANISM_DROP_BELOW_MOUNT`: drawer 0.315, door 0.318, lid 0.070 (others unchanged).

## Shared-code changes worth knowing

- `audit_workspace._geom_half_height` now uses mesh vertices, not `rbound` (mesh geoms
  were reported 5× too tall — would have mounted every mechanism higher). `_GOAL_EXEMPT`
  for Reach is one-sided; `_FIXTURES` no longer get the grasp ceiling.
- `verify_task`: `oracle_hold` (spring-return button), `oracle_root` (peg predicate is
  root-based), `graspable="thin_axis"` (plate/board are pinched on the thin axis).
- `record_task_videos._autoframe_camera`: frames the manipulated scene (robot excluded),
  syncs mocap, runs after the first reset; `render_rollout` camera azimuth 60, elevation −24.
- `tests/test_workspace_placement.py::REACH_EXEMPT` synced with the audit (Throw-To-Bin).
- `publish_v2.sh` tolerates missing optional clips; `site/index.html` falls back to
  `failure.mp4` for characterised failures.

## Traps for the next person

- The Franka hand capsule (r 0.04, half-length 0.06, 0.07 m up the approach axis) fouls on
  any collidable face within ±4 cm of the site xy: a flush carcass front cost the drawer
  teacher 0.906→0.375; recessing the door carcass 15 mm took the cl25 door teacher 0→0.41.
- `nvidia-smi` is aliased to an interactive `srun` in the shell profile — it queues a GPU
  allocation (jobs 20298–20300 were this; all cancelled). Use `srun --jobid=<holder> --overlap … command nvidia-smi`.
- Bare MJCF `range="0 90"` compiles to a hair under π/2 — write radians with an explicit
  `<compiler angle="radian"/>` (all mechanism XMLs now do).
- "Height above centre" teacher constants must be swept, not scaled (cage `RIDE_HEIGHT`,
  topple `CONTACT_Z` both measured 3× / 0.06 SR different from the scaled value).
- Collapsed variants (Lift-Cylinder/Sphere/Ellipsoid, Push-Disc; D5) still load; only
  Lift-Cylinder's spawn was touched (bottle asset shared with Reorient).

## Suggested next work (not done, by design)

Open-Lid hook-under-cap teacher (collider already shaped); Pivot-Lift push jam;
Edge-Grasp side-insert IK; `peg_insertion.carry_tol` 0.012→~0.020 for the chamfer;
`pivot_lift.CONTACT_Z_ABOVE_CENTER` site/pad-space fix (found, not applied so G7
compares the cl25 teachers); shadow-map moiré on `render_asset` floors (cosmetic).
