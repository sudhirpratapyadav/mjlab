# wall — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Pivot-Lift-Franka` (with `free/board`). Static mocap fixture, written
per-env by `PivotLiftCommand`.

## What it is

A **brick wall block, 500 x 105 x 150 mm** — two courses of standard 215 x 102.5 x 65 mm
bricks on 10 mm mortar beds. Box with a 3 mm break on the top edge, textured with
**Poly Haven `brick_wall_001`** (diffuse x AO, 1024^2, tiled at 0.55 m so the courses
come out at brick scale).

* Texture: https://polyhaven.com/a/brick_wall_001 — **CC0**. Geometry: own work, CC0.
* Mass 15.75 kg (2000 kg/m^3 masonry). Mocap body, so the mass is cosmetic.

## Why it grew from 400 x 30 x 150 mm

* **30 mm is not a thickness masonry comes in.** 102.5 mm is the width of a standard
  brick, which is what a single-leaf wall is.
* The board is now 219 mm long across the push direction, so a 400 mm wall would be
  overrun by a board spawned at the edge of its y band (219/2 + 120 = 229 mm > 200 mm).
  500 mm covers it with 21 mm to spare.

## Placement — the cl25 M4 fix

`PivotLiftCommandCfg.wall_spawn_range.x` moved **0.50 -> 0.5605**, i.e. the INNER FACE
moved 0.485 -> **0.508**, and the board band moved 0.32-0.38 -> 0.38-0.41 with it.

cl25 note M4 recorded that the Pivot-Lift teacher never made contact with the board at
all: any single-pad pusher must stand at `board_x - (BOARD_HALF_X + PAD_RADIUS +
clearance)` = `board_x - 0.076`, and with the board at 0.32-0.38 that is **0.244-0.304**,
at or below `workspace.GRASP_RADIAL_MIN` (0.28) — the documented "folds the arm back over
its own base" dead zone. The instrumented rollouts plateaued 10-17 cm short of the seat
target in HOVER, SEAT and PIVOT alike, independent of the orientation constraint.

Moving the board band out to 0.38-0.41 puts the approach point at **0.304-0.334**
(radial 0.327-0.355 at the y-band edge), inside the density peak of comfortable top-down
poses, and the wall has to follow so the board still has something to pivot against.

The three constraints the new number satisfies:

1. **Spawn clearance.** Board yaw-swept +x half-extent = `0.06*cos(0.15) +
   0.109*sin(0.15)` = 0.0756, so a band ending at 0.41 reaches 0.486 against an inner
   face at 0.508 — 22 mm of clearance at the worst yaw, 38 mm at yaw 0.
2. **The pivoted board must be graspable.** Standing against the inner face its centre is
   at ~0.498; radial 0.512 at the y-band edge, inside `GRASP_RADIAL_MAX` (0.55).
3. **The approach point must be reachable** — above.

Verified end to end: `verify_task --num-resets 1000` G4 reports board radial 0.384-0.425,
`frac_in_envelope` 1.000, `frac_dead_zone` 0.000, zero spawn collisions.

## Collision

ONE box at the full envelope (`wall_geom`, original name, original
`friction="0.9 0.03 0.003"` / `condim=3` / `solref="0.01 1"`). Only the inner (-x) face is
ever touched. `wall_constants.WALL_CENTER_Z` stays 0.075 (half the 150 mm height), so the
wall still rests flush on the ground plane.
