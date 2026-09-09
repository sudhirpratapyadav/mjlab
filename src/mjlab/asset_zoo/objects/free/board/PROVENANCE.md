# board — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Pivot-Lift-Franka` (with `free/wall`).

## What it is

**Poly Haven `wooden_cutting_board`** — a scanned wooden carving board, 449.9 x 246.7 x
41.3 mm as scanned — uniformly scaled **0.486** to **219 x 120 x 20.1 mm**, a standard
small chopping / serving board, and **yawed 90 deg so its LONG axis lies along +y**,
across the push direction.

* Source: https://polyhaven.com/a/wooden_cutting_board — **CC0**.
* Scale check: 450 mm is a large carving board; 220 x 120 x 20 mm is the commonest small
  chopping-board size, so the scale-down lands on a real product.
* Mass **170 g** (paulownia, ~320 kg/m^3 — the wood lightweight Japanese boards are cut
  from). The primitive's 60 g implied a 250 kg/m^3 balsa board; a full hardwood board at
  this size would be 370 g and needs ~3x the pivot torque, which is a difficulty change,
  not an asset change, so the lightest real board wood was chosen.

## Why the 90 deg yaw

The pivot happens about a horizontal axis along **y** (the board's leading +x edge jams
against the wall). Putting the long axis along y therefore:

* keeps the x half-extent at **0.060 — exactly the primitive's** — so `BOARD_HALF_X`,
  `BOARD_HALF_Z`, `STANDOFF` and the standing height (120 mm) are all unchanged;
* gives the wall contact a 219 mm line instead of a 100 mm one, so the board rotates
  instead of skewing;
* keeps both in-plane widths (120 mm, 219 mm) far past the 80 mm aperture, which is the
  task's premise.

## Collision

ONE **box** at the AABB, `board_geom`, `size="0.06 0.1094 0.01"`, with the primitive's
`friction="0.7 0.03 0.003"`, `condim=3`, `solref="0.01 1"`. The pivot is an edge-on-edge
contact against the wall and the ground; a rounded scanned edge (or a CoACD hull of one)
makes the tipping point mushy and the pivot non-repeatable.

## Constants re-derived

| constant | old | new | derivation |
|---|---|---|---|
| `pivot_lift.BOARD_HALF_X` | 0.06 | 0.06 (unchanged) | scaled so it would be |
| `pivot_lift.BOARD_HALF_Z` | 0.010 | **0.01004** | measured AABB half-thickness |
| `pivot_lift.BOARD_CENTER_Z` | 0.011 | **0.0101** | resting centre = half-thickness |
| `PivotLiftCommandCfg.object_pose_range.z` | 0.011 | **0.0101** | same |
| `PivotLiftCommandCfg.object_pose_range.x` | (0.32, 0.38) | **(0.38, 0.41)** | the cl25 M4 fix — see `free/wall/PROVENANCE.md` |
| `PivotLiftCommandCfg.object_pose_range.yaw` | (±0.3) | **(±0.15)** | ±0.3 rad on a 219 mm board projects 32 mm onto +x, more than the push travel |
| `board_constants.get_mocap_goal_spec` marker | box 0.06/0.05/0.01 | box 0.06/0.1094/0.01 | matches the new footprint |
