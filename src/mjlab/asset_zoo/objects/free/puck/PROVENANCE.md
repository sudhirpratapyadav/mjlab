# free/puck — provenance

| field | value |
|---|---|
| Asset | **Regulation ice-hockey puck**, built for mjlab (trimesh lathe of the real profile) |
| Geometry source | built — `~/cl_v2_work/W1-b/build/build_puck.py` (revolved profile + moulded face ring) |
| Albedo source | ambientCG **Rubber004** colour map, https://ambientcg.com/view?id=Rubber004 |
| License | Geometry: built here (no third-party license). Texture: **CC0** (ambientCG). |
| Used by | `Mjlab-Strike-Slide-Franka`, `Mjlab-Tool-Pull-Franka` |

## Real dimensions vs the spec

| | real regulation puck | previous primitive | this asset |
|---|---|---|---|
| diameter | 76.2 mm (3 in) | 70 mm | **76.2 mm** |
| thickness | 25.4 mm (1 in) | 24 mm | **25.4 mm** |
| mass | 156–170 g | 30 g | **160 g** |
| material | vulcanised rubber, µ≈0.3–0.5 | — | friction 0.4 (unchanged) |

The 2.5 mm moulded edge bevel and the debossed face ring are modelled; the barrel gets
6 tiles of the rubber grain, the faces a planar projection of the same map.

## Why built rather than downloaded

Neither Poly Haven (521 models scanned for `dimensions`) nor GSO (1033 names) nor YCB
contains a hockey puck, and the real article is a lathe of a 5-point profile — building
it is both exact and 576 triangles, where a scan would be 10 k.

## Collider

One **cylinder primitive** `size="0.0381 0.0127"`. Exact for this shape, analytic against
the ground plane (plane–cylinder is a PRIMITIVE kernel in mujoco_warp), and it keeps the
old geom name `puck_geom` with its `condim`/`friction`/`solref`/`contype` verbatim.
Mass and inertia are the solid-cylinder values at 160 g.

## Known issue carried forward (recorded, not fixed)

cl25 flagged that the `ee_ground_collision` guard leaves under 1 cm of margin over a 24 mm
puck. At 25.4 mm the margin is 1.4 mm larger, not smaller, so the flag stands unchanged:
`FLOOR_MIN_Z = 0.030` on the gripper site puts the pads at ~0.017, i.e. inside the puck's
0–0.0254 span with 8 mm of floor clearance. See `classical/strike_slide.py::RIDE_HEIGHT`.

## Budgets

576 tris, 512² PNG albedo, 1 collider, 496 KB on disk.
