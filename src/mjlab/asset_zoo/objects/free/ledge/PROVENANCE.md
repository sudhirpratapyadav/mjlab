# ledge — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Edge-Grasp-Franka` (with `free/plate`).

## What it is

A **plank-built solid-oak riser / storage box, 200 x 260 x 100 mm**, whose near (-x) edge
is the fixture that manufactures the overhang. Modelled with trimesh (three 4 mm groove
lines round the shell so it reads as planked, 1.5 mm break on the top edge) and textured
with **Poly Haven `oak_wood_planks`** (diffuse x AO, 1024^2).

* Texture: https://polyhaven.com/a/oak_wood_planks — **CC0**.
* Geometry: own work (`scratchpad/build/build_edge.py`), CC0.
* Mass 2.18 kg (oak 700 kg/m^3 at 60% fill). It is a **mocap** body, so the mass is
  cosmetic.

## Why built rather than downloaded

Every scanned wooden box in the reachable catalogues is the wrong shape. The closest,
Poly Haven `CheeseBox_01`, measures **240 x 107 x 66 mm** (`asset_pipeline inspect` on the
glTF; the site's own info block quotes 240 x 136 x 93). Its 107 mm depth cannot hold a
150 mm plate at all, and scaling it up 1.9x to fit would put the top surface at 125 mm —
above the middle of the Franka's measured 60-140 mm top-down grasp band, where the side
pinch has to happen. `wooden_crate_01/02` are 825 mm and 1166 mm long; `Shelf_01` is
furniture. A rectangular oak block is itself a real object (a butcher's block / shelf
riser), so building it costs no realism and buys an exactly flat, exactly-positioned top.

## Geometry decisions

* Widened from the primitive's 180 mm to **200 mm** (half-extent 0.09 -> **0.10**) so the
  150 mm plate spawns fully supported with 25 mm of margin per side.
* **Top surface still at exactly +0.100 above the body origin**, so
  `EdgeGraspCommandCfg.ledge_top_height` (0.10) and `ledge_constants.LEDGE_TOP_HEIGHT`
  are unchanged, and the plate's spawn height and the whole success predicate are
  untouched.
* Depth kept at 260 mm.

## Collision

ONE box at the full 200 x 260 x 100 mm envelope (`ledge_geom`, the original name and the
original `friction="0.6 0.03 0.003"` / `condim=3` / `solref="0.01 1"`). A mesh or CoACD
collider would put the decorative plank grooves into the contact and make the top face
neither exactly flat nor exactly at z = 0.10, which the plate's rest pose and
`lift_clearance` both depend on.

## Constants re-derived

| constant | old | new | derivation |
|---|---|---|---|
| `edge_grasp.LEDGE_HALF_X` | 0.09 | **0.10** | riser half-width |
| `EdgeGraspCommandCfg.ledge_spawn_range.x` | (0.38, 0.43) | **(0.39, 0.43)** | the overhang pinch happens at `ledge_x - LEDGE_HALF_X`; 0.39 keeps it at 0.29-0.33, clear of `GRASP_RADIAL_MIN` = 0.28 |
| `ledge_top_height`, `goal_offset`, `lift_clearance`, `max_drift` | — | unchanged | the top surface height did not move |
