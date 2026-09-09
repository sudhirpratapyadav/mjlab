# Rotate-Valve — asset provenance (CL-V2, W2-a, 2026-09-09)

**Asset:** BUILT in headless **Blender 4.2.23 LTS**, textured from ambientCG (D8).
Build script `~/assets_raw/blender/w2a/build_valve.py`.

**Real object:** a **cast-iron gate-valve cross handwheel** on a bonneted bulkhead — a
flanged valve bonnet with a packing gland, and a dished two-spoke "bar handle" wheel,
212 mm across the knobs.

| part | mesh | texture (source) | license |
|---|---|---|---|
| bulkhead panel 300 x 300 x 20 mm + 4 bolts | `valve_panel_vis.obj` | ambientCG `Metal009` (brushed steel) | CC0 1.0 |
| bonnet: 112 mm flange with 6 studs, tapered neck, hex packing gland | `valve_bonnet_vis.obj` | ambientCG `PaintedMetal006` (industrial green, chipped) | CC0 1.0 |
| handwheel: hub, dish, 2 knobbed spokes, hex stem nut | `valve_wheel_vis.obj` | ambientCG `PaintedMetal004` (signal red, chipped) | CC0 1.0 |

`https://ambientcg.com/view?id=<Metal009|PaintedMetal006|PaintedMetal004>` — CC0 1.0.

## Why a two-spoke cross handle and NOT a rimmed handwheel

The brief asked for a ~110 mm-radius spoked wheel. Two constraints bound it:

1. `object_site` is frozen at (-0.04, **0.09**, 0), so the tracked spoke tip is at
   radius 0.090 and the teacher's `_R_ARM`, `_MARKER_OFF` and the command's
   `goal_marker_offset` are all derived from that number. The wheel is therefore
   **180 mm to the spoke tips / 212 mm over the knobs**, a real gate-valve handwheel
   size, rather than 220 mm radius.
2. The teacher does not push the wheel, it **PINCHES a spoke** at 80% of its length —
   radius 0.072 — and closes the fingers tangentially there. A continuous rim at radius
   0.09 sits exactly in that closing path. A rim would also need 8+ collider segments
   against this task's `nconmax=60`.

A cast two-spoke bar handle is a real gate/globe-valve handle style at this size, so the
task still earns its name. Recorded as a deliberate deviation from "spoked wheel".

## Real dimensions vs. the spec

| dimension | real product | this asset | note |
|---|---|---|---|
| wheel across | 150-250 mm | 212 mm over the knobs | |
| spoke section | 20-25 mm cast | 24 x 24 mm radiused | |
| bonnet flange | 100-130 mm | 112 mm | visual only |
| panel | — | 300 x 300 x 20 mm | UNCHANGED: sets `MECHANISM_DROP_BELOW_MOUNT["valve"] = 0.150` |

## What was kept

Frozen and unchanged: `valve_base` (mocap) / `handle`, joint `valve_hinge`, collider
geoms `valve_body`, `valve_hub`, `handle` (spoke A) and `valve_spoke_b` (spoke B) — the
teacher reads all three by name — **all keeping their exact primitive shape, pose, mass
and contact parameters** — and sites `base_site` / `object_site` at (-0.04, 0.09, 0).
The bonnet is **visual only**: it lives inside radius 0.056, the fingers never come
closer than radius 0.072, and the panel collider already backstops the arm.

Changed: explicit `<compiler angle="radian"/>`; range written `0 6.2831853` rad instead
of `0 360` degrees. Same compiled range.
