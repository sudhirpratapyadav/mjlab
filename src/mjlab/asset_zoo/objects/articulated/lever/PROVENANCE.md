# Turn-Lever — asset provenance (CL-V2, W2-a, 2026-09-09)

**Asset:** BUILT in headless **Blender 4.2.23 LTS**, textured from ambientCG (D8:
PartNet-Mobility unreachable). Build script `~/assets_raw/blender/w2a/build_lever.py`.

**Real object:** a wall-mounted **steel lever handle** on a bolted back plate — the
industrial door-lever / disconnect-lever form, with a rose (boss), a short neck and a
140 mm bar.

| part | mesh | texture (source) | license |
|---|---|---|---|
| back plate 300 x 300 x 20 mm, raised centre panel, 4 proud bolts | `lever_plate_vis.obj` | ambientCG `Metal009` (brushed steel) | CC0 1.0 |
| rose + boss + neck + elbow + 140 mm bar + flared tip (one mesh, one body) | `lever_handle_vis.obj` | ambientCG `Metal032` (bright brushed steel) | CC0 1.0 |

`https://ambientcg.com/view?id=Metal009`, `https://ambientcg.com/view?id=Metal032` — CC0 1.0.

## Real dimensions vs. the spec

| dimension | real product | this asset | note |
|---|---|---|---|
| lever bar length | 110-140 mm | 140 mm (grasp point at 120 mm) | matches the brief's ~14 cm |
| bar cross-section | 20-25 mm | 24 x 24 mm, radiused | keeps the 60 mm pinch limit with room to spare |
| rose diameter | 50-70 mm | 66 mm at the plate, 48 mm boss | |
| back plate | 200-300 mm | 300 x 300 x 20 mm | UNCHANGED: `lever_body` is the contact-sensor geom and its half-extents set `MECHANISM_DROP_BELOW_MOUNT["lever"] = 0.150` |

## What was kept

Frozen and unchanged: `lever_base` (mocap) / `handle`, joint `lever_hinge` (axis +x at
the body origin), collider geoms `lever_body`, `lever_hub`, `handle` — **all three
colliders keep their exact primitive shape, pose, mass and contact parameters** — and
sites `base_site` / `object_site` at (-0.05, 0.12, 0).

Changed: `<compiler angle="radian"/>` is now explicit and the hinge range is written
`-1.5707963 0` rad instead of relying on MuJoCo's degree default (`-90 0`). Same
compiled range, no longer unit-ambiguous. Site marker spheres shrunk 0.010 -> 0.004 m so
they stop covering the new geometry in renders (sites carry no physics).
