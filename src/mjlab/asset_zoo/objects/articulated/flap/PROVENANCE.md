# Push-Flap — asset provenance (CL-V2, W2-a, 2026-09-09)

**Asset:** BUILT in headless **Blender 4.2.23 LTS**, textured from ambientCG (D8).
Build script `~/assets_raw/blender/w2a/build_flap.py`.

**Real object:** a **painted-steel mail-chute / access flap door** hung on a steel jamb
post with three-knuckle barrel hinges. The hinge axis is VERTICAL and the panel is
220 mm wide x 300 mm tall, so this is a small swinging chute door, not a horizontal
letterbox flap — that is the geometry the task's motion profile (a vertical-axis
face-push arc) requires, and it is what was modelled.

| part | mesh | texture (source) | license |
|---|---|---|---|
| jamb post 30 x 30 x 320 mm, hinge pin, 3 static knuckles | `flap_post_vis.obj` | ambientCG `Metal038` (dark scratched steel) | CC0 1.0 |
| flap panel 220 x 300 x 24 mm, raised frame, pull lip, 2 moving knuckles | `flap_panel_vis.obj` | ambientCG `Metal009` re-tinted to postbox red (0.78, 0.11, 0.10) at mean 0.52 — luminance kept, so the brushed micro-detail survives as paint grain | CC0 1.0 |

`https://ambientcg.com/view?id=Metal038`, `https://ambientcg.com/view?id=Metal009` — CC0 1.0.

**Texture note (W2-a, recorded so it is not retried):** the panel was first textured
with `PaintedMetal002` (blue chipped paint). Its chip marks are large and high-contrast,
so at 0.16 m/tile they read as random blotches and at 0.075 m/tile the repeat became
plainly visible — a tiling artefact either way on a flat 220 x 300 mm panel. A tinted
brushed metal has no feature big enough to alias, so it tiles invisibly, and postbox red
also makes the mail-chute reading immediate.

## The flat-face rule

The panel's robot-facing (-x) face is **dead flat at x = -0.012**, exactly on the
`flap_panel` collider face. Every piece of relief — the raised frame, the pull lip, the
hinge straps — is on the +x side, so the pad never meets visual geometry that is proud
of the collision surface. Measured from the exported OBJ: `lo_x = -0.012` exactly.

## UNITS — the bug this asset is famous for

The primitive `flap.xml` shipped `range="-1.4 0"` written as if radians into a field
MuJoCo compiles as DEGREES (no `<compiler angle=...>`), giving -1.4 deg against a
-70 deg target: unsolvable by any policy. It was patched to `-80.2` degrees at HEAD
`1127d12`. This version sets **`<compiler angle="radian"/>` explicitly** and writes
`range="-1.3997541 0"` — the same hard stop, now unit-unambiguous. Verified off the
COMPILED model: `jnt_range = [-1.3997541, 0]`.

## Real dimensions vs. the spec

| dimension | real product | this asset | note |
|---|---|---|---|
| flap panel | 200-300 mm | 220 x 300 x 24 mm | UNCHANGED collider |
| jamb post | 30-50 mm | 30 x 30 x 320 mm | UNCHANGED: sets `MECHANISM_DROP_BELOW_MOUNT["flap"] = 0.160` |
| barrel hinge knuckle | 20-25 mm dia | 23 mm dia | visual only |

## What was kept

Frozen and unchanged: `flap_base` (mocap) / `handle`, joint `flap_hinge` (axis +z at the
body origin), collider geoms `flap_body` and `flap_panel` with their exact primitive
shape, pose, mass and contact parameters, and sites `base_site` (-0.012, 0.12, 0) /
`object_site` (-0.012, 0.18, 0).
