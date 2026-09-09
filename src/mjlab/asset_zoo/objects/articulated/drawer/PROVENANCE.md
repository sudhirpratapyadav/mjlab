# Open-Drawer — asset provenance (CL-V2, W2-b)

## What it is

A **450 x 400 x 465 mm light-oak bedside cabinet** with one working drawer over a fixed
cupboard door, 18 mm panels, a recessed plinth, and two 160 mm-centres square-bar steel
pulls. Built procedurally (not downloaded): PartNet-Mobility is unreachable from this
cluster (STATUS D8), so all nine mechanisms are built.

| | |
|---|---|
| Build script | `~/assets_raw/blender/W2-b/build_drawer.py` (+ `mechbuild.py`) |
| Geometry | procedural chamfered boxes / cylinders, real-world-scale box UV projection |
| Wood texture | ambientCG **Wood048** — https://ambientcg.com/view?id=Wood048 — **CC0 1.0** |
| Metal texture | ambientCG **Metal032** — https://ambientcg.com/view?id=Metal032 — **CC0 1.0** |
| Raw downloads | `~/assets_raw/ambientcg/Wood048_1K`, `~/assets_raw/ambientcg/Metal032_1K` |
| Processed | `xmls/assets/{drawer_carcass,drawer_case_pull,drawer_front,drawer_pull}.obj`, `oak.png` (512²), `steel.png` (256²) |
| Budget | 1092 visual tris, 0.65 MB total (limits: 50k tris, 5 MB) |
| Build record | `build_manifest.json` in this directory |

Both texture sets are 1K JPG albedo + AO; the pipeline convention is followed — AO is
multiplied into the albedo (MuJoCo's classic renderer has no AO pass) and the result is
written as PNG (MuJoCo rejects JPG).

## Real dimensions vs the spec

| Feature | Real product | This asset |
|---|---|---|
| Carcass | bedside cabinet, 450 W x 400 D x 465 H | 450 x 400 x 465 mm |
| Panels | 18 mm MDF/oak veneer | 18 mm |
| Drawer front | 400 x 200 x 20 mm | identical |
| Drawer travel | 3/4-extension runner, 200-250 mm | 250 mm (`drawer_slide` range -0.25..0) |
| Bar pull | 160 mm centres, 20 mm square section, 10 mm standoff | identical |
| Drawer mass | ~1.3 kg (front + box) | 1.29 kg (700 kg/m³ oak on the front, 2700 on the pull) |

## What was scaled / rotated / changed, and why

Nothing was scaled: the build script authors metres directly in the body frame.

**Kept bit-identical from cl25** (this is deliberate — the classical teacher's top-down
hook depends on it and scores 0.888):

* `drawer_panel` collider — box `pos="-0.01 0 0" size="0.01 0.2 0.1"`;
* `handle` collider — box `pos="-0.04 0 0" size="0.01 0.08 0.01"`;
* therefore the **10 mm hook slot** between the bar's rear face (x = -0.03) and the
  drawer front's outer face (x = -0.02) is unchanged;
* `base_site` / `object_site` at `(-0.04, 0, 0)`, `drawer_base` mocap, `handle` body,
  `drawer_slide` range `-0.25 .. 0`.

**Changed:**

1. The cl25 "carcass" was a single translucent 20 x 600 x 600 mm panel floating behind
   the drawer — not a cabinet. It is replaced by a real closed carcass: top slab, two
   side panels, back panel, mid rail, bottom shelf, recessed plinth and a fixed
   cupboard door, all as box colliders (`drawer_body` = the top slab, the piece the
   wrist comes closest to and the geom `ee_drawer_collision` names;
   `drawer_case_{left,right,rail,door}` for the rest).
2. **Full-overlay fronts** (the drawer front and the cupboard door stand 20 mm proud of
   a carcass whose front plane is x = 0). This is a real modern furniture detail and it
   is load-bearing for the task: the Franka `hand_capsule` is radius 0.04 centred 0.07 m
   up the approach axis from the `gripper` site, so while the teacher hooks the bar the
   wrist occupies x up to -0.015. The first build had a flush carcass front at x = -0.02
   with a 5 mm overhanging top, and the teacher fell from 0.906 (cl25 asset, same code
   and GPU) to **0.375**; disabling the carcass colliders restored 0.875, while forcing
   the old mount height changed nothing. Every carcass collider now starts at x >= 0
   (the cupboard door is the exception, and it sits 130 mm below the handle).
3. `MECHANISM_DROP_BELOW_MOUNT["drawer"]` 0.300 -> **0.315** (measured, swept over the
   full slide range). Mount z therefore 0.330 -> **0.345**; the handle rides 15 mm
   higher and the cabinet's plinth sits 30 mm above the floor, as a floor-standing
   cabinet should.
4. `<compiler angle="radian" meshdir="assets" texturedir="assets"/>` added (the asset
   has no angular ranges, but the contract asks for the explicit units).
5. Mass now comes from a real density (oak 700 kg/m³, steel 2700) instead of MuJoCo's
   default 1000 for every box: the drawer went 1.66 kg -> 1.29 kg.

## Teacher

`classical/open_drawer.py` keeps the cl25 top-down HOOK; the hook slot is bit-identical
so `TIP_DROP`, `PANEL_BIAS_X`, `SEATED_TOL`, `HOVER_Z` and `PULL_STEP` all re-derive to
their old values (each derivation is written out in that file). One real change: the
approach is now **staged** — a standoff 0.14 m in FRONT of the cabinet face, then
straight down. Measured on this asset (n=16, CPU): 0.625 without it, **0.875** with it,
and 1.000 with the carcass top slab made non-collidable. cl25's "cabinet" was a 20 mm
wall with open space above and behind, so the arm could swing over it on the way in; a
real 400 mm-deep carcass with a top cannot be swung over, and the contact log showed
the finger pads clipping `drawer_body` during the approach.

`TIP_DROP` is also worth a note: measured from `panda.xml`, the closed pads' lowest
point is 11.8 mm below the `gripper` site, not the 4 mm the cl25 comment claims.
Substituting the true value measured WORSE (0.688 vs 0.750, n=16), so 0.004 stays —
it is a tuned descent bias, not a tool offset.

## Licence summary

Both textures are CC0 1.0 (public domain). The geometry is original, generated by the
build script in this repository's sibling tree, and carries no third-party licence.
