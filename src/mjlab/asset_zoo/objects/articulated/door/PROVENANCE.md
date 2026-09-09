# Open-Door — asset provenance (CL-V2, W2-b)

## What it is

A **300 x 320 x 636 mm walnut wall cabinet** with a 20 mm, 2.7 kg door leaf on a side
hinge, one interior shelf, visible hinge cups, and a 160 mm vertical square-bar steel
pull 50 mm in from the free edge. Built procedurally (PartNet-Mobility is unreachable
from this cluster — STATUS D8).

| | |
|---|---|
| Build script | `~/assets_raw/blender/W2-b/build_door.py` (+ `mechbuild.py`) |
| Wood texture | ambientCG **Wood067** — https://ambientcg.com/view?id=Wood067 — **CC0 1.0** |
| Metal texture | ambientCG **Metal032** — https://ambientcg.com/view?id=Metal032 — **CC0 1.0** |
| Raw downloads | `~/assets_raw/ambientcg/Wood067_1K`, `~/assets_raw/ambientcg/Metal032_1K` |
| Processed | `xmls/assets/{door_case,door_leaf,door_pull}.obj`, `walnut.png` (512²), `steel.png` (256²) |
| Budget | 1000 visual tris, 0.68 MB total |
| Build record | `build_manifest.json` in this directory |

## Real dimensions vs the spec

| Feature | Real product | This asset |
|---|---|---|
| Carcass | 300 mm kitchen wall unit, 300 W x 320 D x 640 H | 346 x 320 x 636 mm outer |
| Door leaf | 300 x 600 x 20 mm, 2.7 kg | identical |
| Hinge | concealed cup hinge on the leaf edge, 0-100 deg | `door_hinge` 0 .. 1.5708 rad |
| Pull | 160 mm centres, 20 mm square bar, 30 mm projection | identical |
| Lever arm | handle 50 mm from the free edge | 0.253 m hinge->handle |

## The geometry change, and why (READ THIS)

The cl25 asset was **not a cabinet door**. It was a 600 x 1200 x 20 mm slab
(`door_panel size="0.01 0.3 0.6"`, 14.4 kg at MuJoCo's default density) hinged 0.55 m
from its handle, inside a 1000 x 1600 mm barrier wall, mounted at z = 0.83 because it
dropped 0.80 m below its mount. Consequences, all measured:

* handle swept **1.1 m** of arc, passing within 0.18 m of the shoulder axis and
  finishing at radial **0.58 behind the robot base**;
* rotational inertia about the hinge ~**1.75 kg m²**;
* the classical teacher scored **0.000** — its own docstring concludes "this task needs
  a learned teacher, or a task-side change that is out of scope".

This asset is the task-side change, made as a **real product substitution**, not a
threshold change (the 0.1 rad / 84.3-of-90-deg success criterion is untouched):

| | cl25 | v2 |
|---|---|---|
| leaf | 600 x 1200 x 20 mm, 14.4 kg | 300 x 600 x 20 mm, 2.69 kg |
| hinge at (body frame) | `y = -0.30` | `y = 0` |
| lever arm handle->hinge | 0.551 m | 0.253 m |
| inertia about hinge | ~1.75 kg m² | ~0.093 kg m² |
| handle radial over the 0-90 deg sweep | 0.18 .. 0.58 | **0.31 .. 0.48** |
| swept drop below mount | 0.800 m | **0.318 m** |
| hinge damping | 0.10 | 0.05 |
| mount z | 0.830 | **0.348** |
| static barrier | 1000 x 1600 mm wall starting at x = 0 | a real 5-sided carcass, front plane recessed to x = +0.015 |
| bar projection from the leaf face | 10 mm | 30 mm (a 10 mm hook slot, exactly the drawer's) |

The **motion profile is unchanged**: a whole-arm arc pull about a vertical hinge from
0 to 90 deg, opening toward the robot, with the carcass walling off the far side so a
push is mechanically unavailable. The task still earns its name — more so, since it is
now recognisably a cabinet door.

## Frozen interface (kept)

`door_base` (mocap), `handle` body, `door_hinge`, geoms `door_body` / `door_panel` /
`handle`, sites `base_site` and `object_site` at **(-0.04, 0.25, 0)**, success target
1.5708 rad with threshold 0.1 rad, mount band x (0.48, 0.52) y (-0.30, -0.20).

## Downstream constants that moved with the hinge

* `mdp/commands.py::OpenDoorCommand._resample_command` — `handle_to_hinge_dist`
  0.55 -> **0.25** (goal-marker placement only; the success metric reads the joint).
* `classical/open_door.py` — `_R0` (-0.040, 0.551, 0) -> **(-0.040, 0.250, 0)**, and the
  whole strategy rewritten from a side pinch to a top-down hook (see that file).
* `workspace.MECHANISM_DROP_BELOW_MOUNT["door"]` 0.800 -> **0.318**.
* `<compiler angle="radian"/>` added: the cl25 XML had none, so its bare `range="0 90"`
  was silently degrees. The range is now written in radians explicitly.

## Teacher

`classical/open_door.py` keeps the cl25 SIDE-PINCH strategy; only `_R0` and `BAR_OUT`
moved. Measured on GPU 2:

| configuration | SR |
|---|---|
| cl25 asset + cl25 pinch teacher | 0.000 (n=128, phase 1) |
| CL-V2 asset + cl25 pinch teacher, `_R0` updated | **0.406 / 0.344** (n=32) |
| CL-V2 asset + a top-down HOOK (a port of open_drawer.py) | 0.219 (n=32) |
| CL-V2 asset + pinch, `BAR_OUT` sign "corrected" to -0.015 | 0.000 (n=32, twice) |

Two things that look like bugs and are not:
* `BAR_OUT` is **+0.016**, which aims the pinch into the 25 mm gap BEHIND the bar
  rather than symmetrically around it. Geometry says -0.015; measurement says +0.016
  by 0.4 to 0.0. Do not "fix" the sign.
* The hook loses to the pinch here even though it wins on the drawer, because the
  drawer's slot is horizontal (the tip descends across it and wedges) while this bar is
  vertical (the tip descends *along* the slot with nothing trapping it).

## Licence summary

Both textures CC0 1.0. Geometry original, generated by the build script.

## Why the carcass front is recessed 15 mm

The Franka `hand_capsule` (`panda.xml`) is a radius-0.04, half-length-0.06 capsule
centred 0.07 m up the approach axis from the `gripper` site. Hooking or pinching a
handle at x = -0.04 therefore puts wrist geometry at x up to +0.004 whatever the
strategy. cl25's barrier wall started at x = 0 and the carcass side panels would have
too. **Measured: recessing the carcass front from x = 0 to x = +0.015 took the
unchanged cl25 pinch teacher from 0.000 to 0.406.** A 15 mm gap between the leaf's back
face and the carcass front is what a concealed cup hinge's crank actually needs, so
this is a real detail, not a clearance hack.
