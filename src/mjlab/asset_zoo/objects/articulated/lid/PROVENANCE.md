# Open-Lid — asset provenance (CL-V2, W2-b)

## What it is

A **220 x 220 x 120 mm plank-built oak chest**: three floor boards, four two-plank
walls on corner battens, a three-board lid on two brass strap hinges with a turned
lifting knob on the near edge and a brass hasp on the front. Built procedurally
(PartNet-Mobility is unreachable from this cluster — STATUS D8); Poly Haven
`CheeseBox_01` was the size/shape reference.

| | |
|---|---|
| Build script | `~/assets_raw/blender/W2-b/build_lid.py` (+ `mechbuild.py`) |
| Wood texture | Poly Haven **oak_wood_planks** — https://polyhaven.com/a/oak_wood_planks — **CC0 1.0** |
| Brass texture | ambientCG **Metal032** tinted (1.00, 0.80, 0.42) — https://ambientcg.com/view?id=Metal032 — **CC0 1.0** |
| Raw downloads | `~/assets_raw/polyhaven_tex/oak_wood_planks`, `~/assets_raw/ambientcg/Metal032_1K` |
| Processed | `xmls/assets/{lid_box,lid_hinge_leaf,lid_boards,lid_straps}.obj`, `oak.png` (512²), `brass.png` (256²) |
| Budget | 1796 visual tris, 0.78 MB total |
| Build record | `build_manifest.json` in this directory |

## Real dimensions vs the spec

| Feature | Real product | This asset |
|---|---|---|
| Chest | 220 x 220 x 120 mm lidded oak box | identical |
| Walls | 20 mm planks | 20 mm |
| Lid | 200 x 220 x 16 mm, three boards + two battens | identical |
| Lifting knob | turned knob, 32 mm across, 36 mm tall | identical |
| Hinge | strap hinge on the far top edge, axis lateral | `lid_hinge` at (0.10, 0, 0.04), axis (0,-1,0) |
| Lid mass | ~0.52 kg (oak at 700 kg/m³) | 0.519 kg |

## Kept from cl25

`lid_base` (mocap), the `handle` body, `lid_hinge` at (0.10, 0, 0.04), the geoms
`lid_body` / `lid_wall_{far,near,left,right}` / `lid_panel` / `handle`, and
`base_site` / `object_site` at **(-0.09, 0, 0.05)**. Every static collider box is
reproduced at its cl25 position and size, so the teacher's lip radius
`_R0 = (-0.19, 0, 0.01)` is unchanged. Target -1.308997 rad and threshold 0.2 rad are
untouched.

## THE HINGE AXIS SIGN — the big change, and why

cl25 hinged the lid with `axis="0 1 0"`, which turns the -75 deg target into a
**drop-down flap**. Two defects follow, both measured on HEAD and both reproducing
identically on the cl25 XML (`~/cl_v2_work/W2-b/lid_orig.xml`), i.e. neither was
introduced here:

1. **The task was winnable by doing nothing.** With zero actions the lid free-falls
   from 0 to the -1.309 rad stop in ~0.25 s (about 12 control steps). cl25: 0.776 kg,
   same trajectory. `verify_task` G3 fails it (`joint_drift_task_gravity` = 1.31 rad
   against a 0.02 limit); `success_at_reset` does not catch it because one step is only
   0.015 rad. The cl25 teacher docstring's "falls to -0.735 rad and stalls there" does
   not reproduce on HEAD.
2. **The lid swept through its own box.** From about -27 deg onward the lid's far edge
   is below the floor plank and inside the carcass footprint; at the target it sits at
   (x = 0.048, z = -0.153), through the floor and out of the bottom. Nothing collides,
   because every mechanism geom is `contype="2" conaffinity="1"` and 2 & 1 = 0. It is
   also why the cl25 swept drop was 0.157 m.

Adding a friction stay fixed (1) and exposed (2): with a 0.62 N m torque hinge the
teacher stalled at 0.30-0.89 rad against the 1.109 rad cutoff, because past ~45 deg the
grasp point is *inside the box*, where the arm cannot follow it.

**CL-V2 therefore reverses the axis to `(0,-1,0)`.** The same -75 deg target now lifts
the lid up and back over the hinge. That is what `open_lid_env_cfg.py` and
`OpenLidCommand` have always documented — "gravity opposes the motion throughout and
the lid falls shut if released" — so this makes the asset match the task, rather than
changing what the task means. Measured after the change:

| | cl25 (axis 0 1 0) | v2 (axis 0 -1 0) |
|---|---|---|
| zero-action drift over a 3 s episode | -1.309 rad (full open) | **-0.0014 rad** |
| lid vs box interpenetration | yes, from -27 deg on | **none** |
| swept drop below mount | 0.157 m | **0.070 m** |
| mount z | 0.187 | **0.100** |
| gravity | assists | **opposes** |

## Other changes

1. Visuals: the five translucent boxes become a plank-built chest with brass hardware.
2. The lifting geometry is a **turned knob** (32 x 32 x 36 mm) instead of a
   30 x 100 x 24 mm batten. The hinge axis is world y, so a pinch whose closing axis is
   world y is invariant under the lid's rotation — the wrist never has to twist about
   the approach axis while lifting — and world y is the closing axis a Franka top-down
   pose already presents. The batten forced a 90 deg wrist twist and measured 0.000.
3. `<compiler angle="radian" meshdir="assets" texturedir="assets"/>` added, and the
   hinge range written as `-1.308997 0` **radians**. The cl25 XML had no compiler tag,
   so its bare `range="-75 0"` was silently degrees.
4. Mass from a real density (oak 700 kg/m³) instead of MuJoCo's default 1000.
5. Mount band pulled in to x (0.45, 0.51) in `env_cfgs.py`: the knob now travels away
   from the robot as it lifts, from radial 0.36-0.44 closed to 0.51-0.58 fully open,
   and 0.58 is `workspace.MECHANISM_HANDLE_RADIAL_MAX`.

## Licence summary

Both textures CC0 1.0. Geometry original, generated by the build script.
