# Flip-Switch — asset provenance (CL-V2, W2-a, 2026-09-09)

**Asset:** BUILT in headless **Blender 4.2.23 LTS**, textured from ambientCG (D8).
Build script `~/assets_raw/blender/w2a/build_switch.py`.

**Real object:** a **heavy industrial toggle** — a black bakelite bat with a ball knob,
on a metal-clad surface-mount switch box. (Not a domestic 30 mm rocker: the frozen
`object_site` sits 62 mm above the pivot, so the lever is 75 mm tall, which is the
industrial toggle / isolator size, and that is what was modelled.)

| part | mesh | texture (source) | license |
|---|---|---|---|
| switch box 120 x 120 x 20 mm, raised face, bezel, 4 screws | `switch_plate_vis.obj` | ambientCG `Metal009` (brushed steel) | CC0 1.0 |
| pivot boss + tapered bat + 27 mm ball knob | `switch_toggle_vis.obj` | ambientCG `Metal029` (matte black), luminance lifted to mean 0.24 | CC0 1.0 |

`https://ambientcg.com/view?id=Metal009`, `https://ambientcg.com/view?id=Metal029` — CC0 1.0.

## The detent is the asset — what must not move

Bistability is an **over-centre weighted lever**, not a joint spring: `handle` carries
4 g at z = 0.026 and `switch_weight` 50 g at z = 0.062, both on the z axis, so centre is
an UNSTABLE equilibrium. Both collider geoms keep their exact size, position, **mass**
and contact parameters, and every visual mesh is `mass="0"`, so the centre of mass stays
directly above the pivot. Any x-offset in the mass would add a constant gravity torque
and destroy bistability — that is what
`tests/test_class_a_expansion.py::test_flip_switch_detent_is_actually_bistable` pins.

The visual pivot boss is part of the MOVING mesh (it rotates with the bat) and is
rotationally symmetric about the hinge axis, so it never clips the static bezel.

## Real dimensions vs. the spec

| dimension | real product | this asset | note |
|---|---|---|---|
| bat length above bushing | 25 mm (domestic) / 60-80 mm (industrial isolator) | 75 mm to the top of the knob | fixed by the frozen `object_site` at z = 0.062 |
| ball knob | 20-30 mm | 27 mm | matches the `switch_weight` 26 mm collider box |
| switch box | 86 x 86 or 120 x 120 mm | 120 x 120 x 20 mm | UNCHANGED: sets `MECHANISM_DROP_BELOW_MOUNT["switch"] = 0.060` |

Changed: explicit `<compiler angle="radian"/>`; range `-0.7853982 0.7853982` rad instead
of `-45 45` degrees. Same compiled range.
