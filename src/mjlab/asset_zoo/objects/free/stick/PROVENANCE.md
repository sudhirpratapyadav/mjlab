# free/stick — provenance

| field | value |
|---|---|
| Asset | **Wooden reach hook** (260 mm turned dowel, 22 mm dia, curving into a 70 mm hook) |
| Geometry source | built — `~/cl_v2_work/W1-b/build/build_stick.py` (lathe + swept tube) |
| Albedo source | ambientCG **Wood051** colour map, https://ambientcg.com/view?id=Wood051 |
| License | Geometry: built here. Texture: **CC0** (ambientCG). |
| Used by | `Mjlab-Tool-Pull-Franka` |

## Real dimensions vs the spec

The primitive it replaces was a 260 × 22 × 22 mm box shaft with a 22 × 70 × 22 mm bar at
`(0.12, 0.035, 0)`. This asset keeps **exactly those extents** — a 22 mm hardwood dowel is
a real stock size — and replaces the T-bar with a quarter-arc bend into the hook, which is
what a real reach hook looks like and what lets the hook CATCH the puck rather than only
push it. Body extent measured after build: 261 × 81 × 22 mm (x −0.130…0.131,
y −0.011…0.070, z ±0.011).

Both sites keep their offsets exactly: `object_site` at `(-0.09, 0, 0)` (the grasp point,
which `tool_spawn_range`'s +0.09 x offset compensates) and `tool_tip_site` at
`(0.12, 0.02, 0)` (on the hook).

## Collider

Two **cylinder primitives** on the same axes and extents as the boxes they replace, with
the original geom names `stick_shaft` / `stick_hook` and their `condim`/`friction`/
`solref`/`contype` verbatim. A dowel *is* a cylinder, so this is exact rather than an
approximation, and MuJoCo composes the body inertia from the two geoms — an L-shaped
inertia, not a convex-hull overestimate. Mass 75 g at 600 kg/m³ (real hardwood): shaft
59 g + hook 16 g, against the primitive's arbitrary 40 + 10 g.

## Note for whoever revisits tool use

`classical/tool_pull.py` documents (with 8 measured configurations) that the stick cannot
be picked up: the pinch ejects it axially. **That was deliberately NOT fixed here** — the
brief for this row is to measure the characterised failure, not to redesign around it —
and a round dowel is, if anything, slightly more slippery than the box was. The teacher
does not use the tool (it is a direct closed-finger drag), so the G7 number is unaffected.
If someone does want tool use to work, the fix the teacher's docstring asks for is a
graspable FEATURE on the shaft (a moulded grip with flats at `object_site`), which this
geometry is set up to accept.

One measured caveat that comes WITH the cylinder colliders, recorded so it is not
rediscovered: mujoco_warp 3.11 emits
`MULTICCD is enabled, but the scene contains CCD pairs without multicontact support:
[('CAPSULE','CYLINDER'), ('CYLINDER','CYLINDER'), ('CYLINDER','BOX')]` for this env —
i.e. a fingertip pad (box) against the shaft (cylinder) resolves to **at most one
contact point**, where the primitive version's box shaft got the full box–box
multicontact. That does not affect the G7 number (the teacher never grasps the stick),
but it makes the squeeze-ejection strictly easier, so anyone attempting a tool-use
teacher should switch `stick_shaft` back to a box of the same extents (its corners then
stand 4.6 mm proud of the visual dowel, which nothing in this task ever touches) or add
flats to the grip.

## Budgets

1016 tris, 512² PNG albedo, 2 primitive colliders, 340 KB on disk.

Note: `stick_package.json` records `mass_kg: 0.157`, which is the pipeline's
density × CONVEX HULL volume — the hull of an L-shape badly overestimates a hook. That
number is a build record only; the XML sets the mass per geom (shaft 0.059 + hook 0.016
= **0.075 kg**, confirmed on the compiled model) precisely so MuJoCo composes the real
L-shaped inertia instead.
