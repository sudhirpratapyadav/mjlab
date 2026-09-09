# free/cylinder — provenance

| field | value |
|---|---|
| Asset | **CoQ10** — Jarrow Formulas amber HDPE "packer" supplement bottle, scanned |
| Source | Google Scanned Objects, via Gazebo Fuel |
| Source URL | https://fuel.gazebosim.org/1.0/GoogleResearch/models/CoQ10 |
| License | CC-BY 4.0 (© 2020 Google LLC) |
| Raw file | `~/assets_raw/gso/CoQ10/meshes/model.obj` (5381 v / 10490 f, 4096² albedo) |
| Used by | `Mjlab-Reorient-Object-Franka` (in scope), `Mjlab-Lift-Cylinder-Franka` (D5, out of scope, kept loading) |

## Real dimensions vs the spec

Scanned extent **47.4 × 47.5 × 84.4 mm** (bottle body Ø47.4 up to z≈60 mm, then a Ø39 mm
cap). The spec it replaces was a primitive `size="0.02 0.02"` cylinder, i.e. a 40 × 40 mm
billet.

## What was scaled / rotated, and why

* **Uniform scale ×0.63 → 29.9 × 29.9 × 53.2 mm.** The reorient teacher grasps this object
  **end face to end face** — `reorient_object.py::_approach_rot` returns
  `down_frame(grasp_bearing)`, and `down_frame` puts the finger-CLOSING axis at that
  bearing, which is the bearing of the bottle's own axis. That grasp is what makes the
  90° wrist roll a rigid-body move (the object's axis *is* the closing axis), and the
  module docstring records it measuring 0.531 against 0.469 for a barrel grasp on the
  same 32 episode-instances. It therefore requires the Franka's **80 mm aperture to span
  the object's LENGTH**. Every candidate can/jar/bottle in the reachable catalogues is
  84–110 mm tall at scanned scale (GSO CoQ10 84.4, Folic_Acid 83.9, Beta_Glucan 98,
  AllergenFree_JarroDophilus 96, Lutein 97, Theanine 97, 5_HTP 89, Inositol 110,
  Quercetin_500 144; Poly Haven has no can-sized model at all), so **none of them can be
  grasped at scanned scale**. ×0.63 lands on 29.9 × 53.2 mm, which is a real bottle of
  the same product family — a 12 cc / 0.4 oz amber packer bottle (catalogue: Ø30 × 51 mm;
  the next size up, 15 cc, is Ø31.8 × 55.6 mm). Aspect ratio 1.78, so it is still stable
  lying down (a squat can would tip onto a face by itself and hand the task a free
  success at reset).
* **No rotation.** The scan's +z is already the bottle axis, which is what
  `ReorientObjectCommandCfg.body_axis = (0,0,1)` measures.
* **Origin: bounding-box centre**, on the bottle axis at mid-length, so `object_site`
  (the drift anchor) is at the centre and the lying-down spawn height is just the radius.

## Processing

`asset_pipeline` (`~/cl_v2_work/W1-b/build/build_bottle.py`): albedo 4096² → 1024² PNG,
visual mesh kept at 10 490 tris (budget 20 k), collider = the convex hull reduced to its
96 Fibonacci-sphere support vertices (188 tris; the raw hull is 2132). Mass 25 g = a
filled 12 cc HDPE bottle; inertia from the hull at that mass. Directory 1.39 MB (budget 5 MB).
