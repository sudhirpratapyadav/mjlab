# free/container — provenance

| field | value |
|---|---|
| Asset | **Spritz Easter Basket, Plastic, Teal** — a moulded open-top plastic basket, scanned |
| Source | Google Scanned Objects, via Gazebo Fuel |
| Source URL | https://fuel.gazebosim.org/1.0/GoogleResearch/models/Spritz_Easter_Basket_Plastic_Teal |
| License | CC-BY 4.0 (© 2020 Google LLC) |
| Raw file | `~/assets_raw/gso/Spritz_Easter_Basket_Plastic_Teal/meshes/model.obj` (7798 v / 12630 f, 4096² albedo) |
| Used by | `Mjlab-Place-In-Container-Franka`, `Mjlab-Throw-To-Bin-Franka` (the XML is SHARED) |

## Real dimensions vs the spec

Scanned extent **191.3 × 182.6 × 128.9 mm**, used at **scanned scale — no rescaling**.
It replaces a 5-box primitive of 140 × 140 mm outer / 108 mm inner clear span / 70 mm rim.

Cavity measured by slicing the mesh every ~1 mm in z:

| feature | value |
|---|---|
| moulded floor thickness | 4 mm (inner floor surface at z = 0.004) |
| inner clear span at the floor | 163 × 163 mm (half-span **0.0815**) |
| inner clear span at the rim | 186 × 178 mm |
| rim, lowest point of the scallop | z = **0.113** |
| rim, scallop peaks | z = 0.129 |
| outer footprint | 191 × 183 mm (half 0.0956 × 0.0913) |

## What was scaled / rotated, and why

Nothing was scaled or rotated. **Origin: `bottom`** — the body frame is the basket's
underside and its xy centre, so `container_spawn_range z = 0` puts the basket ON the
ground. (The primitive's floor slab was centred on the body origin, so the old
`z = 0.02` left the bin floating 12 mm in the air.)

## Why this basket

Real receptacles are big. Of everything reachable — Poly Haven has no container under
266 mm; GSO's are `Curver_Storage_Bin_Black_Small` 286 × 193 × 134, `Target_Basket_Medium`
270 × 268 × 218, `Threshold_Basket_..._Small` 239 × 162 × 148 (66 k tris),
`Hefty_Waste_Basket` 271 × 204 × 300, `Full_Circle_Happy_Scraps` 213 × 155 × 143 (lidded)
— this is the **smallest open-top one**, and even it is 1.4× the primitive's footprint.
That matters because the bin's band and the cube's band have to stay disjoint inside a
grasp envelope only 0.55 m across: at 286 mm the two footprints cannot be separated
without pushing the bin outside `GRASP_RADIAL_MAX + 0.05`. It is also 12.6 k tris, inside
the 20 k budget with no decimation, and its top is genuinely open (verified by slicing:
the footprint stays 166 → 190 mm all the way to the rim; there is no handle).

## Colliders

Floor + four walls as **boxes**, keeping the primitive's five geom names
(`container_floor`, `container_wall_{px,nx,py,ny}`) and their `condim` / `friction` /
`solref` / `contype` verbatim. CoACD was tried first (8 hulls) and rejected: it merged
the 4 mm floor with the flared lower wall into an **18 mm slab**, which would have left
the cube visibly floating 14 mm above the basket's inner floor in every rollout video.
The box walls are vertical at the FLOOR's inner span (0.0815), i.e. up to 11 mm inside
the flared visual wall near the rim — conservative in the safe direction (the collision
cavity is never larger than the real one) and exact where the object actually rests.

## Task constants re-derived from this geometry

| constant | old | new | derivation |
|---|---|---|---|
| `PlaceInContainerCommandCfg.lateral_tolerance` | 0.055 | **0.0585** | 0.0815 inner half-span − 0.0230 cube half-width |
| `PlaceInContainerCommandCfg.rim_height` | 0.05 | **0.093** | 0.113 rim low point − 0.020 site height |
| `PlaceInContainerCommandCfg.floor_tolerance` | 0.04 | **0.020** | site is 0.016 above the inner floor; allows 4 mm of contact penetration |
| `container_spawn_range.z` | 0.02 | **0.0** | body origin is the underside |
| Place `container_spawn_range` | x (0.28,0.48) y (0.07,0.24) | **x (0.30,0.46) y (0.13,0.24)** | the 191 × 183 mm footprint overlapped the cube band by 4 mm at y = 0.07 |
| `place_in_container.DROP_HEIGHT` | 0.10 | **0.15** | scallop peak 0.109 above the site + cube half-height 0.0226 + 18 mm servo margin |
| `place_in_container.lift_height` | 0.22 | **0.26** | required cube rise went from 0.082 to 0.129 |
| `throw_to_bin.DROP_HEIGHT` | 0.12 | **0.15** | same clearance convention |

## Budgets

12 630 tris, 1024² PNG albedo, 5 primitive colliders, 1.10 MB on disk.
