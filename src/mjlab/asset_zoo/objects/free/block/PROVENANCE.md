# block — provenance

| | |
|---|---|
| **Asset** | YCB `003_cracker_box`, `google_16k` textured scan |
| **Source** | YCB Object and Model Set — https://www.ycbbenchmarks.com/ (mirror: `https://ycb-benchmarks.s3.amazonaws.com/data/google/003_cracker_box_google_16k.tgz`) |
| **License** | CC-BY 4.0 |
| **Raw download** | `~/assets_raw/ycb/003_cracker_box/google_16k/textured.obj` (outside the repo) |
| **Used by** | Topple-Block (only) |
| **Build script** | `~/cl_v2_work/W1-a/package_block.py` |

## Real dimensions vs the spec

Scan at real scale, essentially axis-aligned already (yaw search: 0.20 deg).
De-rotated extents **71.8 x 163.9 x 213.4 mm** — the real cracker carton. The
primitive it replaces was 100 x 140 x 180 mm.

## What was changed and why — the x stretch is NOT cosmetic

| Step | Value | Reason |
|---|---|---|
| Rotate about z | +0.20 deg | de-rotate |
| **Non-uniform scale** | **x1.3926, y0.9762, z0.9839 -> 100 x 160 x 210 mm** | see below |
| Origin | bounding-box centre | the mesh CoM lands within 1.1 mm of it, so the tipping analysis and the box collider share one frame |
| Decimation | 16384 -> 11999 tris | budget 20k |
| Texture | 4096^2 -> 1024^2 PNG | budget; PNG only |
| Collider | one `box`, half-extents 0.0500 x 0.0800 x 0.1050 | a carton collides as a box; keeps step time at the primitive baseline, and the topple mechanics depend on a crisp bottom edge to pivot about |
| Mass | 0.550 kg | published 0.411 kg over the published volume = 164 kg/m^3 (a realistic full cracker box), carried across to the scaled volume. **3.7x the primitive's 0.15 kg**, which was unphysically light (60 kg/m^3, lighter than styrofoam) |

**Why x was stretched.** Topple-Block exists because the block cannot be grasped:
`block.xml` states "force closure is impossible by construction", `audit_workspace`
lists it under `_SIDE_APPROACH` ("poked on the near face above its CoM"), and the
whole teacher is a poke. The Franka's fingers travel 0.08 m and the measured pad gap
at full open is ~0.095 m (`lift_object.py` module docstring). A real cracker box is
**71.8 mm** deep — comfortably graspable — so shipping it at real scale would let any
policy pick the box up and lay it down instead of toppling it: a different motion
profile, which the CL-V2 contract does not allow. Stretching x to **100 mm** restores
exactly the primitive's ungraspable width.

The stretch is also the least visible option available: x is the box's depth, so the
two large printed faces (normal +-x) are not distorted at all, and only the two narrow
side panels are stretched. y and z are held within 2.4% of the real product.

**Topple direction is preserved.** Tipping angle over the +-x edges is
`atan(0.050/0.105) = 25.5 deg`, against `atan(0.080/0.105) = 37.3 deg` over the +-y
edges, so the box still preferentially falls about its y-axis, carrying the body
x-axis to vertical — which is what `ReorientObjectCommand(body_axis=(1,0,0),
symmetric_axis=True)` measures. (Primitive: 29.1 deg vs 37.9 deg.)

The tip-vs-slide criterion is mass independent: the box tips rather than slides
whenever the contact height exceeds `half_x / mu = 0.050 / 0.8 = 0.0625 m`. The
teacher contacts at 0.170 m above the ground, so it tips.

Friction (`0.8 0.03 0.003`) / `condim` / `solref` / `contype` / `conaffinity` carried
over verbatim from the primitive geom.

## Size budget

`xmls/assets/` = 1.6 MB (limit 5 MB).
