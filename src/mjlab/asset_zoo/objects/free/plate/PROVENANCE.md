# plate — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Edge-Grasp-Franka` (with `free/ledge`).

## What it is

**Google Scanned Objects `Room_Essentials_Salad_Plate_Turquoise`** — a scanned real
retail ceramic plate, 221.9 x 221.9 x 28.4 mm as scanned — uniformly scaled **0.676** to
**150 x 150 x 19.2 mm**, i.e. a standard **6 in bread-and-butter / side plate**. The scan's
own albedo is the texture. Mass **180 g** (a real 150 mm stoneware side plate is
150-200 g; the scan is not watertight so the pipeline's density path would have used the
convex-hull volume and returned 547 g — the mass is therefore stated, not derived).

* Source: https://fuel.gazebosim.org/1.0/GoogleResearch/models/Room_Essentials_Salad_Plate_Turquoise
* Licence: CC-BY 4.0 (Google Research).
* Scale check: the scan's 222 mm is a salad plate; 150 mm is the next size down in the
  same product family, so the scale-down lands on a real product size rather than a
  convenient number.

## Why not one of the small GSO saucers

All of them are too small to keep the task honest — the plate must be UNSPANNABLE, i.e.
wider than the 80 mm gripper aperture in both in-plane directions:
`Cole_Hardware_Saucer_Glazed_6` 80.0 x 79.9 mm (exactly the aperture),
`Kotobuki_Saucer_Dragon_Fly` 48 x 48 mm, `Ecoforms_Quadra_Saucer_SQ1` 44 x 43 mm,
`Threshold_Bamboo_Ceramic_Soap_Dish` 62 x 49 mm. Measured with `asset_pipeline inspect`.

## Collision

ONE analytic **cylinder**, r 75 mm, half-height 9.6 mm, named `plate_geom`.

CoACD (8 hulls) and the plain convex hull are both geometrically better and neither
SETTLES: MuJoCo's mesh-plane contact returns three points under a flat circular foot and
the plate micro-rocks forever. Measured after 2 s of free settling (verify_task's own G3
settle test): CoACD |v| 5.9 mm/s, |w| 0.0131 rad/s; convex hull |v| 0.7 mm/s,
|w| 0.1013 rad/s; **cylinder |v| = |w| = 0.000**. The gate is |v| < 1 mm/s and
|w| < 0.01 rad/s, so only the cylinder passes.

Nothing in the task needs the dish's concavity: the plate is pushed on its trailing face,
overhung at the riser's edge, and pinched on its thickness — a solid disc of the same
footprint and thickness reproduces all three. `friction="0.5 0.03 0.003"`, `condim=3`,
`solref="0.01 1"` carried over from the primitive verbatim.

## Constants re-derived

| constant | old | new | derivation |
|---|---|---|---|
| `edge_grasp.PLATE_HALF_X` | 0.05 | **0.075** | plate radius |
| `EdgeGraspCommandCfg.plate_rest_offset` | 0.010 | **0.012** | AABB half-height 0.0096 + 2.4 mm settle clearance |
| `EdgeGraspCommandCfg.plate_rel_x` | (-0.02, 0.04) | **(-0.02, 0.02)** | \|rel_x\| <= LEDGE_HALF_X - PLATE_HALF_X = 0.025 keeps the plate fully on the riser at spawn |
| `EdgeGraspCommandCfg.plate_rel_y` | (±0.07) | **(±0.05)** | \|rel_y\| <= 0.13 - 0.075 = 0.055 |
| `edge_grasp.OVERHANG_TARGET` | 0.032 | **0.048** | same 64% of the plate half-length the primitive used (0.032/0.05) |
| `plate_constants.get_mocap_goal_spec` marker | box 0.05/0.045/0.008 | cylinder 0.075 x 0.0096 | matches the new footprint |

The PUSH CONTACT POINT — `ledge_x + plate_rel_x_max + PLATE_HALF_X` — is **unchanged at
`ledge_x + 0.095`**, because `plate_rel_x` shrank by exactly as much as the plate grew.
That is what lets the reach derivation in `ledge_spawn_range` stand.
