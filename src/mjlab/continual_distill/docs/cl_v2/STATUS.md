# CL-V2 — STATUS

> Shared scoreboard: one row per task, seven gates. Mutable — overwritten as things
> change; `LOGS.md` is the append-only trail. **Update only YOUR task's rows.** A gate
> is green only with the evidence the row asks for (n / HEAD / reviewer). Summary
> counts are RECOMPUTED from the rows, never incremented by hand (cl25's drifted twice).

Last updated: 2026-09-09 (recomputed from rows by update_status.py)
Measured on HEAD: `1127d12` (teacher baselines only, from `../cl25/phase_1/STATUS.md`)

## Summary (computed from rows)

| | count |
|---|---|
| Tasks in scope | 25 |
| All seven gates green | 0 |
| Asset chosen (G1) | 1 |
| Physics + init + success verified (G3–G5) | 0 |
| Visual accepted (G6) | 0 |
| Teacher regression recorded (G7) | 0 |
| Blocked / needs decision | 0 |

Gate legend: `-` not started, `~` in progress, `x` failed (see notes), `ok` green.

## Free objects (16)

| Task | Owner | Current asset | Chosen asset | G1 | G2 | G3 | G4 | G5 | G6 | G7 | Teacher SR cl25 → v2 (n, HEAD) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Lift-Cube | W1-a | 40 mm box |  | - | - | - | - | - | - | - | 1.000 → | cube.xml is SHARED by Lift, Stack(object), Place, Throw, Cage-Drag — W1-a owns it; must stay < 55 mm wide for Cage-Drag |
| Stack-Cube | W1-a | box on cuboid |  | - | - | - | - | - | - | - | 0.328 → |  |
| Push-Cuboid | W1-a | 80x80x30 box |  | - | - | - | - | - | - | - | 0.223 → | cuboid.xml is SHARED by Push-Cuboid, Drag-Pull, Stack(base) — W1-a owns it |
| Drag-Pull | W1-a | box |  | - | - | - | - | - | - | - | 0.477 → |  |
| Topple-Block | W1-a | 100x140x180 box |  | - | - | - | - | - | - | - | 0.938 → | CoM-dependent threshold; re-derive |
| Place-In-Container | W1-b | box + 5-box bin |  | - | - | - | - | - | - | - | 0.250 → | container.xml SHARED with Throw-To-Bin — W1-b owns it; object is W1-a's cube |
| Throw-To-Bin | W1-b | cube + 5-box bin (code: cube.xml + container.xml, not a sphere) |  | - | - | - | - | - | - | - | 0.000 → | teacher is a characterised failure; G5 via oracle |
| Reorient-Object | W1-b | cylinder r20 h40 |  | - | - | - | - | - | - | - | 0.547 → |  |
| Cage-Drag | W1-a | small object |  | - | - | - | - | - | - | - | 0.141 → | must fit between OPEN fingers |
| Strike-Slide | W1-b | puck r35 h24 |  | - | - | - | - | - | - | - | 0.000 → | floor-guard margin flagged in cl25; puck.xml SHARED with Tool-Pull — W1-b owns it |
| Tool-Pull | W1-b | stick + hook + puck |  | - | - | - | - | - | - | - | 0.047 → |  |
| Peg-Insertion | W1-c | 24 mm peg, 35 mm hole |  | - | - | - | - | - | - | - | 0.031 → | chamfer = task decision |
| Axial-Extract | W1-c | plug in socket |  | - | - | - | - | - | - | - | 0.781 → | keep ~4 N slide friction |
| Edge-Grasp | W1-c | plate on ledge |  | - | - | - | - | - | - | - | 0.000 → |  |
| Pivot-Lift | W1-c | 120x100x20 board |  | - | - | - | - | - | - | - | 0.000 → | cl25 M4: approach point sits in GRASP_RADIAL_MIN dead zone — G4 applies to SPAWN poses; if the approach is unreachable, tweak geometry (allowed) and log it |
| Reach-Target | W1-a | none | scene only (studio rig) | ok | ok | - | - | - | - | - | 1.000 → | table + lighting only |

## Articulated (9)

| Task | Owner | Current asset | Chosen asset | G1 | G2 | G3 | G4 | G5 | G6 | G7 | Teacher SR cl25 → v2 (n, HEAD) | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Push-Button | W2-a | box button |  | - | - | - | - | - | - | - | 0.984 → |  |
| Open-Drawer | W2-b | box carcass |  | - | - | - | - | - | - | - | 0.888 → |  |
| Open-Door | W2-b | box door |  | - | - | - | - | - | - | - | 0.000 → | 84.3/90 deg threshold: keep or justify |
| Slide-Window | W2-b | box frame + pane |  | - | - | - | - | - | - | - | 1.000 → |  |
| Open-Lid | W2-b | box + hinged lid |  | - | - | - | - | - | - | - | 1.000 → |  |
| Turn-Lever | W2-a | plate + bar |  | - | - | - | - | - | - | - | 0.941 → |  |
| Rotate-Valve | W2-a | plate + spokes |  | - | - | - | - | - | - | - | 0.473 → |  |
| Flip-Switch | W2-a | plate + toggle |  | - | - | - | - | - | - | - | 0.656 → |  |
| Push-Flap | W2-a | panel + flap |  | - | - | - | - | - | - | - | 1.000 → | hinge range bug already FIXED at 1127d12 (range=-80.2 0, degrees); keep angle units explicit in the new MJCF |

## Infrastructure (W0)

| Item | Owner | State | Notes |
|---|---|---|---|
| `asset_pipeline.py` (download → trimesh → CoACD → MJCF) |  | ok | src/mjlab/scripts/asset_pipeline.py — fetch/inspect/package; coacd 1.0.14 + matplotlib added to the dev group (uv add --group dev) |
| `verify_task.py` (G3 + G4 + G5 battery) |  | ok | src/mjlab/scripts/verify_task.py — G3+G4+G5 in one run; validated on Lift-Cube, Open-Drawer, Push-Flap, Place-In-Container (all PASS on the primitives) |
| Scene: table top, lighting rig, shadows | | - | |
| Turntable render + site `/v2/` | | - | |
| Headless Blender tarball in `~/tools/` | | - | D1 |
| git-lfs for `asset_zoo/**/assets/` | | - | D2; 82 MB today |

## Ledger (provenance / license)

| Key | Asset | Source URL | License (any, recorded) | Real scale checked | Used by |
|---|---|---|---|---|---|
| | | | | | |

## Decisions needed

| # | Question | Raised | Decided |
|---|---|---|---|
| D1 | Blender: headless on login node or workstation only? | 2026-09-09 | headless tarball in `~/tools/`, login node |
| D2 | git-lfs for processed assets? | 2026-09-09 | yes, for `asset_zoo/**/assets/` |
| D3 | PartNet-Mobility (non-commercial) allowed? | 2026-09-09 | yes — any license is acceptable, record it |
| D4 | Peg-Insertion: chamfered real toy, or keep chamfer-free 3 mm clearance? | 2026-09-09 | real toy geometry; re-derive clearance |
| D5 | The 4 collapsed variants (Lift-Cylinder/Sphere/Ellipsoid, Push-Disc) stay registered but are OUT of the 25-task gate scope; keep them compiling. W1-a/W1-b may upgrade them opportunistically (golf ball / lemon / can / puck) after their 25-task rows are green | 2026-09-09 | out of scope, must keep loading |
| D6 | git-lfs? REVERSED: no LFS. Remote is GitHub; LFS bandwidth quota (1 GB/month) would be burnt by every clone. Budget is <= 5 MB per asset (25 assets ~125 MB) which plain git handles | 2026-09-09 | no LFS; enforce the 5 MB budget instead |
| D7 | Texture format: MuJoCo 3.11 loads PNG only (JPG rejected). Pipeline converts to PNG <= 2048^2 (default 1024) with AO baked into albedo; no normal/roughness maps (classic renderer ignores them) | 2026-09-09 | PNG only |
| D8 | Asset sources reachable: Poly Haven (CC0 models+textures), GSO via Gazebo Fuel (CC-BY 4.0), YCB (CC-BY 4.0), ambientCG (CC0). SAPIEN/PartNet-Mobility NOT reachable -> all 9 mechanisms are built (Blender/trimesh) with real textures | 2026-09-09 | build mechanisms |
