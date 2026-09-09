# peg_in_hole — provenance (CL-V2, W1-c, 2026-09-09)

Task: `Mjlab-Peg-Insertion-Franka`. Decision **D4** (STATUS.md): use real toy geometry,
re-derive the clearance from the mesh.

## What it is

A wooden **shape-sorter** pair, modelled on the square-post station of a
pound-a-peg / shape-sorting toy (Melissa & Doug "Pound-A-Peg" and the classic wooden
shape-sorting cube are the reference products):

| part | real dimensions | notes |
|---|---|---|
| `peg` | 25 x 25 x 100 mm beech post, 2 mm 45 deg lead chamfer on the insert end, 1 mm break on the top edges | 25 mm is the standard square-dowel section a toddler's hand and a shape sorter both use |
| `hole_board` | 120 x 120 x 30 mm plywood lid, 30 mm square through-bore, 45 deg x 6 mm lead-in chamfer (mouth 42 mm) | every real sorter lid has the lead-in; it is what lets the post find the hole |

**Round vs square.** Real sorters have both; this one is SQUARE. Reason: the collision
representation for the hole must be boxes (MuJoCo collides a mesh as its convex hull,
so a hole cannot be a mesh), and a square bore is represented by boxes EXACTLY. A round
bore approximated by boxes would leave four unmodelled corner voids that contradict the
visual mesh. The square section also keeps the task's yaw constraint meaningful: a
45 deg-rotated 25 mm post has a 35.4 mm diagonal and will not enter the 30 mm bore.

## Clearance, re-derived from the mesh

bore 30.0 mm, peg 25.0 mm -> **2.5 mm per side** (5.0 mm across the flats).
Real toy shape sorters sit in the 2-4 mm per-side band; the previous primitive pair was
24 mm in 30 mm = 3.0 mm per side. The chamfer adds a **6 mm capture radius** on top:
a tip landing anywhere inside the 42 mm mouth is funnelled into the bore.

## Sources / licences

| file | source | licence |
|---|---|---|
| `assets/peg_tex.png` | Poly Haven texture `oak_wood_planks` (diffuse x AO), 512^2 | CC0 |
| `assets/hole_board_tex.png` | Poly Haven texture `plywood` (diffuse x AO), 512^2 | CC0 |
| `assets/peg_vis.obj`, `assets/hole_board_vis.obj` | built with trimesh + manifold3d (CSG) by `scratchpad/build/build_peg.py`; box-projected UVs | CC0 (own work) |

Densities: peg 720 kg/m^3 (beech) -> 44.8 g. Board 700 kg/m^3 (birch plywood) -> 281 g.

## Collision representation

* `peg`: one **box** 25 x 25 x 100 mm. The convex hull of the chamfered visual mesh is
  exact geometry but MuJoCo's mesh-plane contact returns only three points on a square
  base, and the standing peg then micro-rocks forever (|v| 2.8 mm/s, |w| 0.060 rad/s
  after 2 s, never damping). Box-plane is analytic, gives four points, and settles to
  |v| = |w| = 0.000. The peg's 2 mm lead chamfer is therefore cosmetic.
* `hole_board`: **12 boxes** reproducing the visual solid exactly (checked on a 0.9 mm
  grid over the whole bounding box: 0.000 cm^3 of solid without a box, 0.000 cm^3 of
  box outside the solid):
  - `hole_wall_{px,nx,py,ny}` — the 30 mm bore, z in [-0.015, +0.009] (names preserved);
  - `hole_rim_{px,nx,py,ny}` — the flat top face outside the 42 mm mouth;
  - `hole_chamfer_{px,nx,py,ny}` — four boxes rotated 45 deg (given as `quat`, so the
    MJCF degree/radian trap cannot bite) whose inner faces ARE the lead-in cone.
  So the chamfer is **physical**, which is the point of D4.

## Constants re-derived (see the code comments for the derivations)

| constant | old | new | why |
|---|---|---|---|
| `stack_command.stack_height` (env_cfgs.py) | 0.01 | **0.035** | the predicate compares the peg's ROOT to `base_root + stack_height`; an inserted peg's root is at ground + 0.05 and the board root is at 0.015 |
| `stack_command.height_threshold` | 0.03 | **0.015** | 0 seated, 0.030 on the board top, 0.021 stuck on the chamfer |
| `stack_command.success_threshold` (xy) | 0.015 | 0.015 (kept) | now 6x the 2.5 mm per-side clearance: in the bore => success |
| `peg_insertion.INSERT_DEPTH` | 0.025 | **0.050** | `_place` drives `tip_z -> goal_z - INSERT_DEPTH`; goal_z moved 0.025 -> 0.050 |
| `peg_insertion.hover_height` | 0.13 | **0.105** | keeps the carry altitude at the same physical tip z = 0.155 m |

Peg spawn `z=(0.05, 0.05)` and `object_site` at `0 0 -0.05` are unchanged (the peg is
still 100 mm long).
