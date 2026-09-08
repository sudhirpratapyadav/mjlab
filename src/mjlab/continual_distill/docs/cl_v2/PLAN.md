# CL-V2 — PLAN (short)

> Read `GOAL.md` first. `STATUS.md` is the scoreboard, `LOGS.md` the append-only
> trail. No EXPERIMENTS file — the gate table in `STATUS.md` is the run table.
> `../cl25/` is read-only reference: `CONTEXT.md` for the task list,
> `phase_1/STATUS.md` for the teacher numbers we start from.

## Cluster rules

Read `~/use_instructions/README.md` before touching Slurm. Reuse an existing holder,
never `scancel` someone else's job, record the GPU index you took in `LOGS.md`,
`MUJOCO_GL=egl` on a GPU node for rendering. Mesh processing is CPU work — login node.

## Frozen vs free

**Frozen:** body / site / joint names the env cfgs and teachers read
(`object_site`, `base_site`, `drawer_slide`, ...), the `workspace.py` envelope, the
success predicate *shape* (sparse, binary, latched).
**Free:** geometry, materials, mass/inertia, collision representation, thresholds that
depend on geometry (re-derive, do not copy). **Small geometry tweaks are allowed** —
object size, handle offset, mount pose, hinge placement, clearance — when they make the
task better (more graspable, more reachable, more realistic). The test is: the motion
profile is unchanged and the task still justifies its name. Anything that changes the
motion profile (a drawer that swings, a lid that slides) is a new task, not a tweak;
record every tweak in the STATUS row and the W3 handover list. **Classical teachers may
be changed** to fit the new assets; the G7 number is measured on the final teacher.

## Gates (per task; all seven green = done)

| Gate | Passes when |
|---|---|
| G1 asset | chosen, real-scale, any license (record it), ledger row + `PROVENANCE.md` |
| G2 packaged | textured visual mesh + convex/primitive colliders, sites and joints preserved, within budgets |
| G3 physics | settle, drop, grasp-hold, articulation range read from the *compiled* model, step-time <= 1.3x primitive |
| G4 init dist | 1000 resets: all in envelope, none in dead zone, zero spawn collisions, histograms saved |
| G5 success | success-at-reset = 0.000 / 1000; predicate reachable under the physics; threshold stated in physical units |
| G6 visual | still + turntable published; checklist passed; reviewer initials |
| G7 teacher | classical teacher (adapted as needed) at n=128, delta vs cl25 recorded; an unexplained drop blocks G3/G4 |

## Budgets

<= 20k tris per object (50k per mechanism); albedo <= 2048², normal/roughness baked
in (MuJoCo has no PBR); colliders from CoACD (<= 8 pieces free, <= 16 articulated) or
primitives; mass from real density; <= 5 MB processed per asset in git, raw files in
`~/assets_raw/`.

## Waves

- **W0 infra**: `asset_pipeline.py` (download → trimesh → CoACD → MJCF),
  `verify_task.py` (G3+G4+G5), scene lighting/table, turntable render, site `/v2/`.
  Install `coacd`; headless Blender tarball in `~/tools/`; enable git-lfs for `asset_zoo/**/assets/`.
- **W1 free objects** (16, three agents by asset family: blocks/boxes; balls/pucks/cans;
  insertion/edge) — asset selection (G1) starts before W0 lands.
- **W2 articulated** (9, two agents: plate-mounted family; cabinet family) — all
  Blender-built with real textures, or adapted from PartNet-Mobility / Objaverse (any license OK).
- **W3 lead**: G7 sweep, STATUS recomputed from rows, exit check, handover note listing
  every task whose *mechanics* changed.

## Decisions made (2026-09-09)

Headless Blender on the login node (CPU, scripted). git-lfs for processed binaries.
Any asset license, PartNet-Mobility included — record it, that is all. Peg-Insertion
uses the real toy geometry (small chamfer), clearance re-derived from the mesh.
Further open questions: the owner of the row decides and logs it; do not wait.

## Asset direction (decide per row in STATUS.md)

Free: wooden toy blocks (lift/stack), printed cardboard boxes (push/drag), tall carton
(topple), basket + block (place), ball + waste bin (throw), slim can (reorient), puck
(cage-drag/strike), wooden reach-hook + puck (tool-pull), shape-sorter peg + board
(peg), mains plug + power strip (extract), ceramic plate on shelf edge (edge-grasp),
cutting board (pivot). Articulated: mushroom button, cabinet drawer, cabinet door,
sliding pane, hinged wooden box, wall lever, gate-valve handwheel, toggle light switch,
letterbox flap. Gripper opens 80 mm: grasp faces <= 60 mm.

## Traps

MJCF angles default to degrees (the flap bug). Concave meshes collide as their hull.
Scans arrive in random units — check the bounding box against the real spec first. CoM
moves with the mesh (topple/pivot/reorient thresholds). Mechanism placement is set by
the reset EVENT, not the command pose_range. Run `tests/test_classical_teachers.py`
after any registry edit.
