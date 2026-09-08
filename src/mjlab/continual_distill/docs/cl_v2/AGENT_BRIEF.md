# CL-V2 — AGENT BRIEF (read fully before touching anything)

You are one of five wave agents replacing primitive-box assets with realistic ones in
the 25 Class-A Franka tasks. Repo: `/ihub/homedirs/svs_ald/sudhir/mjlab` (HEAD `1127d12`
+ the lead's uncommitted W0 work; **do not commit or run git write commands** — the lead
commits). Read, in order: `GOAL.md`, `PLAN.md`, `CODE_MAP.md` (the plumbing contract:
scene assembly, sim backend, per-task table of names/thresholds/geometry constants,
teacher constants, rendering, tests), `STATUS.md` (your rows), and this file.

## Cluster rules (non-negotiable)

- Read `~/use_instructions/README.md`. Holder job **20277** (`hold_dgx_amit`, dgx1) is
  NOT ours: reuse it, never `scancel` it or any job. Use ONLY the GPU index assigned to
  you (below). GPUs 4–7 are someone's training; GPU 0 is in use by others.
- Every GPU command:
  ```bash
  srun --jobid=20277 --overlap --cpus-per-task=4 bash -c "export CUDA_VISIBLE_DEVICES=<GPU> MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0; cd /ihub/homedirs/svs_ald/sudhir/mjlab; PYTHONPATH=src .venv/bin/python -m <module> <args>"
  ```
  (`MUJOCO_EGL_DEVICE_ID` is relative to `CUDA_VISIBLE_DEVICES`, so it is always 0.)
  Long runs: `bash ~/use_instructions/run_in_holder.sh 20277 <GPU> <log> <cmd...>` and poll
  the log. Log the GPU index you used in your LOGS entries.
- The login node has no GL: renders and teacher SR runs go through `srun`. Mesh work,
  Blender, CoACD and `verify_task.py` (CPU) run on the login node directly.
- `/tmp` is local to the login node and invisible from dgx1. Anything a GPU job reads or
  writes lives under `~/cl_v2_work/<your-agent>/` or the repo. Raw downloads go to
  `~/assets_raw/` (outside the repo), Blender sources to `~/assets_raw/blender/`.
- Python: `PYTHONPATH=src .venv/bin/python ...` (never `python`, never `pip install`;
  a needed package → `uv add --group dev <pkg>`, and say so in your log).

## Tools you have

| Tool | What |
|---|---|
| `python -m mjlab.scripts.asset_pipeline fetch polyhaven:<id> / gso:<name> / ycb:<name> / ambientcg:<id> / polyhaven-texture:<id>` | download to `~/assets_raw` |
| `... asset_pipeline inspect <mesh>` | extents (m), tris, UV, watertight |
| `... asset_pipeline package <mesh> --name <n> --out <xml_dir>/assets --origin ... --collider coacd\|hull\|box\|none --density/--mass ... --ao <armmap>` | writes `<n>_vis.obj`, `<n>_tex.png`, `<n>_col_NN.obj`, `<n>_package.json`, prints the MJCF snippets |
| Python API: `from mjlab.scripts.asset_pipeline import load_mesh, normalise, decimate, decompose, package` | for scripted builds (trimesh primitives + textures → package) |
| `~/tools/blender/blender -b --python <script.py>` | headless Blender 4.2 (CPU) for modelling/UV/decimation. `bpy.ops.wm.obj_export(..., export_uv=True)` |
| `python -m mjlab.scripts.verify_task --task <ID> --num-resets 1000 --out docs/cl_v2/runs/<Task>` | G3+G4+G5, CPU, ~3–8 min |
| `python -m mjlab.scripts.render_asset <object.xml> --out docs/cl_v2/renders/<Task> --joint-sweep` (GPU) | still.png, turntable.mp4, colliders.png |
| `python -m mjlab.continual_distill.classical.test_classical --task <ID> --num-envs 32 --num-episodes 4` (GPU) | **the n=128 G7 measurement** |
| `python -m mjlab.continual_distill.classical.render_rollout --task <ID> --out ~/cl_v2_work/<agent>/<ID> --batch-size 32` (GPU) | teacher.mp4 / failure.mp4 / thumb.jpg / result.json |
| `docs/cl_v2/publish_v2.sh <ID> <dir>` | push to https://cl.untuai.com/v2/<ID>/ |
| `python docs/cl_v2/update_status.py --task <Row> --gate G3=ok ...` | the ONLY way to edit STATUS.md |
| `python -m mjlab.scripts.audit_workspace --keyword <Task>` ; `python -m mjlab.scripts.benchmark_smoke --keyword <Task>` ; `FORCE_CPU=1 .venv/bin/python -m pytest tests/test_workspace_placement.py tests/test_classical_teachers.py -k <task>` | regression checks |

Catalogs (already downloaded lists): Poly Haven has 521 models (`cardboard_box_01`,
`plastic_crate_0[123]`, `CheeseBox_01`, `baseball_01`, `rubber_duck_toy`,
`carved_wooden_plate`, `jug_01`, `pot_enamel_01`, `cleaner_tin_01`, `can_rusted`,
`lightbulb_01`, `mousetrap`, `chess_set`, ...; `/info/<id>` gives dimensions in mm) and
857 textures; GSO has 1033 scanned products (`fuel.gazebosim.org/1.0/GoogleResearch/models?page=N&per_page=100`
lists them; names like `Threshold_Dinner_Plate_Square_Rim_White_Porcelain`,
`Curver_Storage_Bin_Black_Small`, `Target_Basket_Medium`, `GEOMETRIC_SORTING_BOARD`,
`HAMMER_PEG`, `Cole_Hardware_Mug_Classic_Blue`); YCB has the 77 standard objects
(`003_cracker_box`, `009_gelatin_box`, `036_wood_block`, `058_golf_ball`,
`077_rubiks_cube`, `071_nine_hole_peg_test`, `029_plate`, `024_bowl`, ...).
Search: `python3 -c "import json;print([n for n in json.load(open('/tmp/claude-3021/-ihub-homedirs-svs-ald-sudhir-mjlab/4116c241-2d91-4ca5-a0cd-953fad875db8/scratchpad/catalog/gso_models.json')) if 'plate' in n.lower()])"`
(a copy is also at `~/cl_v2_work/catalog/`).

## Packaging contract (what "G2 packaged" means)

1. Files go in `src/mjlab/asset_zoo/objects/<free|articulated>/<asset>/xmls/assets/`;
   the XML at `.../xmls/<asset>.xml` gets `<compiler meshdir="assets" texturedir="assets"/>`
   plus the `<asset>` block from the pipeline. **Add the Franka `assets`-dict pattern**
   to `<asset>_constants.py::get_<asset>_spec()` (see `franka_constants.py:21-37` and
   CODE_MAP §1): `spec.assets = update_assets({}, XML_DIR/"assets", "assets")`.
2. **Keep every body / joint / site / geom name and every site offset** that
   CODE_MAP §3 lists for your task. Keep `mocap="true"` roots. Keep `contype/conaffinity/
   condim/friction/solref` values from the old geoms on the new colliders. Visual mesh
   geoms: `contype="0" conaffinity="0" group="2" mass="0"`; colliders `group="3"`, invisible.
3. Mass from a real density (or the real product mass) via the pipeline's `<inertial>`.
4. Budgets: ≤ 20k tris per object (50k per mechanism), textures PNG ≤ 2048² (1024 default),
   ≤ 8 convex pieces (free) / ≤ 16 (articulated), **≤ 5 MB per asset directory**.
5. Articulated MJCF: the object XMLs have no `<compiler angle="radian"/>`, so bare hinge
   ranges are **degrees**. Write `<compiler angle="radian" .../>` explicitly in every
   mechanism XML you touch and use radians (CODE_MAP §3 lists the targets).
6. Mechanisms hang from a mocap mount: after changing geometry, re-measure the swept
   downward extent and update `workspace.MECHANISM_DROP_BELOW_MOUNT[<asset>]` — the test
   `tests/test_workspace_placement.py::test_mechanism_drops_are_swept_over_the_joint_range`
   is the oracle. Mount x/y bands live in the reset EVENT in `env_cfgs.py`, not the command.
7. Geometry constants baked into cfgs / commands / teachers (CODE_MAP §3 last column and
   §4) must be **re-derived from the new geometry**, never copied. Log each one you change.
8. A **small geometry tweak** (size, handle offset, clearance) is allowed when it makes the
   task better; the motion profile must stay identical and the task must still earn its
   name. Record every tweak in the STATUS row notes AND in your LOGS entry.
9. Concave objects (bins, boxes with holes, hooks) collide as convex hulls → CoACD pieces
   or a few primitive colliders (walls as boxes is fine and cheaper on the GPU).
10. Render check: after packaging, `render_asset` it and LOOK at `still.png` (Read the
    PNG). Black = texture missing; wrong colours = AO/albedo mix-up; floating/sunken =
    origin or collider wrong. Also look at `colliders.png`.

## Gate sequence per task (do them in this order, record evidence)

G1 choose → ledger row (`update_status.py --ledger "key | asset | url | license | scale-checked | used by"`)
and `PROVENANCE.md` in the asset dir (source URL, license, real dimensions vs the spec,
what you scaled/rotated and why).
G2 package → files, XML, constants; `render_asset` still looks right; budgets met.
G3–G5 → `verify_task.py --num-resets 1000` PASS (json under `docs/cl_v2/runs/<Task>/`).
Also `audit_workspace --keyword`, `benchmark_smoke --keyword`, and the two pytest files.
G6 → `render_asset` outputs copied into `docs/cl_v2/renders/<Task>/` (still.png,
colliders.png; the mp4 goes to the site, not git) and the task rollout rendered; publish
with `publish_v2.sh` together with `asset.json`:
```json
{"asset": "YCB 077 Rubik's cube", "source": "YCB", "source_url": "https://...",
 "license": "CC-BY 4.0", "sr_cl25": 1.0,
 "gates": {"G1":"ok","G2":"ok","G3":"ok","G4":"ok","G5":"ok","G6":"ok","G7":"ok"},
 "notes": "46 mm; grasp width 46 mm; teacher OBJ_CENTER_Z re-derived to 0.023"}
```
G7 → teacher adapted (any constant that names a half-extent / ride height / standoff /
face offset), then `test_classical --num-envs 32 --num-episodes 4` (n=128) on your GPU.
Record `cl25 → v2 (n, HEAD)` in the row. An unexplained drop of more than the cl25
noise band (~±0.05 at n=128) blocks G3/G4 until explained; a drop explained by the new
geometry (e.g. a wider object) is recorded, not hidden. Iterate the teacher at n=32 first.

Your task is DONE only when all seven gates read `ok` in STATUS.md, the site card shows
still + turntable + teacher rollout, and your LOGS entry has the numbers.

## Logging (append-only, your own file)

Append dated entries to `docs/cl_v2/logs/<your-agent>.md` using the template in
`LOGS.md` (Context / Did / Found / Decided / Next). The lead merges them. Never edit
`LOGS.md`, `STATUS.md` (except through `update_status.py`), `CODE_MAP.md`, `PLAN.md`,
`GOAL.md`, or another agent's rows/files. Shared code files (`env_cfgs.py`,
`commands.py`, `workspace.py`, `classical/*.py`) ARE edited by several agents: edit only
your task's function/constants, re-read the region right before each edit, keep edits
small, and never reformat a file.

## Traps (from cl25 and W0)

- degrees vs radians in MJCF; `jnt_range` from the COMPILED model is the truth.
- CoM moves with the mesh: topple / pivot / reorient thresholds and the `z=(h,h)` spawn
  heights are half-extents of the COLLISION geometry — read them from `_package.json`.
- Mechanism placement is set by the reset EVENT, not the command pose_range.
- `nconmax=60, njmax=650` on the ten articulated tasks; more collider pieces may need a
  bigger budget (raise it in that task's `*_env_cfg.py` SimulationCfg and say so).
- Textures: PNG only. Poly Haven glTF textures come through `baseColorTexture`; `arm`
  maps are AO/rough/metal packed — the pipeline's `--ao` takes the R channel.
- The gripper opens 80 mm; a pinch-grasped face must be ≤ 60 mm; Cage-Drag's object must
  be < 55 mm wide (aperture latch).
- `render_rollout` needs a working teacher; if your teacher is a characterised failure
  (cl25 SR 0.000) the site still needs `failure.mp4` + `result.json` — that is fine.
- Run `tests/test_classical_teachers.py` after ANY registry or teacher edit.
