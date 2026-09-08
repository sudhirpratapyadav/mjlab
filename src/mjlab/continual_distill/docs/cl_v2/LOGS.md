# CL-V2 — LOGS

> Append-only. Dated entries, newest at the bottom. What was tried, what was measured,
> what was decided and why. The *current* state of a task lives in `STATUS.md`, not here.

## Entry template

```
### YYYY-MM-DD — <Task or topic> — <agent/owner>
**Context:** what you set out to do, HEAD, holder job + GPU index.
**Did:** steps, commands, files touched.
**Found:** measurements with n; screenshots/histograms linked from runs/ or renders/.
**Decided:** anything that changes a STATUS.md row or a plan item, and why.
**Next:** the one thing the next person should do.
```

---

### 2026-09-09 — Program created — lead

**Context:** New program alongside cl25 (which is read-only from here). cl25 Phase 1
resolved all 25 teacher rows on HEAD `1127d12`; its summary block is stale but the
per-task rows are the baselines copied into `STATUS.md`.

**Did:** Created `docs/cl_v2/` with `GOAL.md`, `PLAN.md`, `STATUS.md`, this file, and
empty `runs/` and `renders/`. No task work started.

**Found (tooling, login node):** `mujoco 3.11.1` + `mujoco_warp` — mesh collision
present (convex per geom). `trimesh 4.8.3` in venv. `coacd` and `obj2mjcf` NOT
installed. No Blender on the cluster. Outbound HTTPS works from the login node.
`git-lfs 2.13.3` present, not enabled; `asset_zoo` is 82 MB. All 26 current object
XMLs are pure primitives, no meshes/textures. Franka gripper opens 80 mm. One holder
running: `hold_dgx_amit` (20277) on dgx1 — not ours, reuse per the cluster rules.

**Decided:** Task interface (body/site/joint names) frozen; geometry free. Four
decisions parked in `STATUS.md` (D1–D4).

**Next:** W0 owner installs `coacd`, decides D1/D2, and stubs `verify_task.py`.
Asset selection (G1) for all 25 can begin in parallel.

### 2026-09-09 — Scope clarified, D1–D4 decided — lead

**Decided (user direction):** primary goal is realistic-looking assets with correct
kinematics and physics. Assets may be downloaded (any license) or Blender-built.
Classical teachers MAY be changed to fit new assets. Small geometry tweaks allowed if
the motion profile is unchanged and the task still earns its name. Open questions are
decided by the row owner and logged, not escalated.
D1 headless Blender in `~/tools/`; D2 git-lfs for processed binaries; D3 any license
OK; D4 Peg-Insertion uses real toy geometry, clearance re-derived.

**Next:** unchanged — W0 owner installs `coacd`, fetches Blender, stubs `verify_task.py`.

### 2026-09-09 — W0 infrastructure landed — lead

**Context:** HEAD `1127d12` + uncommitted W0 work. Holder `hold_dgx_amit` (20277) on dgx1;
GPUs 4–7 busy (~38 GB each, someone's training), GPU 0 lightly used, **GPUs 1–3 free** —
CL-V2 uses 1–3 only. Login node has no GL (no OSMesa; EGL needs the GPU node), so every
render is `srun --jobid=20277 --overlap` with `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0`
(device id is relative to `CUDA_VISIBLE_DEVICES`). The verifier and all mesh work run on
the login node CPU (`ManagerBasedRlEnv(device="cpu")` works).

**Did:**
- `src/mjlab/scripts/asset_pipeline.py` — fetch (polyhaven | gso | ycb | ambientcg |
  polyhaven-texture), inspect, package (normalise → Blender decimate → CoACD → PNG
  texture with AO baked → `<asset>/<geom>/<inertial>` snippets + `_package.json`).
  Verified end to end on YCB `036_wood_block` (16k tris, 1 hull) and Poly Haven
  `cardboard_box_01` (17k→6k tris via Blender, 6 hulls, PBR base-colour path).
- `src/mjlab/scripts/verify_task.py` — G3+G4+G5 battery; PASS on the four primitive
  baselines tried (Lift-Cube, Open-Drawer, Push-Flap, Place-In-Container, n=100).
- `src/mjlab/scripts/render_asset.py` — still + turntable + collider view.
- `src/mjlab/tasks/manipulation/studio.py` + `registry.edit_registered_cfgs` — studio
  rig on all Franka tasks (render-only; `MJLAB_NO_STUDIO=1` disables). 33 scene/config
  tests still pass on CPU.
- Site: `docs/cl_v2/site/index.html` published at https://cl.untuai.com/v2/ ;
  `docs/cl_v2/publish_v2.sh <Task-Id> <dir>` pushes a task folder.
- `docs/cl_v2/update_status.py` — flock-protected STATUS row editor + summary recompute.
- Tooling: coacd 1.0.14 + matplotlib via `uv add --group dev`; Blender 4.2.23 LTS in
  `~/tools/blender`; raw downloads in `~/assets_raw/`; GPU-side scratch in `~/cl_v2_work/`
  (the session scratchpad is on the login node's /tmp and NOT visible from dgx1).
- `CODE_MAP.md` — the plumbing brief every task agent reads first.

**Found:** MuJoCo 3.11 rejects JPG textures (PNG only). glTF from Poly Haven loads in
trimesh as a PBRMaterial (`baseColorTexture`), not `material.image`. Poly Haven "arm"
maps are AO/rough/metal packed in R/G/B — baking the mean made the box black.
`mujoco_warp` 3.11 supports every mesh collision pair we need (convex only).
SAPIEN/PartNet is unreachable from the cluster → all mechanisms are built.

**Decided:** D5–D8 (see STATUS). Wave split: W1-a cube/cuboid/block family (7 tasks),
W1-b container/puck/can family (5), W1-c insertion/edge (4), W2-a plate-mounted (5),
W2-b cabinet family (4). GPU assignment: W1-a→1, W1-b→2, W1-c→3, W2-a→1, W2-b→2.

**Next:** launch the five wave agents with `AGENT_BRIEF.md`.
