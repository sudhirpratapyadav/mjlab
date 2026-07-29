# Cluster Port — Errors Faced & Fixes

Log of every problem hit porting the benchmark to the cluster (svs_ald A100) and how
it was solved. Local = RTX A6000 (sm_86); cluster = A100-SXM4-80GB (sm_80).

Cluster path: `/ihub/homedirs/svs_ald/sudhir/mjlab` (branch `benchmark-manip-diversity`).
GitHub: `sudhirpratapyadav/mjlab`. Sync = push local → `git pull` on cluster.

---

## 0. Environment

- venv is **uv-managed** (`~/.local/bin/uv`); no `pip` binary — use `uv pip ...`.
- versions identical local vs cluster: mujoco 3.3.x, warp-lang 1.11.0.dev20251124,
  mujoco-warp 0.0.1, torch 2.6.0+cu124.
- GPU access is **only inside a slurm holder job** (login node has no GPU). See
  `~/use_instructions/README.md`. Run pattern:
  ```bash
  squeue -a -o "%A %j %T %N %b" | grep -i hold        # find HOLDER_JOBID
  RUN_GPU=0 srun --jobid=<HOLDER_JOBID> --overlap --cpus-per-task=8 --export=ALL,RUN_GPU \
    bash -c 'export CUDA_VISIBLE_DEVICES=$RUN_GPU; cd ~/sudhir/mjlab; \
      PYTHONPATH=src .venv/bin/python -m mjlab.scripts.benchmark_smoke --isolate'
  ```

---

## 1. Segfault (exit 139) — dexterous-hand MESH collision

**Symptom.** `Mjlab-*-Leap` tasks: `Segmentation fault (core dumped)`, exit 139, at
env init. faulthandler traceback ended in
`mujoco_warp/_src/collision_driver.py::_narrowphase` inside `sim.py::create_graph`
(CUDA-graph capture). "built" never printed. Worked fine on local A6000.

**Diagnosis.** LEAP hand = 88 geoms, incl. 4 collidable **mesh** geoms (17 more
visual-only). Disabling mesh collision (`contype=conaffinity=0`) → builds + steps clean.
So the crash is mujoco-warp's mesh-mesh narrowphase kernel on this GPU.

**Root cause.** mujoco-warp JIT-compiles CUDA kernels per GPU arch. The mesh-collision
kernel miscompiles / crashes on **sm_80 (A100)** but not sm_86 (A6000). Same code,
same warp version — a GPU-arch-specific kernel bug.

**Fix.** `asset_zoo/robots/leap_hand/leap_constants.py::get_spec` and
`franka_leap/franka_leap_constants.py::get_spec`: set `contype=conaffinity=0` on all
mesh geoms. Per-phalanx **box** colliders (already in the model) preserve contact.
Fingertip-mesh collision fidelity = stage-two TODO (add primitive fingertip colliders).

---

## 2. Segfault (exit 139) — CYLINDER / ELLIPSOID collision

**Symptom.** `Lift-Cylinder-Franka`, `Push-Disc-Franka`, `Lift-Ellipsoid-Franka`
segfault at init, same `_narrowphase` → `create_graph` path. Cube/sphere tasks pass.

**Diagnosis.** Geom-type dump: failing objects use geom **type 5 (CYLINDER)** (disc is
a short cylinder) or **ELLIPSOID**; their mocap-goal markers use the same. Passing
tasks (cube/sphere) use box(6)/sphere(2). Confirmed the crashing kernel is the
cylinder/ellipsoid narrowphase on sm_80.

**Root cause.** Same as §1: mujoco-warp cylinder/ellipsoid collision kernel bug on
sm_80. Not overflow (bumping `nconmax` didn't help), not version, not config.

**Fix.** Swap the object + mocap-goal geom to **capsule** (a rounded cylinder — a
warp-safe primitive, and still a distinct non-box grasp shape):
- `objects/free/cylinder/{xmls/cylinder.xml, cylinder_constants.py}` → capsule
- `objects/free/disc/{xmls/disc.xml, disc_constants.py}` → capsule
- `objects/free/ellipsoid/{xmls/ellipsoid.xml, ellipsoid_constants.py}` → capsule
  (long axis via `fromto`, radius = short semi-axis)

Safe to change: no teacher datasets existed for these objects. obs/action dims
unchanged (60 / 8), so the manifest is unaffected.

---

## 3. Segfault at process exit (harmless) & multi-env-in-one-process crash

**Symptom A.** `benchmark_smoke` printed nothing then exited 139, even for tasks that
pass individually — when run as `python -m ... :main`.

**Diagnosis.** Two separate things:
1. **Teardown segfault:** warp + torch CUDA contexts can segfault during interpreter
   shutdown after a clean run.
2. **Multi-env accumulation:** building/destroying MANY warp envs in one process
   corrupts CUDA state and segfaults mid-sweep, even though each task passes alone.
   (A single task, or 2-3 in a row, is fine; ~20 is not.)

**Fix.**
1. `scripts/benchmark_smoke.py::main` → `os._exit(code)` after flushing, to skip the
   crashing teardown and return an honest exit code.
2. Added `benchmark_smoke.py --isolate`: runs each task in a **fresh subprocess**
   (`_smoke_test_subprocess`), so one env per process. **Always use `--isolate` for a
   full sweep on the cluster.** Slower (warp recompiles per subprocess) but robust.

---

## 4. Other porting gotchas

- **LEAP mesh path bug** (not GPU-related): the vendored LEAP XML had
  `meshdir="./assets/"` → malformed asset keys (`./assets//x.obj`) → MuJoCo looked in
  `assets/robot/`. Fixed to `meshdir="assets"` (franka convention).
- **Contact buffers:** dexterous-hand envs have many finger contacts; raised
  `nconmax=800, njmax=3000` in the hand common helpers (arm bases used 200/1000).
  (Necessary for headroom, but did NOT fix the §1/§2 segfaults — those are kernel bugs.)
- **Cluster `sudhir/mjlab` was a plain copy (no .git).** Converted to a git checkout:
  `git init` + `git remote add origin <gh>` + `git fetch` + `git checkout -f -b <branch>
  FETCH_HEAD`. Untracked `.venv` (8.2G) / wandb / teacher_datasets preserved (gitignored).

---

## Result

**20/20 tasks pass `benchmark-smoke --isolate` on the A100 cluster** (and locally).
All fixes keep obs/action dims and a distinct grasp geometry per task.

## Stage-two follow-ups
- Upgrade mujoco-warp when the sm_80 cylinder/ellipsoid/mesh collision bug is fixed
  upstream, then restore true cylinder/ellipsoid geoms + LEAP fingertip mesh colliders.
- Or: add primitive (capsule/sphere) fingertip colliders to the LEAP hand for grasp
  fidelity without relying on mesh collision.
