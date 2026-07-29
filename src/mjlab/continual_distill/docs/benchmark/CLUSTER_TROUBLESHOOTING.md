# Cluster Port — Errors Faced & Fixes

Log of every problem hit porting the benchmark to the cluster (svs_ald A100) and how
it was solved. Local = RTX A6000 (sm_86); cluster = A100-SXM4-80GB (sm_80).

> **⚠ RESOLVED — §1 and §2 below are HISTORICAL (2026-07-29/30).** They attribute the
> segfaults to miscompiled cylinder/ellipsoid/mesh *collision kernels* on sm_80. An
> isolated repro disproved that: those kernels run **fine** — the crash only happened
> under **CUDA-graph capture** of the convex/CCD narrowphase, and it was **already
> fixed upstream**.
>
> **The fix has been applied (commit fb15756).** mjlab now requires
> `mujoco-warp>=3.11` / `warp-lang>=1.14` (mujoco-warp was pinned to git rev
> `46b4421` = v0.0.1), and **both workarounds are reverted**: cylinder/disc/ellipsoid
> are real CYLINDER/ELLIPSOID geoms again, and LEAP mesh collision is re-enabled.
> Validated on A100: 20/20 `benchmark-smoke --isolate`, 325/325 pytest.
>
> See **`sm80_repro/FINDINGS.md`** for the investigation and
> `tests/test_sm80_graph_capture.py` for the regression guard.

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

## Upgrade — DONE (commit fb15756)

The upgrade landed. What it took, beyond the version bump:

- **`opt.ls_parallel` was removed in MuJoCo Warp 3.9.1** (raises `AttributeError` on
  get *and* set). `Simulation` now sets it best-effort, so mjlab works either side of
  the change.
- **`WarpBridge`/`TorchArray` silently stopped broadcasting shared model arrays.** The
  1 → `nworld` expansion was gated on `stride(0) == 0`, which only held because old
  mujoco-warp built those arrays as zero-stride broadcasts; 3.x allocates them with
  real strides (`jnt_range` stride(0): 0 → 2). The gate silently skipped the
  expansion, so `soft_joint_pos_limits` stayed `(1, njnt, 2)` instead of
  `(nworld, njnt, 2)` → device-side assert on the first per-env index. Now keyed on
  `shape[0] == 1` alone. **This was a silent-correctness bug, not just a crash** —
  worth remembering if other shared model arrays are indexed per-env.
- Two tests needed fixing; one (`test_accelerometer_sensor`) turned out to be a latent
  bad test that only ever passed on a ~1e-15 float artifact. See the commit message.

Result: 20/20 `benchmark-smoke --isolate` and 325/325 pytest on A100, with **real**
cylinder/ellipsoid geoms and **live** LEAP mesh colliders.

## Remaining stage-two follow-ups
- MuJoCo Warp warns `MULTICCD is enabled, but the scene contains CCD pairs without
  multicontact support: [('CYLINDER','BOX')]` — at most 1 contact for those pairs.
  Harmless for smoke (it fires from test fixtures, not the benchmark tasks), but worth
  a look if cylinder grasp stability matters at train time.
- `benchmark-smoke --isolate` is still recommended on the cluster (§3 — the
  multi-env-per-process CUDA state corruption is a separate issue, not retested here).
