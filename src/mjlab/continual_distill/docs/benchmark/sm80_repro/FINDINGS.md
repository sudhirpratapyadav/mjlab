# sm_80 collision segfault — isolated root-cause investigation

Date: 2026-07-29. Node: dgx2, NVIDIA A100-SXM4-80GB (**sm_80**), driver 550.54.15 (CUDA 12.4).
Baseline env: warp 1.11.0.dev20251124, mujoco-warp 0.0.1 (git rev `46b4421`), mujoco 3.3.7.

## TL;DR

The crash is **real and reproducible**, but the previously documented root cause was
**wrong**. It is *not* a miscompiled cylinder/ellipsoid/mesh collision kernel on sm_80.

> **The kernels are fine. The bug is in CUDA-graph capture of the convex/CCD narrowphase,
> and it is already fixed upstream — warp >= 1.14 passes 14/14.**
>
> **STATUS: RESOLVED.** The upgrade has been applied and both geom workarounds reverted
> (commit fb15756) — see [Resolution](#resolution--upgrade-applied-commit-fb15756).

Evidence: every geom that crashes under `wp.ScopedCapture()` runs **perfectly when
stepped eagerly** on the same GPU, same kernels, same data.

## How it was isolated

Pure mujoco-warp, no mjlab: a two-body MJCF (static partner + one free object), pushed
through `put_model`/`put_data`/`step`. One case per subprocess, since SIGSEGV is fatal.

- `repro_geom_collision.py` — single case; `--geom`, `--partner`, `--no-graph`
- `run_matrix.py` — full geom × partner matrix, writes `results.json`
- `repro_leap_mesh.py` — the real LEAP hand XML with mesh colliders re-enabled
- `probe_capture_alloc.py`, `probe_warmup.py`, `probe_capture_modes.py`, `probe_smem.py`
  — hypothesis probes (all negative, see below)

## Result matrix (baseline warp 1.11-dev, A100 sm_80)

| free geom | vs box | vs mesh | eager (no graph) |
|---|---|---|---|
| box | PASS | **SEGV** | PASS |
| sphere | PASS | **SEGV** | PASS |
| capsule | PASS | **SEGV** | PASS |
| cylinder | **SEGV** | **SEGV** | PASS |
| ellipsoid | **SEGV** | **SEGV** | PASS |
| disc (short cyl) | **SEGV** | **SEGV** | PASS |
| mesh | **SEGV** | **SEGV** | PASS |

- **With graph capture: 3/14 pass.** Only box/sphere/capsule vs box survive.
- **Without graph capture (`--no-graph`): 14/14 pass.** ← the decisive result.
- Any pair involving a **mesh** crashes regardless of the other shape.

The split is exactly the narrowphase split: box/sphere/capsule pairs use the analytic
**primitive** narrowphase; cylinder/ellipsoid/mesh route through the **convex/CCD**
(GJK/EPA) path. Crash frame is always:

```
warp/_src/context.py:6675 launch
mujoco_warp/_src/collision_convex.py:1027 convex_narrowphase   <-- CCD kernel launch
mujoco_warp/_src/collision_driver.py:693 _narrowphase
mujoco_warp/_src/forward.py:519 fwd_position -> step
```

Note this is one level deeper than the old note claimed: `collision_convex.py`, not a
generic `_narrowphase` failure.

## Hypotheses tested and ruled out

| # | Hypothesis | Test | Verdict |
|---|---|---|---|
| 1 | Miscompiled cylinder/ellipsoid/mesh kernel on sm_80 | `--no-graph` | **REFUTED** — 14/14 pass eagerly |
| 2 | Illegal GPU alloc during capture (`convex_narrowphase` does ~20 `wp.empty()` per call) | `probe_capture_alloc.py` | REFUTED — mempool supported+enabled; `wp.empty()` inside capture is fine |
| 3 | Lazy JIT / module load inside capture (CCD kernels are built per geom-pair on first use) | `probe_warmup.py` — eager step + `force_load` before capture | REFUTED — still segfaults |
| 4 | Shared-memory over-request on sm_80 | `probe_smem.py` | REFUTED — `forward_smem_bytes = 0` |
| 5 | Capture stream/mode issue | `probe_capture_modes.py` (`stream`, `synced`) | REFUTED — all variants segfault |
| 6 | Contact-buffer overflow (`nconmax`) | earlier work, re-checked | REFUTED — raising limits doesn't help |

Useful side-finding from probe 5: warp **refuses** `verify_cuda` during capture
(`RuntimeError: Cannot use CUDA error verification during graph capture`). So capture
runs with error checking off — an otherwise-catchable launch failure surfaces as a bare
SIGSEGV. That is why this looked like a kernel miscompile.

Environment anomaly, likely the underlying trigger: warp JITs against **CUDA Toolkit
12.9** while the node driver only provides **CUDA 12.4**. Graph capture is the part of
the API most sensitive to that gap.

## Does upgrading help? — YES

Isolated venvs, same node, same repro:

| warp | mujoco-warp | matrix (with capture) | LEAP hand, 21 mesh colliders |
|---|---|---|---|
| 1.11.0.dev20251124 (current) | 0.0.1 | **3/14** | **SEGFAULT** |
| 1.14.0 | 3.11.0 | — | **PASS** |
| 1.15.0 | 3.11.0 | **14/14 PASS** | **PASS** |

The LEAP test is the strongest evidence: it loads the real
`asset_zoo/robots/leap_hand/xmls/leap_right_hand.xml`, **re-enables collision on the 21
mesh geoms** that the workaround currently disables, and graph-captures a step. Old warp
segfaults; warp 1.14/1.15 passes.

Root cause is therefore an upstream warp/mujoco-warp graph-capture bug, already fixed.
**No kernel patching is needed or warranted** — patching mujoco-warp's CCD kernels would
be fixing the wrong layer.

## What this means for the benchmark

The two workarounds were **valid but cost real fidelity** — the suite advertised
cylinder/ellipsoid/disc grasping while actually simulating capsules, and the dexterous
hand grasped with box phalanges. Both have since been **reverted** (see Resolution
below):

1. `leap_constants.py` / `franka_leap_constants.py`: `contype=conaffinity=0` on mesh
   geoms → reverted, restoring true fingertip mesh collision fidelity.
2. `cylinder` / `disc` / `ellipsoid` objects swapped to **capsule** → reverted,
   restoring genuinely distinct grasp geometries (which matters for a benchmark whose
   whole point is shape diversity).

`benchmark-smoke --isolate` remains useful regardless — the multi-env-per-process CUDA
state corruption (§3 of CLUSTER_TROUBLESHOOTING) is a separate issue, not retested here.

## Resolution — upgrade applied (commit fb15756)

Done, in the order above: validated in a cloned scratch venv first, then applied to the
pinned env. `pyproject.toml` now requires `mujoco-warp>=3.11`, `warp-lang>=1.14`,
`mujoco>=3.11` (the git-rev pin is gone; the `mujoco<=3.3.8` cap had to be lifted
because mujoco-warp 3.11 requires mujoco 3.11).

**The upgrade was not drop-in — two real API breaks had to be fixed:**

1. **`opt.ls_parallel` was removed in MuJoCo Warp 3.9.1** and raises `AttributeError`
   on both get and set. `Simulation` now applies it best-effort. Caused all 20 tasks to
   fail instantly.

2. **`WarpBridge`/`TorchArray` silently stopped broadcasting shared model arrays.** The
   1 → `nworld` expansion was gated on `stride(0) == 0`. That held only because
   mujoco-warp <= 0.0.1 built these arrays as zero-stride broadcasts; 3.x allocates
   them with real strides (`jnt_range` stride(0): **0 → 2**, `body_pos`: 0 → 6). The
   gate silently skipped the expansion, leaving `soft_joint_pos_limits` at
   `(1, njnt, 2)` where `(nworld, njnt, 2)` was expected → CUDA device-side assert on
   the first per-env index in `reset_joints_by_offset`. Now keyed on `shape[0] == 1`
   alone. Only the *model* bridge passes `nworld`, so per-world `Data` arrays are
   untouched.

   This one is worth remembering: a stride-dependent broadcast is a **silent
   correctness** trap, not just a crash — anything else indexing shared model arrays
   per-env would have been quietly wrong.

**Both workarounds reverted:**
- `cylinder` / `disc` / `ellipsoid`: capsule → real `CYLINDER` / `CYLINDER` /
  `ELLIPSOID` geoms (verified via `mjtGeom` after compile).
- LEAP hand + Franka-LEAP: mesh collision re-enabled (the `contype=conaffinity=0` loops
  are gone), restoring true fingertip contact instead of box phalanges.

**Two tests needed fixing.** `test_sim` asserted the removed `ls_parallel`. And
`test_builtin_sensor::test_accelerometer_sensor` stepped 100 times then asserted
`|accel| > 0` — but the base starts at z=1.0 and is still in **free fall** at step 100
(z≈0.80), where a proper accelerometer correctly reads ~0. It had only ever passed by
catching a ~1e-15 float artifact on the sampled step. Both stacks produce *identical*
trajectories, so this was a latent bad test, not a behaviour change; it now steps until
the robot lands and asserts a real ground reaction (~g), passing on old and new alike.

**Validation on A100 (sm_80):** 20/20 `benchmark-smoke --isolate` with real geoms and
live mesh colliders; **325/325 pytest**; obs/action dims unchanged for all 20 tasks;
`test_sm80_graph_capture.py` 14/14 (the 4 convex cases went xfail → pass, so they are
now hard assertions rather than expected failures).

Known cosmetic follow-up: MuJoCo Warp warns `MULTICCD is enabled, but the scene
contains CCD pairs without multicontact support: [('CYLINDER','BOX')]` (≤1 contact for
those pairs). It fires from test fixtures, not the benchmark tasks, but is worth a look
if cylinder grasp stability matters at train time.

## Reproducing

```bash
squeue -a -o "%A %j %T %N %b" | grep -i hold     # find HOLDER_JOBID
RUN_GPU=0 srun --jobid=<HOLDER_JOBID> --overlap --cpus-per-task=8 --export=ALL,RUN_GPU \
  bash -c 'export CUDA_VISIBLE_DEVICES=$RUN_GPU; cd ~/sudhir/mjlab; \
    .venv/bin/python src/mjlab/continual_distill/docs/benchmark/sm80_repro/run_matrix.py'

# the decisive contrast:
#   ... run_matrix.py              -> 3/14
#   ... run_matrix.py --no-graph   -> 14/14
```
