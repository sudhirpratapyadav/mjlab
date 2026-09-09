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


## Wave logs (merged by the lead, 2026-09-09; originals in logs/)

<!-- ===== W1-a ===== -->
# CL-V2 — W1-a log (Lift-Cube, Stack-Cube, Push-Cuboid, Drag-Pull, Topple-Block, Cage-Drag, Reach-Target)

Append-only. Template per `LOGS.md`: Context / Did / Found / Decided / Next.

---

### 2026-09-09 — W1-a kickoff: asset survey and the three shared meshes — W1-a

**Context:** HEAD `1127d12` + the lead's uncommitted W0 work. Holder `hold_dgx_amit`
(20277) on dgx1, **GPU index 1** (mine). Seven rows: Lift-Cube, Stack-Cube,
Push-Cuboid, Drag-Pull, Topple-Block, Cage-Drag, Reach-Target. Three shared meshes
carry all six object rows: `free/cube` (Lift, Stack-object, Cage-Drag, and W1-b's
Place-In-Container + Throw-To-Bin), `free/cuboid` (Push-Cuboid, Drag-Pull,
Stack-base) and `free/block` (Topple).

**Did:** Read AGENT_BRIEF / GOAL / PLAN / CODE_MAP §3-§4 / STATUS. Fetched the YCB
candidates (`077_rubiks_cube`, `009_gelatin_box`, `003_cracker_box`, `004_sugar_box`,
`008_pudding_box`) into `~/assets_raw/ycb/`. Measured each with a yaw-minimising
bounding-box search (`~/cl_v2_work/W1-a/find_yaw.py`) because every YCB scan is
parked at an arbitrary yaw.

**Found (de-rotated real extents, mm):**

| YCB | best yaw | x | y | z | published mass |
|---|---|---|---|---|---|
| 077_rubiks_cube | 63.00 deg | 59.0 | 58.1 | 57.9 | 0.094 kg |
| 009_gelatin_box | 76.85 deg | 89.2 | 72.9 | 30.1 | 0.097 kg |
| 003_cracker_box | 0.20 deg | 71.8 | 163.9 | 213.4 | 0.411 kg |
| 004_sugar_box | 1.75 deg | 47.9 | 93.3 | 176.0 | 0.514 kg |
| 008_pudding_box | 62.40 deg | 89.7 | 112.7 | 38.9 | 0.187 kg |

All five scans are already at REAL scale (the Rubik's cube reads 57.9 mm against the
57 mm standard 3x3; the sugar box reads 93 x 176 against the published 95 x 176), so
no unit correction is needed — only de-rotation and, for the cube, an honest
down-scale to a smaller real product.

**Decided (G1 asset choices, three decisions the lead should know about):**

1. **cube = YCB `077_rubiks_cube` scaled to 46 mm** (0.7797x). Rationale: the scanned
   57 mm cube is a real 3x3 but 57 mm > Cage-Drag's 55 mm aperture latch
   (`aperture_min=0.055`), which would make Cage-Drag unwinnable by construction.
   46 mm is a real "mini 3x3" product size, keeps the widest face at 46 mm (well under
   the 60 mm pinch limit that `verify_task`'s `graspable` check enforces) and leaves
   9 mm of latch margin. Mass 0.050 kg = the published 0.094 kg at 57 mm carried across
   by volume (density 508 kg/m^3) — which coincidentally equals the primitive's mass,
   so Lift/Stack/Place/Throw/Cage dynamics keep their baseline.
2. **cuboid = YCB `009_gelatin_box` at native scale, lying flat** (89.2 x 72.9 x
   30.1 mm vs the primitive's 80 x 80 x 30). Mass 0.097 kg, the published product mass
   — nearly 2x the primitive's unphysical 0.050 kg. NOT xy-square any more, so the
   push/drag half-width along the push direction now varies 36.5..44.6 mm; handled in
   the teachers with a direction-aware support width rather than a single constant
   (see the Push-Cuboid entry).
3. **block = YCB `003_cracker_box` standing, NON-UNIFORMLY scaled to 100 x 160 x
   210 mm** (x1.3928, y0.9762, z0.9841). This is a deviation from "uniform scale" and
   the reason is the task's premise: Topple-Block exists because *every* face is wider
   than the gripper can close on (`block.xml`: "force closure is impossible by
   construction"), and the measured open-pad gap is 95 mm. A real cracker box is only
   71.8 mm deep — graspable — which would silently convert a poke task into a pick
   task, i.e. change the motion profile. Stretching x to 100 mm restores exactly the
   primitive's ungraspable width and keeps `FACE_HALF_X = 0.05` (and the hard-coded
   0.05 in `tests/test_class_a_wave1.py::test_topple_scores_either_landing_face`)
   valid unchanged. The stretch only affects the two narrow side faces of the box, so
   the printed front/back faces are undistorted. Mass 0.550 kg = the published 0.411 kg
   at the published volume carried across by volume (density 164 kg/m^3, a realistic
   full cereal/cracker box) — 3.7x the primitive's 0.15 kg, which is a real physics
   change for the topple teacher and is measured, not assumed.

Origin for all three: **bbox centre**, not mesh centroid. Every downstream constant in
CODE_MAP §3 (`z=(h,h)` spawn heights, `stack_height`, `_ENGAGE_INSET`, `_half`,
`_travel`, `goal_z_height`, teacher half-widths) is expressed against the *geometric*
centre of the collision box; putting the body frame anywhere else silently biases all
of them. Collider `box` for all three (a box collides as a box — cheapest on GPU and
keeps `verify_task`'s step-time ratio at ~1.0).

**Next:** package cube first (W1-b is blocked on `cube.xml`), then cuboid, then block.

### 2026-09-09 — G2-G5 green on all seven rows; four pre-existing G5 defects fixed — W1-a

**Context:** HEAD `1127d12` + W0. Holder 20277, **GPU 1** for renders; all mesh work,
`verify_task` and the audits on the login node CPU.
**TRAP for everyone: `nvidia-smi` is ALIASED in the shell profile to
`srun -p 1gpu --gres=gpu:1 --time=02:00:00 --pty bash && nvidia-smi`** — typing it
queues a fresh 2-hour GPU allocation instead of printing anything. It queued job
20298 for me; I cancelled it immediately (mine, no one else's). Use
`srun --jobid=20277 --overlap -n1 bash -c 'command nvidia-smi ...'`.

**Did:**
- Packaged the three shared meshes with `asset_pipeline` (`--collider box`,
  `--origin bbox`, 12k tris, 1024^2 PNG). `block` needed the Python API for its
  non-uniform scale: `~/cl_v2_work/W1-a/package_block.py`.
- Wrote `cube.xml`, `cuboid.xml`, `block.xml` with the `<compiler meshdir="assets"
  texturedir="assets"/>` + `<asset>` + visual-mesh/box-collider pattern, keeping every
  body / joint / geom / site name (`cube_geom`, `cuboid_geom`, `block_geom` are kept
  on the collider) and every `condim`/`friction`/`solref`/`contype`/`conaffinity`.
  Added the Franka assets-dict pattern and `*_HALF_EXTENTS` / `*_HALF_HEIGHT`
  constants to each `<asset>_constants.py`, re-exported from each `__init__.py`, and
  made every env-cfg spawn height import them instead of repeating a literal.
- `PROVENANCE.md` in all three asset dirs; three ledger rows in STATUS.
- `render_asset` on GPU 1 for all three and LOOKED at `still.png` + `colliders.png`:
  Rubik's cube, Jell-O carton and Cheez-It box, all textured, correctly scaled,
  resting exactly on the plane, colliders coincident with the visual mesh.
- `verify_task --num-resets 1000` on all seven rows; `audit_workspace --keyword` on
  all seven (0 problems each).

**Found (measurements):**

| Task | mass | grasp width | settle/drop | step ratio | spawn radial | success@reset | oracle |
|---|---|---|---|---|---|---|---|
| Lift-Cube | 0.050 | 0.0452 | ok/ok | 1.29 | 0.304-0.538 | 0/1000 | 1.00 |
| Stack-Cube | 0.050 + 0.097 | 0.0452 | ok/ok | 1.22 | 0.288-0.530 | 0/1000 | 1.00 |
| Push-Cuboid | 0.097 | n/a | ok/ok | 1.19 | 0.301-0.456 | 0/1000 | 1.00 |
| Drag-Pull | 0.097 | n/a | ok/ok | 1.22 | 0.397-0.507 | 0/1000 | 1.00 |
| Topple-Block | 0.550 | n/a | ok/ok | n/a | 0.371-0.467 | 0/1000 | 1.00 |
| Cage-Drag | 0.050 | n/a | ok/ok | 0.85 | 0.302-0.510 | 0/1000 | 1.00 |
| Reach-Target | — | — | — | — | goal 0.403-0.738 | 0/1000 | n/a |

**Found — FOUR PRE-EXISTING DEFECTS, all in the init distribution, none caused by the
asset swap** (every one of them is geometry-independent and was therefore present on
the primitives at `1127d12`; `verify_task` is new, which is why they surface now):

1. **Push-Cuboid `success_at_reset = 2/1000`.** The object band `x=(x_lo, x_mid)` and
   the goal band `x=(x_mid, x_hi)` MEET at `x_mid` while both span the full
   `GRASP_Y_RANGE`, so a reset can draw the two within the 2 cm success radius.
2. **Drag-Pull `success_at_reset = 5/1000`.** Same defect, mirrored.
3. **Cage-Drag `success_at_reset = 37/1000` (3.7%).** Worse because its object box and
   its goal box are *the same box* — deliberately, since caged transport is
   omnidirectional.
4. **Reach-Target `success_at_reset = 5/1000` AND a G4 goal-bound failure.** The
   default reach box (x 0.40-0.70, y +-0.25, z 0.15-0.50) *contains* the Franka's
   reset EE pose, measured at (0.6774, 0.0000, 0.3819) for `HOME_QPOS`. Separately,
   `audit_workspace._GOAL_EXEMPT` gave Reach a two-sided band `[0.60, 0.75]` copied
   from the tool-pull / throw-to-bin pattern; the audit only ever compares the MAX
   radial so it never fired, but `verify_task` checks every sampled goal and the near
   end of the reach band is 0.403 by design.

**Decided (fixes; each is on a row I own, and each is opt-in for everyone else):**
- Push-Cuboid: a 4 cm dead band between the two x halves (2x the 2 cm success radius).
  Drag-Pull: 5 cm (its radius is 3 cm). Costs ~2 cm of each band; the shortest legal
  push/pull is now 2x the success window instead of 0.
- Cage-Drag: new **opt-in** `PushingCommandCfg.min_goal_distance` (default **0.0**, so
  push-disc / strike-slide / push-cuboid / drag-pull are bit-identical), enforced by
  bounded REJECTION of the goal — a pushed-away goal would walk outside the range the
  workspace audit checked. Set to 0.06 in `cage_drag_env_cfg.py`. This keeps the
  omnidirectional character a half-split would destroy.
- Reach-Target: new **opt-in** `ReachingCommandCfg.min_gripper_clearance` (default
  0.0), set to 0.10 in the Franka cfg. Verified empirically that `site_pos_w` IS fresh
  inside `_resample_command` (0.6774 there vs 0.6772 after reset + one zero step),
  because the robot's reset event runs before command resampling — the opposite of the
  `_spawn_object` staleness trap, so this is safe to read.
- `audit_workspace._GOAL_EXEMPT` Reach floor 0.60 -> 0.0, with the reason written into
  the exemption text: Reach's exemption is ONE-SIDED (its band spans the envelope on
  purpose), unlike tool-pull's and throw-to-bin's.
All four re-verified at 1000 resets: **0/1000 success-at-reset, G3/G4/G5 PASS on all
seven rows.**

**Also re-derived (full list in the final report):** stack_height 0.035 -> 0.0376;
`_ENGAGE_INSET` 0.04 -> 0.0365; topple `_half` 0.07 -> 0.08, `_travel` 0.09 -> 0.105,
spawn z 0.09 -> 0.105; cube spawn z / cage `goal_z_height` 0.02 -> 0.0226; every mocap
goal marker resized to its asset; teachers `OBJ_CENTER_Z` 0.020 -> 0.0226 (lift),
`HALF_WIDTH` 0.02 -> 0.023 + `RIDE_HEIGHT` 0.026 -> 0.028 (cage), `HALF_WIDTH`
0.04 -> 0.0405 (push_cuboid, drag_pull), `CONTACT_Z` 0.055 -> 0.064 (topple).
`aperture_min` 0.055 re-checked and KEPT: it bounds the finger-joint sum, and the pad
gap at 0.055 is ~0.070 m against a 0.046 m cube.

**Next:** G6 rollouts + publish, then the n=128 G7 sweep on GPU 1.

### 2026-09-09 — Teacher probes at n=32; Cage-Drag delta attributed — W1-a

**Context:** GPU 1 (shared with W2-a per the lead's assignment — both of us had jobs
on it, which is fine on an 80 GB A100). n=32 x 1 episode, `test_classical`, the
`play=False` config `render_rollout` also uses, so the numbers are directly
comparable to the n=128 runs.

**Found — Cage-Drag 0.141 (cl25, n=128) -> 0.031 (v2, n=32). ATTRIBUTED, and it is
mostly not the asset:**

| configuration | SR | n |
|---|---|---|
| cl25: 40 mm primitive cube, no goal/object separation | 0.141 | 128 |
| v2 asset (46 mm cube), RIDE_HEIGHT 0.028, **min_goal_distance 0.0** | **0.094** | 32 |
| v2 asset, RIDE_HEIGHT 0.028, min_goal_distance 0.06 (shipped) | **0.031** | 32 |

The asset swap costs ~0.05 and sits inside the n=32 noise band; **the separation
guard costs the rest, and it is a correctness fix, not a regression.** cl25's
Cage-Drag sampled the object and the goal from the SAME box, so ~16% of episodes had
a transport shorter than 6 cm and 3.7% were already successful at t=0. Those
degenerate episodes are the easiest ones in the distribution, so they carried a large
share of the 0.141. Removing them (G5 requires success-at-reset = 0) necessarily
lowers the measured SR while making the number mean something. **The cl25 -> v2 delta
on this row is therefore not an apples-to-apples comparison and must not be read as
one.** The honest statement is: 0.141 was measured on a distribution that included
free wins; 0.094 is the same teacher and same distribution shape as cl25 with the new
asset; 0.031 is the v2 distribution.

**Found — Topple-Block 0.938 (cl25, n=128) -> 0.844 (v2, n=32).** Expected direction:
the block is 3.7x heavier (0.550 vs 0.150 kg, the primitive was 60 kg/m^3 — lighter
than styrofoam) and 17% taller, so the punch needs ~3.3x the force for the same
tipping torque, against a phase budget (align 60 + seat 50 + punch 70 + retreat 20)
that exactly fills the 200-step episode with no slack. Sweeping CONTACT_Z /
PUNCH_STEPS next.

**Decided:** ship `min_goal_distance = 0.06` on Cage-Drag rather than the minimum
that would satisfy G5 (anything > the 0.03 success radius). A goal 3.5 cm from a
3 cm success radius is not a transport, it is noise; 2x the success window is the
smallest value for which every episode is a real caged carry. The cost is measured
and recorded above rather than avoided.

**Next:** topple sweep, then the n=128 sweep + rollouts + publish.

### 2026-09-09 — Cage-Drag RIDE_HEIGHT: the re-derivation was load-bearing — W1-a

**Context:** `cage_drag.py`'s own docstring warns that this "pad height above object
centre" family of constants "does not port between objects even when the rest of the
strategy does, and is worth a real sweep rather than a single guess". The 46 mm cube
moves the object centre 0.020 -> 0.0226, so the constant had to move with it.
Sweep script: `~/cl_v2_work/W1-a/cage_sweep.py` (patches the module constant and the
cfg field in process, so no file edit per point). GPU 1, n=32 per point.

**Found:**

| RIDE_HEIGHT | min_goal_distance | SR (n=32) |
|---|---|---|
| 0.026 (the cl25 value, ported blind) | 0.0 | 0.031 |
| **0.028 (re-derived)** | 0.0 | **0.094** |
| 0.026 | 0.06 (shipped) | 0.000 |
| **0.028** | 0.06 (shipped) | **0.031** |

Carrying the cl25 constant across unchanged would have cost a factor of ~3 on this
row and would have looked exactly like "the new asset broke cage-drag". The
derivation that produced 0.028 — hold the pads at the same 65% of the half-height
they sat at on the 40 mm cube, i.e. `0.65 * 0.0226 + 0.013` — reproduces the sweep
optimum. This is the single most transferable lesson from my seven rows: **every
"height above the object centre" constant must be re-derived as a FRACTION of the new
half-extent, not copied.**

**Decided:** ship RIDE_HEIGHT = 0.028. The n=32 points are 1-3 successes out of 32,
so they separate 0.026 from 0.028 but cannot rank 0.028 against 0.030/0.032 — I did
not spend n=128 per point chasing that on a teacher that is 0.75 below the competence
bar in either direction. Recorded, not hidden.

### 2026-09-09 — Topple-Block: the lever arm, not the phase budget — W1-a

**Context:** the v2 block is 0.550 kg against the primitive's 0.150 kg (the primitive
was 60 kg/m^3 — lighter than styrofoam), so the punch needs ~3.3x the torque for the
same tipping angle, and the phase budget (align 60 + seat 50 + punch 70 + retreat 20)
exactly fills the 200-step episode. Two candidate fixes: more lever arm (CONTACT_Z) or
more sustained force (PUNCH_STEPS, paid for out of RETREAT_STEPS).
Sweep: `~/cl_v2_work/W1-a/topple_sweep.py`, n=32 per point, GPU 1.

**Found:**

| CONTACT_Z | PUNCH:RETREAT | SR (n=32) |
|---|---|---|
| 0.064 (pure half-height scaling of the cl25 0.055) | 70:20 | 0.844 |
| 0.064 | 90:15 | 0.906 |
| **0.078** | **70:20** | **0.906** |
| 0.078 | 90:15 | 0.844 |

Either single change recovers the same ~0.06; **combining them does not, it gives the
deficit back.** Plausibly the two trade against each other — a higher contact point
with a longer punch over-rotates the box past the drift bound, or pushes the pad off
the top edge late in the punch — but at n=32 (27/32 vs 29/32) I am not going to claim
a mechanism I did not instrument.

**Decided:** ship `CONTACT_Z = 0.078`, phase budget untouched. Reasons: it is the
constant the packaging contract names ("any constant that names a half-extent, ride
height, standoff or face offset"), it is physically motivated (a force-limited punch
on a 3.7x heavier box needs a longer moment arm, and 0.078 is 74% of the way from the
centre to the top face against 61% before, still leaving 2.7 cm of face above the
pad), and it is a one-line change instead of a re-timed state machine. The
tip-rather-than-slide criterion (`contact height above ground > half_x / friction =
0.0625 m`) is mass independent and is cleared 2.9x at 0.183 m.

**Also found — Push-Cuboid n=128 = 0.094 (12/128) against cl25's 0.223.** Expected
and attributable: the carton's real 0.097 kg mass doubles the sliding friction the
pusher works against (0.49 N -> 0.95 N), and cl25's own instrumentation recorded this
teacher as *transport-rate* limited, not accuracy limited — it moved the box
0.38 mm/step against the 0.71 mm/step needed, a 1.89x shortfall AT THE OLD MASS.
Doubling the resistance roughly doubles that shortfall. Per the brief this is a
characterised below-bar teacher and the endgame-precision problem is out of scope; the
delta is recorded, not chased. The honest trade is stated plainly: **CL-V2 chose the
published product mass over the primitive's convenient one, and this row paid for it.**

### 2026-09-09 — FOR THE LEAD: rollout camera framing is wrong for every task — W1-a

**Context:** first published rollout (`Mjlab-Push-Cuboid-Franka`, GPU 1). The asset
still and turntable from `render_asset` are excellent — the Jell-O carton fills the
frame, textured, correctly scaled. The TEACHER ROLLOUT video is not: the arm fills the
middle of the frame and the manipulated object sits half-cropped at the very bottom
edge, a few dozen pixels tall.

**Diagnosed, so it is one decision rather than five investigations:** it is not the
studio rig (`studio.py` adds only lights, a skybox and a floor texture — no geoms) and
not the asset. It is `record_task_videos._autoframe_camera`, which `render_rollout`
calls at `:224` after setting `lookat=(0.3, 0, 0.4), distance=1.6`. That helper fits
the camera to the bounding box of **every non-plane geom in env 0**, which for a
manipulation scene is dominated by the Franka: the box centre lands around z = 0.5
(the middle of the ARM), the fitted distance covers a ~1.2 m diagonal, and at
elevation -28 deg an object resting at z = 0.015 near the base falls to the bottom
edge of a 4:3 frame. The helper's own docstring says it exists so "the object [does
not] become a few pixels near the bottom edge" — that is exactly what it now does,
because the arm, not the task, sets the box.

**NOT FIXED BY ME, deliberately.** `render_rollout.py` / `record_task_videos.py` are
run concurrently by all five wave agents and some have already published; a unilateral
global re-framing would silently make the published set inconsistent. This is a W0/W3
call. Suggested one-line shape: fit the box to the manipulated entities plus the
gripper site (the command's `asset_name` entity, the goal marker and the `gripper`
site) instead of all geoms — or simply drop the robot's geoms from `keep`.

**Impact if left:** every `/v2/<Task-Id>/` card's `teacher.mp4` and `failure.mp4` is
poorly framed. The `still.png` / `turntable.mp4` on the same cards are fine, so the
asset work is visible either way. My rows are published with this caveat and can be
re-rendered in minutes once the framing is decided — `~/cl_v2_work/W1-a/rollout.sh`
re-runs any task.

### 2026-09-09 — n=128 measurement protocol note — W1-a

**Decided:** the n=128 G7 numbers on my rows come from
`render_rollout --num-episodes 128 --batch-size 32`, not from a second
`test_classical --num-envs 32 --num-episodes 4` pass. They are the same measurement:
`render_rollout.run_stats_phase` builds the env with `load_env_cfg(task, play=False)`
and `num_envs = batch_size` — byte-identical to `test_classical` — then runs 4 chunks
of 32 one-episode envs and reads `episode_success` inside the loop, exactly the way
`test_classical` does. Using it once instead of twice halves the GPU time on a node
that is shared with W2-a and another user's 8-process job, and it produces the site's
`result.json` (sr / n / head / date) as a by-product, so the published card and the
STATUS row cannot disagree. The n=32 iteration passes and the two teacher sweeps did
go through `test_classical` / in-process harnesses.

### 2026-09-09 — CORRECTION + final n=128 numbers — W1-a

**CORRECTION to my Cage-Drag attribution entry above.** That entry rested on a single
n=32 point (0.094 = 3/32) and I flagged it as noisy at the time. I re-ran the control
properly at **n=128** (`~/cl_v2_work/W1-a/cage_attrib.py`, same teacher, same asset,
only `min_goal_distance` varied):

| Cage-Drag configuration | SR | n |
|---|---|---|
| cl25: 40 mm primitive cube, no separation | 0.141 | 128 |
| v2 asset (46 mm), **min_goal_distance 0.0** (= cl25's sampling) | **0.094** | 128 |
| v2 asset, min_goal_distance 0.035 (shipped) | **0.008** | 128 |

The conclusion **holds and is now properly powered**: the asset swap costs
0.141 -> 0.094, inside the ±0.05 noise band at n=128, i.e. nothing. Everything else is
the G5 fix. cl25 drew Cage-Drag's object and goal from the *same* box, so 3.7% of
episodes were already successful at reset and a further ~5% had a transport shorter
than 3.5 cm — and those turn out to be very nearly all of the teacher's successes.
**cl25's 0.141 on this row was a sampling artefact, not a capability.** The shipped
guard is 0.035, the smallest value that makes success-at-reset exactly 0 by
construction (success is `dist < 0.03`); I had first shipped 0.06 (2x the success
window, better task design) and backed it out to 0.035 precisely so the delta stays
interpretable — 0.06 measured 0.008 too, so the choice costs nothing either way, but
0.035 is the one the gate actually demands.

**Found and fixed — my own regression, caught by the audit, not by a gate.** The
separation dead bands I added to Push-Cuboid and Drag-Pull were split evenly between
the object band and the goal band. On Drag-Pull that pushed the object band from
[0.37, 0.44] to [0.40, 0.44] — the far end of an x range `_ENGAGE_INSET` had already
trimmed — and `audit_workspace` flagged **`sparse-reach`, 3.7% -> 2.0%** of comfortable
top-down poses. `verify_task` did NOT catch this: G4 checks radial bounds and spawn
collisions, not grasp-pose density. Fix: take the whole dead band off the **goal** side
on both tasks. A goal only has to be reachable (bounded by `GOAL_RADIAL_MAX`); the
object band is the one carrying the grasp-density constraint. Object bands are now
back to their pre-fix extents, both audits are clean, both re-verified 0/1000
success-at-reset, and both re-measured at n=128.

**FINAL n=128 numbers (HEAD 41b36cb), all seven rows green:**

| Task | cl25 | v2 (n=128) | delta | reading |
|---|---|---|---|---|
| Lift-Cube | 1.000 | **1.000** | 0.000 | unchanged |
| Stack-Cube | 0.328 | **0.523** | **+0.195** | improved — the 46 mm cube on an 89x73 mm carton top is a far more forgiving place-and-release than a 40 mm cube on an 80x80 mm box |
| Push-Cuboid | 0.223 | **0.086** | -0.137 | the carton's real 0.097 kg doubles sliding friction (0.49 -> 0.95 N) on a teacher cl25 already measured as transport-RATE limited |
| Drag-Pull | 0.477 | **0.391** | -0.086 | same mass effect, smaller because the pull is shorter |
| Topple-Block | 0.938 | **0.898** | -0.040 | inside the noise band, after the CONTACT_Z sweep recovered it from 0.844 |
| Cage-Drag | 0.141 | **0.016** | -0.125 | see the control table above: the asset costs nothing, cl25's baseline was inflated |
| Reach-Target | 1.000 | **1.000** | 0.000 | unchanged |

Every card is published at `https://cl.untuai.com/v2/<Task-Id>/` with `asset.json`
(source, license, sr_cl25, sr_v2, gates, the re-derivation notes), `result.json`,
`still.png`, `colliders.png`, `turntable.mp4` and the rollout clips. Lift-Cube and
Topple-Block have no `failure.mp4` and Cage-Drag has no `teacher.mp4` — at 1.000,
0.898 and 0.016 the render phase legitimately cannot find the other outcome in its
episode budget.

**Two small things the lead may want to pick up:**
1. `publish_v2.sh` lists its files literally, so a task legitimately missing an
   optional clip makes rsync exit 23 with a scary error even though every present file
   transferred. Guarding each path with `[[ -f ... ]]` before adding it to `FILES`
   would silence it.
2. `tests/test_workspace_placement.py::test_mechanism_drops_are_swept_over_the_joint_range`
   failed once for me mid-run and passed on an immediate re-run — an articulated XML
   was being rewritten underneath it by W2. Not a real failure, but worth knowing
   before someone chases it.

<!-- ===== W1-b ===== -->
# CL-V2 — LOG W1-b (container / cylinder / puck / stick family)

> Append-only. Newest at the bottom. Rows owned: Place-In-Container, Throw-To-Bin,
> Reorient-Object, Strike-Slide, Tool-Pull. GPU index 2 inside holder 20277 (dgx1).

---

### 2026-09-09 — Intake and asset selection — W1-b

**Context:** HEAD `41b36cb` (W0 landed). Read AGENT_BRIEF / GOAL / PLAN / CODE_MAP §3+§4 /
STATUS. Holder `hold_dgx_amit` 20277 on dgx1 confirmed RUNNING; using **GPU 2 only**,
never scancelling anything. Work dir `~/cl_v2_work/W1-b`, raw downloads `~/assets_raw`.
Order per the brief: Reorient, Strike-Slide, Tool-Pull first (they do not depend on
W1-a's cube), then Place-In-Container and Throw-To-Bin.

**Did (survey):**
- Measured the *current* geometry from the XMLs rather than trusting the docs:
  `container.xml` floor half 0.07 x 0.07 x 0.008, walls half 0.008 at +-0.062 -> the
  inner clear span is **0.108 m, not the 0.124 quoted in CODE_MAP §3** (0.124 is the
  wall-CENTRE span). The teacher docstring's "~10.8 cm" is the correct figure.
  Rim top is 0.07 above the body origin, `object_site` is at (0,0,0.02), so
  `rim_height = 0.05` is exactly rim_top - site_z.
- The container body also floats: `container_spawn_range z=(0.02,0.02)` with a floor
  geom half-thickness 0.008 at body z=0 puts the bin's underside 12 mm above the ground.
- `cylinder.xml` is r 0.02, half-length 0.02 (a 40x40 mm billet).
  `reorient_object.py` grasps it **end face to end face** (`down_frame(bearing)` puts the
  finger-closing axis ALONG the cylinder axis; the inline comment claiming "across the
  barrel" is stale, the module docstring is right). So the object's LENGTH, not its
  diameter, is what the 80 mm gripper has to span.
- `puck.xml` r 0.035 h 0.024 mass 0.03; `stick.xml` shaft box 0.26x0.022x0.022 with the
  hook bar at (0.12, 0.035, 0).

**Found (asset search):**
- Poly Haven has **no** can-sized or bin-sized model (scanned all 521 `dimensions`
  entries: only `food_lime_01` and `food_lychee_01` fall in a 35-62 mm band, and the
  smallest container is `planter_pot_clay` at 266 mm).
- GSO supplement bottles measured (mm): CoQ10 47.4x47.5x84.4, Folic_Acid 47.5x47.6x83.9,
  Beta_Glucan 54x54x98, AllergenFree_JarroDophilus 54x54x96, Lutein 54x54x97,
  Theanine 54x54x97, 5_HTP 53x53x89, Inositol 62x62x110, Quercetin_500 81x80x144.
  **Every one of them is 84-110 mm tall**, i.e. longer than the gripper's 80 mm
  aperture: at scanned scale none of them can be grasped end-to-end.
- GSO receptacles measured (mm): Spritz_Easter_Basket_Plastic_Teal 191x183x129 (12.6k
  tris, open top, **no handle** — verified by slicing the mesh every 9 mm in z, the
  footprint stays 166->190 mm all the way up), Full_Circle_Happy_Scraps 213x155x143 (has
  a lid rim), Threshold_Basket_..._Small 239x162x148 (66k tris), Curver_Storage_Bin_
  Black_Small 286x193x134, Target_Basket_Medium 270x268x218, Hefty_Waste_Basket 271x204x300.

**Decided:** see the per-task entries below.

**Next:** package the Reorient bottle.

---

### 2026-09-09 — Reorient-Object: GSO CoQ10 amber packer bottle — W1-b

**Context:** HEAD `41b36cb`, holder 20277, GPU 2. First of my five rows; chosen first
because it shares nothing with W1-a.

**Did:**
- G1: `gso:CoQ10` (Jarrow amber HDPE packer bottle, CC-BY 4.0). Ledger row +
  `asset_zoo/objects/free/cylinder/PROVENANCE.md`.
- G2: `~/cl_v2_work/W1-b/build/build_bottle.py` — uniform scale **x0.63**, origin =
  bbox centre (on the bottle axis), albedo 4096²→1024², visual 10 490 tris, collider =
  the convex hull reduced to its 96 Fibonacci-sphere support vertices (188 tris vs the
  raw hull's 2132; helper `~/cl_v2_work/W1-b/build/hullutil.py`). Mass 25 g. Directory
  1.39 MB. Rewrote `cylinder.xml` (kept body `cylinder`, joint `cylinder_joint`, geom
  name `cylinder_geom` on the collider, site `object_site` at the origin) and added the
  Franka `assets`-dict pattern to `cylinder_constants.get_cylinder_spec`.
- G3–G5: `verify_task --num-resets 1000` **PASS** (`runs/Reorient-Object/verify.json`):
  settle ok, drop ok, grasp width 0.0296, step ratio 1.29, radial 0.333–0.532 all in
  envelope, 0 spawn collisions, floor_min_z −0.7 mm, success-at-reset **0/1000**,
  oracle 1.000. `audit_workspace --keyword Reorient` → **0 problems**, 7.4 % approach
  freedom.

**Found (the decision that shaped this row):** `reorient_object.py` grasps the object
**end face to end face** — `_approach_rot` returns `down_frame(grasp_bearing)` and
`down_frame` puts the finger-CLOSING axis at that bearing, which is the bearing of the
object's own axis. (The inline comment claiming "closing axis ACROSS the barrel" is
stale; the module docstring, which records end-face 0.531 vs barrel 0.469 on the same
32 episode-instances, is the correct one.) So the 80 mm aperture has to span the
object's **length**. Measured every plausible can/jar in the reachable catalogues:
GSO CoQ10 84.4 mm tall, Folic_Acid 83.9, 5_HTP 88.8, Beta_Glucan 97.6,
AllergenFree_JarroDophilus 96.0, Lutein 97.2, Theanine 97.4, Mastic_Gum 97.3,
Hyaluronic_Acid 98.0, Krill_Oil 97.6, Inositol 110.2, Quercetin_500 144.1; Poly Haven
has no can-sized model at all (all 521 `dimensions` entries scanned). **Not one of them
can be grasped at scanned scale.**

**Decided:** scale the bottle by 0.63 to 29.9 x 53.2 mm — a real 12 cc packer bottle,
the largest bottle of this form the gripper can span, and still 1.78:1 so it stays put
lying down (a squat can would tip onto a face by itself and hand the task free
successes at reset). Recorded in PROVENANCE.md. Constants re-derived:
- `env_cfgs._pad` **0.05 → 0.031** = hypot(half-length 0.0266, radius 0.0150) = 0.0305
  rounded up. The 0.05 it replaces was never derived — its own comment called 0.05 "the
  half-length" of a cylinder whose half-length was 0.02. Spawn box grows to
  x 0.331–0.489, y ±0.219; corner radial 0.536 < GRASP_RADIAL_MAX.
- spawn `z` **0.025 → 0.016** = collision hull max radius 0.01501 + 1 mm.
- teacher `ROTATE_LIFT` re-derived and **left at 0.10**: the required lift is
  half_length − lying-axis height = 0.0266 − 0.0150 = 0.0116 m (it was 0.0 for the old
  billet, whose half-length equalled its radius), and 0.10 is a commanded servo error
  that clears it by 8.8 cm.
- teacher `grasp_z_offset` re-derived and **left at 0.008**: the `floor_min_z = 0.030`
  guard clamps the commanded site anyway, so the pads land ~1 mm above the axis on a
  15 mm-radius end face — better centred than the old billet, where the same clamp put
  them 4 mm BELOW the axis.
- `body_axis` stays `(0,0,1)`: the scan's +z is the bottle axis.
- Out-of-scope but in my asset's blast radius (D5): Lift-Cylinder's spawn `z` 0.02→0.028
  (the upright resting height is now the half-length 0.0266, so 0.02 buried it 6.6 mm)
  and `LiftCylinderClassicalPolicy.OBJ_CENTER_Z` 0.020→0.0266.

**Next:** G6 rollout + G7.

### 2026-09-09 — Strike-Slide + Tool-Pull: built puck and reach hook — W1-b

**Context:** `puck.xml` is shared by both rows; `stick.xml` is Tool-Pull only. GPU 2.

**Did:**
- G1/G2 puck: **built** a regulation ice-hockey puck (76.2 mm x 25.4 mm, 160 g) as a
  trimesh lathe of the real 5-point profile with a 2.5 mm moulded edge bevel and a
  debossed face ring; albedo = ambientCG **Rubber004** (CC0) desaturated and driven to
  mean 38/255. 576 tris, 512² PNG, 496 KB. Collider: one **cylinder primitive**
  (`puck_geom`, old friction/condim/solref kept). Neither Poly Haven nor GSO nor YCB
  has a puck, and the real article is a lathe — building it is exact and 20x cheaper
  than a scan.
- G1/G2 stick: **built** a wooden reach hook — a 260 mm x 22 mm turned dowel whose far
  end curves through a quarter arc into a 70 mm hook (the primitive was a T-bar);
  ambientCG **Wood051** (CC0), rotated 90° so the grain runs along the shaft. 1016 tris,
  512², 340 KB. Colliders: two cylinder primitives on the same axes/extents as the
  boxes they replace, geom names `stick_shaft` / `stick_hook` kept, sites `object_site`
  (−0.09,0,0) and `tool_tip_site` (0.12,0.02,0) **unchanged**, so `tool_spawn_range`'s
  +0.09 x offset still describes the grasp point. Mass 75 g from 600 kg/m³ hardwood on
  the two geoms (an L-shaped composite inertia, not a hull overestimate).
- G3–G5 both tasks: `verify_task --num-resets 1000` **PASS**. Strike-Slide: puck mass
  0.160, step ratio 1.03, radial 0.301–0.442, 0 spawn collisions, 0/1000 success at
  reset, oracle 1.000. Tool-Pull: puck ratio 1.04 / stick ratio 0.77, puck radial
  0.623–0.709 (inside the 0.58–0.75 exemption), stick 0.288–0.539, 0 spawn collisions,
  0/1000, oracle 1.000. `audit_workspace` → **0 problems** on both.

**Found (a real bug, not just a constant):** `strike_slide.RIDE_HEIGHT` shipped as
**0.026 while the comment right above it derives 0.020**. At 0.026 the commanded site
sits at 0.038 and the pads (≈1.3 cm below the site) reach 0.025 — 1 mm ABOVE the old
24 mm puck's top face, i.e. the "seat" phase was aiming the fingers over the puck, not
at its rim. On the 25.4 mm regulation puck it would be 0.0257 against a 0.0254 top.
Set to 0.020, which is the largest value that keeps the pads inside the puck's vertical
span while leaving 2.7 mm above the `FLOOR_MIN_Z` guard.

**Found (recorded, not fixed):** cl25's flag that `ee_ground_collision` leaves under
1 cm of margin over a 24 mm puck is **unchanged** by the 25.4 mm puck — the margin gets
1.4 mm larger, not smaller. `FLOOR_MIN_Z = 0.030` on the site puts the pads at ~0.017,
inside the puck's 0–0.0254 span with 8 mm of floor clearance.

**Found (new, worth the lead's attention):** mujoco_warp 3.11 warns
`MULTICCD ... CCD pairs without multicontact support: [('CAPSULE','CYLINDER'),
('CYLINDER','CYLINDER'), ('CYLINDER','BOX')]` on the Tool-Pull env — a fingertip pad
(box) against the cylinder shaft resolves to at most ONE contact, where the primitive
box shaft got box–box multicontact. It does not touch the G7 number (the teacher never
grasps the stick — it is a direct closed-finger drag) but it makes the documented
squeeze-ejection strictly easier. Recorded in `free/stick/PROVENANCE.md` with the fix
for anyone attempting a real tool-use teacher.

**Decided:** constants re-derived —
`PUCK_HALF_HEIGHT` 0.012→**0.0127**, `PUCK_RADIUS` 0.035→**0.0381**,
`strike_slide.RIDE_HEIGHT` 0.026→**0.020**, `strike_slide.STANDOFF` 0.06→**0.063**
(radius + 25 mm pad half-width), `tool_pull.GOAL_LOCAL.z` and
`ToolPullCommandCfg.goal_offset.z` 0.012→**0.0127**, `tool_pull.BEHIND` 0.052→**0.055**
(radius + 17 mm pad half-width), `tool_pull.RIDE_Z` re-derived and left at **0.034**
(pads land 83 % up the puck face, was 87 %), `strike_slide_env_cfg.goal_z_height` and
both spawn z 0.012→**0.0127**. The stick's spawn z stays 0.012 (dowel radius 0.011 +
1 mm). **The 5.3x mass increase (30 g → 160 g) is the dominant dynamics change on
Strike-Slide** and is the real article's mass, so it is recorded, not tuned around.

**Next:** G6 rollouts + G7.

### 2026-09-09 — Place-In-Container + Throw-To-Bin: GSO plastic basket — W1-b

**Context:** `container.xml` is shared by both rows. Done last per the brief because
both also use W1-a's cube; by the time I got here W1-a had landed it
(`free/cube/xmls/assets/cube_package.json` present, 46 mm mini Rubik's cube, collision
half-extents 0.0230 x 0.0226 x 0.0226), so **G7 for these two rows is measured against
the real cube, not a placeholder**.

**Did:**
- G1: `gso:Spritz_Easter_Basket_Plastic_Teal` (CC-BY 4.0), **scanned scale, no
  rescaling** (191 x 183 x 129 mm). Ledger row + `free/container/PROVENANCE.md`.
- G2: visual mesh 12 630 tris (inside budget, no decimation), albedo 4096²→1024²,
  1.10 MB. Origin = `bottom`, so the body frame is the basket's UNDERSIDE and it sits ON
  the ground. Colliders: **floor + four walls as boxes**, keeping the five primitive
  geom names and their contype/condim/friction/solref.
- G3–G5 both rows: `verify_task --num-resets 1000` **PASS**.
  Place-In-Container: cube radial 0.292–0.536, container 0.303–0.530, 0 spawn
  collisions, floor_min_z 0.000 for both, success-at-reset **0/1000**, oracle 1.000.
  Throw-To-Bin: container radial 0.781–0.911 inside its 0.75–0.95 exemption, 0/1000,
  oracle 1.000. `audit_workspace` → **0 problems** on both.

**Found:**
- CODE_MAP §3 says the old bin's "inner span ~0.124". It is **0.108** — the walls are
  half-thickness 0.008 centred at ±0.062, so 0.124 is the wall-CENTRE span. The
  teacher's own docstring ("~10.8 cm") was right. Not a problem now, but the number in
  CODE_MAP should be corrected by whoever owns it.
- The old bin **floated 12 mm above the ground**: its floor slab was centred on the body
  origin (half-thickness 0.008) and `container_spawn_range z` was 0.02. Fixed by
  origin=bottom + z=0.
- **CoACD is the wrong tool for this shell** and I nearly shipped it: an 8-hull
  decomposition merged the 4 mm moulded floor with the flared lower wall into an **18 mm
  slab**, which would have left the cube visibly floating 14 mm above the basket's floor
  in every rollout video. Verified from the hull bounds, then replaced with boxes.
- Real receptacles are big. The smallest open-top container in any reachable catalogue
  is this basket at 191 x 183 mm — 1.4x the primitive's footprint — and the next ones up
  (Curver 286x193, Target_Basket 270x268, Threshold 239x162, Hefty 271x204) cannot be
  separated from the cube band inside a 0.55 m grasp envelope at all.
- At the primitive's bands the real basket's footprint **overlapped the cube band by
  4 mm**, i.e. every reset would have been a spawn collision.

**Decided:** constants re-derived from the measured cavity (floor top 0.004, inner
half-span 0.0815, rim low point 0.113, scallop peaks 0.129, site at 0.020):
- `lateral_tolerance` 0.055 → **0.0585** = 0.0815 − 0.0230 (cube half-width), i.e.
  exactly "an axis-aligned cube's face touching the wall". Checked the predicate stays
  physically meaningful: a cube resting OUTSIDE against the wall is at lateral ≥ 0.1065,
  and a cube balanced on the rim is at vertical 0.116 > `rim_height`, so neither counts.
- `rim_height` 0.05 → **0.093** = 0.113 − 0.020.
- `floor_tolerance` 0.04 → **0.020** (4 mm of contact penetration; the old value allowed
  a centre 28 mm BELOW the old bin's floor).
- `container_spawn_range.z` 0.02 → **0.0**; Place bands **cube y (−0.25,−0.07)** and
  **container y (0.10,0.24)**, x kept at 0.28–0.48. I moved the CUBE band rather than
  pushing the bin outward, so the bin stays near the centreline where the arm can hold
  the cube above it; that keeps the audit's approach-freedom at 3.3 % (a 0.30–0.46 x band
  measured 2.7 %, under the 3 % floor). Throw-To-Bin's bin band is unchanged.
- `place_in_container.DROP_HEIGHT` 0.10 → **0.15**: the binding constraint is the lateral
  CARRY, not the release — the cube's underside must clear the scallop peaks
  (0.109 above the site) plus its own 0.0226 half-height = 0.132, +18 mm servo margin.
- `place_in_container.lift_height` 0.22 → **0.26** (required cube rise 0.082 → 0.129).
- `throw_to_bin.DROP_HEIGHT` 0.12 → **0.15** (same clearance convention).
The task gets *easier* in one respect and harder in another, both recorded: the drop
grows from 9.2 cm to 14.3 cm, but the effective landing radius grows from 3.4 cm to
5.85 cm — about 3x the area.

**Next:** G6 rollouts + G7 for all five rows.

### 2026-09-09 — G6 + G7 for all five rows, and two ablations — W1-b

**Context:** GPU 2, holder 20277. cl25 protocol exactly: `test_classical --num-envs 32
--num-episodes 4` = n=128. Rollouts: `render_rollout --num-episodes 32 --batch-size 32
--max-render-episodes 6`. All five published with `publish_v2.sh` + `asset.json`.

**Measured (n=128, HEAD 41b36cb + this wave's uncommitted work):**

| task | cl25 | v2 | note |
|---|---|---|---|
| Strike-Slide | 0.000 | **0.000** | characterised failure confirmed on the real puck |
| Tool-Pull | 0.047 | **0.000** | 0.047 was 6/128; see below |
| Reorient-Object | 0.547 | **0.258** | asset-driven; two ablations below |
| Place-In-Container | 0.250 | **0.359** | improved |
| Throw-To-Bin | 0.000 | **0.000** | characterised failure confirmed |

**Found — Reorient's drop is the ASSET, and I proved it rather than asserting it.**
0.547 → 0.258 is far outside the ±0.05 noise band, so I ran the two obvious
alternatives and both came back worse:
- **Spawn box.** Reverting `_pad` to cl25's exact 0.05 (so the spawn box is
  byte-identical to the measurement's) gave **0.188 at n=128** against 0.258 for the
  derived 0.031. The re-derived, WIDER box is the better of the two; the box is not the
  cause.
- **Descent tolerance.** The obvious teacher adaptation to a smaller end face is to make
  the descent converge tighter before closing; `descend_tol` 0.012 → 0.008 gave
  **0.156 at n=64**, below the baseline's whole per-32 spread (0.219–0.344). The
  mechanism is the one this teacher's docstring already names: P_DESCEND has a 60-step
  timeout and cannot beat the DLS lateral bias, so a tighter tolerance only keeps the
  wrist low near the floor for longer and buys more `ee_ground_collision` terminations.
Both reverted; the shipped configuration is the measured best of the three. What is
left is geometry: the pads' target faces went from 40 mm to 30 mm across (−44 % of
area), the aperture clearance per side from 20 mm to 13.4 mm, and the object from 50 g
to 25 g against a task whose documented dominant failure is drift under `max_drift`.
Recorded in the row, the asset.json and here — not hidden, and not tuned around.

**Found — Place-In-Container got BETTER, and for the reason the geometry predicts.**
0.250 → 0.359. The real basket's 16.3 cm inner clear span makes the effective landing
radius 5.85 cm where the primitive's was 3.4 cm (0.054 inner half-span − the old 2 cm
cube half-width) — about 3x the area — which more than pays for the release height
growing from 9.2 cm to 14.3 cm to clear the taller rim.

**Found — Tool-Pull 0.047 → 0.000.** 0.047 is 6/128, and this teacher's own docstring
records the number as unstable between batches (0.250 and 0.000 on two consecutive
16-env episodes). On top of that noise the puck is now 5.3x heavier, which directly cuts
the steering authority of a closed-finger shove at radial 0.65. Recorded as a real
geometry-explained change.

**Did (on the lead's mid-task note, all three points):**
1. **`throw_to_bin.py::OBJ_CENTER_Z` 0.020 → 0.0226.** Confirmed the diagnosis: this file
   keeps its OWN copy of the lift_object grasp constants, so W1-a's edit to
   `LiftCubeClassicalPolicy` did not reach it. `GRASP_SITE_Z` deliberately stays 0.045 —
   lift_object.py documents it as the MEASURED ground-collision floor for the gripper
   site, not a function of the object, and the taller cube only improves where the pads
   land. `CLOSE_STEPS` 15 already matches W1-a's. `DROP_HEIGHT` was re-derived against
   the new bin rim earlier today and already used 0.0226. `place_in_container.py` has no
   half-size constants (its grasp offset is relative to the object site) — confirmed.
   **The Throw-To-Bin G7 above was measured AFTER this fix.**
2. `publish_v2.sh`: all five of my publishes returned 0 and listed their files on the
   remote, so nothing needed re-running.
3. **Rollout camera framing: reverted, and handed over instead.** I had already found and
   fixed it before the note arrived; on the note I reverted both edits so the lead's
   global fix lands on a clean tree, and wrote the diagnosis + the exact patch to
   `docs/cl_v2/handover/W1-b-rollout-camera-framing.md` (+ `.patch`). **The cause is not
   "fits to the robot":** `_autoframe_camera` copies only `qpos` into its scratch
   `MjData`, so the 13 mocap-mounted assets are measured at their MJCF pose, and it also
   runs BEFORE the first `env.reset()`, which is when a command term first writes them.
   Throw-To-Bin's bin is written to x 0.78–0.90 but sits at 0.55 in the XML, so it was
   outside the frame for the entire clip. Two lines fix it (sync `mocap_pos`/`mocap_quat`,
   and `env.reset()` before framing) — verified by re-rendering, see the thumbs for
   Place-In-Container and Throw-To-Bin, which DO show the basket.
   **Caveat the lead needs:** those two clips were rendered while the patch was applied
   locally, so a global re-render without it will regress them. Throw-To-Bin's clip was
   also rendered before point 1 landed; its published SR (0.000, n=128) is from the
   corrected teacher, the clip is illustrative of a 0.000 teacher either way.

**Checks (all green):** `verify_task --num-resets 1000` PASS on all five (evidence in
`runs/<Task>/verify.json`, success-at-reset 0/1000 and oracle 1.000 everywhere);
`audit_workspace --keyword` 0 problems on all five; `benchmark_smoke --keyword` PASS on
all five; `FORCE_CPU=1 pytest tests/test_workspace_placement.py
tests/test_classical_teachers.py tests/test_class_a_wave1.py
tests/test_class_a_expansion.py -q` → **149 passed**.

**Decided:** Ship `_pad = 0.031` and `descend_tol = 0.012` for Reorient (the measured
best), record the drop rather than tune it. One placement decision worth flagging: the
Place-In-Container container band's approach-freedom sits at 3.3 % against the audit's
3 % floor. I got there by moving the CUBE band out (y max −0.04 → −0.07) instead of
pushing the bin further from the centreline, because the pose the task turns on is the
gripper holding the cube ABOVE the bin. A narrower container x band (0.30–0.46) measured
2.7 % and would have failed the audit.

**Next (for the lead):** apply `docs/cl_v2/handover/W1-b-rollout-camera-framing.md`
before the global re-render, and correct CODE_MAP §3's "bin inner span ~0.124" — the
primitive's inner clear span was 0.108 (0.124 is the wall-CENTRE span).

<!-- ===== W1-c ===== -->
# CL-V2 — W1-c log (Peg-Insertion, Axial-Extract, Edge-Grasp, Pivot-Lift)

> Append-only. Template: Context / Did / Found / Decided / Next.
> Owner: wave agent W1-c. GPU index inside holder 20277: **3**.

---

### 2026-09-09 — W1-c kickoff: survey, asset shortlist, baselines — W1-c

**Context:** HEAD `1127d12` + the lead's uncommitted W0 work. Holder `hold_dgx_amit`
(20277, dgx1) — reused, never cancelled; my GPU index is 3. Read AGENT_BRIEF, GOAL,
PLAN, CODE_MAP §3/§4, STATUS. Four rows: Peg-Insertion, Axial-Extract, Edge-Grasp,
Pivot-Lift.

**Did:** Read every current asset XML + constants module, the three commands
(`StackingCommand` reuse, `AxialExtractCommandCfg`, `EdgeGraspCommand(Cfg)`,
`PivotLiftCommand(Cfg)`), the four teachers, `workspace.py`, `verify_task.py`,
`asset_pipeline.py`. Searched the GSO (1033) and Poly Haven (521) catalogs and fetched
candidates.

**Found (catalog measurements, `asset_pipeline inspect`, metres):**
- GSO `HAMMER_PEG` 0.126x0.093x0.089 (the whole bench, not a peg),
  `GEOMETRIC_PEG_BOARD` 0.088x0.087x0.086, `SHAPE_SORTER` 0.169x0.094x0.046,
  `GEOMETRIC_SORTING_BOARD` 0.087x0.087x0.061.
- GSO saucers are all far too small to be unspannable: `Cole_Hardware_Saucer_Glazed_6`
  0.080x0.080x0.030 (exactly the 0.080 aperture), `Kotobuki_Saucer_Dragon_Fly`
  0.048x0.048x0.024, `Ecoforms_Quadra_Saucer_SQ1` 0.044x0.043x0.021,
  `Threshold_Bamboo_Ceramic_Soap_Dish` 0.062x0.049x0.051.
- Poly Haven `/info`: `wooden_cutting_board` 450x247x41 mm, `CheeseBox_01`
  240x136x93 mm, `carved_wooden_plate` 272x269x38 mm, `power_box_01` 377x418x503 mm.

**Next:** package the four asset families; baselines of `verify_task` on the current
primitives are running on the login node for reference.

### 2026-09-09 — All four assets packaged; G1-G5 green on all four — W1-c

**Context:** HEAD `41b36cb` (the lead committed W0 mid-session). Holder 20277, **GPU 3**.
Mesh work, CSG and `verify_task` on the login node CPU; renders and teacher runs through
`srun --jobid=20277 --overlap` with `MUJOCO_GL=egl`.

**Did — baselines first.** Ran `verify_task --num-resets 100` on all four tasks against
the PRIMITIVE assets before touching anything, so any change could be attributed:

| task | G3 | G4 | G5 | what was already broken |
|---|---|---|---|---|
| Peg-Insertion | PASS | PASS | **FAIL** | oracle 0.000 |
| Axial-Extract | PASS | PASS | PASS | — |
| Edge-Grasp | **FAIL** | PASS | PASS | `grasp_w` 0.09 > 0.06 |
| Pivot-Lift | **FAIL** | PASS | PASS | `grasp_w` 0.10 > 0.06 |

Two of those are verifier defects, not asset defects, and both reproduce on the
primitives (so they are pre-existing, not something I introduced):

1. **Edge-Grasp / Pivot-Lift G3 `graspable`.** `verify_task` measured `min(ext[0],
   ext[1])` — the object's in-plane width in its SPAWN pose. These two tasks are the
   extrinsic-dexterity pair: their premise is that the object is unspannable where it
   lies and is pinched on its THICKNESS only after the ledge edge / wall has re-presented
   it. The check therefore failed them for doing exactly what they are designed to do.
   Added `graspable="thin_axis"` (min over all three extents) and tagged those two rows.
2. **Peg-Insertion G5 oracle.** The verifier teleports the entity so its `object_site`
   lands on `target_pos`, but `StackingCommand`'s predicate compares the ROOT — and the
   peg's site is 50 mm away at its tip. The teleport therefore dropped the peg in mid-air
   with 25 mm still to fall, and three 0.02 s steps cover 17.6 mm of a 0.071 s drop.
   Added a per-task `oracle_root=True` (place the root, since the predicate is
   root-based). This alone did not make the oracle pass — see the `stack_height`
   re-derivation below, which was the real fix.

**Did — assets.** All four packaged, `render_asset`-ed and looked at (still + colliders).

| task | asset | source | tris | dir size |
|---|---|---|---|---|
| Peg-Insertion | 25x25x100 mm chamfered shape-sorter peg + 120x120x30 mm plywood board with a 30 mm bore and a 45 deg x 6 mm lead-in | trimesh + manifold3d; Poly Haven `oak_wood_planks` / `plywood` (CC0) | 28 / 56 | 876 KB |
| Axial-Extract | CEE 7/4 Schuko plug (50 mm round moulding, 34 mm thick) in an 80x80x100 mm surface-mount outlet box | trimesh; ambientCG `Plastic010` (CC0) | 480 / 402 | 356 KB |
| Edge-Grasp | GSO `Room_Essentials_Salad_Plate_Turquoise` scaled 0.676 -> a 150 mm side plate; plank-built oak riser 200x260x100 mm | GSO (CC-BY 4.0) + Poly Haven `oak_wood_planks` (CC0) | 2608 / 116 | 676 KB + 1.6 MB |
| Pivot-Lift | Poly Haven `wooden_cutting_board` scaled 0.486 -> a 219x120x20 mm chopping board, yawed 90 deg; 500x105x150 mm brick wall | Poly Haven (CC0) x2 | 3520 / 20 | 1.9 MB + 1.9 MB |

**Found — three collider decisions, all forced by MuJoCo's mesh-plane contact.**
A mesh geom resting on the ground plane gets only THREE contact points, and every one of
my flat-bottomed objects then micro-rocks forever and fails the G3 settle gate
(|v| < 1 mm/s, |w| < 0.01 rad/s after 2 s):

| object | mesh collider measured | primitive collider measured |
|---|---|---|
| peg (chamfered hull, exact) | \|v\| 2.8 mm/s, \|w\| 0.0598 rad/s, ncon 3 | box: \|v\| = \|w\| = **0.000**, ncon 4 |
| plate (8 CoACD hulls) | \|v\| 5.9 mm/s, \|w\| 0.0131 rad/s | cylinder: **0.000** |
| plate (convex hull) | \|v\| 0.7 mm/s, \|w\| 0.1013 rad/s | — |

So: peg -> box collider (its 2 mm lead chamfer becomes cosmetic), plate -> cylinder
collider, board -> box collider, ledge/wall -> box colliders. **The chamfer that matters
is the BOARD's, and that one IS physical:** the hole board's collision is 12 boxes —
`hole_wall_*` (bore), `hole_rim_*` (top face), `hole_chamfer_*` (four boxes rotated 45 deg,
given as `quat` so the MJCF degree/radian trap cannot bite) — verified against the visual
solid on a 0.9 mm grid over the whole bounding box: **0.000 cm^3 of solid without a box,
0.000 cm^3 of box outside the solid**. That gives the peg a 6 mm capture radius on top of
its 2.5 mm/side clearance, which is the entire point of decision D4.

**Found — the Pivot-Lift M4 blocker is a PLACEMENT bug and it is now fixed.** cl25
recorded that the teacher never touched the board. Any single-pad pusher must stand at
`board_x - (BOARD_HALF_X + PAD_RADIUS + clearance)` = `board_x - 0.076`; with the board
band at 0.32-0.38 that is **0.244-0.304**, at or below `GRASP_RADIAL_MIN` (0.28) — the
documented "folds the arm back over its own base" dead zone. Moving the board band to
**0.38-0.41** puts it at **0.304-0.334** and the wall's inner face follows, 0.485 ->
0.508 (`wall_spawn_range.x` 0.50 -> 0.5605). Checked against all three constraints the
old docstring lists plus the new one; `verify_task --num-resets 1000` reports board
radial 0.380-0.426, `frac_in_envelope` 1.000, `frac_dead_zone` 0.000, 0 spawn collisions.
The 90 deg board yaw is what makes this cheap: with the long axis ACROSS the push
direction, `BOARD_HALF_X`, `BOARD_HALF_Z` and `STANDOFF` are all unchanged.

**Found — two more shared-tool defects, both fixed minimally and both pre-existing.**
* `verify_task._hist` crashed the whole run (not the gate — the RUN) on a fixture written
  to a fixed pose: numpy refuses 30 bins over a 1.19e-07 span (the float noise of adding
  and subtracting the env origin). Degenerate ranges are now widened, and the plotting is
  wrapped so a histogram can never take a gate down with it.
* `audit_workspace` applied the top-down GRASP ceiling (0.55) to entities in its own
  `_FIXTURES` table, while `verify_task.g4_g5` — which imports that very table to stay in
  sync — gives them `GRASP_RADIAL_MAX + 0.05`. The two tools disagreed about the same
  entity. Matched the audit to `verify_task`.
* Separately, the pivot wall's `object_site` moved to its INNER face. That face is the
  only point on the wall the task touches and the only one every `wall_spawn_range`
  derivation is written in terms of, and it is what the placement audit and
  `tests/test_workspace_placement.py` measure — so the audited radial is now 0.508, and
  no exemption is needed anywhere.

**Found — the LEAP peg-insertion variant would have silently broken.**
`leap_peg_insertion_env_cfg` sets its own `stack_height=0.01` on the same shared assets.
With the goal 25 mm below the peg's resting root height and the default 0.02
`height_threshold`, that Class-C task becomes unsatisfiable. Raised to 0.035 with the
same derivation. Out of my 25-task scope but on my assets, so fixed.

**G3/G4/G5 — all four PASS at `--num-resets 1000`** (json under `docs/cl_v2/runs/<Task>/`):

| task | success_at_reset | oracle | spawn collisions | envelope |
|---|---|---|---|---|
| Peg-Insertion | 0/1000 | 1.000 | 0 | object 0.286-0.526, base 0.285-0.531 |
| Axial-Extract | 0/1000 | 1.000 | 0 | plug 0.400-0.489 |
| Edge-Grasp | 0/1000 | 1.000 | 0 | plate 0.372-0.453, ledge 0.390-0.434 |
| Pivot-Lift | 0/1000 | 1.000 | 0 | board 0.380-0.426, wall 0.508 |

`audit_workspace --keyword {Peg,Axial,Edge,Pivot}`: **0 placement problems** each.
`benchmark_smoke`: all PASS (including `Mjlab-Peg-Insertion-Leap`).
`pytest tests/test_workspace_placement.py tests/test_classical_teachers.py
tests/test_class_a_wave1.py tests/test_class_a_expansion.py`: **149 passed**.

**Decided (logged, not escalated):**
* **Square peg, square hole**, not round. The collision representation for a hole must be
  boxes, and a square bore is represented by boxes EXACTLY; a round bore approximated by
  boxes leaves four corner voids that contradict the visual mesh. It also keeps the
  teacher's yaw constraint meaningful (a 45 deg-rotated 25 mm post has a 35.4 mm diagonal
  and will not enter a 30 mm bore).
* **Clearance 2.5 mm per side** (bore 30.0, peg 25.0), inside the 2-4 mm band a real toy
  sorter uses and slightly tighter than the primitive's 3.0.
* **The ledge is built, not downloaded.** Every scanned wooden box in reach is the wrong
  shape — `CheeseBox_01`, the closest, measures 240 x 107 x 66 mm and its 107 mm depth
  cannot hold a 150 mm plate at all. A rectangular oak riser is itself a real object.
* **Board mass 0.17 kg** (paulownia). The primitive's 0.06 kg implied a 250 kg/m^3 balsa
  board; full hardwood at this size is 0.37 kg and needs 3x the pivot torque, which would
  be a difficulty change smuggled in as an asset change.

**Next:** G6 (rollouts + publish) and G7 (n=128 on GPU 3) for all four.

### 2026-09-09 — G7 diagnostics: what the two zero-SR teachers actually do now — W1-c

**Context:** n=32 screen on GPU 3 came back Peg-Insertion 0.156 / 0.031 (two runs),
Axial-Extract 1.000 / 1.000, Edge-Grasp 0.000, Pivot-Lift 0.000. Before spending the
n=128 budget I instrumented the two zeros on the LOGIN NODE CPU (n=4, one episode,
true state read from `scene[...].data`, not the noisy obs) to find out whether the
CL-V2 placement change had moved the failure.

**Found — Pivot-Lift: the cl25 M4 blocker IS fixed; a second, different mechanism is
now the blocker.**
* cl25's signature was the arm plateauing **10-17 cm** short of the seat target and
  never touching the board. It is gone: the gripper now converges to
  `board_x - 0.049..0.072` against a `STANDOFF` of 0.076 — a **2.6 cm** residual, and
  the pad is genuinely in contact with the board's near face (the board moves).
* What blocks it now: the board only creeps. Over the 100-step PIVOT phase the board
  advances **5 mm** (0.397 -> 0.402) while the gripper advances 16 mm into it, and the
  board is simultaneously pressed **3 mm into the ground plane** (spawn z 0.0101 ->
  0.007). `cos_tilt` never leaves 1.00, i.e. it never starts to rotate, and the board's
  leading edge stops ~46 mm short of the wall. The push is jamming, not sliding: the
  normal force the position servo develops through a stuck contact multiplies the
  ground friction (mu 0.7) faster than it adds forward force.
* **A real constant bug found on the way, tested, and deliberately NOT applied mid-run:**
  `CONTACT_Z_ABOVE_CENTER` is used as a **site-space** offset (`pos_err` is a `gripper`
  site error) but its comment reasons in **pad space** — the two differ by
  `workspace.SITE_TO_FINGERTIP` = 0.019. At 0.0 the commanded site sits at the board's
  mid-thickness, which puts the PAD 19 mm below it, i.e. 9 mm underground, and the
  resulting downward press is part of the jam. Monkey-patched to 0.019 and re-run:
  board travel 5 mm -> 8 mm, the sinking is unchanged and SR is still 0.000. Left the
  teacher alone so the n=128 number below measures the teacher cl25 measured; flagged
  for the lead in the handover list.

**Found — Edge-Grasp: the push now works; the SIDE approach knocks the plate off.**
Per-step trace: `overhang` goes -0.009 -> **+0.053** by t=60, past
`OVERHANG_TARGET` (0.048), and the state machine advances PUSH -> RETREAT -> SIDE_HOVER
cleanly. Then in SIDE_INSERT the plate is shoved a further 80-230 mm toward the robot
and drops off the riser: `lift` goes +0.008 -> **-0.062** (the plate is on the ground,
not on the riser) in 3/4 envs. The gripper site is ~34 mm ABOVE the plate's centre when
it commits, so the lower jaw arrives at the plate's mid-thickness instead of under it
and strikes the rim edge-on. This is the 4-9 cm side-approach IK residual the teacher's
own `SIDE_STANDOFF_X` docstring already documents, and the round plate makes it worse
than the primitive box did: the pinch point on a circle is a tangent, so any y error
slides the jaw along the curve instead of onto a flat face.

**Decided:** both stay **characterised failures** for this wave, exactly as the brief
scopes them (`render failure.mp4`, publish, record). What CL-V2 buys is that
Pivot-Lift's blocker is no longer a placement defect — G4 proves the spawn distribution
is sound and the approach point is reachable — so a future teacher fix has something to
work with. Neither failure is caused by the new assets: both reproduce the cl25 0.000.

### 2026-09-09 — G6 + G7 measured and published; all four rows green — W1-c

**Context:** holder 20277, **GPU 3**. The dgx1 node was carrying all five wave agents'
teacher runs at once (load average 17, each Python IK loop getting 50-68% of a core),
so the two sweeps below took roughly two hours of wall time between them.

**Did:** `test_classical --num-envs 32 --num-episodes 4` (the cl25 n=128 protocol) and
`render_rollout --num-episodes 128 --batch-size 32` for all four tasks, then
`publish_v2.sh` with `asset.json`.

**Found — G7 at n=128 on HEAD `41b36cb`:**

| task | cl25 | v2 | rollout's own independent n=128 |
|---|---|---|---|
| **Axial-Extract** | 0.781 | **1.000** (128/128) | 0.992 (127/128) |
| **Peg-Insertion** | 0.031 | **0.117** (15/128) | 0.078 (10/128) |
| Edge-Grasp | 0.000 | 0.000 | 0.000 |
| Pivot-Lift | 0.000 | 0.000 | 0.000 |

Both improvements are far outside the ~+-0.05 cl25 noise band and both are explained by
the geometry:
* **Axial-Extract +0.219.** The pinched face went from a 40 mm x 24 mm cylinder to the
  real Schuko moulding, 50 mm across and 34 mm tall with 29 mm of it standing proud of
  the faceplate. A 50 mm round face is both wider (more margin inside the 80 mm aperture)
  and taller (a vertical miss no longer slips off the cap), and the pinch stays
  orientation-free because it is round. Nothing else about the task changed: same
  frictionloss, same joint range, same mount band, same `object_site`.
* **Peg-Insertion +0.086 (3.8x).** The 45 deg x 6 mm lead-in is real collision geometry,
  so the capture radius went from the bare 2.5 mm clearance to ~8.5 mm; a descent that
  used to wedge the peg on a square rim now funnels it in. Screening runs at n=32 agreed
  (0.156 / 0.062 / 0.031 across three independent 32-env batches).
  The teacher's own re-derived constants (`INSERT_DEPTH` 0.025 -> 0.050,
  `hover_height` 0.13 -> 0.105) were required by the `stack_height` change, not tuning.

**Published** (still + colliders + turntable + rollout + asset.json):
https://cl.untuai.com/v2/Mjlab-Peg-Insertion-Franka/ ,
`.../Mjlab-Axial-Extract-Franka/` , `.../Mjlab-Edge-Grasp-Franka/` ,
`.../Mjlab-Pivot-Lift-Franka/` .
For the two-asset tasks the card's `still.png` / `colliders.png` are a side-by-side
composite of both objects, and the turntable is the more distinctive one (board / plate /
chopping board). The mp4s were then deleted from `docs/cl_v2/renders/` — they live on
the site, per the brief; the PNGs stay in git.

**One shared-site fix:** `site/index.html` claimed "no rollout yet" for any task whose
teacher is a characterised failure, because it only probes `teacher.mp4`. It now falls
back to `failure.mp4` and labels it "teacher rollout (failure)". That affects four cards
beyond mine (Throw-To-Bin, Strike-Slide, ...). Peg-Insertion also renders only a failure
clip: at SR 0.078-0.117 the render phase's 12 single-env episodes had a 63% chance of
catching a success and did not.

**Verified after everything:** `pytest tests/test_workspace_placement.py
tests/test_classical_teachers.py tests/test_class_a_wave1.py tests/test_class_a_expansion.py`
-> **149 passed**; `audit_workspace --keyword {Peg,Axial,Edge,Pivot}` -> 0 placement
problems each; `benchmark_smoke` -> PASS on all four plus `Mjlab-Peg-Insertion-Leap`.

**For the lead (handover):**
1. **Mechanics changed** on Pivot-Lift only: the board spawn band and the wall pose moved
   (see `free/wall/PROVENANCE.md`), and the board yaw range narrowed. Peg-Insertion's
   success THRESHOLDS moved (`stack_height` 0.01 -> 0.035, `height_threshold` 0.03 ->
   0.015) — the predicate shape is unchanged and it is now strictly better posed (the
   old goal was 25 mm below the peg's own resting height, which is why G5's oracle read
   0.000 on the primitives). Edge-Grasp's `plate_rel_x/y` and `ledge_spawn_range.x`
   narrowed with the bigger plate and riser; the push contact point is unchanged.
2. **Shared-tool edits I made**, all pre-existing defects, all reproduced on the
   primitives: `verify_task` `graspable="thin_axis"` (Edge-Grasp / Pivot-Lift),
   `oracle_root` (Peg-Insertion), degenerate-range histograms; `audit_workspace`
   `_FIXTURES` now exempts the grasp ceiling the way `verify_task` already did;
   `site/index.html` failure-clip fallback; `leap_hand/env_cfgs.py` peg `stack_height`.
3. **Two teacher fixes I found but did NOT apply**, so the G7 numbers measure the same
   teachers cl25 measured: `pivot_lift.CONTACT_Z_ABOVE_CENTER` is a site-space offset
   used as if it were pad-space (off by `workspace.SITE_TO_FINGERTIP` = 0.019; tested,
   board travel 5 mm -> 8 mm, SR unchanged), and `peg_insertion.carry_tol` (0.012) was
   chosen when the hole had no lead-in — the chamfer's 8.5 mm capture radius would now
   justify ~0.020 and should recover some of the CARRY-phase deadlocks that file's own
   `CARRY_TIMEOUT` note documents.
4. Peg-Insertion keeps one predicate weakness the primitives also had: a peg lying FLAT
   on the board top is within `height_threshold`. It is now much harder to reach (a
   25 mm peg lying across a 30 mm hole falls in), and no teacher produces that pose, but
   a one-sided height test would close it properly.

<!-- ===== W2-a ===== -->
# CL-V2 — W2-a log (plate-mounted mechanisms)

Owner of: Push-Button, Turn-Lever, Rotate-Valve, Flip-Switch, Push-Flap.
Holder job 20277 (`hold_dgx_amit`, dgx1), **GPU index 1** for every GPU command.
Repo HEAD `1127d12` + the lead's uncommitted W0 work. Never commit.

---

### 2026-09-09 — Survey + build plan — W2-a

**Context:** Read AGENT_BRIEF / GOAL / PLAN / CODE_MAP §3-§4 / STATUS. PartNet-Mobility
unreachable (D8), so all five mechanisms are BUILT in headless Blender 4.2.23
(`~/tools/blender/blender`) and textured from ambientCG (CC0) via
`asset_pipeline fetch ambientcg:<id>`.

**Found (textures screened, ambientCG CC0, 1K JPG -> PNG by the pipeline):**
`Metal009` brushed steel, `Metal029` matte black, `Metal032` bright brushed steel,
`Metal038` dark scratched steel, `PaintedMetal001` yellow/orange chipped,
`PaintedMetal002` blue chipped, `PaintedMetal004` red chipped, `PaintedMetal006`
green chipped, `Plastic007` smooth red-orange plastic, `Rubber004` dark rubber.
`Metal012` is a chrome ball with an HDRI baked into the albedo — unusable, rejected.

**Found (a real trap for every mesh-using agent):**
`audit_workspace._geom_half_height` falls back to `geom_rbound` (a bounding SPHERE)
for mesh geoms. `_floor_penetration` and
`tests/test_workspace_placement.py::test_mechanism_drops_are_swept_over_the_joint_range`
both call it, so the moment a mechanism grows a textured visual mesh, its measured
"drop below mount" jumps to the mesh's bounding-sphere radius (e.g. a flat 200x200x10
plate reports 0.14 m instead of 0.005 m). That inflates `MECHANISM_DROP_BELOW_MOUNT`,
raises `min_mechanism_mount_z`, and silently moves every mechanism out of the band the
task was tuned for.

**Decided:** fix `_geom_half_height`'s mesh branch to use the compiled mesh vertices
(`mesh_vert` projected on the world z row of `geom_xmat`), which is exact and reduces
to the existing behaviour for primitives. Verified against a rotated tetrahedron:
exact min-z, vs. `rbound` under-reporting by 25 mm on that toy case. Shared-file edit,
one branch only — flagged here for the lead and the other wave agents.

**Decided (design, per task):** keep every joint / body / site / geom NAME and every
site offset from CODE_MAP §3; keep the collision representation as PRIMITIVES (allowed
explicitly by the brief, and much cheaper against `nconmax=60`), sized to the new
visual meshes; add textured mesh geoms as `contype=0 conaffinity=0 group=2 mass=0`
visuals. Contact params (`condim=3 friction="1 0.03 0.003" contype=2 conaffinity=1
solref="0.01 1"`) carried verbatim onto every collider. `contype=2/conaffinity=1`
means mechanism parts do not self-collide, so overlapping a plunger with its barrel is
free.

**Next:** build the five mechanisms in Blender, package per part, then G2->G7.

---

### 2026-09-09 — Five plate-mounted mechanisms BUILT and packaged (G1+G2) — W2-a

**Context:** HEAD `1127d12` + the lead's uncommitted W0 tree (`verify_task` reports
`@ 41b36cb`, the working-tree hash it computes). Holder 20277, **GPU 1** for every
render / SR run. All mesh work on the login node CPU.

**Did:**
- Screened ambientCG (CC0) and picked ten material sets; rejected `Metal012` (a chrome
  ball with an HDRI reflection baked into the albedo — unusable as a texture).
- Wrote `~/assets_raw/blender/w2a/{mechlib.py,build_*.py,pack.py}`: a Blender 4.2
  primitive/bevel/boolean kit with a hand-rolled OBJ writer. The writer computes UVs
  itself (box or cylindrical projection at a PHYSICAL texel density, metres per tile)
  and splits every triangle, because (a) `bpy.ops.uv.*` needs a 3D-view context that
  headless Blender does not have and (b) `bpy.ops.wm.obj_export` re-merges UVs and
  needs axis flags to stay identity. `pack.py` then loads each part with trimesh,
  attaches the (optionally re-tinted) ambientCG albedo and calls
  `asset_pipeline.package(..., collider="none")`.
- Built and packaged all five mechanisms. Per-part textures at **512 px** to stay
  inside the 5 MB per-asset budget.
- Added the Franka `spec.assets = update_assets(...)` pattern to all five
  `<asset>_constants.py`, and updated the Push-Button mocap goal marker from a
  60x60x10 mm box to a cylinder matching the new 45 mm cap.

**Found (sizes, all within budget):** button 2.9 MB / 6 972 tris over 5 parts;
lever 1.3 MB / 4 388 over 2; valve 2.8 MB / 8 308 over 3; switch 1.5 MB / 4 348 over 2;
flap 1.4 MB / 3 236 over 2. Every mesh geom is `contype="0" conaffinity="0" group="2"
mass="0"`; every collider stays a primitive with the ORIGINAL contact parameters.

**Found (the important one — nothing had to move):** the swept
`MECHANISM_DROP_BELOW_MOUNT` is **unchanged for all five** — button 0.0300, lever
0.1500, valve 0.1500, switch 0.0600, flap 0.1600 — because in each case the mount
plate/post still governs the lowest swept point and I sized the visual meshes to sit
inside it. So `workspace.py` needed **no edit**, mount heights are identical, and the
G4 reach distributions are the ones the tasks were tuned for. (Measured with the fixed
`_geom_half_height`; with the old `rbound` fallback the button alone would have jumped
0.030 -> 0.161 and been mounted 13 cm higher.)

**Decided (per task, the open questions):**
1. **Push-Button — keep the 50 mm stroke.** Shortening it would mean editing
   `PushButtonCommandCfg`'s target and threshold in the shared `commands.py`, i.e.
   changing the benchmark's difficulty for a cosmetic gain. Instead the geometry was
   made honest about it: the cap rides a 62 mm bright-steel plunger rod that retracts
   into a 41 mm black guide barrel — a palm/plunger button, which really does travel
   this far. Logged as the one non-real dimension. Clearance verified over the whole
   travel: 5 mm of daylight between the cap underside and the collar at full press.
2. **Rotate-Valve — a two-spoke CROSS handle, not a rimmed handwheel.** `object_site`
   is frozen at radius 0.09 and the teacher PINCHES a spoke at radius 0.072; a rim at
   0.09 sits exactly in the fingers' closing path, and a ring of 8+ collider segments
   would eat the `nconmax=60` budget. A cast two-spoke bar handle is a real gate-valve
   style at 212 mm, so the task keeps its name. Recorded as a deviation from the
   brief's "spoked wheel, ~11 cm radius".
3. **Colliders stay primitives** on all five (explicitly allowed by the brief). A flat
   panel, a square bar, a round cap and a flat flap ARE primitives; a convex-hull mesh
   would cost GJK for zero fidelity and risk the tight contact budget.
4. **Bonnet / barrel / bezel are visual-only.** They live inside the radius the fingers
   ever reach and the mount plate already backstops the arm, so a collider there would
   only spend contacts. `contype="2" conaffinity="1"` means mechanism parts never
   self-collide, so the plunger may sink into its barrel.
5. **Masses held at the cl25 values** so the G7 numbers isolate the geometry/visual
   change: the button cap collider needed an explicit `mass="0.036"` (the old box's
   implicit mass) because it changed from box to cylinder; every other collider keeps
   its implicit mass. The switch's 4 g / 50 g detent masses are the mechanism itself
   and were carried over exactly.
6. **Site marker spheres shrunk 0.010 -> 0.004 m** on all five. Sites carry no physics;
   at 0.010 the blue `base_site` ball poked through the new button cap in every render.

**Next:** G3-G5 verify, then the teacher measurements.

---

### 2026-09-09 — G3-G7 green on all five; published — W2-a

**Context:** HEAD `1127d12` + the W0 working tree (verifier stamps `41b36cb`).
Holder 20277, **GPU 1** for every render / rollout / SR run (shared with W1-a; the runs
are small). Mesh work, `verify_task` and pytest on the login node CPU.

**G3-G5 — `verify_task --num-resets 1000`, all five PASS**
(`docs/cl_v2/runs/{Push-Button,Turn-Lever,Rotate-Valve,Flip-Switch,Push-Flap}/verify.json`):
`success_at_reset 0/1000` and `oracle 1.0` everywhere; step-time ratios vs the 1127d12
primitives 1.15 / 1.05 / 1.11 / 1.05 / — (flap has no baseline: its asset dir is
UNTRACKED at 1127d12, so `git show` cannot fetch one) — all under the 1.3 budget.
Compiled joint ranges read off the LIVE env: button [-0.05, 0], lever [-1.570796, 0],
valve [0, 6.283185], switch [-0.785398, 0.785398], **flap [-1.399754, 0]**.
`audit_workspace --keyword` = 0 placement problems x5; `benchmark_smoke` PASS x5;
`pytest test_workspace_placement + test_classical_teachers + test_class_a_wave1 +
test_class_a_expansion` = **149 passed** (includes
`test_flip_switch_detent_is_actually_bistable` and
`test_mechanism_drops_are_swept_over_the_joint_range`).

**Found + FIXED — a G5 verifier artefact, not an asset defect.** Push-Button failed G5
with `oracle 0.000`. The oracle writes the joint to the target and takes three free
20 ms steps, but `button_slide` is SPRING-RETURNED (stiffness 1000 on a 36 g plunger,
zeta 0.42, period 38 ms): measured trajectory after the write is +0.0163, -0.0045,
+0.0010, i.e. the spring throws the cap back through zero before the predicate is ever
sampled. **Proved it is pre-existing** by running the same trajectory on
`git show HEAD:.../button.xml` — bit-for-bit identical numbers, same 36 g moving mass.
Added an `oracle_hold` flag to `verify_task.TASKS` (mirroring W1-c's `oracle_root`):
for a spring-returned mechanism the oracle re-seats the joint after each step and
refreshes the metrics, i.e. asks the real G5 question — with the mechanism AT the goal
configuration, does the predicate fire? Same move `test_class_a_expansion.py` already
makes for the switch's detent. Push-Button then reads `oracle 1.0`.

**Found + FIXED — `audit_workspace._geom_half_height` used `geom_rbound` for meshes**
(a bounding SPHERE). This feeds `_floor_penetration` AND
`test_mechanism_drops_are_swept_over_the_joint_range`, so the first textured mesh on any
mechanism inflates its recorded drop (the button's flat 200 mm panel alone would have
gone 0.030 -> 0.161 and been mounted 13 cm higher). Replaced the mesh branch with an
exact projection of the compiled `mesh_vert` onto the world z row of `geom_xmat`;
reduces to the existing behaviour for primitives. **This one matters for every other
wave agent, not just me.**

**Found + FIXED — `publish_v2.sh` failed on a task with no `failure.mp4`.** The file
list is literal paths, so `nullglob` never drops the missing ones; rsync exited 23 and
`set -e` killed the script AFTER everything else had already copied — a successful
publish reported as a failure (hit on Push-Button and Push-Flap, whose teachers are at
0.984 / 1.000 so `render_rollout` never captures a failure clip). Now each candidate is
tested with `-f` before being added.

**G7 — `test_classical --num-envs 32 --num-episodes 4` (n=128) on GPU 1:**

| Task | cl25 | v2 (n=128) | delta | independent n=128 from `render_rollout` |
|---|---|---|---|---|
| Push-Button | 0.984 | **0.969** | -0.015 | 0.984 |
| Turn-Lever | 0.941 | **0.922** | -0.019 | 0.961 |
| Rotate-Valve | 0.473 | **0.461** | -0.012 | 0.438 |
| Flip-Switch | 0.656 | **0.609** | -0.047 | 0.672 |
| Push-Flap | 1.000 | **1.000** | 0.000 | 1.000 |

Every delta is inside the +-0.05 cl25 noise band, and each task's two independent n=128
measurements (test_classical vs render_rollout's own stats phase) bracket the cl25
number. Nothing is blocked.

**Turn-Lever looked like a regression at n=32 and was not.** The first n=32 read 0.812
against a 0.941 baseline. Ran a CONTROL: `git show HEAD:.../lever.xml` swapped in,
measured twice on this same working tree -> 0.906 and 0.844; new asset -> 0.812 and
0.969. The two distributions overlap completely, so the task is simply noisy at n=32 on
this tree and the cl25 0.941 was not reproducible here even on the primitive. n=128
settles it at 0.922. Recorded so nobody re-investigates it.

**Teacher constants — re-derived, not copied.** The colliders were deliberately kept
identical on lever / valve / switch / flap, so the constants that name their half-extents
re-derive to the same numbers, and I state the derivation rather than the value:
- `turn_lever.FACE_STANDOFF = 0.018` = bar collider half-thickness 0.012 + 6 mm pad
  clearance -> unchanged; `CONTACT_FRAC 0.90 x _R_ARM 0.12 = 0.108` still lands on the
  bar (which spans 0..0.140); `_R_ARM`, `_X_OFF`, `_MARKER_OFF` are the frozen site
  offsets -> unchanged.
- `rotate_valve.FACE_STANDOFF = 0.014` = spoke half-thickness 0.012 + 2 mm -> unchanged;
  `PINCH_FRAC 0.80 x 0.09 = 0.072` is on the spoke (0..0.110) -> unchanged.
- `flip_switch.PUSH_STANDOFF = 0.030` clears the 0.013 weight half-width by 17 mm;
  `CONTACT_Z = -0.012` puts contact at z 0.050 on the bat (collider spans -0.004..0.056)
  -> both unchanged. `STROKE_X/STROKE_Z` are ballistic, not geometric.
- `push_flap.FACE_STANDOFF = 0.030` / `CONTACT_FRAC = 0.92` / `PUSH_DEPTH = 0.12` key off
  the panel's -x collider face at x = -0.012, which is unchanged and is exactly where
  the new visual face sits -> unchanged.
- `push_button`: **one constant genuinely moved.** `HOVER_XY_TOL` IS the cap's
  half-width, and the cap changed from a 60 x 60 mm square face to a 45 mm round one:
  0.030 -> **0.020**. At 0.030 the press could commit with the pad entirely off the new
  cap. Measured both at n=128: 0.977 (old) vs 0.969 (derived). A one-episode difference,
  so the derived value ships. `HOVER_HEIGHT 0.07` (>> cap half-height 0.009 + the 0.019
  site-to-fingertip offset) and `PRESS_DEPTH 0.03` (< the 0.05 stroke) re-derive
  unchanged.

**G6 — published.** `render_asset --joint-sweep` (still + turntable + colliders) and
`render_rollout --num-episodes 128 --batch-size 32` (teacher.mp4 / thumb.jpg /
result.json, plus failure.mp4 where the teacher ever fails) for all five, then
`publish_v2.sh` with `asset.json`. Live at
`https://cl.untuai.com/v2/Mjlab-{Push-Button,Turn-Lever,Rotate-Valve,Flip-Switch,Push-Flap}-Franka/`
(verified by listing the remote tree over ssh; a `curl` from the login node returns 000,
so the site is not fetchable from inside the cluster — check it from a browser).
`still.png` + `colliders.png` also copied into `docs/cl_v2/renders/<Task>/`; the mp4s
stay out of git.

**Decided (late, on the renders):** the Push-Flap panel was first textured with ambientCG
`PaintedMetal002` (blue chipped paint). Its chip marks are large and high-contrast, so on
a flat 220 x 300 mm panel they read as random blotches at 0.16 m/tile and as a plainly
visible repeat at 0.075 m/tile — a tiling artefact either way. Replaced with `Metal009`
re-tinted to postbox red: brushed metal has no feature big enough to alias, so it tiles
invisibly, and red makes the mail-chute reading immediate. Ledger, PROVENANCE and
asset.json updated; teacher rollout re-rendered so the video matches the shipped texture.

**Next (for the lead):** three shared-file fixes landed here that the other agents
benefit from and should know about — `audit_workspace._geom_half_height` (mesh branch),
`verify_task` `oracle_hold`, and `publish_v2.sh` (optional files). None of them changes
any task's physics.

<!-- ===== W2-b ===== -->
# CL-V2 — W2-b log (cabinet family: Open-Drawer, Open-Door, Slide-Window, Open-Lid)

Append-only. Template: Context / Did / Found / Decided / Next.

---

### 2026-09-09 — W2-b day 1: four cabinet-family assets built, three big findings — W2-b

**Context:** Build realistic assets for Open-Drawer, Open-Door, Slide-Window, Open-Lid.
HEAD `1127d12` + the lead's W0 work. Holder job **20277** (`hold_dgx_amit`, dgx1),
**GPU index 2** for every GPU command. PartNet-Mobility unreachable (D8), so all four
are BUILT procedurally.

**Did:**
- Wrote a small procedural builder, `~/assets_raw/blender/W2-b/{mechbuild,build_drawer,
  build_door,build_window,build_lid}.py`. Every cabinet part is a chamfered board, so
  the library is a chamfered-box / chamfered-cylinder generator plus a real-world-scale
  box UV projector; one mesh + one texture per part, colliders written as MJCF box
  primitives. Textures: ambientCG Wood048 / Wood067 / Metal032 and Poly Haven
  `oak_wood_planks`, all CC0, AO multiplied into the albedo, PNG, 256-512².
  Budgets: 920-1404 visual tris and 0.45-0.70 MB per asset (limits 50k / 5 MB).
- Assets: a 450x400x465 mm light-oak bedside cabinet (drawer); a 300 mm walnut wall
  cabinet with a 20 mm door leaf (door); a 560x440 mm aluminium sliding window with two
  glazed sashes (window); a 220 mm plank-built oak chest with brass strap hinges (lid).
  `PROVENANCE.md` in each asset dir; `build_manifest.json` next to it.
- Added the Franka `assets`-dict pattern to all four `*_constants.py`
  (`spec.assets = update_assets({}, XML_DIR/"assets", spec.meshdir)`).
- `<compiler angle="radian" meshdir="assets" texturedir="assets"/>` in all four XMLs;
  door and lid ranges rewritten in radians.
- Re-measured `MECHANISM_DROP_BELOW_MOUNT` (swept over the joint range):
  drawer 0.300 -> **0.315**, door 0.800 -> **0.318**, window 0.220 (unchanged),
  lid 0.157 -> **0.158**.

**Found (three things that cost real SR, all measured on GPU 2, n=32):**

1. **The Franka hand capsule is the binding constraint for every cabinet.** From
   `panda.xml`: `hand_capsule` is radius 0.04, half-length 0.06, centred 0.07 m up the
   approach axis from the `gripper` site, with its long axis along the FINGER-CLOSING
   axis. So the wrist always occupies `site_xy +- 0.04` (and +-0.10 along the closing
   axis). Any carcass face inside that envelope fouls the hand.
   - Drawer: my first build put the carcass front flush at x = -0.02 with a 5 mm
     overhanging top. Teacher 0.906 (cl25 asset, same code/GPU) -> **0.375**. A/B:
     carcass colliders disabled -> 0.875; mount height alone -> no effect. Fix: real
     **full-overlay fronts** — the drawer front and cupboard door stand 20 mm proud and
     the carcass front plane is x = 0, so nothing collidable is ahead of the wrist.
   - Door: recessing the carcass front to x = +0.015 (a real concealed-hinge crank gap)
     took the OLD cl25 pinch teacher from 0.000 to **0.406** with no other change.
2. **Open-Lid was winnable by doing nothing, on the cl25 asset too.** With gravity on
   and the hinge on the far edge, the lid is a drop-down flap: it free-falls from 0 to
   the -1.309 rad stop in ~0.25 s (~12 control steps) with zero actions. cl25 XML:
   identical (0.776 kg, same trajectory). `verify_task` G3 fails it
   (`joint_drift_task_gravity` = 1.31 rad vs a 0.02 limit); `success_at_reset` does not
   catch it because one step is only 0.015 rad. The teacher docstring's "falls to
   -0.735 rad and stalls there" does not reproduce on HEAD. Fixed with a real
   **0.62 N m friction ("torque") hinge** — 7% above the 0.577 N m peak gravity torque,
   so the lid stays where it is put. NOTE: `frictionloss` magnitude alone does nothing
   here — the residual creep (0.143 rad / 2 s) is set by the friction constraint's
   softness, so `solimpfriction="0.99 0.999 0.001 0.5 2"` is what actually holds it
   (0.0138 rad / 2 s, 0.021 rad over a whole 3 s episode).
3. **Open-Lid's lid sweeps through its own box** at every angle past ~-27 deg, and
   nothing collides because every mechanism geom is `contype="2" conaffinity="1"`
   (2 & 1 = 0). With the hinge position, axis and target the brief pins, this is
   geometrically unavoidable — see the asset's PROVENANCE.md. Reported, not fixed.

**Decided:**
- Door: real product substitution rather than a threshold change. 300 x 600 x 20 mm,
  2.7 kg leaf; hinge moved from y = -0.30 to y = 0 (lever arm 0.551 -> 0.253 m);
  damping 0.10 -> 0.05; carcass recessed to x = +0.015; 30 mm-projection bar pull
  giving the SAME 10 mm hook slot the drawer has. Success threshold untouched.
  Downstream: `OpenDoorCommand.handle_to_hinge_dist` 0.55 -> 0.25 (goal marker only),
  teacher `_R0` (-0.040, 0.551, 0) -> (-0.040, 0.250, 0).
- Door teacher rewritten from the cl25 side pinch (0.000) to a **top-down hook**, a
  port of `open_drawer.py`: closed fingertip wedged in the bar/leaf corner, dragged
  along the hinge arc recomputed every step.
- Window joint range 0..0.25 -> **0..0.24 m** (the real travel of a 240 mm sash in a
  480 mm opening); command target 0.22 and threshold 0.03 untouched.
- Door joint range written as `0 1.5708`, not `0 1.5707963`: a bare `range="0 90"`
  compiles a hair BELOW `OpenDoorCommand`'s 1.5708 target and `verify_task` fails G3
  on it. That trap is latent in cl25 too.

**Next:** finish the teacher re-derivation (drawer 0.750 vs the 0.906 cl25 baseline
still has a gap), then G3-G7 for all four.

### 2026-09-09 — Open-Lid: the axis sign, and what it cost — W2-b

**Context:** Open-Lid's G3 failed (`joint_drift_task_gravity` = 1.31 rad vs a 0.02
limit) on both the CL-V2 and the cl25 asset. GPU 2, holder 20277.

**Did / Found (in order, all measured):**
1. Confirmed the drop-down lid is degenerate on cl25 too: zero actions reach the
   -1.309 rad stop in ~0.25 s. cl25 0.776 kg, v2 0.543 kg — identical trajectory,
   because gravity torque and inertia both scale with mass.
2. Tried a **friction ("torque") hinge**. Learned that `frictionloss` magnitude alone
   changes nothing: 0.58 .. 0.90 N m all creep 0.143 rad in 2 s, because the creep rate
   is set by the friction constraint's *softness*. `solimpfriction="0.99 0.999 0.001
   0.5 2"` is what actually holds it (0.0138 rad / 2 s). With a 0.62 N m stay (7% over
   the 0.577 N m peak gravity torque) G3 passed — and the teacher fell to 0.125,
   stalling at 0.30-0.89 rad.
3. Diagnosed why it stalls: **the lid sweeps through its own box.** Past ~-27 deg the
   grasp point is inside the carcass, where the arm cannot follow. With the pinned
   hinge position, axis and target this is unavoidable — any lid long enough to put
   `object_site` at x = -0.09 sweeps a 0.19 m arc down through wherever the box is.
   So the drop-down lid is either winnable by doing nothing, or unsolvable.
4. **Reversed the hinge axis to `(0,-1,0)`.** The same -75 deg target now lifts the lid
   up and back over the hinge. Zero-action drift over a whole 3 s episode: 0.0014 rad.
   No interpenetration. Swept drop 0.157 -> **0.070 m**, mount z 0.187 -> 0.100.
   This makes the asset match what `open_lid_env_cfg.py` and `OpenLidCommand` have
   always documented ("gravity opposes the motion throughout").

**Teacher rewrite** (a lift cannot be a push, so this is the only cabinet-family task
that needs a real grasp). Four measured dead ends before it worked:
- 30 x 100 mm lifting batten, pads closing along the lid's local x: **0.000**. The full
  3x3 orientation target fixes the yaw to one of two equivalent signs and picks the far
  one, so the wrist spends the episode fighting a 180 deg roll. **Axis-only orientation
  targets fixed it** — a Franka top-down pose already presents world y as its closing
  axis, and the hinge axis IS world y, so a y-closing pinch is invariant under the
  lid's rotation.
- 30-step descent timeout closed the fingers on air 0.18 m away, then pressed the
  closed pads down on the lid (joint driven +0.24 rad past its closed stop). Timeouts
  are now gated on a distance test.
- Integral gain 0.3 with no band wound up 0.09 m of bias on the 0.3 m approach and
  overshot the knob by 100 mm. Anti-windup band 0.10 m.
- `GRASP_UP` 0.014 put the pads' lower faces through the lid panel (the descent settles
  ~8 mm below target), so the hand pressed instead of gripping.
Asset changes that came out of this: the lifting batten became a **mushroom knob**
(28 mm stem under a 44 mm cap — the cap is a mechanical stop against the pads sliding
off during the lift, which took the best-reached angle from -0.60 to -1.01 rad), and
the lid boards went to 450 kg/m³ (plywood, not oak).

**Decided:**
- `episode_length_s` 3.0 -> **5.0** for Open-Lid only. Under cl25 the lid free-fell to
  the target in 12 steps so the budget never mattered; it is now a 1.0 m reach + grasp +
  1.309 rad lift against gravity. 5.0 s is in line with Rotate-Valve (8.0), Tool-Pull
  (12.0), Edge-Grasp and Pivot-Lift (6.0). **The success threshold is untouched.**
- Mount band x (0.52, 0.59) -> **(0.40, 0.46)**: the knob now travels AWAY from the
  robot as it lifts, and the wrist sits 0.068 m beyond the knob at full travel.
- The arc lead is the most sensitive constant in the teacher (n=16, CPU):
  0.35/0.15 -> 0.000, 0.55/0.15 -> 0.000, **0.70/0.30 -> 0.125**, 0.95/0.50 -> 0.063.

**Next:** the remaining failure mode is retention at the top of the arc — the cap stops
the pads sliding along the knob's axis, but near -75 deg the load direction has rotated
90 deg relative to it. A hook that engages UNDER the cap (friction-free, like the
drawer's) is the obvious next teacher.

### 2026-09-09 — W2-b G3-G7 results — W2-b

**Context:** All four assets final. Holder 20277, **GPU 2** for every measurement.
HEAD at measurement time `41b36cb` (the lead's uncommitted W0 tree).

**verify_task, 1000 resets each, CPU** (`docs/cl_v2/runs/<Task>/verify.json`):

| Task | G3 | G4 | G5 | notes |
|---|---|---|---|---|
| Open-Drawer | PASS | PASS | PASS | 7 colliders, radial 0.420-0.529, success@reset 0/1000, oracle 1.0 |
| Open-Door | PASS | PASS | PASS | 7 colliders, radial 0.440-0.482, range [0, 1.5708] covers the 1.5708 target |
| Slide-Window | PASS | PASS | PASS | 6 colliders, radial 0.460-0.521 |
| Open-Lid | PASS | PASS | PASS | 8 colliders, radial 0.310-0.376; joint drift under task gravity 0.0014 rad |

`audit_workspace --keyword` on all four: **0 placement problems**.
`benchmark_smoke`: 4/4 PASS. `pytest test_workspace_placement + test_classical_teachers
+ test_class_a_wave1 + test_class_a_expansion`: **149 passed**.

**Contact budget:** measured max per-world contacts during a teacher rollout —
drawer 4, door 6, window 4, lid 8, against `nconmax=60` (MJWarp's nconmax is
per-world; its own default heuristic is 45). No change needed; left at 60/650.

**G7, n=128 on GPU 2:**

| Task | cl25 | v2 | delta |
|---|---|---|---|
| Open-Drawer | 0.888 | see the final row (re-measured after the staged approach) | |
| Open-Door | 0.000 | **0.297** | +0.297 |
| Slide-Window | 1.000 | **0.969** | -0.031 (inside the ±0.05 band) |
| Open-Lid | 1.000 | **0.102** | -0.898, and the whole point: cl25's 1.000 was gravity's |

**Budgets:** 920-1812 visual tris and 0.48-0.83 MB per asset directory (limits 50k tris,
5 MB). Frozen-interface check (bodies, joints, geoms, site offsets, mocap root, compiled
joint ranges) passes on all four.

### 2026-09-09 — W2-b DONE: four rows green, all published — W2-b

**Final G7 (n=128, GPU 2, HEAD `41b36cb`):**

| Task | cl25 | v2 | note |
|---|---|---|---|
| Open-Drawer | 0.888 | **0.938** | +0.050. First measurement on the real carcass was 0.734; the staged approach (front standoff, then straight down) recovered it and then some. |
| Open-Door | 0.000 | **0.297** | +0.297 on an unchanged success threshold. The task is now solvable at all. |
| Slide-Window | 1.000 | **0.969** | -0.031, inside the ±0.05 band. |
| Open-Lid | 1.000 | **0.102** | -0.898. cl25's 1.000 was gravity's: a null policy scored it. See the hinge-axis entry above. |

Published to https://cl.untuai.com/v2/ for all four (still.png, colliders.png,
turntable.mp4, teacher.mp4, failure.mp4 where one exists, thumb.jpg, result.json,
asset.json). `still.png` is a front-3/4 shot from the same studio rig as `render_asset`
(`~/assets_raw/blender/W2-b/front_still.py`) — these assets have a FRONT, and
render_asset's fixed azimuth lands behind them, which would have put the back of a box
on every site card.

**Shared files I touched, and only my task's lines:**
* `tasks/manipulation/workspace.py` — `MECHANISM_DROP_BELOW_MOUNT` for door/drawer/lid,
  plus the stale "the door drops 0.80m" sentence in the module docstring.
* `tasks/manipulation/mdp/commands.py` — `OpenDoorCommand.handle_to_hinge_dist`
  0.55 -> 0.25 (goal-marker maths only).
* `tasks/manipulation/config/franka/env_cfgs.py` — the lid's reset-event mount band.
* `tasks/manipulation/open_lid_env_cfg.py` — `episode_length_s` 3.0 -> 5.0.
* `classical/{open_drawer,open_door,open_lid,slide_window}.py`.

**Final tests:** `pytest tests/test_workspace_placement.py tests/test_classical_teachers.py
tests/test_class_a_wave1.py tests/test_class_a_expansion.py` -> **149 passed**.
`audit_workspace` 0 problems on all four; `benchmark_smoke` 4/4.

**Three things the lead should know:**
1. **The Franka hand capsule sets the clearance budget for every cabinet.** r=0.04,
   half-length 0.06, centred 0.07 m up the approach axis from the `gripper` site, long
   axis along the finger-closing axis. Any collidable face inside `site_xy ± 0.04`
   fouls the wrist. Two of my three big SR swings were this, and W2-a's plate-mounted
   family will hit it too if a back plate ever sits in front of a handle.
2. **cl25's Open-Lid was winnable by doing nothing**, and its lid swept through its own
   box. Both reproduce on the cl25 XML. I reversed the hinge axis so the task matches
   what its own cfg documents; the SR fell from a free 1.000 to a real 0.102. If the
   program needs distillable teacher data from this task more than it needs the task to
   be sound, the one-line revert is `axis="0 -1 0"` -> `axis="0 1 0"` in lid.xml (plus
   the drop back to 0.158 and the mount band) — but the G3 joint-drift failure comes
   back with it.
3. **Open-Lid's remaining failure mode is retention at the top of the arc.** The knob's
   cap stops the pads sliding along the knob axis, but by -75 deg the load direction has
   rotated 90 deg relative to it. A hook that engages UNDER the cap (friction-free, like
   the drawer's) is the obvious next teacher, and the collider is already shaped for it
   (`handle` stem + `handle_cap`).
