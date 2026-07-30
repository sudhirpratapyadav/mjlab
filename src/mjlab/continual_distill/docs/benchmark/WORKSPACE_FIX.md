# Class A workspace + placement fix

> Systematic fix for object placement / layout / workspace-boundary errors, found by
> watching the recorded task videos (2026-07-30). Companion to CLASS_A_EXPANSION.md.

## The problem

Reviewing the 28 preview videos showed objects at implausible positions: at the very
edge of reach, mechanisms the arm can only paw at, one object not placed per-env at all.
Spawn ranges had been chosen per task by eye and copied between tasks, with no shared
definition of where the arm can actually work.

## Step 1 — define the workspace ONCE (done)

Class A is one embodiment, so it has one workspace. `tasks/manipulation/workspace.py`
is now the single source of truth, derived by measurement rather than intuition:
FK over 600k joint samples drawn from 90% of each joint range (staying off the hard
stops that `joint_pos_limits` penalises at -10.0), recording the `gripper` site pose.
Reproduce with `python -m mjlab.scripts.audit_workspace --measure`.

TOP-DOWN radial reach (sqrt(x^2+y^2)), by site height:

| site z | p5 | p50 | p90 | p95 |
|---|---|---|---|---|
| 0.06-0.14 (grasp a floor object) | 0.274 | 0.415 | 0.704 | 0.771 |
| 0.14-0.25 (low carry) | 0.322 | 0.485 | 0.759 | 0.813 |
| 0.25-0.40 (mid carry) | 0.303 | 0.580 | 0.789 | 0.831 |
| 0.40-0.60 (mechanism height) | 0.267 | 0.572 | 0.730 | 0.768 |

**p90/p95 are not usable workspace.** They are near-singular, fully-extended poses.
The density of comfortable top-down grasp poses peaks at **x ~ 0.30-0.35**.

### The metric that matters: approach freedom

For a spawn box, the fraction of comfortable top-down grasp poses landing inside it.
Low = the arm can only enter that region near singularities, so grasp orientation is
heavily constrained and the reach reward fights the joint-limit penalty. A box in the
tail is not "harder" — it is mis-specified.

| spawn box | approach freedom |
|---|---|
| old lift / stack (x 0.60-0.80) | **1.2%** |
| old tool-pull puck (x 0.78-0.88) | **0.1%** |
| `workspace.GRASP_*` (x 0.30-0.52) | **11.7%** |

## Step 1b — audit (done): 22 problems across 19 of 20 Class A tasks

`python -m mjlab.scripts.audit_workspace` builds every Class A task, resets 64 envs,
and reports actual spawn extents + approach freedom + flags. This is systemic, not a
handful of bad tasks. Full output in the commit; summary:

- **All 4 lift variants + both pushes**: x 0.60-0.80, radial to 0.81, ~1% freedom.
- **Peg-insertion, stack**: x 0.55-0.70, radial ~0.71, ~0.4% freedom.
- **All 7 articulation mechanisms**: handle radial 0.64-0.70, over the 0.58 ceiling.
- **Tool-pull puck**: radial 0.88, 0.1% — the worst in the suite.
- **Place-in-container `container`**: spans +-7.0 with radial 9.9. Real bug — the
  container is a static mocap body that is never written per-env, so it sits at the
  world origin while each env's robot is at its own origin. Only env 0 is correct.

## Step 2 — per-task placement rules

Placement is task-dependent, so each task gets its own rule *inside* the shared
envelope rather than a single global box:

- **lift / push / reorient** (one free object, grasp or push it): `GRASP_*` box.
- **stack / peg-insertion** (two objects, must not overlap): split the `GRASP_*` box
  laterally, one object either side of y=0, keeping both within `GRASP_RADIAL_MAX`.
- **place-in-container**: cube in the `GRASP_*` box; container offset laterally so the
  transport is a genuine lateral carry — AND actually written per-env.
- **tool-pull**: stick inside `GRASP_*` (must be grasped); puck deliberately OUTSIDE
  `GRASP_RADIAL_MAX` but inside the *tool-extended* reach — being out of reach is the
  task's premise, so it is exempt from the grasp ceiling but still bounded.
- **articulation** (door/drawer/button/lever/valve/switch/window/lid): mount so the
  HANDLE — not the mount body — lands within `MECHANISM_HANDLE_RADIAL_MAX`. The handle
  protrudes back toward the robot, so mount x is offset from handle x per asset.

## Method

Step 1 was done centrally because everything depends on it. Step 2 is parallelised per
task family via subagents, each given the measured envelope and the audit output, with
`audit_workspace` as the acceptance gate.

## Findings from step 2

**`GRASP_X_RANGE x GRASP_Y_RANGE` was self-inconsistent.** The ranges bound each axis
independently, so the box CORNER escapes the radial ceiling:
hypot(0.52, 0.25) = 0.577 > GRASP_RADIAL_MAX (0.55). A task sampling uniformly in that
box intermittently spawns outside the envelope — passing one audit and failing the next
depending on the draw. Fixed centrally with `workspace.grasp_box(y_max=...)`, which
keeps the full lateral spread (the y variation is what gives lift/push their character)
and pulls x back to sqrt(RADIAL_MAX^2 - y_max^2) - margin. Shipped in workspace.py so
every task family shares one implementation instead of each rediscovering the clamp.

**The Franka `*_pose_range` command overrides were DEAD CODE for placement.**
`franka_open_door/drawer/button_env_cfg` set `door_command.door_pose_range` etc., but
the mount pose is actually written by the `reset_<asset>_position` RESET EVENT in the
base env cfg. That is why the door's declared `x=(1.5, 1.5)` had no effect and its
handle measured x~0.64. Anyone tuning placement must edit the reset event, not the
command's pose_range.

**Mount position != handle position.** Handles protrude back toward the robot by an
asset-specific offset (door: +0.25 in y; lid: -0.09 in x, +0.05 in z; button: +0.10 in
z). The gripper must reach the HANDLE, so mount ranges are derived from the handle
offset read out of each asset XML — not set directly.

## Results

Articulation — handle radial, all now under MECHANISM_HANDLE_RADIAL_MAX (0.58):

| task | handle radial before -> after |
|---|---|
| Turn-Lever | 0.651 -> 0.513 |
| Rotate-Valve | 0.653 -> 0.514 |
| Flip-Switch | 0.639 -> 0.476 |
| Slide-Window | 0.651 -> 0.521 |
| Open-Lid | 0.532 -> 0.505 |
| Open-Door | 0.700 -> 0.481 |
| Open-Drawer | 0.654 -> 0.525 |
| Push-Button | 0.676 -> 0.486 |

Free objects — approach freedom:

| task | before -> after |
|---|---|
| Lift-Cube / Cylinder / Sphere / Ellipsoid | ~1.1% -> 9.8-10.5% |
| Push-Cuboid / Push-Disc | ~1% -> 8.2-8.3% |
| Reorient-Object | 1.7% -> 5.2% |

Push splits the x band (spawn 0.30-0.41, target 0.41-0.52) so each episode is a real
forward push rather than a nudge. Reorient insets x/y by 0.05 because the cylinder lies
on its side at random yaw and its body extends ~5cm from the spawn point.

## !! DATASET INVALIDATION — door / drawer / button moved

All three had out-of-bounds handles (radial 0.65-0.70) and were moved substantially;
the door most of all (mount shifted ~0.25m in -y and ~0.18m in -x to bring the handle
onto the midline). **Existing teacher datasets and the 0.93 CL baseline for
`Mjlab-Open-Door-Franka`, `Mjlab-Open-Drawer-Franka` and `Mjlab-Push-Button-Franka`
were collected under the old geometry and must be re-collected.** The classical
teachers in `continual_distill/classical/` are hand-tuned to the old handle poses and
will need re-tuning too. This is a deliberate correctness-over-continuity call: the old
placements were only reachable near singularities.

## Status

- [x] Step 1: workspace.py measured + committed
- [x] Step 1b: audit script + full Class A audit (22 problems)
- [x] Step 2: lift/push/reorient (7 tasks)
- [x] Step 2: articulation (8 tasks)
- [x] Step 2: two-object tasks (stack, peg, place-in-container, tool-pull)
- [x] Re-audit: **0 placement problems** (was 22)
- [x] 319 passed / 18 skipped; 23/23 benchmark-smoke; 42 new placement tests
- [x] Videos re-recorded

## Two-object results

| task / entity | radial before -> after | free% before -> after |
|---|---|---|
| Stack object / base | 0.702 / 0.701 -> ~0.51 | 0.4% / 0.2% -> 4.4% / 5.0% |
| Peg object / base | 0.708 / 0.691 -> ~0.52 | 0.4% / 0.2% -> 4.3% / 4.9% |
| Place cube | 0.631 -> 0.508 | 0.9% -> 4.6% |
| Place container | **9.899 (bug)** -> 0.524 | n/a -> 3.8% |
| Tool-pull stick | 0.477 -> 0.53 | 0.7% -> 3.9% |
| Tool-pull puck | 0.884 -> 0.707 | exempt by design |

### The container per-env bug, fixed

Mocap bodies are not reset by the scene, so the bin inherited the single world-frame
pose from its MJCF: it sat at the world origin while each robot sat at its own env
origin, and only env 0 was correct. `PlaceInContainerCommand._resample_command` now
samples a per-env position, adds `scene.env_origins`, and writes the mocap pose.

Subtlety worth keeping: `target_pos` is computed ANALYTICALLY as
`container_pos + site_offset`, not read back from `site_pos_w`. Forward kinematics has
not re-run at that point in the step, so reading the cached site position would return
the PRE-write value and silently reintroduce the bug.

### Audit exemption, made explicit

The tool-pull puck is deliberately outside the grasp envelope — being unreachable by
hand is the task's premise. Rather than leave a permanent misleading flag, the auditor
now has a `_REACH_EXEMPT` table giving the bound that DOES apply (0.58-0.75: outside
direct reach, inside the stick's extension). `tests/test_workspace_placement.py` keeps
the same table, so an exempt entity drifting out of ITS band still fails.

## Regression tests

`tests/test_workspace_placement.py` (42 tests) pins the envelope, because placement
bugs are invisible to every other gate: `benchmark-smoke` still passes (the env builds
and steps), and the success-predicate tests teleport objects into success states so
they never exercise spawn. Verified the per-env test actually catches the container bug
by reintroducing it — it fails with "varies by 6.17m across envs".
