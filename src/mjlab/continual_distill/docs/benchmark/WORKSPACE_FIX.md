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

## Status

- [x] Step 1: workspace.py measured + committed
- [x] Step 1b: audit script + full Class A audit (22 problems)
- [ ] Step 2: per-family placement fixes
- [ ] Re-audit clean, re-record videos, re-run smoke + tests
