# CL-V3 — PLAN

**Next stage:** [CL-V4 RL teachers](../cl_v4_rl/README.md): one independent RL teacher per active task, each above 90% success.

## Current scope — 2026-09-12

**24 active tasks**, using the common 60D observations and normalized 8D actions.
Tool-Pull is **deferred**, not deleted: it needs both the stick and puck poses, while
this observation layout represents only one object and its goal. Keep its task,
assets, registration, teacher code and historical results for future work.
Revisit it when an agreed observation design supports both objects.

Use `mjlab.tasks.manipulation.benchmark.active_cl_tasks()` for new suite runs.
The reviewable list is [active_tasks.json](active_tasks.json). The full registry
continues to contain Tool-Pull and the four object variants outside this CL suite.
Older 25-task scoreboards and results below are historical; Tool-Pull is excluded
from current RL/CL training, evaluation aggregates and completion requirements.


> `GOAL.md` first. `STATUS.md` is the scoreboard (edit via `update_status.py` only),
> `LOGS.md` the append-only trail (agents write `logs/<agent>.md`, the lead merges).
> Starting point: `../cl_v2/STATUS.md` + `HANDOVER.md` (v2 numbers, mechanics changes),
> `../cl25/phase_1/STATUS.md` "Handover to Phase 2" (mechanisms M1–M4, per-task failure
> characterisations — read your task's row and LOGS entries there before anything else),
> `../cl_v2/CODE_MAP.md` (plumbing: scene assembly, task table, teacher constants, tests).

## 1. Order of work

```
W0  infra + frozen init spec + re-baseline          (one agent, GPUs 1-3, ~2 h)
W1  five strategy agents, grouped by mechanism      (GPUs 1-3, two agents per GPU)
W2  lead: n=128 confirmation sweep on every row, videos, site, handover
```

W1 agents may start reading code and the phase-1 failure analyses while W0 runs, but
**no SR is recorded in STATUS before W0 has frozen that task's init spec and posted its
baseline**. A teacher iterated on the old distribution is re-measured on the new one.

## 2. Gates per task (all green = done)

| Gate | Passes when |
|---|---|
| I  init spec | The row's init spec (§3) is implemented in `config/franka/env_cfgs.py` / the task cfg, `verify_task.py --num-resets 1000` PASS (in envelope, no spawn collisions, success-at-reset 0/1000), `audit_workspace --keyword` clean, `tests/test_workspace_placement.py` green. Spec text in the row. |
| B  baseline | Current teacher measured on the frozen spec at n = 128, HEAD named. |
| A  analysis | Failure analysis on the baseline: per failing env, the phase it ended in and the physical event (`diagnose.py` dump + your reading). Written in LOGS *before* the strategy change. |
| S  strategy | The new way of doing the task is written down (why it should work, what could still fail), then implemented. |
| T  teacher | SR >= 0.90 at n = 128 on the frozen spec; the remaining failures classified. Target ~1.0: if you are between 0.90 and 0.97, say what the residual is and whether another pass would close it. |
| V  video | `render_rollout` success clip published to `/v3/<Task-Id>/` with `result.json` + `task.json`. |

A task whose predicate/budget/spawn is proven impossible gets a decision row (`update_status.py --decision`) with the measurement, and only then a task change.

## 3. Init-distribution spec (the "reasonable coverage" contract)

Principles: keep the audited xy bands (they were derived from measured reach); **add
orientation** wherever the object's symmetry does not make it a no-op; give every
mechanism a small yaw and height band; start the robot with joint noise on every task;
never re-randomize the scene mid-episode.

Global (W0, all 25):
- `reset_robot_joints.position_range = (-0.1745, 0.1745)` (±10°) on every task (the 15 free-object tasks currently use (0, 0)).
- `resampling_time_range` >= episode length on every task (Lift, Stack, Peg, Reach, Place, Reorient re-spawn the object/goal at 8–12 s inside a 20 s episode today — a mis-specification, decision D1).
- Mechanisms: mount yaw uniform ±0.26 rad (±15°) about the facing-the-robot pose; mount z band = current fixed z ± 0.03 m, floored at `min_mechanism_mount_z`; xy bands unchanged. Mechanism joint starts unchanged (closed / OFF).

Per task (free objects): yaw is uniform in the stated band; positions unchanged unless stated.

| Task | Object yaw | Notes |
|---|---|---|
| Lift-Cube | ±π (already) | — |
| Stack-Cube | cube ±π, base ±π (was 0/0) | goal tracks the base; teacher must read base yaw only if it grasps it — it does not |
| Push-Cuboid | ±π (was 0) | any face may face the robot; the push contact face is whichever is behind along the goal line |
| Drag-Pull | ±π (was 0) | same carton; same rule |
| Topple-Block | ±π (already) | — |
| Place-In-Container | cube ±π (already); bin ±π (was 0) | goal = bin interior site |
| Throw-To-Bin | cube ±π; bin ±π | bin distance: see D3 |
| Reorient-Object | yaw ±π (already), roll fixed π/2 | lying bottle; keep |
| Cage-Drag | ±π (was 0) | 46 mm cube |
| Strike-Slide | ±π (symmetric puck; no-op, set for uniformity) | — |
| Tool-Pull | stick ±0.35 rad (±20°, was 0); puck ±π | stick must still point roughly at the puck |
| Peg-Insertion | peg ±π/6, board ±π/6 (was 0/0) | orientation matching is part of insertion; widen later if the teacher handles it |
| Axial-Extract | mechanism rule | — |
| Edge-Grasp | riser ±0.26, plate ±π (was 0) | — |
| Pivot-Lift | board ±0.15 (already); wall follows the board | — |
| Reach-Target | no object | robot noise only |

## 4. Strategy hypotheses per task (lead's reading — verify, do not trust)

Grouping follows the phase-1 mechanisms so that one fix moves several tasks.

### Group G (grasp retention, M1) — Stack-Cube 0.523, Place-In-Container 0.359, Peg-Insertion 0.117, Tool-Pull 0.000, Edge-Grasp 0.000
- **The strongest clue: Lift-Cube is 1.000 on the SAME 46 mm cube** with `lift_object.py`,
  while Stack and Place lose it during CLOSE with `stack_object.py` /
  `place_in_container.py`. Diff the two grasp phases (approach height, seat tolerance,
  close timing, whether the hand keeps descending while closing, the yaw of the hand
  relative to the cube faces). First experiment: make Stack/Place/Peg *reuse Lift's
  grasp phase verbatim* (subclass or shared helper), then only the transport differs.
- Squeeze ejection means the pads meet the object off-centre or on an edge: align the
  hand yaw to the cube's faces (the cube now has ±π yaw), centre on `gripper_to_object`
  in the hand frame, stop descending before closing, close with the pads at the object's
  mid-height, and hold a beat before lifting. Closed-loop check: if the aperture
  collapses below the object width, the grasp is lost — reopen, re-centre, retry
  (finger joint positions are in the observation).
- Fingertip friction is domain-randomized 0.3–1.5 at startup (per env, fixed for the run).
  Check whether the persistent failing envs are the low-friction ones; if so a slower
  lift / firmer squeeze is the strategy, not a task change.
- Tool-Pull: phase-1 measured the hook stick ejected laterally during the squeeze. The
  stick is a thin plank; pinch it across its thin axis, centred along its length near the
  CoM, hand yaw aligned to the stick, then drag. Alternative that avoids the grasp
  entirely: hook the puck with the stick *without lifting* — push the stick's hook end
  behind the puck with the closed gripper on top of the stick (pressing down), and pull.
- Peg-Insertion: real clearance + 45° lead-in. After the grasp is retained, insertion is a
  compliance problem: hover above the hole, descend slowly, and on stall do a small
  spiral/wiggle in xy while keeping z pressure; exit `_place` on xy AND z jointly. Match
  peg yaw to board yaw (±π/6 spec).
- Edge-Grasp: the plate overhangs the riser edge; instead of a dual-constrained tilted
  pinch, push the plate until it overhangs enough that a *vertical* pinch across the rim
  from the side works, or slide one finger under the overhang (the user's "finger in the
  gap") and close.

### Group P (planar push, M2) — Push-Cuboid 0.086, Drag-Pull 0.391, Cage-Drag 0.016
- The servo holds a position offset behind the object and cannot sustain the push (real
  97 g carton, pad friction). Strategy: push with the **closed gripper as a flat paddle**
  (both pads together, wider contact, no risk of the object entering the aperture),
  contact at the object's mid-height, hand aligned to the contact face; command a lead
  point *proportional to the remaining distance* along the goal line (larger lead = more
  force, saturating early, shrinking near the goal); correct cross-track error by
  re-approaching from the corrected line rather than pushing sideways; stop when inside
  the tolerance (success latches).
- Budget check first: Push-Cuboid and Drag-Pull have 150 steps (3 s) for approach +
  up to ~22 cm of travel. Measure the fastest feasible trajectory (max_dq, PD tracking);
  if it exceeds 150 steps the budget is mis-specified — a decision row, then lengthen.
- Cage-Drag must keep the aperture > 55 mm: the "cage" is the two open fingers straddling
  the cube; contact through the finger *insides*; ride height so the pads meet the cube
  mid-face; approach from behind along the goal line; same lead rule.
- Yaw ±π now: pick the contact face as the one whose outward normal is most opposite to
  the goal direction; the teacher reads object yaw from `object_orientation`.

### Group M (mechanisms) — Open-Door 0.297, Open-Lid 0.102, Rotate-Valve 0.461, Flip-Switch 0.609
- Open-Door: 90° swing in 150 steps on a real 0.3 m door with a bar pull. Strategies:
  (a) hook the fingers *behind* the bar (closed gripper through the gap between bar and
  leaf, then pull — no grasp retention needed), following the hinge arc with the wrist
  yawing; (b) pinch the bar and re-grasp once at ~45°. Also check the budget: 90° at the
  door's damping vs 3 s — measure the fastest feasible swing.
- Open-Lid: reversed hinge, mushroom knob. Pinch the knob and lift along the hinge arc
  (wrist pitch follows the lid angle), or slide the closed gripper under the lid's front
  edge (finger in the gap) and lift — the second needs no grasp at all. The v2 handover
  suggests a hook-under-cap teacher; the collider is already shaped.
- Rotate-Valve: 270° cross handwheel, no rim. Insert one *finger between two spokes*
  and push tangentially (a closed gripper sweeping a spoke through ~120°), then retreat,
  re-insert at the next spoke, repeat: three sweeps of 90–120° each, each phase with an
  unconditional timeout. Drop consumed 40 % of the budget in phase-1: the retreat must be
  short (a few cm along the axis).
- Flip-Switch: a toggle at −45° → +30°. The stroke lands 10–15 % single-shot. Use the
  closed gripper as a paddle placed *below* the toggle tip on the OFF side at contact
  height, then sweep along the arc with a small radial lead so contact is never lost;
  verify with the joint angle in the observation and only retreat after the threshold.

### Group D (dynamic / contact-mode) — Throw-To-Bin 0.000, Strike-Slide 0.000, Pivot-Lift 0.000, Reorient-Object 0.258
- Throw-To-Bin: the bin is at 0.78–0.90 m, beyond reach *by design*; a drop is impossible,
  a toss is required. Measure the peak end-effector speed the action interface allows
  (joint-position targets, `FRANKA_ACTION_SCALE` 0.04, PD gains) with a scripted
  wind-up + release; the required launch speed for 0.4–0.5 m at 45° is ~2.1–2.3 m/s. If
  the arm cannot reach it, the task is physically impossible as specified → decision D3
  (bring the bin to the farthest *throwable* distance, still beyond reach). Release timing:
  open the gripper at the top of the swing when the velocity vector points at the bin.
- Strike-Slide: 66 % of strikes were voided by `ee_ground_collision` (link7 touching the
  floor). Keep the hand vertical (link7 well above the floor) and strike with the closed
  fingertips at puck mid-height (12 mm); wind up along the goal line; the impulse needed
  scales with the required slide distance and the floor friction — calibrate with a
  lookup from the wind-up length. Measure feasibility as for Throw.
- Pivot-Lift: the approach point sits in the IK dead zone (M4 posture trap). Approach the
  board from the side/top instead of from behind; press the far edge down and pull the
  near edge up (pivot about the wall), or pinch the near edge on its thin axis and lift
  while pushing toward the wall. Reset the IK posture bias when a phase starts.
- Reorient-Object: perfect axis alignment yet drift > 0.18 m from ~10 collision resets.
  Do it in one motion: pinch the lying bottle near its top end (thin axis), lift 10 cm,
  rotate the wrist 90° about the horizontal so the bottle points up, lower until it
  touches, open, retreat straight up. No pushing, no repeated attempts.

### Group N (near the bar + regression guard) — Topple-Block 0.898, Turn-Lever 0.922, Open-Drawer 0.938, Push-Button 0.969, Slide-Window 0.969, and the 1.000s (Lift-Cube, Reach-Target, Axial-Extract, Push-Flap)
- Under the new init spec (yaw, mount yaw/height, robot noise) these will move. Baseline
  first; then close the residual: Topple (CONTACT_Z sweep on the cracker box, hand yaw to
  the face), Open-Drawer (near-misses 1–2 mm short as the budget runs out: pull further /
  faster), Turn-Lever (re-seat gate), Push-Button / Slide-Window (find the 4/128).

## 5. Measurement protocol

| purpose | n | command |
|---|---|---|
| iterate | 32 | `test_classical --num-envs 32 --num-episodes 1` |
| claim | 128 | `test_classical --num-envs 32 --num-episodes 4` |
| failure analysis | 32 | `classical/diagnose.py --task <ID> --num-envs 32` → `docs/cl_v3/diag/<Task>/` |

Always `SR (n, HEAD)`. n = 32 is never quoted as a result. A task at 0.90–0.97 gets a
second independent n = 128 read before it is called done.

## 6. Waves / agents / GPUs

Holder 20277 (`hold_dgx_amit`, dgx1). GPUs 4–7 belong to someone's training; GPU 0 is
reserved for others. **Use 1, 2, 3 only.** Two agents per GPU is fine for teacher evals.

| Agent | Scope | GPU |
|---|---|---|
| W0 | init spec on all 25, `diagnose.py`, D1 fix, re-baseline all 25 at n=128, `/v3/` site skeleton | 1–3 |
| W1-G | Stack, Place, Peg, Tool-Pull, Edge-Grasp | 1 |
| W1-P | Push-Cuboid, Drag-Pull, Cage-Drag | 2 |
| W1-M | Open-Door, Open-Lid, Rotate-Valve, Flip-Switch | 3 |
| W1-D | Throw-To-Bin, Strike-Slide, Pivot-Lift, Reorient-Object | 1 |
| W1-N | Topple, Turn-Lever, Open-Drawer, Push-Button, Slide-Window + the 1.000s | 2 |
| W2 lead | confirmation reads, videos, site, handover | 3 |

## 7. Decisions (mirror of the STATUS decision table)

- D1 (proposed, W0 applies): `resampling_time_range` >= episode length everywhere —
  mid-episode re-spawn is a mis-specification.
- D2 (open): robot joint noise ±10° on free-object tasks — coverage, applies to all.
- D3 (open, needs measurement): Throw-To-Bin bin distance vs. achievable launch speed.
- D4 (open, needs measurement): 150-step budgets on Push-Cuboid / Drag-Pull / Open-Door
  vs. fastest feasible motion.
- D5 (open): Strike-Slide `ee_ground_collision` margin — first try the vertical-hand
  strike; only if the geometry makes it impossible, revisit.
