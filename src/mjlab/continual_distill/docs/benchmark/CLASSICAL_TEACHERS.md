# Classical teachers for Class A

> Hand-coded scripted teacher policies for every Class A task, with measured success
> rates. Companion to CLASS_A_EXPANSION.md and WORKSPACE_FIX.md.

## What a classical teacher is

A drop-in replacement for an NN teacher: it receives ONLY the policy observation vector
and returns 8-D joint-position actions in the same convention as the RL policies
(`JointPositionAction`, `use_default_offset=True`, `scale=FRANKA_ACTION_SCALE`).

All the machinery lives in `classical/base.py::ClassicalPolicyBase`: a private CPU
MuJoCo Franka for FK/Jacobians, damped-least-squares differential IK solved to
convergence per control step, uniform step scaling, optional command-lead for sustained
pulling force. **A subclass implements exactly one method**, `_target_error(i, obs_i)`,
returning `(pos_err, target_rot | None, gripper_action)`.

Two rules the base class docstrings call out, both learned the hard way:

- **Use RELATIVE observations only.** Observations are world-frame and include per-env
  scene-origin offsets. Building a target from absolute terms works in env 0 and fails
  everywhere else. `gripper_to_object` (obs[40:43]) and `object_to_goal` (obs[43:46])
  are offset-free.
- **Use the right `DEFAULT_QPOS`.** `joint_pos_rel` and the action offset are both
  relative to the robot cfg's init pose, which DIFFERS PER TASK: articulation tasks use
  `NEUTRAL_QPOS`, lift/push/stack use `HOME_QPOS`. The wrong one puts FK and the
  Jacobian at a fantasy configuration.

## Baseline: the workspace fix invalidated the existing teachers

Measured with `classical.test_classical`, 16 envs x 2 episodes, CPU. The four
pre-existing teachers were hand-tuned against the OLD object placements; WORKSPACE_FIX
moved every object.

| task | pre-fix (commit 8e7fd5c) | post-fix (babf036) |
|---|---|---|
| Push-Button | **1.00** | 0.16 |
| Push-Cuboid | 0.31 | 0.00 |
| Open-Drawer | (not re-measured) | **0.81** |
| Open-Door | **0.00** | 0.00 |

Two things this measurement settles, which guesswork would have got wrong:

1. **Button and cuboid were genuinely broken by the workspace fix** (1.00 -> 0.16,
   0.31 -> 0.00). They need re-tuning, exactly as WORKSPACE_FIX predicted.
2. **Open-Door was ALREADY at 0.00 before the workspace fix.** Its teacher is not a
   regression from the move — it does not work and apparently has not for some time.
   The `classical/sweep_door.py` script in the tree suggests it was being tuned and
   never landed. Do not attribute this one to the placement change.

Open-Drawer at 0.81 survived the move, which is consistent with its documented strategy
(a top-down fingertip hook that keys off the handle's local geometry rather than an
absolute pose).

## Scope

Build a classical teacher for all 20 Class A registered tasks (16 motion profiles).
Re-tune the 3 broken existing ones; author the rest.

Accuracy is measured, not asserted: every teacher reports a success rate from
`test_classical` over >= 32 episode-instances, and that number is recorded here
whatever it says. A teacher that does not work is reported as not working.

## A classical teacher is a TASK-VALIDITY check

The single most valuable result so far was a teacher scoring **0.000**.

`Mjlab-Open-Lid-Franka` was **unsolvable as specified**. Success needed the hinge at
<= -1.109 rad, and at that angle the lid's lip sat at z = -0.026 — *below the floor*.
No policy, scripted or learned, could ever have passed it. Cause: the lid's flap swings
DOWN as it opens (swept drop 0.157m vs 0.070m closed), and
`MECHANISM_DROP_BELOW_MOUNT` had recorded the REST-pose extent, so the mount was 9cm
too low.

Every structural gate passed this task: benchmark-smoke builds and steps it, the
placement audit sees a healthy radial reach and nothing buried *at rest*, and the
success predicate fires when the joint is teleported to target. Only trying to actually
SOLVE it exposed the defect. After fixing the mount (commit 96ebbb4), the **unchanged**
teacher went 0.000 -> 1.000.

Lesson worth keeping: scripted teachers are a task-validity gate that no structural
check replaces. Build them early.

## Measured results

Independently re-verified by the parent session, not taken from agent reports.
16 envs x 2 episodes = 32 episode-instances, CPU.

### Articulation

| task | success | notes |
|---|---|---|
| Slide-Window | **1.000** | closed pad pushes the grab bar's face; pure normal force |
| Open-Lid | **1.000** | after the task fix; arc-follower, unchanged |
| Open-Drawer | **0.812** | pre-existing; survived the workspace move |
| Turn-Lever | **0.812** | arc-follower about the approach axis, no regrasp needed |
| Flip-Switch | **0.500** | ballistic open-loop stroke; +-0.08 run-to-run |
| Rotate-Valve | **0.03-0.08** | see below — effectively a failure |

**Rotate-Valve is effectively a failure and is reported as such.** The authoring agent
measured 0.083; the parent session's independent re-run measured **0.031**. Both are
near-noise, and the disagreement between two 32-episode runs is itself the finding: at
this level the teacher is not reliably doing anything. The pad seats reliably but
most envs stall after ~1.2 rad of the required 4.71: the spoke is a thin 2.4cm arm, so
the pad contacts a small face and slips off under the hinge damping. An aggressive lead
angle made it *worse* (0.000), so it is a contact-retention problem, not a force
problem. Getting past ~0.3 likely needs a caging grasp, which reintroduces exactly the
friction fragility the drawer's geometric-hook approach exists to avoid. Open question
for the CL work: is a 0.08 teacher useful, or should the task be softened?

### Reach / grasp-and-lift

| task | success |
|---|---|
| Reach-Target | **1.000** |
| Lift-Cube | **1.000** |
| Lift-Cylinder | **1.000** |
| Lift-Sphere | **0.969** |
| Lift-Ellipsoid | **0.906** |
| Push-Disc | **0.406** (capped by the task, see below) |

One `LiftObjectClassicalPolicy` with four thin subclasses differing only in object
centre height and squeeze duration — the same collapse the motion-profile counting rule
applies to the tasks themselves. Hover open -> descend -> close -> climb -> drive
`object_to_goal` to zero. Sphere/ellipsoid get tighter alignment and longer squeezes;
their lower scores are genuine grasp-geometry difficulty, not tuning debt.

### Two-object / novel success shapes

| task | success | note |
|---|---|---|
| Stack-Cube | 0.41-0.44 | release is essential — the height test cannot fire while held |
| Place-In-Container | 0.344 | drops the cube from above; it does not fit inside with the gripper |
| Reorient-Object | **0.062** | teacher limitation, diagnosed below |
| Peg-Insertion | 0.03-0.06 | tolerance below the controller noise floor |
| Tool-Pull | 0.031 | measured BEFORE the observability fix |

### Task-side issues these teachers exposed

**Push-Disc (0.406) is capped by its success predicate, not the teacher.** The goal sits
at fixed z=0.03 while the disc rests at z=0.020, and success is a 3D distance under
0.02. The 1cm vertical residual is unavoidable, leaving ~1.7cm of lateral budget versus
the 5cm ball lift is scored against. Setting the goal z to the disc's resting height
would roughly double the budget. NOT changed — altering a threshold to improve a number
is exactly what these teachers exist to detect, so it is flagged for a human decision.

**Reorient-Object (0.062): a genuine teacher limitation, NOT an env bug.** The
authoring agent attributed this to a stale `target_pos` anchor. That bug was real and is
now fixed (d8e5115) — but re-measuring afterwards gave the SAME 0.062, so it was not the
limiter. Instrumented over a full episode x 16 envs:

- only **2/16 envs ever met the angular criterion** at any point;
- best axis alignment in the other 14 was ~0.04-0.17, i.e. still essentially horizontal;
- when the angle WAS met, drift was 0.048-0.134, comfortably inside the 0.18 bound.

So the drift bound is not binding and the anchor fix, while correct, does not rescue
this task. The teacher does not reliably stand the cylinder up: grasping a lying
cylinder across its flat end faces and rotating the wrist to vertical demands both a
secure grasp on a rolling object and a large wrist rotation, and it usually loses the
object mid-rotation. Reported as a failing teacher. Worth noting the fix was still
worth making — an unwinnable-at-spawn env is a defect regardless of whether it was the
dominant one.

**Peg-Insertion (0.062): the tolerance is below the controller's noise floor.** A 2.4cm
peg in a 3cm hole is 3mm clearance per side against a 1.5cm xy tolerance, ~1cm
observation noise, and a measured 2-3.6cm DLS lateral steady-state bias that integral
action only partly removes. Insertion also wants compliance a position servo does not
have: residual lateral error at contact wedges the peg on the rim instead of sliding in.
This is a genuine limit of scripted control, and a fair argument that peg-insertion
needs a learned teacher.

### Two general controller findings (they generalise to any teacher here)

1. **Stale phase state after mid-episode auto-resets.** `ee_ground_collision` ends
   episodes mid-flight and respawns the object, but `policy.reset()` is only called
   BETWEEN episodes. The state machine then runs "carry" against an object it never
   picked up. Detecting the discontinuity and rewinding took cube-lift from
   **0.125 -> 1.000**. Any teacher on a task with a non-timeout termination needs this.
2. **The DLS solve has a 2-3.6cm lateral steady-state bias** — larger than Stack's 3cm
   and Peg's 1.5cm tolerances, so proportional-only placement can never succeed on the
   tighter tasks. Integral action on the carry/place phases is required, not optional.

### Design principle that generalises

Every teacher above follows `open_drawer`'s winning idea: **fingers stay closed and the
pad acts as a geometric face-normal pusher**. Nothing depends on grip friction, which
domain randomisation drives as low as 0.3. The two lowest scorers (valve at 0.083,
switch at 0.500) are precisely the two where geometric engagement is hardest to
maintain.

### Re-tuned existing teachers

| task | pre-fix | post-fix (broken) | after re-tune | verified |
|---|---|---|---|---|
| Push-Button | 1.00 | 0.16 | **1.000** | yes |
| Push-Cuboid | 0.31 | 0.00 | **0.12-0.16** | yes |
| Open-Door | 0.00 | 0.00 | **0.000** | yes, still broken |

**Push-Button fully recovered.** The break was NOT geometry but TRAVEL TIME: the button
mount dropped to z~0.16, so the gripper starts ~0.85m above the cap instead of level
with it, and at the inherited `max_dq=0.05` the descent ate ~130 of the 150 steps. Fixed
by making the rate phase-dependent — brisk while travelling, gentle while pressing.
Worth remembering: after a geometry change, check the TIME budget, not just reachability.

**Push-Cuboid recovered only to ~0.12-0.16** against a 0.31 pre-fix baseline. Two real
bugs fixed (constant ground terminations once the table vanished; the pusher aiming
*through* the object and shoving it away from the goal), but ride height is genuinely
tight: the box is 3cm tall, so too high hovers over it and too low trips
`ee_ground_collision`. Reported as a partial recovery, not a success.

**Open-Door remains broken at 0.000 and the premise is wrong.** Three genuine defects
were found and fixed along the way (a wrong hinge-arc radius `_R0`, corrected by circle-
fitting the measured handle path; height not held during the drag; too-slow arc
advance). None rescued it. The blocker is geometric: the bar **cams out of the finger
gap along the approach axis**. Alignment at closure is good (~0.02-0.03, inside the 2cm
bar), but over ~8 steps the approach-axis error grows to 0.13 and the grasp is gone —
the arc waypoint pushes the hand into the panel, driving the bar out through the open
finger gap, the one direction the cage cannot resist. Confirmed not force-limited
(`cmd_lead_max` at 0.35/0.8/1.5 all gave 0.00 deg) and not speed-limited (84 deg in 3s
needs only 0.26 m/s).

**Verified independently:** `GRIPPER_CAGE = -0.5` leaves a ~4.2cm finger gap around a
2cm bar (fingers span 0-0.08m total), so the bar is NEVER gripped. The "geometric
containment" premise in the module docstring does not hold. A full pinch is worse — the
fingers close before clearing the bar. This needs a strategy change (hook the panel
edge, or close *after* seating with an explicit approach-axis preload), not tuning.

Caution left in the code: `TIP_VEC` must stay at 0.04, NOT the physically-correct
site-to-fingertip offset. The "correct" value parks the bar at the very fingertips and
the door does not move at all.

## Final results — all 20 Class A tasks

| task | success | |
|---|---|---|
| Reach-Target | 1.000 | |
| Lift-Cube | 1.000 | |
| Lift-Cylinder | 1.000 | |
| Push-Button | 1.000 | re-tuned |
| Slide-Window | 1.000 | |
| Open-Lid | 1.000 | after task fix |
| Lift-Sphere | 0.969 | |
| Lift-Ellipsoid | 0.906 | |
| Turn-Lever | 0.812 | |
| Open-Drawer | 0.62-0.81 | pre-existing |
| Flip-Switch | 0.500 | |
| Stack-Cube | 0.41-0.44 | |
| Push-Disc | 0.406 | capped by predicate |
| Place-In-Container | 0.344 | |
| Push-Cuboid | 0.12-0.16 | partial recovery |
| Reorient-Object | 0.062 | teacher limitation |
| Peg-Insertion | 0.03-0.06 | below controller noise floor |
| Rotate-Valve | 0.031 | failure |
| Tool-Pull | 0.031 | needs a stick-using rewrite |
| Open-Door | 0.000 | broken; premise wrong |

All 20 numbers were re-measured independently by the parent session, not taken from
agent reports. Where the two disagreed the LOWER figure is recorded (Peg 0.062 -> 0.031,
Rotate-Valve 0.083 -> 0.031): on a 32-episode sample the spread between two runs of a
weak teacher is itself the finding, and rounding toward the flattering number is exactly
how a benchmark's teacher table stops being trustworthy.

**11 of 20 at >= 0.4; 6 at >= 0.9.** Six teachers are weak or failing and are reported
as such — none of these numbers was obtained by relaxing a task.

## Status

- [x] Baseline measured for the 4 existing teachers
- [x] Articulation family: lever, valve, switch, window, lid
- [x] Open-Lid task-design bug found and fixed (96ebbb4)
- [x] Re-tune: Push-Button (1.000), Push-Cuboid (0.12-0.16), Open-Door (still 0.000)
- [x] Author: Reach, Lift x4, Push-Disc
- [x] Author: Stack, Peg-Insertion, Place-In-Container, Reorient, Tool-Pull
- [x] Register all teachers — `classical.CLASSICAL_POLICIES` is the single source of
      truth (20/20 Class A tasks); `test_classical` reads it
- [ ] Open decisions: Push-Disc goal z; Rotate-Valve task softening; Open-Door strategy;
      Tool-Pull teacher rewrite now that the stick is observable


## Improvement pass (2026-07-30, user: "improve all except peg-insertion")

Constraint: **strategy changes only** — no thresholds, tolerances, spawn ranges, assets
or task definitions, and nothing under `src/mjlab/tasks/`. A number obtained by
loosening a task would defeat the purpose of having these teachers at all.

### Rotate-Valve: 0.031 -> 0.50 (verified; agent conservatively reported 0.344)

The 16x win came from two changes, and the SECOND mattered more:

1. **Pad-push -> genuine pinch.** Fingers close on the spoke at 80% of its length so
   the driving load is carried face-on between the pads; friction is only asked to
   resist *radial* sliding, keeping the 0.3-friction randomisation off the critical
   path. The pinch alone measured ~0.125.
2. **A stall watchdog.** The spoke handoff gate keyed on *swept angle*, so a jammed
   engagement made no progress, never reached the gate, and deadlocked for the rest of
   the episode — envs were traced frozen at a constant angle (one sat at exactly 85.81
   deg from t=200 to t=400) with the EE motionless. Forcing a handoff on lack of
   progress is what unlocked the ratchet.

Root cause of the old failure, verified live: the pad walks inboard along the 2.4cm
spoke until it rests against the hub, where the moment arm vanishes. The old
lost-contact test never fired because the pad *is* still near the arc — just at the
useless end of it.

### Open-Door: 0.000 -> 0.000 (not improved; two corrections to the old diagnosis)

Reported honestly as a failure. Mechanics did improve — mean peak angle **2.5 -> 12.0
deg**, with individual envs now reaching 80-90 deg where the old teacher never passed
~7 — but 26/32 envs still never clear 10 deg. The cam-out IS fixed (wrist rotated 90
deg so the pads sandwich the bar normal to the pull).

Two measured corrections to what was previously believed:

- **Opening is a PULL, not a push.** Driving the hinge through its range and reading
  `object_site` in the robot frame: hinge centre (0.4873, -0.5395), radius 0.5514,
  handle polar angle tracking hinge angle 1:1, handle moving toward -x throughout. The
  face-push idea was tested directly and measured 0.1 deg mean over 32 episodes; the
  static `door_body` barrier also walls off the far side.
- **Reach is NOT the constraint.** The handle stays 0.53-0.77m from the shoulder,
  inside the ~0.85m envelope. The difficulty is that it passes within 0.18m of the base
  axis (shoulder singularity) and finishes behind the base.

**Why it still fails — a throughput budget, verified independently:** the episode is
150 steps and success needs 1.471 of 1.571 rad, i.e. **84.3 of 90 deg with no partial
credit**. A holding pull advances ~1-2 deg per 25 steps and the approach alone costs
~40 steps, so success requires one seat to hold almost the entire episode. Every
speed-up made it worse (arc rate 22->40 deg: 6.7 deg mean; faster approach: 1.4 deg;
higher drag rate + command lead: neutral; stall watchdog: 2.4 deg) — the pinch is
retention-limited, so hurrying loses the bar sooner.

The valve was rescued by a ratcheting multi-regrasp cycle over 400 steps; the door has
150. That is the honest difference. This needs a learned teacher, or a task-side change
that is out of scope under the current constraint.

### Reorient-Object: 0.125 -> 0.41-0.59, from ONE constant

Parent-verified at **0.594**; the authoring agent reported its lower batch (0.406).

The entire gain is `floor_min_z` 0.022 -> 0.030. Head-to-head: 0.022 -> 0.125,
0.030 -> 0.531, 0.034 -> 0.500.

**This corrects a repo-wide error.** Teachers assumed the lowest end-effector COLLISION
geom sits 1.24cm below the `gripper` site. Measured over only the genuinely collidable
link7-subtree geoms (`hand_capsule`, `left_finger_pad`, `right_finger_pad` — the rest
are `contype=conaffinity=0` visual-only), it is **1.38-1.4cm**. A guard at 0.022 leaves
~8mm of pad material, so `ee_ground_collision` fires on transient dips and silently
auto-resets the env under a state machine the harness never informs: **45 collisions in
600 steps x 8 envs, 34 of them during descent.**

The earlier diagnosis ("the teacher fails to stand the cylinder up") was right about the
symptom and wrong about the cause. Envs that SURVIVED to the rotate phase already
reached 0.99 axis alignment with the unmodified grasp strategy. The teacher was not
losing the cylinder — it was being killed on the way down.

Both strategies proposed in the improvement brief measured WORSE and were reverted, with
the negative results recorded in the module docstring so they are not re-tried:
- barrel grasp **0.469** vs 0.531 (the end-face grasp makes the cylinder's axis *be*
  the closing axis, so the wrist rotation cannot disturb the grip);
- phase-dependent rate limits **0.188** vs 0.531 (no travel leg worth accelerating; the
  larger stride just throws the wrist into the floor).

Worth noting for future work: a suggestion from a brief is a hypothesis, not an
instruction. Measuring both and keeping the incumbent was the right call.

### Tool-Pull: 0.031, unimproved — and tool use is not scriptable with this gripper

The same floor fix moved it 0.000 -> 0.031/0.042, within noise of its own baseline
(per-batch variance 0.000-0.250 across five batches of 16). The authoring agent first
reported 0.125 from a standalone harness, then re-measured through `test_classical`,
got 0.031, and RETRACTED the higher figure. The official path wins.

Genuine tool use was built and measured **0.000**. The blocker is mechanical and
isolated:
- the approach works — the gripper converges to 0.023-0.028m of the stick's grasp site
  with the closing axis at exactly (0.00, 1.00, 0.00), perpendicular to the shaft;
- the fingers genuinely grip, stalling at qpos ~0.010 per side, **exactly half the 22mm
  shaft**;
- but the stick never leaves the ground: during the squeeze `gripper_to_tool`'s x
  component grows monotonically -0.004 -> -0.042. **The pinch ejects it axially.** A
  0.26m box with its mass 9cm off the grasp point turns any residual misalignment into
  axial force, and randomised slide friction (as low as 0.3) cannot arrest it.
- reproduced across 7 grasp heights x 3 grasp points (`object_site`, CoM, hook bar) x 2
  orientation modes x 2 close windows.

Even granting a grasp, the plan needs the stick carried ~0.12m in +x and ~0.23m in +y —
a carry a marginal grip would not survive. Conclusion: this needs a learned teacher, or
a stick with an actual graspable feature rather than a smooth shaft. The task is now
well-posed (the stick is observable since d8e5115); it is the GRIPPER-object pair that
defeats scripting.


## How to read a weak teacher's number (measurement variance)

Tool-pull is the cautionary case. SIX measurements of *identical code*:

| run | episodes | result |
|---|---|---|
| agent, standalone harness | 32 | 0.125 |
| agent, `test_classical` | 32 | 0.031 |
| agent, `test_classical` | 48 | 0.042 |
| parent, `test_classical` | 32 | 0.125 |
| **parent, `test_classical`** | **128** | **0.039  <- settles it** |

Per-batch figures across all of them span **0.000 to 0.250**. The 128-episode run
(4x any earlier sample) puts the true value near 0.04, so BOTH 0.125 readings were
favourable-batch noise — including the parent's, which had reproduced the agent's
retracted figure exactly and briefly looked like vindication of it. The agent saw 0.125,
re-measured at 0.031, and retracted its higher claim as a favourable-batch artifact —
then the parent's independent run reproduced 0.125 exactly. Neither party was careless;
**32 episodes is simply not enough resolution to separate 0.03 from 0.13.**

Practical rules this implies for anything below ~0.3:
- 32 episodes resolves "works" vs "does not work" and nothing finer;
- report a RANGE, not a point, and say how many episodes produced it;
- a 2-4x disagreement between two 32-episode runs is expected, not a contradiction, and
  should never be resolved by picking the flattering one;
- a claimed improvement from 0.03 to 0.13 on 32 episodes is not evidence of anything;
- when a weak number matters, spend the 128 episodes. It cost one run to convert a
  four-way disagreement into a settled 0.039.

The strong teachers do not have this problem: 1.000 over 32 episodes, reproduced twice,
is a real result. The variance caveat applies specifically to the tail.


### Push-Cuboid, Flip-Switch, Stack, Place-In-Container

Measured at **96 episode-instances** (32 envs x 3 episodes) with matched 96-instance
baselines re-measured at HEAD in the same session — not carried over from earlier docs.

| task | baseline (96) | after (96) | verdict |
|---|---|---|---|
| Flip-Switch | 0.406 | **0.594-0.615** | improved ~1.5x |
| Push-Cuboid | 0.104 | **0.167-0.240** | improved ~2x |
| Stack-Cube | 0.354 | 0.28-0.375 | unchanged |
| Place-In-Container | 0.302 | 0.27-0.29 | unchanged |

**Flip-Switch: the seat gate was unsigned.** It fired on an unsigned 3D residual, so it
was equally satisfied 5cm SHORT of the toggle and 5cm PAST it. Instrumented failures
entered the ballistic stroke at `gto.x ~ -0.05` (gripper already beyond the toggle) and
the hinge never left -45 deg. A signed gate — must still need +x travel, aligned in y/z
— plus removing x from the integrator fixed it. Parent-verified at 0.615/96.

**Push-Cuboid: the pusher overtakes its own workpiece.** Tracking `along` (the component
of object-minus-gripper along the push direction), successes hold +0.03..+0.05
throughout; failures start at +0.04 and decay through ZERO to -0.05, after which the
gripper sits BETWEEN object and goal and every advance shoves the box backwards. Cause:
as the goal nears, the tapering lead lets net penetration reach 0 and the DLS bias walks
the site through the 8x8cm box. Fixed with "behind the object" as a hard precondition
plus a cheap lateral re-seat and hysteresis. (A full return to the hover phase was tried
first and measured much worse — it thrashed p2->p0->p1 every ~10 steps.) push_cuboid
also had NO absolute floor guard at all; every height was relative to a +-1cm-noisy
observation.

**Stack and Place: the bottleneck is GRASP RETENTION, and it was not fixable here.**
Instrumented grasp-loss: Stack drops the cube in 22/32 runs (13 of them inside the lift
phase), Place-In-Container in 19/32 (12 in lift). Of the 10 Stack runs that KEPT hold,
9 succeeded. So these scores are made of retention, not placement precision — which
means the obvious remedies (drop lower, damp longer, tune the integrator) target a phase
most failures never reach. Deeper grasp, longer squeeze and a ramped lift were all
measured at 96 instances, all worse, all reverted with the negative result recorded
in-code.

### A correction to the floor-guard constant

The agent reported the `hand_capsule` reaching ~3.1cm below the gripper site. That
figure is `geom_rbound` — a bounding SPHERE — which for a capsule mounted above the site
vastly overstates its downward reach. Projecting its true half-extent onto world z puts
its lowest point **1.69cm ABOVE** the site, so it cannot touch the floor before the pads
do. The deepest genuinely-colliding geom is a finger pad at **1.38cm below** the site,
as originally measured. `floor_min_z = 0.030` is still correct; only the rationale
needed fixing. Same trap as the phantom "buried in the floor" readings in the workspace
audit: use true extents, never rbound.

**Peg-insertion is pinned at `floor_min_z = 0.022`** and must stay there: its place phase
deliberately drives the peg DOWN THROUGH THE HOLE to the ground, so the raised guard
clamps the insertion itself and peg measures 0.000. Excluded from improvement by the
user; the pin exists purely to keep it from regressing.
