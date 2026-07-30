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

### Design principle that generalises

Every teacher above follows `open_drawer`'s winning idea: **fingers stay closed and the
pad acts as a geometric face-normal pusher**. Nothing depends on grip friction, which
domain randomisation drives as low as 0.3. The two lowest scorers (valve at 0.083,
switch at 0.500) are precisely the two where geometric engagement is hardest to
maintain.

## Status

- [x] Baseline measured for the 4 existing teachers
- [x] Articulation family: lever, valve, switch, window, lid
- [x] Open-Lid task-design bug found and fixed (96ebbb4)
- [ ] Re-tune: Push-Button, Push-Cuboid, Open-Door
- [ ] Author: Reach, Lift (x4 objects), Push-Disc
- [ ] Author: Stack, Peg-Insertion, Place-In-Container, Reorient, Tool-Pull
