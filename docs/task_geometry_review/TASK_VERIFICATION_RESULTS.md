# Verification of all 25 task definitions — 2026-09-11

All 25 tasks were reviewed individually through code/configuration inspection, 8 sampled resets, a constructed MuJoCo goal/contact state, and one real teacher rollout to first success, termination or timeout. Each has saved qpos/qvel/mocap state, goal and rollout renders, and an MP4. The constructed states test predicates; they are not demonstrated policy trajectories.

Regression validation: **290 tests passed** on the CPU MuJoCo Warp path. The final run uses the same source fingerprint for every task.

Focused stress validation: **1,024 resets each for Peg-Insertion and Cage-Drag**. Both pass placement/collision and success gates, with zero success-at-reset and all goal witnesses accepted. Raw reports: [Peg-Insertion](stress_validation/Peg-Insertion.json), [Cage-Drag](stress_validation/Cage-Drag.json).

Test mode now preserves each training task's episode budget and failure conditions. Only observation corruption and external robot pushes are disabled. Simulation ran on CPU in Slurm holder 20277, dgx1, with one task per command; rendering used EGL with software rendering requested.

**The 25 single-episode outcomes below are diagnostics, not teacher success rates. The September 10 teacher scoreboard must be remeasured under these corrected contracts.**

[Open the visual review](verified_tasks/index.html)

| Task | Correction or checked invariant | Goal state | Reset checks | Diagnostic rollout |
|---|---|---|---|---|
| [Axial-Extract](verified_tasks/Mjlab-Axial-Extract-Franka/result.json) | Joint-progress reward; extraction target and plug/mount contacts checked. | pass | 8 clear | success |
| [Cage-Drag](verified_tasks/Mjlab-Cage-Drag-Franka/result.json) | Geometric cage + transport; yaw-safe aperture; violations revoke success and reward. | pass | 8 clear | timeout |
| [Drag-Pull](verified_tasks/Mjlab-Drag-Pull-Franka/result.json) | Ground-level point goal, spawn separation, transport direction and termination rules checked. | pass | 8 clear | success |
| [Edge-Grasp](verified_tasks/Mjlab-Edge-Grasp-Franka/result.json) | Actual bilateral pad contact and low velocity required; transient flicks rejected. | pass | 8 clear | timeout |
| [Flip-Switch](verified_tasks/Mjlab-Flip-Switch-Franka/result.json) | Joint-progress reward; switch target, detent motion and housing clearance checked. | pass | 8 clear | success |
| [Lift-Cube](verified_tasks/Mjlab-Lift-Cube-Franka/result.json) | Goal requires a slow actual two-pad grasp; free-flight proximity rejected. | pass | 8 clear | success |
| [Open-Door](verified_tasks/Mjlab-Open-Door-Franka/result.json) | Joint-progress reward; hinge target and rendered door pose checked. | pass | 8 clear | timeout |
| [Open-Drawer](verified_tasks/Mjlab-Open-Drawer-Franka/result.json) | Joint-progress reward; slide target and drawer/handle geometry checked. | pass | 8 clear | timeout |
| [Open-Lid](verified_tasks/Mjlab-Open-Lid-Franka/result.json) | Joint-progress reward; hinge target, lid and handle goal geometry checked. | pass | 8 clear | success |
| [Peg-Insertion](verified_tasks/Mjlab-Peg-Insertion-Franka/result.json) | Tip/center frames reconciled; goal lowered 50 mm; seated square-bore fit, release and speed checked. | pass | 8 clear | timeout |
| [Pivot-Lift](verified_tasks/Mjlab-Pivot-Lift-Franka/result.json) | Recorded wall-pivot contact plus actual slow grasp required for the airborne goal. | pass | 8 clear | timeout |
| [Place-In-Container](verified_tasks/Mjlab-Place-In-Container-Franka/result.json) | Full cube/basket-frame containment, support, release and low speed; resting goal geometry corrected. | pass | 8 clear | success |
| [Push-Button](verified_tasks/Mjlab-Push-Button-Franka/result.json) | Joint-progress reward; pressed target and spring-return behavior checked. | pass | 8 clear | success |
| [Push-Cuboid](verified_tasks/Mjlab-Push-Cuboid-Franka/result.json) | Ground-level point goal, spawn separation, object dimensions and termination rules checked. | pass | 8 clear | timeout |
| [Push-Flap](verified_tasks/Mjlab-Push-Flap-Franka/result.json) | Joint-progress reward; negative hinge target, panel and housing clearance checked. | pass | 8 clear | success |
| [Reach-Target](verified_tasks/Mjlab-Reach-Target-Franka/result.json) | Position-only wrist goal, target reachability and matching reward checked. | pass | 8 clear | success |
| [Reorient-Object](verified_tasks/Mjlab-Reorient-Object-Franka/result.json) | Axis-based reward matches success; fast orientation crossings rejected. | pass | 8 clear | timeout |
| [Rotate-Valve](verified_tasks/Mjlab-Rotate-Valve-Franka/result.json) | Monotonic joint reward removes 270-degree Cartesian shortcut; wheel/hub geometry checked. | pass | 8 clear | timeout |
| [Slide-Window](verified_tasks/Mjlab-Slide-Window-Franka/result.json) | Joint-progress reward; slide target and sash/frame clearance checked. | pass | 8 clear | success |
| [Stack-Cube](verified_tasks/Mjlab-Stack-Cube-Franka/result.json) | Slow released placement in actual contact with the base; hovering rejected. | pass | 8 clear | success |
| [Strike-Slide](verified_tasks/Mjlab-Strike-Slide-Franka/result.json) | Beyond-reach ground goal, puck dimensions/contact configuration and point success checked. | pass | 8 clear | success |
| [Throw-To-Bin](verified_tasks/Mjlab-Throw-To-Bin-Franka/result.json) | Beyond-reach bin retained; full containment, support, release and low speed required. | pass | 8 clear | timeout |
| [Tool-Pull](verified_tasks/Mjlab-Tool-Pull-Franka/result.json) | Actual pad/tool then tool/puck contacts required; direct robot/puck contact invalidates success. | pass | 8 clear | terminated |
| [Topple-Block](verified_tasks/Mjlab-Topple-Block-Franka/result.json) | Symmetric axis reward matches the two accepted faces; fast transient crossings rejected. | pass | 8 clear | terminated |
| [Turn-Lever](verified_tasks/Mjlab-Turn-Lever-Franka/result.json) | Joint-progress reward; lever target, mount and housing clearance checked. | pass | 8 clear | success |

The goal-state witnesses include contact history for Cage-Drag, Tool-Pull and Pivot-Lift. They establish that the intended state can satisfy the predicate without deep interpenetration; they do not prove the teacher can reliably reach that state from every reset.

[Task contract decisions](VERIFY_FIXES.md) · [Pre-fix audit](SEMANTICS_AUDIT.md)
