# Task definitions, goals and rewards audit — 2026-09-11

The 25-task suite is **not semantically clean**. This audit found a reproducible
50 mm Peg-Insertion goal/reward/rendering mismatch, incomplete Cage-Drag semantics,
and missing orientation rewards on Reorient-Object and Topple-Block.
No task, reward, teacher, or success threshold was changed in this audit.

## Evidence and scope

Inspected all 25 registered CL-V3 task configurations, their shared command and
reward implementations, and compiled asset tracking-site offsets. Ran constructed
state probes in two CPU environments per focal task inside holder 20277 on dgx1,
using `srun --overlap -n 1 --cpus-per-task=4`, FORCE_CPU=1 and no visible CUDA devices.
These probes test what the implementation accepts, not policy reachability or
teacher success rates. Data: `semantics_audit.json`; reproduction:

```bash
srun --jobid=20277 --overlap -n 1 --cpus-per-task=4 bash -c \
  'export FORCE_CPU=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4; .venv/bin/python docs/task_geometry_review/audit_task_semantics.py'
```

Existing regression suites also ran inside the same allocation on CPU:
`tests/test_manipulation_geometry.py`, `tests/test_class_a_expansion.py`, and
`tests/test_class_a_wave1.py`: **92 passed, 14 warnings, 73.63 seconds**.
Passing these tests does not refute the findings: their assertions do not cover
reward optima versus insertion depth, peg replica/root anchor agreement, or the
cage episode-success latch after a violation.

## 1. Peg-Insertion: center and tip frames disagree (confirmed bug)

The board root is 15 mm above the floor. `stack_height=35 mm` therefore puts
`StackingCommand.target_pos` at z=50 mm, the correct **center** of the fully seated
100 mm peg. The hole itself is centered in XY; this probe found no XY offset bug.

However, the peg's `object_site` is its tip, 50 mm below its center.
`object_at_goal_reward`, the bringing component of `staged_manipulation_reward`,
and `object_to_goal_vector` choose that site whenever it exists. They compare
the **tip** to the **center target** without converting frames.

| Constructed state | Center z | Tip z | Current success | Precise reward |
|---|---:|---:|---|---:|
| Fully seated | 50 mm | 0 mm | true | 0.353019 |
| Hovering above the hole | 100 mm | 50 mm | false | 1.000000 |
| Partly inserted, moving upward at 1 m/s | 64 mm | 14 mm | true | 0.582879 |

Both environments reproduced these values. The reward actually improves when
lifting out of the correct seated position toward a hover above the hole.

The goal replica has the same offset bug: `make_goal_spec` anchors geometry at
the source tracking site, but `StackingCommand` places that marker at its center
target. Thus the orange peg's tip is at 50 mm and its center at 100 mm: **the
rendered target peg is 50 mm too high**. This agrees with the reward optimum but
disagrees with success.

The success predicate additionally checks only center XY (<15 mm) and center
height error (<15 mm). It has no seating duration, velocity, orientation, or
release check. The moving partial-insertion probe demonstrates that it does not
mean fully seated and settled. Whether partial insertion should count is a task
definition choice; the coordinate mismatch above is an implementation error.

Recommended correction: explicitly distinguish desired root pose from desired
tracking-site pose. Use one consistent reference for goal observations and bringing
rewards, and place the marker using its own anchor convention. Preserve the teacher's
tip-based approach observations unless its offsets are updated together. Separately
specify insertion depth, acceptable tilt, and whether stable seating is required.

Sources: `config/franka/env_cfgs.py:669`, `mdp/commands.py:1002`,
`mdp/rewards.py:17`, `mdp/observations.py:42`,
`asset_zoo/objects/goal.py:8`, `peg_in_hole/xmls/peg.xml`.

## 2. Cage-Drag: open fingers do not establish caging

The goal and cube use the same center reference. Goal z and the actual cube
collider half-height are both 22.6 mm (23 mm is its X half-extent, not its height).
The 35 mm minimum initial goal distance prevents immediate success
from overlapping spawn/goal regions.

Three distinct issues remain:

1. **No cage geometry requirement.** `caged` means only the episode minimum finger
   joint sum exceeds 55 mm. There is no cube-in-gripper-frame, contact, or transport
   history check. Placing the cube at the goal with the open gripper far away counts
   as success. This establishes a predicate gap; it does not establish that a specific
   exploiting policy has been trained. An open-finger shove can satisfy the code
   without demonstrating the advertised caging skill.
2. **Two success definitions disagree after a violation.** Reach goal, then close
   fingers: `compute_success()` becomes false, while `episode_success` stays 1.
   Reopening does not reconcile them. This contradicts the documented whole-episode
   prohibition. The current strict teacher evaluator stops scoring at first success,
   so the effect on that evaluator differs from a full-episode evaluator. The
   existing cage test checks `at_goal`, not `episode_success`.
3. **Rewards still pay after permanent invalidation.** Both transport rewards ignore
   the minimum-aperture latch. Closing has a per-step -0.5 penalty, but reopening
   removes that penalty even though success remains impossible for the episode.
   The precise reward remains 1 at the goal after the violation. This is a training
   objective mismatch, not evidence of a measured RL exploit.

Recommended correction: specify an actual caging relation and how much transport
must occur while it holds. Choose either an episode-long prohibition with revocable
success or a trial that ends immediately on valid success. Align rewards, metrics,
and evaluation with that choice. Resetting a latch alone cannot enforce caging.

Sources: `mdp/commands.py:2085`, `cage_drag_env_cfg.py:254`,
`mdp/rewards.py:235`, `tests/test_class_a_wave1.py:172`,
`continual_distill/classical/episode_evaluation.py`.

## 3. Reorient-Object and Topple-Block: no orientation reward

Both use angle-based success but their positive rewards are generic Cartesian
reach/bring kernels. There is no angular progress or success reward. For the same
position, changing orientation from success to failure leaves those kernels
unchanged. The probes confirm this for upright versus 90-degree-rotated states.
For Topple-Block, the Cartesian target also retains the initial standing center
height, while the rendered goal uses the lower resting height of the toppled block.

Recommended correction: use the command's angle error (including Topple's symmetric
axis convention) in the task reward and use positional drift only as a constraint.
The existing generic position bonus should not be the sole completion objective.

## 4. Additional definition gaps found by code inspection

- **Stack-Cube:** shares Peg's position-only success predicate. Correct height is
  described as resting/released, but neither rest nor release is checked.
- **Edge-Grasp:** lift height and lateral drift suffice; neither a grasp nor retention
  is required. The existing CL-V3 D7 decision already records transient flicks scoring
  success. That proposal remains unapplied.
- **Tool-Pull:** puck-at-goal is sufficient. `tool_grasped` is a proximity diagnostic
  and is not required for success. Rewards likewise do not enforce using the tool.
- **Place-In-Container / Throw-To-Bin:** a center containment proxy and low linear
  velocity are used. Low speed alone does not establish release; a stationary held
  cube can satisfy it. The radial XY test is also a conservative proxy rather than
  exact orientation-aware containment of a cube inside a rectangular basket.
- **Pivot-Lift:** inherits point-goal lifting success without checking the wall-pivot
  sequence or a retained grasp. Geometry is intended to induce the skill, but the
  success predicate does not verify the sequence.

These are differences between stated behavior and measured behavior. Whether to
enforce a particular strategy is a benchmark choice, unlike Peg's inconsistent
coordinate frames.

## 5. Coverage across all 25 tasks

| Tasks | Review outcome |
|---|---|
| Peg-Insertion | Confirmed reward/observation/marker frame mismatch; seating semantics incomplete |
| Cage-Drag | Confirmed caging omission, success-latch inconsistency, reward/constraint mismatch |
| Reorient-Object, Topple-Block | Confirmed missing angular reward; transient orientation can latch |
| Stack-Cube, Edge-Grasp, Tool-Pull, Pivot-Lift | Definition gaps listed above |
| Place-In-Container, Throw-To-Bin | Containment proxy and release semantics need specification |
| Lift-Cube, Push-Cuboid, Drag-Pull, Strike-Slide, Reach-Target | No analogous center/site mismatch found; position-reaching objectives are consistent at this review level |
| Open-Door, Open-Drawer, Push-Button, Turn-Lever, Rotate-Valve, Flip-Switch, Slide-Window, Open-Lid, Push-Flap, Axial-Extract | Joint-based success and site-based Cartesian shaping inspected; exact mechanism goal FK covered by geometry tests |

This is not a certification that every reset, contact trajectory, reward optimum,
or teacher behavior is valid. New teacher scores should follow corrections to the
task contract, rather than being used to decide whether the contract is correct.
