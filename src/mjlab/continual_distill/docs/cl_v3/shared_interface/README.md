# Shared Franka interface v1 — 2026-09-12

**Superseded:** the user requested restoring the working 60D field set. See the [current 60D contract](../shared_60/README.md). This report and its scripts/results describe the provisional 143D implementation.

All 25 CL tasks now use **the same 143D observation and 8D action contract** through the task registry. This also applies to the four additional Franka object variants. Train, test and play modes use the same fields, order, units, padding and action semantics. Actor and critic receive identical state. There is no task-ID one-hot.

Implementation: `src/mjlab/tasks/manipulation/franka_interface.py`. The Franka registration finalizer applies the contract after each task-specific config is built; base config builders remain available for legacy observation reconstruction. Load runnable configurations through `load_env_cfg`.

## Actions

The order is joint1–joint7, then finger_joint1 (the second finger is mechanically coupled). Each input is bounded to [-1, 1], clipped inside the action term even when bypassing the PPO wrapper. The common mapping is:

```
target = midpoint + half_range * clip(action, -1, 1)
```

These are absolute position targets, independent of the task's reset/default pose. There is no per-substep accumulation. For the gripper, -1 = fully closed, 0 = half open, +1 = fully open. The commanded finger travel is 0–0.04 m; total jaw aperture is 0–0.08 m. Every task retains that same controllable gripper channel, including Cage.

| Channel | Target at -1 | Target at +1 | Units |
|---|---:|---:|---|
| joint1 | -2.8973 | 2.8973 | rad |
| joint2 | -1.7628 | 1.7628 | rad |
| joint3 | -2.8973 | 2.8973 | rad |
| joint4 | -3.0718 | -0.0698 | rad |
| joint5 | -2.8973 | 2.8973 | rad |
| joint6 | -0.0175 | 3.7525 | rad |
| joint7 | -2.8973 | 2.8973 | rad |
| finger_joint1 | 0 | 0.04 | m |

Registered PPO configs use `clip_actions=1.0` and initial noise standard deviation `0.2` in these normalized units. This gives initial arm target standard deviations of approximately 0.30–0.58 rad, instead of 0.04 rad under the old mapping. It is an initial exploration setting, not a claim of optimal PPO tuning. Existing joint-limit penalties continue to operate on physical state.

## Observations

Positions are in **environment-local, world-aligned coordinates**, in meters. Subtracting the environment origin removes vectorized-grid offsets. Relative vectors keep their geometric meanings. Joint positions are normalized using the same fixed physical range for every task, rather than each task's different reset pose. Arm velocities are divided by 10 rad/s; finger velocities by 0.2 m/s. Linear velocities use m/s and angular velocities are divided by 10 rad/s. Control error is divided by each actuator's half range.

All absent object, fixture or tool fields are zero; five presence flags identify object, fixture, tool, mechanism and axis-goal availability. The fixture role contains the stack base/hole board, container, ledge or wall as appropriate. Tool position uses its tracking/grasp site. Object position and velocity use the tracking site where present; object quaternion/rotation describe its root frame, with mechanism displacement/velocity supplied separately.

The observation is clean simulator state in this phase: task-dependent noise, including the old ±1 cm Peg noise, is removed. Any later sensor-noise model should be shared and unit-aware. Running actor/critic observation normalization remains enabled in the registered PPO configs.

| Field | Slice (Python, end exclusive) | Width |
|---|---|---:|
| `joint_position` | `0:9` | 9 |
| `joint_velocity` | `9:18` | 9 |
| `gripper_position` | `18:21` | 3 |
| `gripper_rotation` | `21:27` | 6 |
| `gripper_velocity` | `27:33` | 6 |
| `object_position` | `33:36` | 3 |
| `object_quaternion` | `36:40` | 4 |
| `object_rotation` | `40:46` | 6 |
| `object_velocity` | `46:52` | 6 |
| `goal_position` | `52:55` | 3 |
| `goal_rotation` | `55:61` | 6 |
| `gripper_to_object` | `61:64` | 3 |
| `object_to_goal` | `64:67` | 3 |
| `goal_orientation_diff` | `67:73` | 6 |
| `fixture_position` | `73:76` | 3 |
| `fixture_quaternion` | `76:80` | 4 |
| `fixture_velocity` | `80:86` | 6 |
| `tool_position` | `86:89` | 3 |
| `tool_rotation` | `89:95` | 6 |
| `tool_velocity` | `95:101` | 6 |
| `mechanism` | `101:105` | 4 |
| `axis_goal` | `105:112` | 7 |
| `history` | `112:117` | 5 |
| `contacts` | `117:121` | 4 |
| `control_error` | `121:129` | 8 |
| `last_action` | `129:137` | 8 |
| `presence` | `137:142` | 5 |
| `time_remaining` | `142:143` | 1 |

Rotation6 fields are the last two rows of the rotation matrix; quaternions are wxyz. `goal_orientation_diff` retains the legacy first-two-row goal-minus-object difference. `mechanism` contains current position, target, velocity and a hinge flag; its units are pi radians for hinges and 0.25 m for slides. `axis_goal` contains the object's scored body axis, desired world axis and sign-symmetry flag.

History slots are minimum episode aperture / 0.08 m, accumulated caged progress in meters, tool-used flag, direct-contact flag, and pivoted flag. Contact slots are robot/object contact, bilateral object grasp, bilateral tool grasp, and tool/object contact. `last_action` records the bounded action for the transition just taken (zero after reset). `time_remaining` is the fraction of the configured episode remaining, or 1 for infinite play episodes.

## Teacher and checkpoint compatibility

Scripted teachers automatically recognize the 143D schema. Their boundary adapter reconstructs their original 38/51/60/69D inputs, restores physical joint units and translates their old actions into the common normalized targets. Their internal controllers retain the old conventions. Legacy-shaped calls still use legacy action units.

Old neural checkpoints and datasets are **not directly compatible**. They lack the new state fields and use different action units; regenerate datasets through the updated teachers and train new networks/normalizers, or build an explicitly versioned migration adapter. Do not feed old raw actions into the new environments. The schema key `franka_shared_v1` is present in saved environment configs.

## Validation

- 48 unit/regression tests passed: all 29 registered Franka tasks in all three modes, layout continuity, clipping order, gripper endpoints, teacher unit/slot conversion, and existing teacher waypoint/transition/evaluation checks.
- GPU integration: all 25 CL tasks, four environments each, 16 teacher steps and eight random-action steps per environment: **2,400 transitions**. Checked finite observations/rewards/MuJoCo positions, identical actor/critic inputs, 143D/8D dimensions, physical joint round trips, local gripper coordinates, action endpoints, and normalized teacher commands. See [verification.json](verification.json) and [verify.py](verify.py).
- Independent legacy comparison: Reach, Lift, Peg and Tool cover all four previous layouts. Original observation-manager terms were evaluated on the same live simulation state, then compared against teacher reconstruction after removing scene origins. See [legacy_comparison.json](legacy_comparison.json) and [compare_legacy.py](compare_legacy.py).
- New interface/adapter/tests pass Ruff. No PPO convergence claim is made by these checks.

Reward functions, success predicates and physics were not changed in this phase. The other reward/physics findings in the previous RL audit remain follow-up work; the changed action units naturally change the numerical size of action-rate penalties.
