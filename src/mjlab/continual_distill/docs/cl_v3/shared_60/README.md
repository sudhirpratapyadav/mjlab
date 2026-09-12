# Restored common 60D observations — 2026-09-12

**Current scope: 24 active tasks.** Tool-Pull is deferred, with its code and registration retained. Use [the active task list](../active_tasks.json). The 25-task verification below predates this scope change.

Supersedes the provisional 143D interface. All 25 CL tasks (and the four additional Franka variants) now use the **working 60D observation field order**, with the normalized **8D absolute joint-position action mapping retained**. Registered train/test/play modes share this interface; actor and critic use identical, clean state.

| Field | Slice, end exclusive | Dimensions |
|---|---|---:|
| `joint_position` | `0:9` | 9 |
| `joint_velocity` | `9:18` | 9 |
| `object_position` | `18:21` | 3 |
| `object_quaternion` | `21:25` | 4 |
| `gripper_position` | `25:28` | 3 |
| `gripper_rotation` | `28:34` | 6 |
| `object_rotation` | `34:40` | 6 |
| `gripper_to_object` | `40:43` | 3 |
| `object_to_goal` | `43:46` | 3 |
| `goal_orientation_diff` | `46:52` | 6 |
| `control_error` | `52:60` | 8 |

Joint positions and velocities are relative to robot defaults in physical units, as in the original working layout. Object/gripper positions retain the environment-local coordinate correction. Object orientation is still provided as both wxyz quaternion and the last two matrix rows. Goal orientation difference is the first two rows of goal-minus-object rotation matrices. Control error is the actual actuator control minus joint position in physical units. Running PPO normalization remains enabled.

There are no fixture, extra tool, mechanism, velocity-of-object, contact/history, presence-mask, last-action or clock fields. Goal-relative vectors retain destination information for Stack/Peg/Place without a separate fixture slot. Reach treats its target marker as a virtual object, so gripper-to-object is the reaching error and object-to-goal is zero.

The action mapping remains identical on every task: [-1, 1] maps linearly to physical arm joint limits; the gripper maps -1/0/+1 to closed/half-open/open. Inputs are clipped inside the action term. PPO initial noise remains 0.2 and wrapper clipping remains 1.0. Reward functions, success predicates and physics are unchanged.

## Teacher compatibility and the Tool-Pull limitation

Registry teachers explicitly opt into the common observation contract and normalized action output. Direct legacy classes retain their old action output; width alone cannot identify an action contract now that both old and new standard inputs have 60 values.

Standard 60D controllers consume the restored layout directly. Reach and Stack/Peg adapt to their older internal layouts. The removed Stack/Peg base-position slot is unused by these controllers and is zero-filled internally; it is not an extra policy observation.

**Tool-Pull observes the puck and its goal, but not the independently randomized stick.** This is a real information loss required by retaining precisely these 60 fields. Its scripted teacher explicitly rejects 60D input with a descriptive error, while still accepting an explicitly supplied legacy 69D input containing the stick pose. The RL environment runs with the same 60D contract as all others; this does not establish that a feedforward Tool-Pull policy can solve it. No tool pose is silently fabricated or placed into another field.

Old neural checkpoints still need an action-conversion adapter because the retained action scaling differs from the historical one. Provisional 143D checkpoints/datasets are also incompatible with this restored layout. Use the version key `franka_shared_60_v2` to distinguish saved configurations.

## Verification

- 49 focused contract and teacher-regression tests passed, including all 29 registered Franka configurations in train/test/play modes, exact slice order, normalized action endpoints, and explicit handling of Tool's missing information.
- GPU verification: 25 tasks × four environments × 16 transitions = 1,600 transitions. Checked finite rewards/observations/MuJoCo positions, 60D observations, identical actor/critic state, and bounded 8D action targets.
- Independently evaluated original observation-manager terms on the same MuJoCo state. Compared their fields against restored observations after removing environment origins. For Stack/Peg the intentionally removed, controller-unused base slot is excluded from this comparison. Tool's extra nine stick fields are removed when comparing its original 69D layout.
- Compared adapted and original teacher physical action calculations for the other 24 tasks. Tool's rejection is verified explicitly.

Raw results and reproducer: [verification.json](verification.json), [verify.py](verify.py).
