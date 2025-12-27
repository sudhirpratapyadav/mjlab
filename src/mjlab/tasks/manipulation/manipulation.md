# Harmonization Analysis: Manipulation Tasks

## Overview
- **Free Object Task**: lift_cube (manipulate a free floating cube)
- **Articulated Object Tasks**: open_door, open_drawer, push_button (manipulate objects with joints)

---

## KEY DIFFERENCES

### 1. **OBSERVATION NAMING & STRUCTURE**

#### Lift Cube (Free Object):
```python
- "joint_pos" (no explicit params)
- "joint_vel" (no explicit params)
- "ee_to_cube" → uses ee_to_object_distance()
- "cube_to_goal" → uses object_position_error()
- "actions"
```
**Total: ~5 observations**

#### Articulated Objects (Door/Drawer/Button):
```python
- "joint_pos" (explicit asset_cfg param)
- "joint_vel" (explicit asset_cfg param)
- "handle_pos" → handle_geom_position()
- "handle_quat" → handle_body_quaternion()
- "gripper_pos" → gripper_position()
- "gripper_orientation" → gripper_orientation()
- "handle_body_orientation" → handle_body_orientation()
- "gripper_to_handle" → gripper_to_handle_vector()
- "target_to_handle" → target_to_handle_vector()
- "target_orientation_diff" → target_orientation_diff()
- "control_qpos_diff" → control_qpos_difference()
```
**Total: ~11 observations (60 dims)**

**INCONSISTENCY IDENTIFIED:**
- Lift cube uses **"ee_to_cube"** vs articulated uses **"gripper_to_handle"**
- Lift cube uses **"cube_to_goal"** vs articulated uses **"target_to_handle"**
- Lift cube has **NO** gripper position/orientation observations
- Lift cube has **NO** control_qpos_diff observation
- Articulated tasks have **MUCH richer** observation space

---

### 2. **ACTION SPACE**

#### Lift Cube:
```python
JointPositionActionCfg(scale=0.5, use_default_offset=True)
```
- Absolute position control
- Scale: 0.5

#### Articulated Objects:
```python
JointDeltaPositionActionCfg(scale=0.04, offset=0.0)
```
- Delta position control
- Scale: 0.04
- No offset

**MAJOR DIFFERENCE:**
- **Position control type**: Absolute vs Delta
- **Scale magnitude**: 0.5 vs 0.04 (12.5x difference!)

---

### 3. **COMMAND NAMING & STRUCTURE**

#### Lift Cube:
- Command name: **"lift_height"**
- Command class: `LiftingCommand`
- Target type: 3D position (`target_pos`)
- Config: `LiftingCommandCfg` with `ObjectPoseRangeCfg`

#### Articulated Objects:
- Command names: **"open_door"**, **"open_drawer"**, **"push_button"**
- Command classes: `OpenDoorCommand`, `OpenDrawerCommand`, `PushButtonCommand`
- Target types:
  - Door: angle (`target_angle`) + position (`target_pos`)
  - Drawer: distance (`target_distance`) + position (`target_pos`)
  - Button: distance (`target_distance`) + position (`target_pos`)
- Configs: `{Door/Drawer/Button}PoseRangeCfg`

**INCONSISTENCY:**
- Lift uses **"lift_height"** (implies Z-only) but actually 3D position
- Articulated uses **"open_X"** or **"push_X"** (action-oriented naming)
- All articulated tasks track **reached_box** (latch), lift cube does NOT

---

### 4. **REWARD STRUCTURE**

#### Lift Cube:
```python
"lift" → staged_position_reward (reaching + bringing gating)
"lift_precise" → bring_object_reward
"action_rate_l2" → -0.01
"joint_pos_limits" → -10.0
"joint_vel_hinge" → -0.01 (with curriculum)
```
**Total: 5 rewards**

#### Articulated Objects (Door/Drawer/Button):
```python
"gripper_to_handle" → gripper_to_handle_reward (weight: 4.0)
"handle_to_target" → handle_to_target_reward (weight: 8.0)
"no_X_collision" → no_door_body_collision_reward (weight: 0.25)
```
**Total: 3 rewards**

**CRITICAL DIFFERENCES:**
1. **Reward functions**:
   - Lift: Uses `staged_position_reward` (Gaussian kernels)
   - Articulated: Uses `tanh`-based rewards (1 - tanh(k * dist^4))

2. **Naming inconsistency**:
   - Lift cube has NO "gripper_to_handle" equivalent (has "ee_to_cube" in obs but not reward)
   - Articulated has explicit two-stage rewards, lift uses single staged reward

3. **Regularization**:
   - Lift cube: Has action_rate, joint_limits, joint_vel penalties
   - Articulated: ONLY has collision penalty, NO action/velocity penalties

4. **Reward weights**:
   - Lift: 1.0, 1.0 (main rewards)
   - Articulated: 4.0, 8.0 (main rewards) - **2x higher for handle_to_target**

---

### 5. **TERMINATION CONDITIONS**

#### Lift Cube:
```python
"time_out"
"ee_ground_collision"
"object_out_of_bounds" → UNIQUE (x_bounds, y_bounds check)
```

#### Articulated Objects:
```python
"time_out"
"ee_ground_collision"
```

**DIFFERENCE:**
- Lift cube has **out_of_bounds** termination (spatial limits)
- Articulated tasks have **NO** spatial termination

---

### 6. **SENSORS**

#### Lift Cube:
```python
ee_ground_collision_cfg
```
**Total: 1 sensor**

#### Articulated Objects:
```python
ee_ground_collision_cfg
ee_{door/drawer/button}_collision_cfg → collision with articulated body
```
**Total: 2 sensors**

**DIFFERENCE:**
- Articulated tasks track collision with **object body** (barrier/cabinet/base)
- Lift cube does NOT track cube collision

---

### 7. **CURRICULUM**

#### Lift Cube:
```python
"joint_vel_hinge_weight" → Increases penalty over time
  Step 0: -0.01
  Step 24000: -0.1
  Step 36000: -1.0
```

#### Articulated Objects:
```python
curriculum = {}  # Empty!
```

**DIFFERENCE:**
- Lift cube has **progressive curriculum** on velocity penalty
- Articulated tasks have **NO curriculum**

---

### 8. **EPISODE LENGTH**

#### Lift Cube:
```python
episode_length_s=20.0
```

#### Articulated Objects:
```python
episode_length_s=3.0
```

**MAJOR DIFFERENCE:**
- Lift cube: **20 seconds** (6.7x longer!)
- Articulated: **3 seconds**

---

### 9. **METRICS TRACKED**

#### Lift Cube:
```python
"object_height"
"position_error"
"at_goal"
"episode_success"
```

#### Articulated Objects (Door):
```python
"door_angle"  # or drawer_distance, button_distance
"angle_error"  # or distance_error
"at_goal"
"episode_success"
"reached_handle"
"gripper_to_handle_distance"
```

**DIFFERENCE:**
- Articulated tracks **"reached_handle"** (latch metric)
- Articulated tracks **joint-specific metrics** (angle/distance)
- Lift tracks **"object_height"** (Z-position specific)

---

### 10. **PHYSICS SETTINGS**

#### Lift Cube:
```python
gravity=(0.0, 0.0, -9.81)  # Default gravity
```

#### Articulated Objects:
```python
gravity=(0.0, 0.0, 0.0)  # DISABLED for debugging
```

**CRITICAL DIFFERENCE:**
- Articulated tasks have **gravity disabled**!
- This is a **major physics difference**

---

### 11. **OBSERVATION FUNCTION NAMING PATTERNS**

#### Lift Cube Functions:
- `ee_to_object_distance()` - specific to free objects
- `object_position_error()` - specific to free objects

#### Articulated Functions:
- `handle_geom_position()` - articulated-specific
- `handle_body_quaternion()` - articulated-specific
- `gripper_position()` - general
- `gripper_orientation()` - general
- `gripper_to_handle_vector()` - articulated-specific
- `target_to_handle_vector()` - articulated-specific
- `target_orientation_diff()` - articulated-specific
- `control_qpos_difference()` - articulated-specific

**PATTERN INCONSISTENCY:**
- Lift uses **"object"** terminology
- Articulated uses **"handle"** terminology
- Could unify as **"target_object"** or **"manipulation_object"**

---

### 12. **REWARD FUNCTION NAMING PATTERNS**

#### Lift Cube:
- `staged_position_reward()` - describes reward structure
- `bring_object_reward()` - action-oriented

#### Articulated:
- `gripper_to_handle_reward()` - state-oriented
- `handle_to_target_reward()` - state-oriented
- `no_door_body_collision_reward()` - constraint-oriented

**PATTERN:**
- Articulated uses **state-based** naming
- Lift uses **action/structure-based** naming

---

## HARMONIZATION RECOMMENDATIONS

### A. **Unified Naming Convention**

**Current Issues:**
1. "ee" vs "gripper" (end-effector terminology)
2. "cube"/"object" vs "handle" (manipulation target)
3. "lift_height" vs "open_X"/"push_X" (command naming)

**Proposed Standard:**
```python
# Observation naming pattern:
"robot_joint_pos"
"robot_joint_vel"
"gripper_pos"
"gripper_orientation"
"target_pos"  # Position of manipulation target (handle or object)
"target_orientation"
"gripper_to_target"  # Vector from gripper to target
"target_to_goal"  # Vector from target to goal position
"goal_orientation_diff"  # For articulated objects
"control_qpos_diff"  # Delta control feedback

# Command naming pattern:
"manipulate_{object_type}"  # e.g., "manipulate_cube", "manipulate_door"
```

---

### B. **Unified Observation Structure**

**Minimal Common Set (all tasks):**
```python
"robot_joint_pos"
"robot_joint_vel"
"gripper_pos"
"gripper_to_target"
"target_to_goal"
"actions"
```

**Extended Set (articulated objects):**
```python
+ "gripper_orientation"
+ "target_orientation"
+ "goal_orientation_diff"
+ "control_qpos_diff"
```

**Free object specific:**
```python
+ "target_velocity"  # For dynamic objects
```

---

### C. **Unified Reward Structure**

**Proposed common reward naming:**
```python
# Phase 1: Reach target
"reach_target" → distance-based (gripper to object/handle)

# Phase 2: Manipulate target
"manipulate_target" → task-based (lift/open/push)

# Regularization (optional, configurable)
"action_regularization" → action smoothness
"velocity_regularization" → joint velocity limits
"collision_penalty" → illegal contacts
"pose_regularization" → stay near init pose
```

**Common weights:**
```python
reach_target: 4.0
manipulate_target: 8.0
regularization: -0.01 to -1.0 (curriculum-based)
collision: -10.0 or termination
```

---

### D. **Unified Command Structure**

**All commands should have:**
```python
class ManipulationCommand:
    # Common attributes
    target_pos: torch.Tensor  # 3D goal position
    target_orientation: torch.Tensor | None  # Optional quaternion
    episode_success: torch.Tensor  # Latch
    reached_target: torch.Tensor  # Latch (gripper reached object)

    # Common metrics
    metrics["target_state"]  # Object height / joint angle / joint distance
    metrics["state_error"]  # Distance to goal
    metrics["at_goal"]
    metrics["episode_success"]
    metrics["reached_target"]
    metrics["gripper_to_target_distance"]
```

---

### E. **Unified Metric Naming**

**Current inconsistencies:**
- "object_height" vs "door_angle" vs "drawer_distance" vs "button_distance"
- "position_error" vs "angle_error" vs "distance_error"

**Proposed:**
```python
"{task_type}_state"  # Generalized state measurement
"goal_error"  # Generalized error measurement
"at_goal"
"episode_success"
"reached_target"  # Instead of "reached_handle" or implicit
"gripper_target_distance"
```

---

### F. **Action Space Standardization**

**Decision needed:**
1. Should all use **Delta** or **Absolute** position control?
2. Standard scale factor?

**Recommendation:**
- Use **JointDeltaPositionActionCfg** for all (more stable)
- Scale: **0.04** for articulated, **maybe 0.1** for free object (needs tuning)
- OR: Make scale a **task-dependent hyperparameter**

---

### G. **Physics Consistency**

**Critical fix:**
- Remove `gravity=(0.0, 0.0, 0.0)` from articulated tasks
- Use standard gravity for all, OR
- Make it a configurable parameter with clear documentation

---

### H. **Episode Length**

**Decision needed:**
- Should episode length be task-specific or unified?

**Recommendation:**
- Keep task-specific but use consistent naming:
  - `episode_length_s` (already consistent)
  - Document why different (lift is slower, articulated is fast precision task)

---

## SUMMARY TABLE

| **Aspect** | **Lift Cube** | **Articulated (Door/Drawer/Button)** | **Harmonization Priority** |
|------------|---------------|--------------------------------------|----------------------------|
| **Observations** | 5 terms (~23 dims) | 11 terms (~60 dims) | **HIGH** - Add gripper obs to lift |
| **Action Type** | Absolute Position | Delta Position | **CRITICAL** - Choose one |
| **Action Scale** | 0.5 | 0.04 | **CRITICAL** - Standardize or document |
| **Command Name** | "lift_height" | "open_X", "push_X" | **MEDIUM** - Use "manipulate_X" |
| **Reward Functions** | Gaussian kernels | Tanh-based | **HIGH** - Unify functional form |
| **Reward Count** | 5 | 3 | **MEDIUM** - Add regularization to articulated |
| **Terminations** | 3 (incl. out_of_bounds) | 2 | **LOW** - Task-specific OK |
| **Sensors** | 1 | 2 | **LOW** - Task-specific OK |
| **Curriculum** | Yes (velocity) | No | **MEDIUM** - Add to articulated? |
| **Episode Length** | 20s | 3s | **LOW** - Document difference |
| **Gravity** | Enabled | **DISABLED** | **CRITICAL** - Fix bug or document |
| **Metrics** | 4 | 6 | **LOW** - Already similar |
| **Latch State** | No "reached_target" | Yes "reached_box" | **MEDIUM** - Add to lift cube |

---

## RECOMMENDED HARMONIZATION STEPS

### Phase 1: Critical Fixes
1. **Fix gravity** in articulated tasks or document why disabled
2. **Standardize action type** (Delta vs Absolute)
3. **Standardize action scale** or make it explicit config parameter

### Phase 2: Naming Unification
4. Rename observations to use consistent terminology (gripper/target instead of ee/handle/cube)
5. Rename commands to "manipulate_X" pattern
6. Rename metrics to use generic patterns

### Phase 3: Feature Parity
7. Add gripper position/orientation to lift_cube observations
8. Add "reached_target" latch to lift_cube command
9. Consider adding regularization rewards to articulated tasks
10. Consider adding curriculum to articulated tasks

### Phase 4: Code Structure
11. Create base classes: `ManipulationCommand`, `ManipulationObservation`, `ManipulationReward`
12. Refactor task configs to inherit from common base with overrides
13. Document design decisions (why different episode lengths, etc.)

---

## FILE REFERENCES

### Task Configuration Files:
- `lift_cube_env_cfg.py` - Free object manipulation (cube lifting)
- `open_door_env_cfg.py` - Articulated object (door hinge)
- `open_drawer_env_cfg.py` - Articulated object (drawer slide)
- `push_button_env_cfg.py` - Articulated object (button slide)

### MDP Component Files:
- `mdp/observations.py` - All observation functions
- `mdp/rewards.py` - All reward functions
- `mdp/terminations.py` - Termination conditions
- `mdp/commands.py` - Command definitions (LiftingCommand, OpenDoorCommand, etc.)

---

## NOTES

The biggest issues requiring immediate attention are:
1. **Action space mismatch** (absolute vs delta, scale difference of 12.5x)
2. **Gravity disabled** in articulated tasks (likely debugging artifact)
3. **Observation richness disparity** (lift cube missing gripper obs that articulated tasks have)
4. **Inconsistent naming** (ee/gripper, cube/handle, object/target terminology)

These differences make it difficult to:
- Transfer learning between tasks
- Create unified training curricula
- Compare task difficulty objectively
- Build composable task primitives
