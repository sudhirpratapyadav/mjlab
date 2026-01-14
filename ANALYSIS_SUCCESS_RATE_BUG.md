# Analysis: False 100% Success Rate for Lift-Cube Task

## Problem Statement
- **Training sequence**: [OpenDoor, LiftCube, OpenDrawer, PushButton]
- **During LiftCube training (Task 1)**: Success rate fluctuates 20-40% ✓
- **During OpenDrawer training (Task 2)**: LiftCube success jumps to 100% ✗
- **eval_student.py with final checkpoint**: Shows 0% success for LiftCube ✓ (correct)
- **Wandb logs**: Show 100% success for LiftCube ✗ (incorrect)

## Key Findings

### 1. Episode Length Difference
**lift_cube (test mode)**:
- `episode_length_s = 5.0` (250 steps at 0.02s control_dt)
- Removes terminations: `ee_ground_collision`, `object_out_of_bounds`

**Other tasks (test mode)**:
- `episode_length_s = 3.0` (150 steps)
- Removes terminations: `ee_ground_collision` only

### 2. Command Resampling Configuration
All tasks have resampling_time_range > episode_length in test mode:
- lift_cube: resample every 8.0-12.0s, episode=5.0s
- open_door/drawer/button: resample every 10.0-15.0s, episode=3.0s

**Result**: Commands (cube/door positions, goals) are NEVER resampled during test episodes - only at episode reset.

### 3. Success Metric Extraction in continual_distill.py

```python
# Lines 441-447: Student evaluation
for step in range(episode_length):  # Runs for 250 steps (lift_cube)
    obs, reward, terminated, truncated, info = env.step(student_action_torch)
    episode_returns += reward.cpu().numpy()

# Extract success AFTER loop
if "log" in info and isinstance(info["log"], dict):
    for key, value in info["log"].items():
        if "episode_success" in key.lower():
            student_successes = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)
```

**Issue**: `info["log"]["Metrics/lift_object/episode_success"]` is a **scalar mean** (not per-env array), computed in `command_manager.py:47`:
```python
extras[metric_name] = torch.mean(metric_value[env_ids]).item()
```

### 4. When info["log"] Gets Populated

From `manager_based_rl_env.py:275-316`:
```python
def step(self, action):
    # ... physics simulation ...
    self.episode_length_buf += 1

    # Check terminations
    self.reset_buf = self.termination_manager.compute()

    # IF envs timeout/terminate:
    if len(reset_env_ids) > 0:
        self._reset_idx(reset_env_ids)  # <-- POPULATES extras["log"]

    self.command_manager.compute(dt=self.step_dt)  # <-- Updates episode_success metric

    return (obs, reward, terminated, truncated, self.extras)  # <-- extras contains "log"
```

**Critical order of operations**:
1. Check which envs timed out
2. **IF timeout**: Call `_reset_idx()` which populates `extras["log"]` with metrics
3. Call `command_manager.compute()` which updates `episode_success` for next step
4. Return `self.extras`

### 5. The _reset_idx() Function

```python
def _reset_idx(self, env_ids):
    # ... reset scene ...

    self.extras["log"] = dict()  # CLEARS extras["log"]

    # Each manager.reset() returns metrics for env_ids that reset
    info = self.command_manager.reset(env_ids)  # <-- Returns MEAN of episode_success
    self.extras["log"].update(info)
```

`command_manager.reset()` (from command_manager.py:43-51):
```python
def reset(self, env_ids: torch.Tensor) -> dict[str, float]:
    extras = {}
    for metric_name, metric_value in self.metrics.items():
        extras[metric_name] = torch.mean(metric_value[env_ids]).item()  # <-- MEAN!
        metric_value[env_ids] = 0.0  # Reset for next episode
    self.command_counter[env_ids] = 0
    self._resample(env_ids)  # Resample command (cube position, goal)
    return extras
```

## ROOT CAUSE HYPOTHESIS

### Scenario: Evaluating lift-cube during open-drawer training

1. **Environment reuse**: Each task's environment is created once and stored in `task_buffers` (continual_distill.py:1038)

2. **Evaluation call**: `evaluate_environment()` is called for lift-cube task while training open-drawer

3. **Student rollout** (lines 428-439):
   ```python
   obs, _ = env.reset()  # All 64 envs reset, new cube positions/goals sampled

   for step in range(250):  # Run full episode
       obs, reward, terminated, truncated, info = env.step(student_action_torch)
       episode_returns += reward.cpu().numpy()
   ```

4. **On step 250**: All 64 envs timeout simultaneously
   - `_reset_idx([0,1,2,...,63])` is called
   - `command_manager.reset()` computes: `mean(episode_success[0:64])`
   - This mean is stored in `info["log"]["Metrics/lift_object/episode_success"]`

5. **After loop**: Extract `student_successes` from `info["log"]`

### Potential Issues

**ISSUE A: extras["log"] persistence across evaluations**
- `extras["log"]` is an instance variable that persists between calls
- If somehow `info["log"]` from a previous successful evaluation persists, it would show false 100%
- **BUT**: `extras["log"]` is cleared in `_reset_idx()` (line 398), which IS called when envs timeout

**ISSUE B: episode_success latching across evaluations**
- `episode_success` is latched: `self.episode_success = torch.maximum(self.episode_success, at_goal)`
- Reset in `_resample_command()` (line 87): `self.episode_success[env_ids] = 0.0`
- `_resample_command()` is called from `command_manager.reset()` (line 50)
- **Should be fine**: Reset happens BEFORE metric extraction

**ISSUE C: Timing of command_manager.compute() vs metric extraction**
Looking at the order in `env.step()`:
1. IF envs timeout: `_reset_idx(env_ids)` extracts metrics, resamples command
2. THEN: `command_manager.compute()` updates metrics for NEXT step

So metrics extracted in `info["log"]` are from BEFORE the reset, which is correct!

**ISSUE D: Scalar handling bug**
Lines 446-447 in continual_distill.py:
```python
student_successes = value.cpu().numpy() if hasattr(value, 'cpu') else np.array(value)
```

If `value` is a Python float (0.078125 from your test), then:
- `np.array(0.078125)` creates a 0-dimensional array
- Line 450: `student_successes = student_successes.tolist()` converts to Python float
- Line 459: `float(np.mean(student_successes))` → `np.mean(0.078125)` → `0.078125`

So the code correctly extracts the mean! **Not the bug**.

## MOST LIKELY ROOT CAUSE

Based on your description that lift-cube shows **100% during open-drawer training**, the most likely cause is:

### **Hypothesis: Goal position coincides with object spawn position**

When the environment resets during evaluation:
1. Cube spawns at random position within `x=(0.6, 0.8), y=(-0.15, 0.15), z=(0.02, 0.05)`
2. Goal sampled at random position within `x=(0.6, 0.8), y=(-0.15, 0.15), z=(0.2, 0.4)`
3. Success threshold is 0.05m (5cm)

**IF** by chance:
- Cube spawns at `(0.7, 0.0, 0.05)`
- Goal is at `(0.7, 0.0, 0.2)`
- Distance = `√((0)² + (0)² + (0.15)²)` = 0.15m

The cube would need to move up 15cm vertically. But in the **very first step** of the episode, **before the robot acts**, the goal_error is already computed.

**Wait** - let me check if there's an issue with when metrics are updated vs when the episode actually runs...

## RECOMMENDATION

Add debug logging to continual_distill.py to capture:
1. Cube spawn positions
2. Goal positions
3. Initial goal_error at episode start
4. Final goal_error at episode end
5. Per-environment episode_success values (not just mean)

This will reveal if there's a geometric/sampling issue causing false positives.
