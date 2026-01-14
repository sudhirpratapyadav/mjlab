# Comprehensive Debugging Guide for Success Rate Bug

## What Was Added

I've added extensive debugging output to `continual_distill.py` to track down the false 100% success rate for lift-cube task. The debug output will help identify:

1. **Environment state** before and after evaluations
2. **Command manager state** (episode_success values, goal positions, object positions)
3. **Success metric extraction** from info["log"]
4. **WandB logging** - what values are actually being logged
5. **Task evaluation flow** - which tasks are evaluated and when

## Debug Output Structure

### 1. When `evaluate_all_tasks_env()` is called:

```
####################################################################################################
### EVALUATE_ALL_TASKS_ENV CALLED ###
### Current training task: X
### Global step: XXXXX, Epoch: X
### Will evaluate tasks 0 to X (inclusive)
####################################################################################################
```

This shows:
- Which task is currently being trained
- At which training step this evaluation happens
- Which tasks will be evaluated

### 2. For Each Task Evaluation:

```
>>> Evaluating Task X (task_name) in environment...
```

Then for each task, you'll see:

#### a) Evaluation Start:
```
================================================================================
DEBUG EVAL START: Task X (task_name) | Step XXXXX | Epoch X
================================================================================
Environment info:
  num_envs: 64
  max_episode_length: 250
  episode_length_buf: [0 0 0 0 0] (first 5)
  Command 'lift_object':
    episode_success: [0. 0. 0. 0. 0.] (first 5)
    metrics['episode_success']: [0. 0. 0. 0. 0.] (first 5)
```

**What to check:**
- Are `episode_success` values reset to 0 at start?
- If not, this is stale data from previous evaluation!

#### b) Teacher Rollout:
```
--- TEACHER ROLLOUT START ---
After env.reset() (Teacher):
  num_envs: 64
  episode_length to run: 250

After teacher episode complete:
  'log' in info: True
  info['log'] keys: [list of keys]
    Metrics/lift_object/episode_success: 0.15625
    >>> EXTRACTED teacher episode_success: 0.15625
  Teacher successes mean: 0.1562
--- TEACHER ROLLOUT END ---
```

**What to check:**
- Is the teacher success rate reasonable?
- Does the extracted value match expectations?

#### c) Student Rollout (Most Important!):
```
--- STUDENT ROLLOUT START ---
After env.reset():
  episode_length_buf: [0 0 0 0 0] (first 5)
  lift_object.episode_success: [0. 0. 0. 0. 0.] (first 5)
  lift_object.target_pos[0]: [0.4 0.0 0.3]
  lift_object.object_pos[0]: [0.65 0.05 0.03]
  lift_object.initial_goal_error[0]: 0.3234m

Step 0: episode_length_buf=1
    lift_object.episode_success[:5]: [0. 0. 0. 0. 0.]

Step 125: episode_length_buf=126
    lift_object.episode_success[:5]: [0. 0. 0. 0. 1.]

Step 249: episode_length_buf=250
    lift_object.episode_success[:5]: [0. 0. 0. 1. 1.]

After episode complete (step 250):
  episode_length_buf: [250 250 250 250 250] (first 5)
  terminated: [False False False False False False...] (first 5)
  truncated: [ True  True  True  True  True...] (first 5)
  lift_object state BEFORE extraction:
    episode_success[:10]: [0. 0. 0. 1. 1. 0. 0. 0. 0. 1.]
    metrics['episode_success'][:10]: [0. 0. 0. 1. 1. 0. 0. 0. 0. 1.]

  Extracting from info['log']:
    'log' in info: True
    info['log'] keys: ['Episode_Reward/reach_object', ... 'Metrics/lift_object/episode_success', ...]
    Metrics/lift_object/episode_success: 0.078125
      (Scalar) type=<class 'float'>
    >>> EXTRACTED episode_success: 0.078125
        Type: <class 'numpy.ndarray'>, Shape: ()
  >>> student_successes is 0-dim array (scalar): 0.078125
  Final student_successes (list): [0.078125, 0.078125, ...] (first 10)
  Mean: 0.0781
--- STUDENT ROLLOUT END ---
```

**What to check:**
1. **Initial goal_error**: Is the cube already at the goal at episode start? If yes, that's the bug!
2. **episode_success during rollout**: Does it latch to 1 prematurely?
3. **episode_success BEFORE extraction**: What are the actual per-env values?
4. **Extracted value**: Is it a scalar mean (correct) or something else?

#### d) Computed Metrics:
```
================================================================================
COMPUTED METRICS FOR TASK 1 (lift_cube):
  env_loss: 0.123456
  teacher_return: 12.3456
  student_return: 5.6789
  teacher_success: 0.1562
  student_success: 0.0781
================================================================================
```

**This is the final computed value.**

#### e) WandB Logging:
```
LOGGING TO WANDB (step XXXXX):
  EnvEval/task_1_lift_cube/ts_loss/total: 0.1235
  Rewards/task_1_lift_cube: 5.6789
  Accuracy/task_1_lift_cube: 0.0781
  Training/epoch: 5
```

**What to check:**
- Does `Accuracy/task_1_lift_cube` match the computed `student_success`?
- **If this shows 0.0781 but WandB dashboard shows 1.0, then the bug is in WandB logging or dashboard display, NOT in the code!**

### 3. Evaluation Complete:
```
  >>> Task 1 (lift_cube) evaluation complete!
      Student success: 0.0781

####################################################################################################
### EVALUATE_ALL_TASKS_ENV COMPLETE ###
####################################################################################################
```

## What to Look For

### Scenario 1: Stale episode_success values
If at "Environment info" or "After env.reset()", you see `episode_success` is NOT all zeros, this means:
- The command manager didn't reset properly
- Values from previous successful evaluation are persisting

### Scenario 2: Cube spawns at goal
If `initial_goal_error[0]` is very small (< 0.05m), this means:
- The cube randomly spawned at or very near the goal position
- This would immediately trigger success even with no policy action

### Scenario 3: Success latches too early
Monitor the `episode_success` values at steps 0, 125, 249:
- They should start at 0
- Only change to 1 when the cube actually reaches the goal
- If they're 1 from the start, goal positions are wrong

### Scenario 4: Extraction vs WandB mismatch
If:
- `COMPUTED METRICS` shows correct low value (e.g., 0.078)
- `LOGGING TO WANDB` shows correct low value (e.g., 0.078)
- But WandB dashboard shows 1.0

Then the problem is:
- WandB API issue
- Dashboard caching
- Looking at wrong metric/step

### Scenario 5: Scalar mean issue
The extracted value SHOULD be a scalar mean. If you see:
```
>>> EXTRACTED episode_success: [1. 1. 1. ... 1. 1.]  # Array of per-env values
```
Instead of:
```
>>> EXTRACTED episode_success: 0.078125  # Scalar mean
```

Then `command_manager.reset()` is not computing the mean correctly.

## How to Run

```bash
python src/mjlab/continual_distill/continual_distill.py --config your_config.yaml
```

The debug output will be VERY verbose. Redirect to a file:

```bash
python src/mjlab/continual_distill/continual_distill.py --config your_config.yaml 2>&1 | tee debug_output.log
```

Then search the log for:
- `DEBUG EVAL START: Task 1 (lift_cube)` - Find lift-cube evaluations
- `initial_goal_error` - Check if cube spawns at goal
- `COMPUTED METRICS FOR TASK 1` - See final values
- `LOGGING TO WANDB` - See what's actually logged

## Expected vs Problematic Output

### GOOD (Working Correctly):
```
After env.reset():
  lift_object.episode_success: [0. 0. 0. 0. 0.] (first 5)  # ✓ Reset to 0
  lift_object.initial_goal_error[0]: 0.3234m  # ✓ Cube far from goal

Step 249:
    lift_object.episode_success[:5]: [0. 0. 0. 1. 1.]  # ✓ Only some succeeded

COMPUTED METRICS:
  student_success: 0.0781  # ✓ Low success rate

LOGGING TO WANDB:
  Accuracy/task_1_lift_cube: 0.0781  # ✓ Matches computed
```

### BAD (Bug Present):
```
After env.reset():
  lift_object.episode_success: [1. 1. 0. 1. 1.] (first 5)  # ✗ NOT reset!
  OR
  lift_object.initial_goal_error[0]: 0.0023m  # ✗ Cube at goal!

COMPUTED METRICS:
  student_success: 1.0000  # ✗ False 100%
```

## Next Steps After Getting Debug Output

1. **Find the lift-cube evaluation** during open-drawer training
2. **Check the "initial_goal_error"** - is it suspiciously small?
3. **Check "episode_success BEFORE extraction"** - are values already 1?
4. **Compare "COMPUTED METRICS" with "LOGGING TO WANDB"** - do they match?
5. **Share the relevant debug section** with me for analysis

The debug output will definitively show WHERE the bug occurs in the pipeline!
