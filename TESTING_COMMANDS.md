# Testing Commands for Franka Pick Task

## What We Built

Complete implementation of Franka pick task in mjlab format:

### Files Created

**Robot Assets:**
- `src/mjlab/asset_zoo/robots/franka_emika_panda/franka_constants.py`
- `src/mjlab/asset_zoo/robots/franka_emika_panda/xmls/panda.xml` (copied + adapted)
- `src/mjlab/asset_zoo/robots/franka_emika_panda/xmls/hand.xml` (copied)
- `src/mjlab/asset_zoo/robots/franka_emika_panda/xmls/assets/*` (meshes)

**Task Implementation:**
- `src/mjlab/tasks/pick/pick_env_cfg.py` - Base task config
- `src/mjlab/tasks/pick/mdp/observations.py` - 11 observation functions
- `src/mjlab/tasks/pick/mdp/rewards.py` - 4 reward functions
- `src/mjlab/tasks/pick/mdp/events.py` - 3 event functions
- `src/mjlab/tasks/pick/config/franka/env_cfgs.py` - Franka-specific config
- `src/mjlab/tasks/pick/config/franka/rl_cfg.py` - PPO config
- `src/mjlab/tasks/pick/config/franka/__init__.py` - Task registration

**Registered Task ID:** `Mjlab-Pick-Cube-Franka`

---

## Testing Steps

### 1. Verify Robot Loads

Test that Franka robot can be loaded in MuJoCo viewer:

```bash
python src/mjlab/asset_zoo/robots/franka_emika_panda/franka_constants.py
```

**Expected:** MuJoCo viewer opens with Franka arm in home pose

---

### 2. List Available Tasks

Check that the pick task is registered:

```bash
uv run python -c "from mjlab.tasks.registry import list_tasks; print(list_tasks())"
```

**Expected:** Should include `'Mjlab-Pick-Cube-Franka'` in the list

---

### 3. Test with Zero Agent

Run environment with zero actions (no learning):

```bash
uv run play Mjlab-Pick-Cube-Franka --agent zero --env.scene.num-envs 4
```

**Expected:**
- Environment launches without errors
- Robot stays mostly still (zero actions)
- Can observe cube and target marker
- Episode resets after 150 steps (~3 seconds)

---

### 4. Test with Random Agent

Run environment with random actions:

```bash
uv run play Mjlab-Pick-Cube-Franka --agent random --env.scene.num-envs 4
```

**Expected:**
- Robot moves randomly
- Rewards are computed and displayed
- No crashes or NaN values
- Terminations trigger correctly (out of bounds, timeout)

---

### 5. Short Training Test

Run a very short training to verify the training loop works:

```bash
MUJOCO_GL=egl uv run train Mjlab-Pick-Cube-Franka \
  --env.scene.num-envs 64 \
  --rl.max_iterations 10
```

**Expected:**
- Training starts without errors
- Losses and rewards are logged
- No NaN or inf values
- Checkpoints are saved

---

### 6. Full Training (if short test passes)

Start full training run:

```bash
MUJOCO_GL=egl uv run train Mjlab-Pick-Cube-Franka \
  --env.scene.num-envs 4096 \
  --rl.max_iterations 10000
```

**Training params:**
- Episode length: 150 steps (3 seconds)
- Num envs: 4096 (adjust based on GPU memory)
- Max iterations: 10,000
- Save interval: Every 50 iterations

---

### 7. Evaluate Trained Policy

After training, evaluate with a checkpoint:

```bash
uv run play Mjlab-Pick-Cube-Franka \
  --wandb-run-path your-org/mjlab/run-id
```

Or with local checkpoint:

```bash
uv run play Mjlab-Pick-Cube-Franka \
  --checkpoint_file /path/to/checkpoint.pt
```

---

## Expected Issues & Fixes

### Issue 1: Missing Site/Body Names

**Error:** `KeyError: 'gripper'` or similar

**Fix:** Check panda.xml for actual site/body names and update in `env_cfgs.py`:
- Gripper site name
- Cube body name
- Hand/finger geom names

### Issue 2: Mocap Bodies Not Working

**Error:** Mocap position not updating

**Fix:** May need to define mocap body in scene or use different approach for target visualization. Check how velocity task handles visual markers.

### Issue 3: Multi-Entity Scene

**Error:** Cube entity not loading

**Fix:** mjlab may need specific configuration for multiple entities. May need to include cube in robot's MJCF instead of separate entity.

### Issue 4: Delta Action Mode

**Error:** Actions not working as expected

**Fix:** Verify `JointPositionActionCfg` with `use_default_offset=False` works as delta control. May need to check action implementation.

### Issue 5: State Persistence

**Error:** `reached_box` flag not working

**Fix:** Verify `env.extras` dict persists across steps. May need different mechanism for state tracking.

---

## Debugging Tips

### Print Observation Shape

```bash
uv run python -c "
from mjlab.tasks.pick.config.franka.env_cfgs import franka_pick_cube_env_cfg
cfg = franka_pick_cube_env_cfg()
print('Num observation terms:', len(cfg.observations['policy'].terms))
# Should be 11 terms, ~60D total
"
```

### Check Reward Weights

```bash
uv run python -c "
from mjlab.tasks.pick.config.franka.env_cfgs import franka_pick_cube_env_cfg
cfg = franka_pick_cube_env_cfg()
for name, reward_cfg in cfg.rewards.items():
    print(f'{name}: {reward_cfg.weight}')
"
```

### Inspect Scene Config

```bash
uv run python -c "
from mjlab.tasks.pick.config.franka.env_cfgs import franka_pick_cube_env_cfg
cfg = franka_pick_cube_env_cfg()
print('Entities:', list(cfg.scene.entities.keys()))
print('Sensors:', [s.name for s in cfg.scene.sensors])
"
```

---

## Success Metrics

Task is working correctly if:

1. ✅ Environment loads without errors
2. ✅ Robot moves in response to actions
3. ✅ Rewards are computed correctly:
   - `gripper_box` increases as gripper approaches cube
   - `box_target` activates after reaching cube
   - `no_floor_collision` = 1 when no contact
   - `robot_target_qpos` highest when near home pose
4. ✅ Observations are sensible (no NaN, reasonable ranges)
5. ✅ Terminations work (timeout, out of bounds)
6. ✅ Training converges (rewards increase over iterations)
7. ✅ Success rate > 0 after sufficient training

---

## Next Steps After Testing

If basic functionality works:

1. **Tune rewards**: Adjust weights for better learning
2. **Add curriculum**: Gradually increase difficulty
3. **Domain randomization**: Vary object mass, friction, etc.
4. **Multiple objects**: Pick different shapes/colors
5. **Orientation constraints**: Require upright placement
6. **Vision**: Add camera observations for visual policies
