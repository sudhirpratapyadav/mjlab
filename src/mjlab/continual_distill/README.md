# Continual Distillation for mjlab

This module provides tools for extracting data from mjlab teacher policies for continual learning.

## Quick Test

Test that JAX teacher network can load PyTorch checkpoint and run mjlab environments:

```bash
# From mjlab root directory
cd /media/cvlab/EXTDRIVE/sudhir/continual_learning/mjlab

# Run test with a checkpoint (text output)
python -m mjlab.continual_distill.test_jax_teacher \
    --checkpoint logs/rsl_rl/yam_lift_cube/2025-12-02_21-19-35/model_1100.pt \
    --task franka-lift-cube-v0 \
    --num-envs 4 \
    --num-episodes 3 \
    --max-steps 100

# Run with viser viewer (visual)
python -m mjlab.continual_distill.test_jax_teacher \
    --checkpoint logs/rsl_rl/franka_open_door/2025-12-29_15-42-50/model_100.pt \
    --task Mjlab-Open-Door-Franka \
    --num-envs 1 \
    --viewer
```

## What the test does:

1. **Loads PyTorch checkpoint** - Extracts actor network weights, std, and observation normalizer (no critic)
2. **Converts to JAX** - Transposes weights and creates JAX/Flax parameters
3. **Creates JAX teacher network** - MLP with ELU activation matching mjlab structure
4. **Runs mjlab environment** - Tests that everything works together
5. **Prints statistics** - Episode rewards, lengths, action ranges, etc.

## Extract Teacher Dataset

Extract training data from teacher policies:

```bash
python -m mjlab.continual_distill.extract_teacher_dataset \
    --checkpoint logs/rsl_rl/franka_open_door/2025-12-29_15-42-50/model_100.pt \
    --task Mjlab-Open-Door-Franka \
    --num-samples 150000 \
    --num-envs 512 \
    --episode-length 100
    # Output saved to: continual_distill/teacher_datasets/ (default)
```

**What it does:**
- Loads PyTorch checkpoint → converts to JAX
- Runs vectorized environments (fast!)
- Collects observations + action distributions (mean + logstd)
- Saves as pickle file with metadata

**Output format:**
```python
{
    'observations': array (N, obs_dim),      # State observations
    'action_targets': array (N, action_dim*2),  # [mean, logstd] concatenated
    'metadata': dict                          # Task info, checkpoint, etc.
}
```

## Files:

- `test_jax_teacher.py` - Test script with viewer support (for verification)
- `utils.py` - JAX models and PyTorch→JAX conversion utilities
- `extract_teacher_dataset.py` - Fast data extraction (no viewer overhead)
