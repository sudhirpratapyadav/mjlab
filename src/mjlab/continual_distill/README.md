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

## Next Steps:

Once the test works:
- [ ] Separate into modules (models.py, checkpoint_loader.py, data_extractor.py)
- [ ] Add vectorized data collection
- [ ] Add data saving/loading
- [ ] Support all manipulation tasks
