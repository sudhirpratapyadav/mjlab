# Continual Distillation for mjlab

Continual learning with policy distillation for mjlab tasks.

## Usage

### 1. Extract Teacher Dataset

```bash
python -m mjlab.continual_distill.extract_teacher_dataset \
    --checkpoint logs/rsl_rl/franka_open_door/model_100.pt \
    --task Mjlab-Open-Door-Franka \
    --num-samples 150000 \
    --num-envs 512
```

Output: `teacher_datasets/{task}_{checkpoint}_{timestamp}/` with `data.pkl` and `teacher.pkl`

### 2. Train Student

```bash
python -m mjlab.continual_distill.continual_distill \
    --config config/tasks.yaml \
    --learning-rate 1e-4 \
    --batch-size 512 \
    --si-coeff 1.0 \
    --track
```

**YAML Config:**
```yaml
task_info:
  - task_name: "OpenDoor"
    env_id: "Mjlab-Open-Door-Franka"
    dataset_folder: "teacher_datasets/Mjlab_Open_Door_Franka_model_100_20260108_123456"
    obs_dim: 60
    action_dim: 8
    ep_len: 100
    num_epochs: 500
```

## Files

- `extract_teacher_dataset.py` - Extract data from teacher policies
- `continual_distill.py` - Train multi-task student with SI
- `utils.py` - JAX models and PyTorch→JAX conversion
