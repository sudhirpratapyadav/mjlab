#!/bin/bash
# Common continual-distill runner for the iHub-Drishti cluster.
#
# Usage:
#   bash slurm/_cl_run.sh <RUN_NAME> <TASKS_CONFIG> <WANDB_MODE> <TASK...>
#
# Designed to be invoked inside the existing holder allocation:
#   srun --jobid=$HOLDER_JOB --overlap --gres=gpu:1 --cpus-per-task=8 \
#        bash slurm/_cl_run.sh seq5 config/tasks.yaml online PushCuboid PushButton OpenDoor OpenDrawer
#
# Notes:
#   * uv sync is broken on this cluster (CUDA 12.4 driver needs torch<2.7,
#     repo pins torch>=2.7).  The .venv here was built manually with
#     torch 2.6.0+cu124 and jax[cuda12]==0.7.2; we use PYTHONPATH=src.
#   * wandb credentials are loaded from src/mjlab/continual_distill/.env
#     by continual_distill.py itself.

set -euo pipefail

RUN_NAME=${1:?run name required}
TASKS_CONFIG=${2:?tasks config filename required (relative to continual_distill/config)}
WANDB_MODE=${3:?wandb mode required (online/offline/disabled)}
SEED=${4:?seed required}
shift 4
TASK_SEQUENCE=("$@")
[[ ${#TASK_SEQUENCE[@]} -ge 1 ]] || { echo "task sequence required"; exit 1; }

REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
CD_DIR=$REPO/src/mjlab/continual_distill
LOG=$REPO/logs/cl_${RUN_NAME}.log
mkdir -p "$REPO/logs"

# Pin to a specific GPU: srun --overlap steps inherit the full allocation and
# slurmstepd OVERWRITES any CUDA_VISIBLE_DEVICES passed via --export, so the
# pin must happen here, inside the step.
if [[ -n "${RUN_GPU:-}" ]]; then
    export CUDA_VISIBLE_DEVICES=$RUN_GPU
fi

cd "$REPO"

echo "============================================================"
echo "Continual distill run: $RUN_NAME"
echo "  tasks config:   $TASKS_CONFIG"
echo "  sequence:       ${TASK_SEQUENCE[*]}"
echo "  seed:           $SEED"
echo "  wandb mode:     $WANDB_MODE"
echo "  log file:       $LOG"
echo "  host:           $(hostname)"
echo "  date:           $(date)"
echo "  CUDA_VISIBLE:   ${CUDA_VISIBLE_DEVICES:-unset}"
echo "  SLURM_JOB:      ${SLURM_JOB_ID:-unset}"
echo "============================================================"

# EXTRA_ARGS: optional additional CLI flags (e.g. "--si-coeff 0.1").
MUJOCO_GL=egl \
MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES:-0} \
PYTHONPATH=$REPO/src \
.venv/bin/python -m mjlab.continual_distill.continual_distill \
    --tasks-config "$CD_DIR/config/$TASKS_CONFIG" \
    --task-sequence "${TASK_SEQUENCE[@]}" \
    --run-name "$RUN_NAME" \
    --seed "$SEED" \
    --wandb-mode "$WANDB_MODE" \
    --no-tqdm \
    ${EXTRA_ARGS:-} \
2>&1 | tee "$LOG"
