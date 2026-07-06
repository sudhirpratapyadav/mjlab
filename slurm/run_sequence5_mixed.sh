#!/bin/bash
# Mixed-teacher sequence5: RL teachers for PushCuboid/OpenDoor/OpenDrawer,
# classical (hand-coded) teacher for PushButton (tasks_mixed.yaml).
# Pinned to a free GPU inside the holder allocation.
#
# Usage: bash slurm/run_sequence5_mixed.sh [HOLDER_JOBID] [GPU_ID] [SEED]

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2 -o %A | head -1)}
GPU_ID=${2:-1}
SEED=${3:-0}
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }

RUN_NAME="sequence5_mixed_seed${SEED}_$(date +%s)"
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
mkdir -p "$REPO/logs"

nohup srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
    --export=ALL,CUDA_VISIBLE_DEVICES=$GPU_ID \
    bash "$REPO/slurm/_cl_run.sh" \
    "$RUN_NAME" tasks_mixed.yaml online "$SEED" \
    PushCuboid PushButton OpenDoor OpenDrawer \
    > "$REPO/logs/srun_${RUN_NAME}.log" 2>&1 &

echo "Launched $RUN_NAME (pid $!) on GPU $GPU_ID"
echo "Follow with: tail -f $REPO/logs/cl_${RUN_NAME}.log"
