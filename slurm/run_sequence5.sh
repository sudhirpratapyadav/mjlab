#!/bin/bash
# Full continual-distill run: sequence5 = [PushCuboid, PushButton, OpenDoor, OpenDrawer]
# Student (4096, 2048, 1024), 500 epochs/task, wandb project continual_rl_mjlab.
# Runs inside the existing holder allocation on dgx2, detached via nohup so it
# survives ssh disconnect.
#
# Usage: bash slurm/run_sequence5.sh [HOLDER_JOBID]

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2 -o %A | head -1)}
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }
echo "Using holder job: $HOLDER"

RUN_NAME="sequence5_$(date +%s)"
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
mkdir -p "$REPO/logs"

nohup srun --jobid="$HOLDER" --overlap --gres=gpu:1 --cpus-per-task=8 \
    bash "$REPO/slurm/_cl_run.sh" \
    "$RUN_NAME" tasks.yaml online 0 \
    PushCuboid PushButton OpenDoor OpenDrawer \
    > "$REPO/logs/srun_${RUN_NAME}.log" 2>&1 &

echo "Launched $RUN_NAME (pid $!)"
echo "Follow with: tail -f $REPO/logs/cl_${RUN_NAME}.log"
