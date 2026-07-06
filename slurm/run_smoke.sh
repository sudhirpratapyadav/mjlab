#!/bin/bash
# Smoke test: 4-task sequence, 2 epochs per task, wandb offline.
# Runs inside the existing holder allocation (all 8 GPUs held on dgx2).
#
# Usage: bash slurm/run_smoke.sh [HOLDER_JOBID]

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2 -o %A | head -1)}
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }
echo "Using holder job: $HOLDER"

srun --jobid="$HOLDER" --overlap --gres=gpu:1 --cpus-per-task=8 \
    bash /ihub/homedirs/svs_ald/sudhir/mjlab/slurm/_cl_run.sh \
    smoke_cluster tasks_smoke.yaml offline \
    PushCuboid PushButton OpenDoor OpenDrawer
