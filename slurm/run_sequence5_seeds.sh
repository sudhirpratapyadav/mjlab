#!/bin/bash
# Launch sequence5 with multiple seeds in parallel, one GPU each, inside the
# existing holder allocation on dgx2.
#
# Usage: bash slurm/run_sequence5_seeds.sh [HOLDER_JOBID] [SEED...]
#        (default seeds: 1 2 3)

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2 -o %A | head -1)}
shift || true
SEEDS=("${@:-}")
[[ -n "${SEEDS[0]:-}" ]] || SEEDS=(1 2 3)
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }
echo "Using holder job: $HOLDER, seeds: ${SEEDS[*]}"

REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
mkdir -p "$REPO/logs"
TS=$(date +%s)

# srun --overlap --gres=gpu:1 hands every step GPU 0, so pin each run to its
# own device explicitly (seed i -> GPU i) via CUDA_VISIBLE_DEVICES.
GPU_ID=1
for SEED in "${SEEDS[@]}"; do
    RUN_NAME="sequence5_seed${SEED}_${TS}"
    nohup srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
        --export=ALL,CUDA_VISIBLE_DEVICES=$GPU_ID \
        bash "$REPO/slurm/_cl_run.sh" \
        "$RUN_NAME" tasks.yaml online "$SEED" \
        PushCuboid PushButton OpenDoor OpenDrawer \
        > "$REPO/logs/srun_${RUN_NAME}.log" 2>&1 &
    echo "Launched $RUN_NAME (pid $!) on GPU $GPU_ID"
    GPU_ID=$((GPU_ID+1))
done

echo "Follow with: tail -f $REPO/logs/cl_sequence5_seed*_${TS}.log"
