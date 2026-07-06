#!/bin/bash
# Seed-instability diagnostics: parallel sequence5 runs on separate GPUs.
# Seeds 1-3 collapsed (student KL diverged under SI) while seed 0 succeeded.
# These runs test (a) whether the failure reproduces deterministically and
# (b) whether the SI coefficient is the destabilizer.
#
# Usage: bash slurm/run_si_diagnostics.sh [HOLDER_JOBID]

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2 -o %A | head -1)}
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }

REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
mkdir -p "$REPO/logs"
TS=$(date +%s)

launch() {  # launch <gpu> <name-suffix> <seed> <extra-args...>
    local GPU=$1 SUFFIX=$2 SEED=$3; shift 3
    local RUN_NAME="sequence5_diag_${SUFFIX}_${TS}"
    RUN_GPU=$GPU EXTRA_ARGS="$*" nohup srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
        --export=ALL,RUN_GPU,EXTRA_ARGS \
        bash "$REPO/slurm/_cl_run.sh" \
        "$RUN_NAME" tasks.yaml online "$SEED" \
        PushCuboid PushButton OpenDoor OpenDrawer \
        > "$REPO/logs/srun_${RUN_NAME}.log" 2>&1 &
    echo "GPU $GPU: $RUN_NAME (seed=$SEED extra='$*')"
}

launch 2 seed1_repro      1
launch 3 seed1_si0.1      1 --si-coeff 0.1
launch 4 seed1_si0.01     1 --si-coeff 0.01
launch 5 seed2_si0.1      2 --si-coeff 0.1

echo "Follow with: tail -f $REPO/logs/cl_sequence5_diag_*_${TS}.log"
