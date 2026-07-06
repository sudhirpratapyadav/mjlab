#!/bin/bash
# 5-task LiftCube-position sweep: the 4 base tasks keep their standard order
# (PushCuboid, PushButton, OpenDoor, OpenDrawer); LiftCube is inserted at each
# of the 5 positions -> 5 experiments, one per GPU. All RL teachers, lr 3e-5.
#
# Usage: bash slurm/run_5task_liftpos.sh [HOLDER_JOBID]

set -euo pipefail
HOLDER=${1:-$(squeue -u svs_ald -h -n hold_dgx2_sudhir_7 -o %A | head -1)}
[[ -n "$HOLDER" ]] || { echo "no holder job found"; exit 1; }

REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
mkdir -p "$REPO/logs"
TS=$(date +%s)

launch() {  # launch <gpu> <pos_label> <task1> <task2> <task3> <task4> <task5>
    local GPU=$1 LABEL=$2; shift 2
    local RUN_NAME="seq5task_liftpos${LABEL}_${TS}"
    RUN_GPU=$GPU EXTRA_ARGS="--learning-rate 3e-5" nohup srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
        --export=ALL,RUN_GPU,EXTRA_ARGS \
        bash "$REPO/slurm/_cl_run.sh" \
        "$RUN_NAME" tasks_5task.yaml online 0 \
        "$@" \
        > "$REPO/logs/srun_${RUN_NAME}.log" 2>&1 &
    echo "GPU $GPU: $RUN_NAME  seq=[$*]"
}

# LiftCube at positions 1..5; base order PushCuboid PushButton OpenDoor OpenDrawer
launch 0 1 LiftCube  PushCuboid PushButton OpenDoor  OpenDrawer
launch 1 2 PushCuboid LiftCube  PushButton OpenDoor  OpenDrawer
launch 2 3 PushCuboid PushButton LiftCube  OpenDoor  OpenDrawer
launch 3 4 PushCuboid PushButton OpenDoor  LiftCube  OpenDrawer
launch 4 5 PushCuboid PushButton OpenDoor  OpenDrawer LiftCube

echo "TS=$TS"
echo "Follow: tail -f $REPO/logs/cl_seq5task_liftpos*_${TS}.log"
