#!/usr/bin/env bash
# CL-V5 SI-15 Block 4: fragile-last ordering (reverse of fragile-first),
# motivated by Block 3's finding that mid-sequence placement may be worse
# than either extreme at N=15. 3 seeds, GPUs 1-3.
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl15-si-20260918
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15"
LOGDIR="$DOCS/run_logs"
mkdir -p "$LOGDIR" "$DOCS/results"

FRAGILE_LAST="ReachTarget PushFlap TurnLever SlideWindow OpenLid PushButton OpenDoor AxialExtract FlipSwitch OpenDrawer RotateValve PushCuboid ThrowToBin ToppleBlock DragPull"

JOBS=(
  "fl-s0 0 GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"
  "fl-s1 1 GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"
  "fl-s2 2 GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"
)

run_one() {
  local name="$1" seed="$2" gpu_uuid="$3"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name seed=$seed gpu=$gpu_uuid seq=[$FRAGILE_LAST]" | tee "$log"
  srun --jobid=20277 --overlap -n1 --cpus-per-task=8 bash -c "
    export FORCE_CPU=0
    export CUDA_VISIBLE_DEVICES=$gpu_uuid
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH='$WORKTREE/src'
    export WANDB_API_KEY=\$(cat ~/.config/mjlab-cl24/wandb_api_key)
    export WANDB_ENTITY=sudhirpratapyadav-indian-institute-of-technology-jodhpur
    export WANDB_PROJECT=mjlab-cl15-si-20260918
    export WANDB_MODE=online
    export WANDB_CONFIG_DIR=~/.config/mjlab-cl24/sdk
    export NETRC=~/.config/mjlab-cl24/netrc
    unset WANDB_USERNAME
    cd '$WORKTREE'
    $MAIN/.venv/bin/python src/mjlab/continual_distill/continual_distill.py \
      --tasks-config src/mjlab/continual_distill/config/tasks_cl_v5_si15.yaml \
      --task-sequence $FRAGILE_LAST \
      --regularizer si --si-coeff 1.0 \
      --student-hidden-dims 4096 2048 1024 \
      --learning-rate 3e-5 \
      --checkpoint-dir '$DOCS/results' \
      --seed $seed \
      --track --wandb-project mjlab-cl15-si-20260918 --wandb-entity sudhirpratapyadav-indian-institute-of-technology-jodhpur \
      --run-name cl15-si-$name \
      --no-tqdm
  " >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE $name exit=$?" | tee -a "$log"
}

pids=()
for entry in "${JOBS[@]}"; do
  read -r name seed gpu_uuid <<< "$entry"
  run_one "$name" "$seed" "$gpu_uuid" &
  pids+=($!)
done
wait "${pids[@]}"
echo "FRAGILE-LAST RUNS COMPLETE"
