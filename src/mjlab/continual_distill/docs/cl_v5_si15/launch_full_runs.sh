#!/usr/bin/env bash
# CL-V5 SI-15 full launches: 2 orderings x 3 seeds, SI-only, width 4096, lr 3e-5.
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl15-si-20260918
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15"
LOGDIR="$DOCS/run_logs"
mkdir -p "$LOGDIR" "$DOCS/results"

FRAGILE_FIRST="DragPull ToppleBlock ThrowToBin PushCuboid RotateValve OpenDrawer FlipSwitch AxialExtract OpenDoor PushButton OpenLid SlideWindow TurnLever PushFlap ReachTarget"
RANDOM_ORDER="ToppleBlock RotateValve OpenDrawer FlipSwitch ThrowToBin ReachTarget SlideWindow PushButton PushFlap PushCuboid AxialExtract TurnLever OpenDoor OpenLid DragPull"

# run_name  ordering_var  seed  gpu_uuid
JOBS=(
  "ff-s0 FRAGILE_FIRST 0 GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"
  "ff-s1 FRAGILE_FIRST 1 GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"
  "ff-s2 FRAGILE_FIRST 2 GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"
  "rnd-s0 RANDOM_ORDER 0 GPU-0b3441b1-580c-9c76-f27d-cbec321e1937"
  "rnd-s1 RANDOM_ORDER 1 GPU-7d5c3740-990d-15b7-5c54-6dc6270633fa"
  "rnd-s2 RANDOM_ORDER 2 GPU-46a4f4dc-5916-8d5a-a71b-6193e01b5723"
)

run_one() {
  local name="$1" order_var="$2" seed="$3" gpu_uuid="$4"
  local -n seq_ref="$order_var"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name seed=$seed gpu=$gpu_uuid seq=[$seq_ref]" | tee "$log"
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
    $WORKTREE/../mjlab/.venv/bin/python src/mjlab/continual_distill/continual_distill.py \
      --tasks-config src/mjlab/continual_distill/config/tasks_cl_v5_si15.yaml \
      --task-sequence $seq_ref \
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
  read -r name order_var seed gpu_uuid <<< "$entry"
  run_one "$name" "$order_var" "$seed" "$gpu_uuid" &
  pids+=($!)
done
wait "${pids[@]}"
echo "ALL 6 FULL RUNS COMPLETE"
