#!/usr/bin/env bash
# CL-V6 first full-scale wave: shared_resnet architecture, random ordering
# (established best from CL-V5), validated P1 training settings.
# Primary: width 2048 / 6 blocks, 3 seeds (GPUs 1-3).
# Parallel capacity/depth probes at full 15-task scale (GPUs 4-5).
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl-arch-20260920
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v6_arch"
LOGDIR="$DOCS/run_logs"
mkdir -p "$LOGDIR" "$DOCS/results"

TASKS_CFG="$WORKTREE/src/mjlab/continual_distill/config/tasks_cl_v5_si15.yaml"
RANDOM_ORDER="ToppleBlock RotateValve OpenDrawer FlipSwitch ThrowToBin ReachTarget SlideWindow PushButton PushFlap PushCuboid AxialExtract TurnLever OpenDoor OpenLid DragPull"

# name  seed  width  blocks  gpu_uuid
JOBS=(
  "w2048b6-s0 0 2048 6 GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"
  "w2048b6-s1 1 2048 6 GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"
  "w2048b6-s2 2 2048 6 GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"
  "w4096b6-s0 0 4096 6 GPU-0b3441b1-580c-9c76-f27d-cbec321e1937"
  "w2048b10-s0 0 2048 10 GPU-7d5c3740-990d-15b7-5c54-6dc6270633fa"
)

run_one() {
  local name="$1" seed="$2" width="$3" blocks="$4" gpu_uuid="$5"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name seed=$seed width=$width blocks=$blocks gpu=$gpu_uuid" | tee "$log"
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
      --tasks-config '$TASKS_CFG' \
      --task-sequence $RANDOM_ORDER \
      --regularizer si --si-coeff 1.0 \
      --architecture shared_resnet --residual-width $width --num-residual-blocks $blocks --task-embed-dim 32 \
      --learning-rate 3e-5 \
      --checkpoint-dir '$DOCS/results' \
      --seed $seed \
      --track --wandb-project mjlab-cl15-si-20260918 --wandb-entity sudhirpratapyadav-indian-institute-of-technology-jodhpur \
      --run-name cl-v6-$name \
      --no-tqdm
  " >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE $name exit=$?" | tee -a "$log"
}

pids=()
for entry in "${JOBS[@]}"; do
  read -r name seed width blocks gpu_uuid <<< "$entry"
  run_one "$name" "$seed" "$width" "$blocks" "$gpu_uuid" &
  pids+=($!)
done
wait "${pids[@]}"
echo "CL-V6 WAVE 1 COMPLETE"
