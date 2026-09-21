#!/usr/bin/env bash
# CL-V6 Wave 2: full 15-task shared_resnet runs with the FIXED evaluation
# code (episode-length bug, EXPERIMENTS.md Block 7) and all three FiLM
# conditioning layers (block-level, out_head, out_logit; Blocks 2-5).
# width 2048 / 6 blocks (Wave 1's primary config), si-coeff 5.0, random
# ordering (established best), 3 seeds.
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl-arch-20260920
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v6_arch"
LOGDIR="$DOCS/run_logs"
mkdir -p "$LOGDIR" "$DOCS/results_v2"

TASKS_CFG="$WORKTREE/src/mjlab/continual_distill/config/tasks_cl_v5_si15.yaml"
RANDOM_ORDER="ToppleBlock RotateValve OpenDrawer FlipSwitch ThrowToBin ReachTarget SlideWindow PushButton PushFlap PushCuboid AxialExtract TurnLever OpenDoor OpenLid DragPull"

JOBS=(
  "v2-s0 0 GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"
  "v2-s1 1 GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"
  "v2-s2 2 GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"
)

run_one() {
  local name="$1" seed="$2" gpu_uuid="$3"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name seed=$seed gpu=$gpu_uuid" | tee "$log"
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
      --regularizer si --si-coeff 5.0 \
      --architecture shared_resnet --residual-width 2048 --num-residual-blocks 6 --task-embed-dim 32 \
      --learning-rate 3e-5 \
      --checkpoint-dir '$DOCS/results_v2' \
      --seed $seed \
      --track --wandb-project mjlab-cl15-si-20260918 --wandb-entity sudhirpratapyadav-indian-institute-of-technology-jodhpur \
      --run-name cl-v6-$name \
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
echo "CL-V6 WAVE 2 COMPLETE"
