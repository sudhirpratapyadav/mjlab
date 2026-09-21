#!/usr/bin/env bash
# Apply a PRIOR VALIDATED fix (GRASP_FIX_EXPERIMENTS.md) to the ORIGINAL
# per-task-head architecture (CL-V5, not CL-V6's shared_resnet): delta_action
# distillation weighting up-weights success-critical (high action-change)
# steps, addressing plain-KL's blindness to which actions actually decide
# success. Prior work found +0.13 to +0.18 avg improvement when a fragile
# task is present, safe/neutral otherwise. si-coeff 3.0, floor 0.05, clip 20,
# weight-tasks all -- the doc's own recommended "safe default".
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl-arch-20260920
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v6_arch"
LOGDIR="$DOCS/run_logs"
mkdir -p "$LOGDIR" "$DOCS/results_deltaaction"

TASKS_CFG="$WORKTREE/src/mjlab/continual_distill/config/tasks_cl_v5_si15.yaml"
RANDOM_ORDER="ToppleBlock RotateValve OpenDrawer FlipSwitch ThrowToBin ReachTarget SlideWindow PushButton PushFlap PushCuboid AxialExtract TurnLever OpenDoor OpenLid DragPull"

JOBS=(
  "da-s0 0 GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"
  "da-s1 1 GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"
  "da-s2 2 GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"
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
      --regularizer si --si-coeff 3.0 \
      --architecture heads --student-hidden-dims 4096 2048 1024 \
      --distill-weight-mode delta_action --distill-weight-floor 0.05 --distill-weight-clip 20 --distill-weight-tasks all \
      --learning-rate 3e-5 \
      --checkpoint-dir '$DOCS/results_deltaaction' \
      --seed $seed \
      --track --wandb-project mjlab-cl15-si-20260918 --wandb-entity sudhirpratapyadav-indian-institute-of-technology-jodhpur \
      --run-name cl-deltaaction-$name \
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
echo "DELTA-ACTION RUNS COMPLETE"
