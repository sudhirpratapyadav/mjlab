#!/usr/bin/env bash
# Quick LR / SI-coeff / batch-size / width bracket on GPU7 (idle), 2 tasks x
# 100 epochs each, sequential, while the 6 main full runs use GPUs 1-6.
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl15-si-20260918
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15"
LOGDIR="$DOCS/bracket_logs"
mkdir -p "$LOGDIR" "$DOCS/bracket_results"
GPU_UUID="GPU-811bf62a-ec3a-649f-1167-bfff22107497"  # 7

# name  extra_args
JOBS=(
  "width8192 --student-hidden-dims 8192 4096 2048"
  "width16384 --student-hidden-dims 16384 8192 4096"
  "lr1e-4 --learning-rate 1e-4"
  "lr1e-5 --learning-rate 1e-5"
  "si0.3 --si-coeff 0.3"
  "si3.0 --si-coeff 3.0"
  "batch256 --batch-size 256"
  "batch1024 --batch-size 1024"
)

for entry in "${JOBS[@]}"; do
  name="${entry%% *}"
  extra="${entry#* }"
  log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START bracket-$name ($extra)" | tee "$log"
  srun --jobid=20277 --overlap -n1 --cpus-per-task=8 bash -c "
    export FORCE_CPU=0
    export CUDA_VISIBLE_DEVICES=$GPU_UUID
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
      --tasks-config src/mjlab/continual_distill/config/tasks_bracket2.yaml \
      --task-sequence DragPull ToppleBlock \
      --regularizer si --si-coeff 1.0 \
      --student-hidden-dims 4096 2048 1024 \
      --learning-rate 3e-5 \
      --checkpoint-dir '$DOCS/bracket_results' \
      --seed 0 \
      --track --wandb-project mjlab-cl15-si-20260918 --wandb-entity sudhirpratapyadav-indian-institute-of-technology-jodhpur \
      --run-name bracket-$name \
      --no-tqdm \
      $extra
  " >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE bracket-$name exit=$?" | tee -a "$log"
done

echo "BRACKET COMPLETE"
