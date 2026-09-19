#!/usr/bin/env bash
# Render all 15 tasks from the final rnd-s2 checkpoint (current best run,
# avg 0.837), spread across GPUs 4-7 (idle: fragile-last is using 1-3).
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl15-si-20260918
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
DOCS="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15"
CKPT="$DOCS/results/cl15-si-rnd-s2/checkpoint_task14_epoch500_step2352000.pkl"
OUT="$DOCS/videos/rnd-s2"
LOGDIR="$DOCS/render_logs"
mkdir -p "$OUT" "$LOGDIR"

GPU_UUIDS=(
  "GPU-0b3441b1-580c-9c76-f27d-cbec321e1937"  # 4
  "GPU-7d5c3740-990d-15b7-5c54-6dc6270633fa"  # 5
  "GPU-46a4f4dc-5916-8d5a-a71b-6193e01b5723"  # 6
  "GPU-811bf62a-ec3a-649f-1167-bfff22107497"  # 7
)

TASKS=(AxialExtract FlipSwitch OpenDoor OpenDrawer OpenLid PushButton PushFlap ReachTarget SlideWindow TurnLever RotateValve PushCuboid ThrowToBin ToppleBlock DragPull)

run_one() {
  local task="$1" gpu_uuid="$2"
  local log="$LOGDIR/${task}.log"
  echo "[$(date -u +%FT%TZ)] START render $task on $gpu_uuid" | tee "$log"
  srun --jobid=20277 --overlap -n1 --cpus-per-task=4 bash -c "
    export FORCE_CPU=0
    export CUDA_VISIBLE_DEVICES=$gpu_uuid
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH='$WORKTREE/src'
    cd '$WORKTREE'
    $MAIN/.venv/bin/python src/mjlab/continual_distill/docs/cl_v5_si15/render_student.py \
      --checkpoint '$CKPT' \
      --task-name $task \
      --output-dir '$OUT' \
      --num-envs 8
  " >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE render $task exit=$?" | tee -a "$log"
}

i=0
pids=()
for task in "${TASKS[@]}"; do
  gpu_idx=$(( i % 4 ))
  run_one "$task" "${GPU_UUIDS[$gpu_idx]}" &
  pids+=($!)
  i=$((i+1))
  if (( i % 4 == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done
wait "${pids[@]}" 2>/dev/null

echo "ALL RENDERS COMPLETE"
