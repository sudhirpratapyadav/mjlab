#!/usr/bin/env bash
# Extract fresh teacher datasets for all 15 CL-V5 tasks from their certified
# CL-V4 checkpoints, round-robined across GPUs 1-7 (GPU0 stays unused).
set -u

WORKTREE=/ihub/homedirs/svs_ald/sudhir/mjlab-cl15-si-20260918
MAIN=/ihub/homedirs/svs_ald/sudhir/mjlab
OUT="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15/teacher_datasets"
LOGDIR="$WORKTREE/src/mjlab/continual_distill/docs/cl_v5_si15/extract_logs"
mkdir -p "$OUT" "$LOGDIR"

GPU_UUIDS=(
  "GPU-2496e2db-e138-0dec-41fc-27d2535b6e86"  # 1
  "GPU-23a5dcb4-5248-01aa-94a3-d0f998660161"  # 2
  "GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc"  # 3
  "GPU-0b3441b1-580c-9c76-f27d-cbec321e1937"  # 4
  "GPU-7d5c3740-990d-15b7-5c54-6dc6270633fa"  # 5
  "GPU-46a4f4dc-5916-8d5a-a71b-6193e01b5723"  # 6
  "GPU-811bf62a-ec3a-649f-1167-bfff22107497"  # 7
)

# task_name  env_id  checkpoint(relative to MAIN)
# AxialExtract and ReachTarget already extracted successfully in the first attempt.
TASKS=(
  "FlipSwitch Mjlab-Flip-Switch-Franka docs/cl_v4_rl/runs/RL-009-flip-mechanism/model_800.pt"
  "OpenDoor Mjlab-Open-Door-Franka docs/cl_v4_rl/runs/RL-010-door-mechanism/model_900.pt"
  "OpenDrawer Mjlab-Open-Drawer-Franka docs/cl_v4_rl/runs/RL-008-drawer-mechanism/model_1000.pt"
  "OpenLid Mjlab-Open-Lid-Franka docs/cl_v4_rl/runs/RL-015-V3-lid-bounded/model_700.pt"
  "PushButton Mjlab-Push-Button-Franka docs/cl_v4_rl/runs/RL-007-button-mechanism/model_700.pt"
  "PushFlap Mjlab-Push-Flap-Franka docs/cl_v4_rl/runs/RL-011-flap-mechanism/model_900.pt"
  "SlideWindow Mjlab-Slide-Window-Franka docs/cl_v4_rl/runs/RL-014-window-mechanism/model_300.pt"
  "TurnLever Mjlab-Turn-Lever-Franka docs/cl_v4_rl/runs/RL-012-lever-mechanism/model_1000.pt"
  "RotateValve Mjlab-Rotate-Valve-Franka docs/cl_v4_rl/runs/RL-016-valve-mechanism/model_1999.pt"
  "PushCuboid Mjlab-Push-Cuboid-Franka docs/cl_v4_rl/runs/RL-003-R3-push-extend/model_1998.pt"
  "ThrowToBin Mjlab-Throw-To-Bin-Franka docs/cl_v4_rl/runs/RL-024-R7-throw-box/model_14098.pt"
  "ToppleBlock Mjlab-Topple-Block-Franka docs/cl_v4_rl/runs/RL-006-topple-resume-wandb/model_499.pt"
  "DragPull Mjlab-Drag-Pull-Franka docs/cl_v4_rl/runs/RL-004-R2-drag-extend/model_1300.pt"
)

run_one() {
  local name="$1" env_id="$2" ckpt_rel="$3" gpu_uuid="$4" cpus="$5"
  local ckpt="$MAIN/src/mjlab/continual_distill/$ckpt_rel"
  local log="$LOGDIR/${name}.log"
  echo "[$(date -u +%FT%TZ)] START $name ($env_id) on $gpu_uuid" | tee "$log"
  srun --jobid=20277 --overlap -n1 --cpus-per-task=8 bash -c "
    export FORCE_CPU=0
    export CUDA_VISIBLE_DEVICES=$gpu_uuid
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    export PYTHONPATH='$WORKTREE/src'
    cd '$WORKTREE'
    $MAIN/.venv/bin/python src/mjlab/continual_distill/extract_teacher_dataset.py \
      --checkpoint '$ckpt' \
      --task $env_id \
      --num-samples 150000 --num-envs 512 --seed 0 \
      --output-dir '$OUT'
  " >> "$log" 2>&1
  echo "[$(date -u +%FT%TZ)] DONE $name exit=$?" | tee -a "$log"
}

# Two waves of up to 7 parallel extractions each (15 tasks / 7 GPUs).
i=0
pids=()
for entry in "${TASKS[@]}"; do
  read -r name env_id ckpt_rel <<< "$entry"
  gpu_idx=$(( i % 7 ))
  gpu_uuid="${GPU_UUIDS[$gpu_idx]}"
  cpus="$(( 2 + gpu_idx*2 )),$(( 130 + gpu_idx*2 ))"
  run_one "$name" "$env_id" "$ckpt_rel" "$gpu_uuid" "$cpus" &
  pids+=($!)
  i=$((i+1))
  if (( i % 7 == 0 )); then
    wait "${pids[@]}"
    pids=()
  fi
done
wait "${pids[@]}" 2>/dev/null

echo "ALL EXTRACTIONS COMPLETE"
