#!/bin/bash
# Pre-flight for P1-6: confirm all SIX teachers work on current HEAD before spending
# 18 runs on them. Waits for a free GPU rather than displacing a running experiment.
#
# This exists because five teachers have already been silently invalidated by the
# workspace-audit placement change (a0cc9be, 97703b4). The failure mode is silent —
# obs_dim stays 60, nothing errors, the numbers are just wrong — so the only safe
# move is to measure before launching.
#
# Expected: PushCuboid ~0.82, OpenDrawer ~1.00, OpenDoor ~0.99, PushButton ~1.00,
#           LiftCube ~0.95 (after 97703b4), OpenLid ~1.00.
#
# Usage: bash verify_p1b_teachers.sh [HOLDER]
set -uo pipefail
HOLDER=${1:-19736}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
LOG=$REPO/logs/verify_p1b_teachers.log

echo "waiting for a free GPU..."
GPU=""
while [ -z "$GPU" ]; do
  BUSY=$(srun --jobid="$HOLDER" --overlap bash -c \
    "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader" 2>/dev/null | sort -u | wc -l)
  if [ "${BUSY:-8}" -lt 8 ]; then
    # find which index is idle by memory use
    GPU=$(srun --jobid="$HOLDER" --overlap bash -c \
      "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits" 2>/dev/null \
      | awk -F', ' '$2 < 500 {print $1; exit}')
  fi
  [ -z "$GPU" ] && sleep 120
done
echo "using GPU $GPU"

RUN_GPU=$GPU srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
  --export=ALL,RUN_GPU bash -c '
    export CUDA_VISIBLE_DEVICES=$RUN_GPU
    cd '"$REPO"'
    MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 PYTHONPATH='"$REPO"'/src \
    .venv/bin/python -u -m mjlab.continual_distill.continual_distill \
      --tasks-config src/mjlab/continual_distill/config/tasks_6task_check.yaml \
      --task-sequence PushCuboid OpenDrawer OpenDoor PushButton LiftCube OpenLid \
      --run-name verify6 --seed 0 --wandb-mode disabled --no-track --no-tqdm \
      --env-eval-episodes 128 --eval-every 100000 --checkpoint-every 100000' \
  > "$LOG" 2>&1

echo "=== teacher competence on HEAD ==="
grep -E "COMPUTED METRICS FOR TASK|teacher_success" "$LOG"
