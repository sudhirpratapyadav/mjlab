#!/bin/bash
# Block A: P1-5 regularizer comparison + the per-width LR sweep that finishes
# de-confounding the capacity curve. 21 runs into wandb project continual_rl_mjlab_p1.
#
#   P1-5  ewc_best / ewc_worst / l2_best        x3 seeds =  9
#   LRW   width {512,1024,2048,4096} x lr {1e-5,5e-6} x3 = 24 -> trimmed to 12 (see below)
#
# Why the LR sweep: P0-3 de-confounded ONLY 8192. The rest of the capacity curve
# (0.66/0.79/0.87/0.96 for 512..4096) still shares the LR tuned at 4096, so a
# reviewer can ask whether those widths would also improve at their own optimum.
# We sweep lr 1e-5 (8192's optimum) at each width x 3 seeds = 12 runs, which is
# enough to show whether the ranking is LR-invariant without a full grid.
#
# Continuous scheduling, same as run_p0.sh: every free GPU gets the next queued job.
#
# Usage: bash run_p1a.sh [HOLDER] [WANDB_PROJECT]
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
STATE=$REPO/logs/p1a_state
mkdir -p "$REPO/logs" "$STATE"

BEST="PushCuboid OpenDrawer OpenDoor PushButton"   # p06
WORST="PushButton OpenDoor OpenDrawer PushCuboid"  # p10
CFG=tasks_p0.yaml
NGPU=8

JOBS=()
# --- P1-5: regularizer comparison (SI baseline already exists from SWEEP24) ---
for s in 0 1 2; do
  JOBS+=("ewc_best_s${s}|${s}|--regularizer ewc|$BEST")
  JOBS+=("ewc_worst_s${s}|${s}|--regularizer ewc|$WORST")
  JOBS+=("l2_best_s${s}|${s}|--regularizer l2|$BEST")
done
# --- per-width LR sweep at lr 1e-5 (the optimum P0-3 located at 8192) ---
for w in "512 256 128" "1024 512 256" "2048 1024 512" "4096 2048 1024"; do
  tag=$(echo "$w" | cut -d' ' -f1)
  for s in 0 1 2; do
    JOBS+=("lrw${tag}_lr1e5_s${s}|${s}|--student-hidden-dims $w --learning-rate 1e-5|$BEST")
  done
done

echo "P1-A scheduler: ${#JOBS[@]} jobs, holder=$HOLDER project=$PROJ gpus=$NGPU"

is_done()  { grep -q "Training complete" "$REPO/logs/cl_${1}.log" 2>/dev/null; }
is_live()  { local p; p=$(cat "$STATE/$1.pid" 2>/dev/null) || return 1
             [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch() {  # launch <gpu> <name> <seed> <extra> <seq>
  local GPU=$1 NAME=$2 SEED=$3 EXTRA=$4 SEQ=$5
  RUN_GPU=$GPU EXTRA_ARGS="--wandb-project $PROJ --learning-rate 3e-5 $EXTRA" \
    nohup srun --jobid="$HOLDER" --overlap --cpus-per-task=8 \
    --export=ALL,RUN_GPU,EXTRA_ARGS \
    bash "$REPO/slurm/_cl_run.sh" "$NAME" "$CFG" online "$SEED" $SEQ \
    > "$REPO/logs/srun_${NAME}.log" 2>&1 &
  echo $! > "$STATE/$NAME.pid"
  echo "  [dispatch] GPU$GPU  $NAME  (pid $!)"
}

while true; do
  declare -A BUSY=()
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$j"
    if is_live "$name" && ! is_done "$name"; then
      g=$(cat "$STATE/$name.gpu" 2>/dev/null); [ -n "$g" ] && BUSY[$g]=1
    fi
  done

  for g in $(seq 0 $((NGPU-1))); do
    [ -n "${BUSY[$g]:-}" ] && continue
    for j in "${JOBS[@]}"; do
      IFS='|' read -r name seed extra seq <<< "$j"
      is_done "$name" && continue
      is_live "$name" && continue
      [ -f "$STATE/$name.claimed" ] && is_live "$name" && continue
      launch "$g" "$name" "$seed" "$extra" "$seq"
      echo "$g" > "$STATE/$name.gpu"
      touch "$STATE/$name.claimed"
      BUSY[$g]=1
      sleep 20
      break
    done
  done

  nd=0; for j in "${JOBS[@]}"; do IFS='|' read -r n _ _ _ <<< "$j"; is_done "$n" && nd=$((nd+1)); done
  echo "[$(date +%H:%M:%S)] complete: $nd/${#JOBS[@]}"
  [ $nd -ge ${#JOBS[@]} ] && { echo "ALL P1-A RUNS COMPLETE."; break; }
  sleep 120
done
