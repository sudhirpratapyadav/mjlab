#!/bin/bash
# P1-5b: bracket EWC's lambda optimum.
#
# The first sweep was monotone increasing (40 -> 0.539, 400 -> 0.732, 5000 -> 0.923),
# so 5000 is the EDGE of the grid, not a proven peak. Claiming SI (0.960) beats EWC
# (0.923) on that evidence would be overclaiming — EWC might still be climbing.
#
# Extends to {20000, 50000} x 3 seeds. Either the curve turns over (optimum located,
# SI > EWC is then a fair claim) or it keeps rising (EWC matches SI, and that is the
# honest result to report).
#
# Usage: bash run_p15b.sh [HOLDER] [WANDB_PROJECT]
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
STATE=$REPO/logs/p15b_state
mkdir -p "$REPO/logs" "$STATE"

BEST="PushCuboid OpenDrawer OpenDoor PushButton"
CFG=tasks_p0.yaml
NGPU=8

WORST="PushButton OpenDoor OpenDrawer PushCuboid"

JOBS=()
for lam in 20000 50000; do
  for s in 0 1 2; do
    JOBS+=("ewc_l${lam}_best_s${s}|${s}|--regularizer ewc --reg-coeff $lam|$BEST")
  done
done
# EWC on the WORST ordering at the best lambda found so far. P1-5 only ever ran EWC
# on the best ordering, so the SI-vs-EWC comparison is currently one-sided; SI has
# both (0.960 / 0.659). Uses 5000 (the current best); if 20000/50000 turn out better
# this arm should be repeated at that value.
for s in 0 1 2; do
  JOBS+=("ewc_l5000_worst_s${s}|${s}|--regularizer ewc --reg-coeff 5000|$WORST")
done

echo "P1-5b scheduler: ${#JOBS[@]} jobs, holder=$HOLDER project=$PROJ"

is_done()  { grep -q "Training complete" "$REPO/logs/cl_${1}.log" 2>/dev/null; }
is_live()  { local p; p=$(cat "$STATE/$1.pid" 2>/dev/null) || return 1
             [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch() {
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
  [ $nd -ge ${#JOBS[@]} ] && { echo "ALL P1-5b RUNS COMPLETE."; break; }
  sleep 120
done
