#!/bin/bash
# P1-5 (re-run): regulariser comparison with a corrected Fisher and a LAMBDA SWEEP.
#
# The first attempt was invalid twice over (see 2517a83): the Fisher was the BATCHED
# variant, and EWC was run at SI's coefficient of 1.0, which left the penalty
# numerically inert (0.269 vs a 0.267 no-regulariser floor).
#
# EWC's raw Fisher measures ~6e-4 here, and the literature uses lambda ~40-5000, so a
# single value would be a guess. Sweeping {40,400,5000} x 3 seeds on the BEST ordering
# locates EWC's own optimum; the winner is then run on the worst ordering for the
# best-vs-worst comparison. Reporting EWC at a swept optimum against SI at its own is
# the only version of this comparison a reviewer will accept.
#
#   ewc lambda {40,400,5000} x best  x3 =  9
#   l2  lambda {0.1,1,10}    x best  x3 =  9   (same problem: L2's uniform weight is
#                                               also on a different scale to SI's omega)
#  -> 18 runs. The worst-ordering arms are launched afterwards for the winning lambda.
#
# Usage: bash run_p15.sh [HOLDER] [WANDB_PROJECT]
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
STATE=$REPO/logs/p15_state
mkdir -p "$REPO/logs" "$STATE"

BEST="PushCuboid OpenDrawer OpenDoor PushButton"
CFG=tasks_p0.yaml
NGPU=8

JOBS=()
for lam in 40 400 5000; do
  for s in 0 1 2; do
    JOBS+=("ewc_l${lam}_best_s${s}|${s}|--regularizer ewc --reg-coeff $lam|$BEST")
  done
done
for lam in 0.1 1 10; do
  tag=$(echo "$lam" | tr -d '.')
  for s in 0 1 2; do
    JOBS+=("l2_l${tag}_best_s${s}|${s}|--regularizer l2 --reg-coeff $lam|$BEST")
  done
done

echo "P1-5 scheduler: ${#JOBS[@]} jobs, holder=$HOLDER project=$PROJ gpus=$NGPU"

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
  [ $nd -ge ${#JOBS[@]} ] && { echo "ALL P1-5 RUNS COMPLETE."; break; }
  sleep 120
done
