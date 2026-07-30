#!/bin/bash
# P0 experiment block (floor / ceiling / 8192-LR sweep) -> wandb project
# continual_rl_mjlab_p0.
#
# Unlike run_24seq_sweep.sh (wave-based, idles GPUs waiting for the slowest run
# in each wave), this is a CONTINUOUS scheduler: it keeps every free GPU busy and
# dispatches the next queued job the moment a slot frees.
#
# Job list (18 total):
#   P0-1 nosi_{best,worst}_s{0,1,2}      6   --si-coeff 0
#   P0-3 w8192_lr{1e-5,5e-6,1e-6}_s{0,1,2}  9   width 8192, best ordering
#   P0-2 joint_s{0,1,2}                  3   requires --joint-distill (added separately)
#
# P0-2 is only dispatched when JOINT=1 is passed in the environment, so the
# 15 no-new-code runs can start while joint mode is still being written.
#
# Usage: bash run_p0.sh [HOLDER] [WANDB_PROJECT]
#        JOINT=1 bash run_p0.sh ...   # include the 3 joint runs
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p0}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
STATE=$REPO/logs/p0_state
mkdir -p "$REPO/logs" "$STATE"

BEST="PushCuboid OpenDrawer OpenDoor PushButton"   # p06
WORST="PushButton OpenDoor OpenDrawer PushCuboid"  # p10
CFG=tasks_p0.yaml
NGPU=8

# ---- build job list: name|seed|extra_args|sequence -------------------------
JOBS=()
for s in 0 1 2; do
  JOBS+=("nosi_best_s${s}|${s}|--si-coeff 0|$BEST")
  JOBS+=("nosi_worst_s${s}|${s}|--si-coeff 0|$WORST")
done
for lr in 1e-5 5e-6 1e-6; do
  tag=$(echo "$lr" | tr -d '-')
  for s in 0 1 2; do
    JOBS+=("w8192_lr${tag}_s${s}|${s}|--student-hidden-dims 8192 4096 2048 --learning-rate $lr|$BEST")
  done
done
if [ "${JOINT:-0}" = "1" ]; then
  for s in 0 1 2; do
    JOBS+=("joint_s${s}|${s}|--joint-distill --si-coeff 0|$BEST")
  done
fi

echo "P0 scheduler: ${#JOBS[@]} jobs, holder=$HOLDER project=$PROJ gpus=$NGPU"

# ---- helpers ---------------------------------------------------------------
# A job is DONE if its log has "Training complete"; RUNNING if a marker file
# exists and its srun client is still alive. Both survive a scheduler restart,
# so re-running this script resumes rather than duplicating work.
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

# ---- continuous scheduling loop -------------------------------------------
# NOTE: later --learning-rate in EXTRA_ARGS overrides the earlier 3e-5 default,
# which is how the 8192 LR sweep sets its own rate.
while true; do
  # which GPUs are occupied by our own live jobs?
  declare -A BUSY=()
  for j in "${JOBS[@]}"; do
    IFS='|' read -r name _ _ _ <<< "$j"
    if is_live "$name" && ! is_done "$name"; then
      g=$(cat "$STATE/$name.gpu" 2>/dev/null); [ -n "$g" ] && BUSY[$g]=1
    fi
  done

  # dispatch into free GPUs
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
      sleep 20   # stagger: avoid 8 simultaneous JAX/warp inits on one node
      break
    done
  done

  # progress report
  nd=0; for j in "${JOBS[@]}"; do IFS='|' read -r n _ _ _ <<< "$j"; is_done "$n" && nd=$((nd+1)); done
  echo "[$(date +%H:%M:%S)] complete: $nd/${#JOBS[@]}"
  [ $nd -ge ${#JOBS[@]} ] && { echo "ALL P0 RUNS COMPLETE."; break; }
  sleep 120
done
