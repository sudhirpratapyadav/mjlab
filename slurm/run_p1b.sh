#!/bin/bash
# Block B — P1-6: six-task scalability x capacity. 18 runs into continual_rl_mjlab_p1.
#
#   3 orderings x 2 widths x 3 seeds
#
# Orderings (PC=PushCuboid is the fragile task, from P0-1/P0-3):
#   ff  fragile-first : PC OW OD PB LC OL
#   fl  fragile-last  : OL LC PB OD OW PC
#   rnd random        : OD PC OL OW PB LC   (fixed shuffle, PC neither first nor last)
#
# Widths run at their OWN best LR, which is the whole point of pairing this with
# P0-3: 4096 @ 3e-5 and 8192 @ 1e-5. Holding LR fixed across widths would re-create
# exactly the confound P0-3 removed, so a capacity difference here would be
# unreadable. With each width at its optimum, a real 8192 > 4096 gap at N=6 means
# capacity genuinely binds once the sequence is long enough -- which is the question
# this block exists to answer, since at N=4 the two are tied (0.960 vs 0.946).
#
# Usage: bash run_p1b.sh [HOLDER] [WANDB_PROJECT]
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
STATE=$REPO/logs/p1b_state
mkdir -p "$REPO/logs" "$STATE"

CFG=tasks_6task.yaml
NGPU=8

FF="PushCuboid OpenDrawer OpenDoor PushButton LiftCube OpenLid"
FL="OpenLid LiftCube PushButton OpenDoor OpenDrawer PushCuboid"
RND="OpenDoor PushCuboid OpenLid OpenDrawer PushButton LiftCube"

JOBS=()
for s in 0 1 2; do
  # width 4096 at its optimum lr 3e-5
  JOBS+=("t6_ff_w4096_s${s}|${s}|--student-hidden-dims 4096 2048 1024 --learning-rate 3e-5|$FF")
  JOBS+=("t6_fl_w4096_s${s}|${s}|--student-hidden-dims 4096 2048 1024 --learning-rate 3e-5|$FL")
  JOBS+=("t6_rnd_w4096_s${s}|${s}|--student-hidden-dims 4096 2048 1024 --learning-rate 3e-5|$RND")
  # width 8192 at its optimum lr 1e-5 (located by P0-3)
  JOBS+=("t6_ff_w8192_s${s}|${s}|--student-hidden-dims 8192 4096 2048 --learning-rate 1e-5|$FF")
  JOBS+=("t6_fl_w8192_s${s}|${s}|--student-hidden-dims 8192 4096 2048 --learning-rate 1e-5|$FL")
  JOBS+=("t6_rnd_w8192_s${s}|${s}|--student-hidden-dims 8192 4096 2048 --learning-rate 1e-5|$RND")
done

echo "P1-B scheduler: ${#JOBS[@]} jobs, holder=$HOLDER project=$PROJ gpus=$NGPU"

is_done()  { grep -q "Training complete" "$REPO/logs/cl_${1}.log" 2>/dev/null; }
is_live()  { local p; p=$(cat "$STATE/$1.pid" 2>/dev/null) || return 1
             [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }

launch() {  # launch <gpu> <name> <seed> <extra> <seq>
  local GPU=$1 NAME=$2 SEED=$3 EXTRA=$4 SEQ=$5
  RUN_GPU=$GPU EXTRA_ARGS="--wandb-project $PROJ $EXTRA" \
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
  [ $nd -ge ${#JOBS[@]} ] && { echo "ALL P1-B RUNS COMPLETE."; break; }
  sleep 120
done
