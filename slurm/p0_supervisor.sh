#!/bin/bash
# Autonomous supervisor for the P0 block.
#
# Every 10 min it:
#   1. restarts run_p0.sh if the scheduler died (it is idempotent — finished runs
#      are skipped via their "Training complete" marker, so nothing is duplicated)
#   2. refreshes the results tables in docs/P0_EXPERIMENTS.md
#   3. exits once all 18 runs report complete
#
# JOINT=1 is set so the 3 joint runs are dispatched alongside the other 15; they
# simply queue until a GPU frees.
#
# Usage: setsid bash slurm/p0_supervisor.sh &
set -uo pipefail
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p0}
cd "$REPO"

stall_ticks=0
last_done=-1

while true; do
  # The holder owning our GPUs must still be alive; without it every srun step is
  # dead and restarting the scheduler just spins. Bail loudly instead.
  if ! squeue -h -j "$HOLDER" -o "%T" 2>/dev/null | grep -q RUNNING; then
    echo "[$(date +%F' '%T)] FATAL: holder $HOLDER is no longer RUNNING — stopping."
    echo "  Resubmit a holder, then re-run: setsid bash slurm/p0_supervisor.sh <NEW_HOLDER>"
    break
  fi

  if ! pgrep -f "run_p0.sh $HOLDER" > /dev/null; then
    echo "[$(date +%F' '%T)] scheduler not running -> (re)starting"
    JOINT=1 nohup bash "$REPO/slurm/run_p0.sh" "$HOLDER" "$PROJ" \
      >> "$REPO/logs/p0_scheduler.log" 2>&1 &
  fi

  "$REPO/.venv/bin/python" "$REPO/slurm/collect_p0.py" --write \
    > "$REPO/logs/p0_status.txt" 2>&1
  ndone=$(grep -oP 'complete: \K\d+' "$REPO/logs/p0_status.txt" | head -1)
  nbusy=$(srun --jobid="$HOLDER" --overlap bash -c \
    "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader" 2>/dev/null \
    | sort -u | wc -l)
  echo "[$(date +%F' '%T)] complete=${ndone:-?}/18  gpus_busy=${nbusy:-?}"

  # Surface runs that died without finishing — otherwise a crashloop looks like
  # "still training" forever.
  for f in "$REPO"/logs/cl_nosi_*.log "$REPO"/logs/cl_w8192_*.log "$REPO"/logs/cl_joint_*.log; do
    [ -f "$f" ] || continue
    grep -q "Training complete" "$f" 2>/dev/null && continue
    if grep -qE "Traceback|CUDA out of memory|Segmentation fault|srun: error" "$f" 2>/dev/null; then
      echo "  WARN: $(basename "$f") shows a failure signature and has not completed"
    fi
  done

  if [ "${ndone:-0}" -ge 18 ]; then
    echo "[$(date +%F' '%T)] ALL 18 P0 RUNS COMPLETE."
    "$REPO/.venv/bin/python" "$REPO/slurm/collect_p0.py" --write >> "$REPO/logs/p0_status.txt" 2>&1
    break
  fi

  # No progress AND no GPU activity for ~1h => wedged, not merely slow.
  if [ "${ndone:-0}" = "$last_done" ] && [ "${nbusy:-0}" -eq 0 ]; then
    stall_ticks=$((stall_ticks+1))
    if [ $stall_ticks -ge 6 ]; then
      echo "[$(date +%F' '%T)] FATAL: no progress and no GPU activity for ~1h — stopping."
      break
    fi
  else
    stall_ticks=0
  fi
  last_done=${ndone:-0}

  sleep 600
done
