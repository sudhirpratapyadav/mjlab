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

while true; do
  if ! pgrep -f "run_p0.sh $HOLDER" > /dev/null; then
    echo "[$(date +%F' '%T)] scheduler not running -> (re)starting"
    JOINT=1 nohup bash "$REPO/slurm/run_p0.sh" "$HOLDER" "$PROJ" \
      >> "$REPO/logs/p0_scheduler.log" 2>&1 &
  fi

  "$REPO/.venv/bin/python" "$REPO/slurm/collect_p0.py" --write \
    > "$REPO/logs/p0_status.txt" 2>&1
  ndone=$(grep -oP 'complete: \K\d+' "$REPO/logs/p0_status.txt" | head -1)
  echo "[$(date +%F' '%T)] complete=${ndone:-?}/18"

  if [ "${ndone:-0}" -ge 18 ]; then
    echo "[$(date +%F' '%T)] ALL 18 P0 RUNS COMPLETE."
    break
  fi
  sleep 600
done
