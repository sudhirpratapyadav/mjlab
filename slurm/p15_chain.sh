#!/bin/bash
# Chain P1-5 behind P1-6.
#
# run_p1b.sh and run_p15.sh each pick GPU indices from their own view of which of
# their OWN jobs are live, so running them concurrently double-books GPUs — observed
# directly: three GPUs ended up with two training processes each. Serialising is the
# simple fix; P1-6 is already most of the way through.
#
# Usage: setsid bash slurm/p15_chain.sh [HOLDER] [PROJECT] &
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
cd "$REPO"

echo "[$(date +%F' '%T)] waiting for P1-6 to finish before starting P1-5..."
while pgrep -f "run_p1b.sh $HOLDER" > /dev/null; do sleep 180; done

echo "[$(date +%F' '%T)] P1-6 done. Launching P1-5 lambda sweep."
setsid nohup bash "$REPO/slurm/run_p15.sh" "$HOLDER" "$PROJ" \
  >> "$REPO/logs/p15_scheduler.log" 2>&1 < /dev/null &
echo "[$(date +%F' '%T)] P1-5 scheduler started."
