#!/bin/bash
# Chain P1-6 behind the pre-flight teacher check.
#
# Gate: every one of the six teachers must clear a competence threshold on HEAD
# before 18 runs are spent on them. Five teachers have already been silently
# invalidated by the workspace-audit placement change; the failure leaves obs_dim
# at 60 and raises no error, so an unchecked launch produces 18 plausible-looking
# runs of garbage. That has happened once already in this project.
#
# Threshold is 0.5: comfortably below every expected value (0.82-1.00) and far above
# the broken signature (0.00-0.03), so it separates the two cases without being
# sensitive to ordinary eval noise.
#
# Usage: setsid bash slurm/p1b_chain.sh [HOLDER] [PROJECT] &
set -uo pipefail
HOLDER=${1:-19736}
PROJ=${2:-continual_rl_mjlab_p1}
REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
VLOG=$REPO/logs/verify_p1b_teachers.log
cd "$REPO"

echo "[$(date +%F' '%T)] waiting for the six-teacher pre-flight to finish..."
# The verify script waits for its own GPU, so this can be a long wait.
while pgrep -f "verify_p1b_teachers.sh" > /dev/null; do sleep 120; done

if [ ! -s "$VLOG" ]; then
  echo "[$(date +%F' '%T)] FATAL: no verification log at $VLOG — refusing to launch."
  exit 1
fi

echo "[$(date +%F' '%T)] pre-flight results:"
grep -E "COMPUTED METRICS FOR TASK|teacher_success" "$VLOG" || true

# Every reported teacher_success must clear the threshold.
BAD=$("$REPO/.venv/bin/python" - "$VLOG" <<'PY'
import re, sys
t = open(sys.argv[1], errors="ignore").read()
vals = [float(v) for v in re.findall(r"teacher_success: ([\d.]+)", t)]
if not vals:
    print("NO_VALUES"); raise SystemExit
bad = [v for v in vals if v < 0.5]
print("OK" if not bad else f"BAD:{bad}")
PY
)

if [ "$BAD" != "OK" ]; then
  echo "[$(date +%F' '%T)] FATAL: teacher check failed ($BAD). NOT launching P1-6."
  echo "  Diff the task's placement against 69b2896 — that has been the cause every time."
  exit 1
fi

echo "[$(date +%F' '%T)] all six teachers healthy. Waiting for Block A to finish..."
while pgrep -f "run_p1a.sh $HOLDER" > /dev/null; do sleep 300; done

echo "[$(date +%F' '%T)] launching P1-6."
setsid nohup bash "$REPO/slurm/run_p1b.sh" "$HOLDER" "$PROJ" \
  >> "$REPO/logs/p1b_scheduler.log" 2>&1 < /dev/null &
echo "[$(date +%F' '%T)] P1-6 scheduler started."
