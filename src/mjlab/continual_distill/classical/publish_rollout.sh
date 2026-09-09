#!/usr/bin/env bash
# Publish a task's rollout clips to the CL-25 Phase-1 site.
#
# One wrapper so ~13 phase-1 agents are not each inventing an scp/rsync command.
# Pushes a local directory (the --out of render_rollout.py: teacher.mp4,
# failure.mp4, thumb.jpg, result.json — whichever of the clips exist) to
#   untu_vps:~/sudhir/continual_learning/phase1/<Task-Id>/
# which Caddy serves immediately at
#   https://cl.untuai.com/phase1/<Task-Id>/
# with no build or deploy step.
#
# Uses rsync (verified installed both ends) so re-publishing after a re-render
# only transfers what changed, and a partial local dir never clobbers a file it
# doesn't contain.
#
# Usage:
#   src/mjlab/continual_distill/classical/publish_rollout.sh <Task-Id> <local-dir>
#
# Example:
#   src/mjlab/continual_distill/classical/publish_rollout.sh \
#       Mjlab-Lift-Cube-Franka videos/rollouts/Mjlab-Lift-Cube-Franka
#
# <Task-Id> is the gym task ID exactly as registered (e.g. Mjlab-Lift-Cube-Franka)
# — it becomes the directory name under phase1/, so use the real ID, not a
# shortened name.
set -euo pipefail

REMOTE_HOST="untu_vps"
REMOTE_BASE="~/sudhir/continual_learning/phase1"
SITE_URL="https://cl.untuai.com/phase1"

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <Task-Id> <local-dir>" >&2
  echo "  e.g.: $0 Mjlab-Lift-Cube-Franka videos/rollouts/Mjlab-Lift-Cube-Franka" >&2
  exit 1
fi

TASK_ID="$1"
LOCAL_DIR="$2"

if [[ ! "$TASK_ID" =~ ^Mjlab-[A-Za-z0-9-]+-Franka$ ]]; then
  echo "warning: '$TASK_ID' does not look like a Mjlab-<Name>-Franka task ID — continuing anyway" >&2
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "error: local dir '$LOCAL_DIR' does not exist" >&2
  exit 1
fi

shopt -s nullglob
FILES=("$LOCAL_DIR"/*.mp4 "$LOCAL_DIR"/*.jpg "$LOCAL_DIR"/result.json)
if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "error: '$LOCAL_DIR' has none of teacher.mp4 / failure.mp4 / thumb.jpg / result.json — nothing to publish" >&2
  exit 1
fi

echo "Verifying ssh to ${REMOTE_HOST}..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p ${REMOTE_BASE}/${TASK_ID}"; then
  echo "error: could not ssh to ${REMOTE_HOST} or create the remote directory. STOP — this is a blocker, do not guess around it." >&2
  exit 1
fi

echo "Publishing ${LOCAL_DIR} -> ${REMOTE_HOST}:${REMOTE_BASE}/${TASK_ID}/"
rsync -avz --progress "${LOCAL_DIR}"/ "${REMOTE_HOST}:${REMOTE_BASE}/${TASK_ID}/"

echo
echo "Verifying remote listing..."
ssh "$REMOTE_HOST" "ls -la ${REMOTE_BASE}/${TASK_ID}/"

echo
echo "Published. View at: ${SITE_URL}/${TASK_ID}/"
echo "The phase1 index (${SITE_URL}/) reads each task's result.json client-side at"
echo "page load — no site rebuild needed. Just refresh ${SITE_URL}/ to see the card."
