#!/usr/bin/env bash
# Publish a CL-V3 task folder (or the site index) to https://cl.sudhirpratapyadav.com/v3/
#
#   publish_v3.sh <Task-Id> <local-dir>   rsync teacher.mp4 / failure.mp4 / thumb.jpg /
#                                          result.json / task.json
#                                          -> untu_vps:~/sudhir/continual_learning/v3/<Task-Id>/
#   publish_v3.sh --index                  push docs/cl_v3/site/index.html to .../v3/index.html
#
# The index reads every task's task.json / result.json client-side; publishing a task
# folder is enough for its card to update. The corrected-physics review includes
# below-bar teachers with honest result.json rates and failure clips.
set -euo pipefail
REMOTE_HOST="untu_vps"
REMOTE_BASE="~/sudhir/continual_learning/v3"
SITE_URL="https://cl.sudhirpratapyadav.com/v3"
# Reuse one authenticated connection instead of opening several per clip folder.
SSH_COMMAND="ssh -o BatchMode=yes -o ConnectTimeout=10 -o ControlMaster=auto -o ControlPersist=300 -o ControlPath=/tmp/mjlab-clv3-%C"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--index" ]]; then
  $SSH_COMMAND "$REMOTE_HOST" "mkdir -p ${REMOTE_BASE}"
  rsync -az -e "$SSH_COMMAND" "$HERE/site/index.html" "${REMOTE_HOST}:${REMOTE_BASE}/index.html"
  echo "Published index: ${SITE_URL}/"
  exit 0
fi
if [[ $# -ne 2 ]]; then echo "usage: $0 <Task-Id> <local-dir> | $0 --index" >&2; exit 1; fi
TASK_ID="$1"; LOCAL_DIR="$2"
[[ "$TASK_ID" =~ ^Mjlab-[A-Za-z0-9-]+-Franka$ ]] || { echo "error: invalid task id" >&2; exit 1; }
[[ -d "$LOCAL_DIR" ]] || { echo "error: '$LOCAL_DIR' missing" >&2; exit 1; }
CANDIDATES=(teacher.mp4 failure.mp4 thumb.jpg result.json task.json)
FILES=()
for f in "${CANDIDATES[@]}"; do
  [[ -f "$LOCAL_DIR/$f" ]] && FILES+=("$LOCAL_DIR/$f") || true
done
[[ ${#FILES[@]} -gt 0 ]] || { echo "error: nothing publishable in $LOCAL_DIR" >&2; exit 1; }
$SSH_COMMAND "$REMOTE_HOST" "mkdir -p ${REMOTE_BASE}/${TASK_ID}" \
  || { echo "error: ssh to ${REMOTE_HOST} failed — STOP, do not guess around it" >&2; exit 1; }
rsync -az -e "$SSH_COMMAND" "${FILES[@]}" "${REMOTE_HOST}:${REMOTE_BASE}/${TASK_ID}/"
# Remove only obsolete clip names; prior versions are retained in the archived gallery.
for f in teacher.mp4 failure.mp4; do
  if [[ ! -f "$LOCAL_DIR/$f" ]]; then
    $SSH_COMMAND "$REMOTE_HOST" "rm -f ${REMOTE_BASE}/${TASK_ID}/$f"
  fi
done
$SSH_COMMAND "$REMOTE_HOST" "ls -la ${REMOTE_BASE}/${TASK_ID}/"
echo "Published: ${SITE_URL}/${TASK_ID}/   (index: ${SITE_URL}/)"
