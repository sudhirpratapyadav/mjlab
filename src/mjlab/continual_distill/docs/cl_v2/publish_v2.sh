#!/usr/bin/env bash
# Publish a CL-V2 task folder (or the site index) to https://cl.untuai.com/v2/
#
#   publish_v2.sh <Task-Id> <local-dir>     rsync still.png / turntable.mp4 / colliders.png /
#                                            teacher.mp4 / failure.mp4 / thumb.jpg / result.json /
#                                            asset.json  ->  untu_vps:~/sudhir/continual_learning/v2/<Task-Id>/
#   publish_v2.sh --index                    push docs/cl_v2/site/index.html to .../v2/index.html
#
# The index reads every task's asset.json / result.json client-side, so publishing a
# task folder is enough for its card to update — no rebuild.
set -euo pipefail
REMOTE_HOST="untu_vps"
REMOTE_BASE="~/sudhir/continual_learning/v2"
SITE_URL="https://cl.untuai.com/v2"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--index" ]]; then
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p ${REMOTE_BASE}"
  rsync -az "$HERE/site/index.html" "${REMOTE_HOST}:${REMOTE_BASE}/index.html"
  echo "Published index: ${SITE_URL}/"
  exit 0
fi
if [[ $# -ne 2 ]]; then echo "usage: $0 <Task-Id> <local-dir> | $0 --index" >&2; exit 1; fi
TASK_ID="$1"; LOCAL_DIR="$2"
[[ "$TASK_ID" =~ ^Mjlab-[A-Za-z0-9-]+-Franka$ ]] || echo "warning: '$TASK_ID' does not look like a task id" >&2
[[ -d "$LOCAL_DIR" ]] || { echo "error: '$LOCAL_DIR' missing" >&2; exit 1; }
shopt -s nullglob
FILES=("$LOCAL_DIR"/still.png "$LOCAL_DIR"/turntable.mp4 "$LOCAL_DIR"/colliders.png "$LOCAL_DIR"/teacher.mp4 \
       "$LOCAL_DIR"/failure.mp4 "$LOCAL_DIR"/thumb.jpg "$LOCAL_DIR"/result.json "$LOCAL_DIR"/asset.json)
[[ ${#FILES[@]} -gt 0 ]] || { echo "error: nothing publishable in $LOCAL_DIR" >&2; exit 1; }
ssh -o BatchMode=yes -o ConnectTimeout=10 "$REMOTE_HOST" "mkdir -p ${REMOTE_BASE}/${TASK_ID}" \
  || { echo "error: ssh to ${REMOTE_HOST} failed — STOP, do not guess around it" >&2; exit 1; }
rsync -az "${FILES[@]}" "${REMOTE_HOST}:${REMOTE_BASE}/${TASK_ID}/"
ssh "$REMOTE_HOST" "ls -la ${REMOTE_BASE}/${TASK_ID}/"
echo "Published: ${SITE_URL}/${TASK_ID}/   (index: ${SITE_URL}/)"
