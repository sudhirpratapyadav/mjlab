#!/bin/bash
# Record a 3s random-agent clip of every benchmark task, then serve them as an HTML
# gallery. Rendering needs a real GL device, so this must run on a GPU node — the
# login node has no EGL device and mujoco.Renderer will fail there.
#
# Usage:
#   bash slurm/record_and_serve_videos.sh [HOLDER_JOBID] [PORT]
#
# Then, from your laptop:
#   ssh -N -L 8000:<node>:8000 <user>@<login-host>
#   open http://localhost:8000
set -euo pipefail

REPO=/ihub/homedirs/svs_ald/sudhir/mjlab
HOLDER=${1:-$(squeue -u "$USER" -h -n hold_dgx2 -o %A | head -1)}
PORT=${2:-8000}
[[ -n "$HOLDER" ]] || { echo "no holder job found; pass one explicitly"; exit 1; }
echo "Using holder job: $HOLDER"

srun --jobid="$HOLDER" --overlap --gres=gpu:1 --cpus-per-task=8 bash -c "
  cd $REPO
  export MUJOCO_GL=egl PYTHONPATH=src
  .venv/bin/python -m mjlab.scripts.record_task_videos \
      --out-dir benchmark_videos --seconds 3 --width 640 --height 480
  echo
  echo 'Serving on \$(hostname):$PORT — Ctrl-C to stop'
  .venv/bin/python -m mjlab.scripts.serve_task_videos \
      --video-dir benchmark_videos --port $PORT
"
