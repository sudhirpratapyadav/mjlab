"""Single-task launcher for an already UUID-pinned Slurm step (no GPU reselection)."""

import argparse
import os
from pathlib import Path

from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.manipulation.benchmark import active_cl_tasks


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=active_cl_tasks())
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--iterations", type=int, default=500)
  parser.add_argument("--seed", type=int, default=20260912)
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
    parser.error("--run-id must be a single directory name")
  if args.num_envs < 1 or args.iterations < 1:
    parser.error("environment and iteration counts must be positive")
  visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if not visible or "," in visible:
    parser.error("Run inside srun with exactly one assigned GPU UUID exported")
  run = Path(__file__).resolve().parent / "runs" / args.run_id
  if run.exists():
    parser.error(f"Run directory already exists: {run}")
  cfg = TrainConfig.from_task(args.task)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.iterations
  cfg.agent.seed = args.seed
  cfg.agent.logger = "tensorboard"
  cfg.agent.run_name = args.run_id
  print(
    f"{args.task}: {args.num_envs} envs, {args.iterations} iterations, seed {args.seed}, GPU {visible}, output {run}"
  )
  if not args.dry_run:
    run_train(args.task, cfg, run)


if __name__ == "__main__":
  main()
