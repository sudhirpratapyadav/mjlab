"""Single-task launcher for an already UUID-pinned Slurm step (no GPU reselection)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import yaml

from wandb_config import ENTITY, KEY_FILE, PROJECT, configure
from rl_recipes import RECIPES, apply_recipe
from teacher_runner import TeacherRunner
from allocation import GPUS

from mjlab.scripts.train import TrainConfig, run_train
from mjlab.tasks.manipulation.benchmark import active_cl_tasks


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=active_cl_tasks())
  parser.add_argument("--run-id", required=True)
  parser.add_argument("--num-envs", type=int, default=1024)
  parser.add_argument("--iterations", type=int, default=500)
  parser.add_argument("--seed", type=int, default=20260912)
  parser.add_argument("--wandb-project", default=PROJECT)
  parser.add_argument("--wandb-entity", default=ENTITY)
  parser.add_argument(
    "--wandb-api-key-file", type=Path,
    default=KEY_FILE,
  )
  parser.add_argument("--resume-checkpoint", type=Path)
  parser.add_argument("--recipe", choices=RECIPES, default="baseline")
  parser.add_argument("--wandb-offline", action="store_true", help="Save W&B locally until this entity is accessible")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
    parser.error("--run-id must be a single directory name")
  if args.num_envs < 1 or args.iterations < 1:
    parser.error("environment and iteration counts must be positive")
  if not args.dry_run:
    try:
      configure(args.wandb_entity, args.wandb_project, args.wandb_api_key_file, args.wandb_offline)
    except (OSError, ValueError):
      parser.error("Run this experiment's setup_wandb.py to provide its W&B key first")
  visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if visible not in GPUS.values():
    parser.error("Export exactly one assigned GPU1–7 UUID inside srun; GPU0 is excluded")
  run = Path(__file__).resolve().parent / "runs" / args.run_id
  if run.exists():
    parser.error(f"Run directory already exists: {run}")
  cfg = TrainConfig.from_task(args.task)
  apply_recipe(cfg, args.recipe)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.iterations
  cfg.agent.seed = args.seed
  cfg.agent.logger = "wandb"
  cfg.agent.wandb_project = args.wandb_project
  checkpoint = None
  if args.resume_checkpoint:
    checkpoint = args.resume_checkpoint.resolve()
    if not checkpoint.is_file() or checkpoint.parent.parent != run.parent:
      parser.error("Resume checkpoint must be an existing checkpoint in this worktree's stage runs")
    previous_cfg = yaml.load((checkpoint.parent / "params/agent.yaml").read_text(), Loader=yaml.FullLoader)
    if previous_cfg["experiment_name"] != cfg.agent.experiment_name:
      parser.error("Resume checkpoint belongs to another task")
    if previous_cfg["policy"]["noise_std_type"] != cfg.agent.policy.noise_std_type:
      parser.error("Resume requires the same policy std parameterization; use a matching recipe")
    previous_manifest = json.loads((checkpoint.parent / "manifest.json").read_text())
    previous_train_cfg = TrainConfig.from_task(args.task)
    apply_recipe(previous_train_cfg, previous_manifest.get("recipe", "baseline"))
    if previous_train_cfg.agent.policy.class_name != cfg.agent.policy.class_name:
      parser.error("Resume requires the same policy architecture; bounded-mean variants start fresh")
    cfg.agent.resume = True
    cfg.agent.load_run = "^" + re.escape(checkpoint.parent.name) + "$"
    cfg.agent.load_checkpoint = "^" + re.escape(checkpoint.name) + "$"
  cfg.agent.run_name = args.run_id
  print(
    f"{args.task}: {args.num_envs} envs, {args.iterations} iterations, seed {args.seed}, GPU {visible}, output {run}"
  )
  if not args.dry_run:
    worktree = next(parent for parent in Path(__file__).resolve().parents if (parent / ".git").exists())
    run.mkdir(parents=True, exist_ok=False)
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip()
    source_patch = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=worktree)
    (run / "source.patch").write_bytes(source_patch)
    manifest = {
      "task": args.task, "run_id": args.run_id, "worktree": str(worktree),
      "source_head": source_head, "source_patch_sha256": hashlib.sha256(source_patch).hexdigest(),
      "started_utc": datetime.now(timezone.utc).isoformat(),
      "seed": args.seed, "num_envs": args.num_envs, "requested_updates": args.iterations,
      "interface": "franka_shared_60_v2", "gpu_uuid": visible,
      "recipe": args.recipe,
      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
      "wandb_entity": args.wandb_entity, "wandb_project": args.wandb_project,
      "resume_checkpoint": str(checkpoint) if checkpoint else None,
      "resume_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
      "resume_note": "Optimizer and normalizers restored; environment and RNG restart from recorded seed. RSL starts labels at the saved iteration." if checkpoint else "Fresh policy",
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
      run_train(args.task, cfg, run, runner_cls_override=TeacherRunner)
    finally:
      import sys
      import wandb
      if wandb.run is not None:
        wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
  main()
