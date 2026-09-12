"""Single-task launcher for an already UUID-pinned Slurm step (no GPU reselection)."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from functools import partial
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
  parser.add_argument("--resume-gripper-std", type=float, help="Explicit gripper-only exploration reset after checkpoint load")
  parser.add_argument("--resume-gripper-mean", type=float, help="Reset only the gripper output row after loading a bounded actor")
  parser.add_argument("--capture-pre-step", action="store_true", help="Save preceding physics state if the numerical guard fails")
  parser.add_argument("--elliptic-hessian", action="store_true", help="Opt in to stable native-equivalent dense cone Hessian")
  parser.add_argument("--free-body-gyro", action="store_true", help="Opt in to CPU-compatible standalone free-body gyroscopic integration")
  parser.add_argument("--learning-rate", type=float, help="Explicit optimizer LR override, including after checkpoint load")
  parser.add_argument("--recipe", choices=RECIPES, default="baseline")
  parser.add_argument("--wandb-offline", action="store_true", help="Save W&B locally until this entity is accessible")
  parser.add_argument("--dry-run", action="store_true")
  args = parser.parse_args()
  if Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
    parser.error("--run-id must be a single directory name")
  if args.num_envs < 1 or args.iterations < 1:
    parser.error("environment and iteration counts must be positive")
  if args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
    parser.error("--learning-rate must be finite and positive")
  if args.resume_gripper_std is not None and (not args.resume_checkpoint or not math.isfinite(args.resume_gripper_std) or args.resume_gripper_std <= 0):
    parser.error("--resume-gripper-std requires a resume checkpoint and a finite positive value")
  if args.resume_gripper_mean is not None and (not args.resume_checkpoint or not math.isfinite(args.resume_gripper_mean) or not -1 < args.resume_gripper_mean < 1):
    parser.error("--resume-gripper-mean requires a resume checkpoint and a finite value strictly between -1 and 1")
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
  cfg.env.sim.elliptic_hessian_compat = args.elliptic_hessian
  cfg.env.sim.free_body_implicitfast_compat = args.free_body_gyro
  if args.resume_gripper_mean is not None and cfg.agent.policy.class_name != "BoundedActorCritic":
    parser.error("--resume-gripper-mean requires a bounded-mean policy")
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
    cfg.env.sim.elliptic_hessian_compat |= previous_manifest.get("elliptic_hessian_compat", False)
    cfg.env.sim.free_body_implicitfast_compat |= previous_manifest.get("free_body_implicitfast_compat", False)
    previous_train_cfg = TrainConfig.from_task(args.task)
    apply_recipe(previous_train_cfg, previous_manifest.get("recipe", "baseline"))
    if previous_train_cfg.agent.policy.class_name != cfg.agent.policy.class_name:
      parser.error("Resume requires the same policy architecture; bounded-mean variants start fresh")
    if args.learning_rate is None:
      args.learning_rate = previous_manifest.get("learning_rate_override")
    cfg.agent.resume = True
    cfg.agent.load_run = "^" + re.escape(checkpoint.parent.name) + "$"
    cfg.agent.load_checkpoint = "^" + re.escape(checkpoint.name) + "$"
  if args.learning_rate is not None:
    cfg.agent.algorithm.learning_rate = args.learning_rate
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
      "elliptic_hessian_compat": cfg.env.sim.elliptic_hessian_compat,
      "free_body_implicitfast_compat": cfg.env.sim.free_body_implicitfast_compat,
      "learning_rate_override": args.learning_rate,
      "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
      "wandb_entity": args.wandb_entity, "wandb_project": args.wandb_project,
      "resume_checkpoint": str(checkpoint) if checkpoint else None,
      "resume_gripper_std": args.resume_gripper_std,
      "resume_gripper_mean": args.resume_gripper_mean,
      "capture_pre_step": args.capture_pre_step,
      "capture_history_steps": 8 if args.capture_pre_step else 0,
      "resume_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,
      "resume_note": "Optimizer and normalizers restored; environment and RNG restart from recorded seed. RSL starts labels at the saved iteration." if checkpoint else "Fresh policy",
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    try:
      runner = partial(TeacherRunner, resume_gripper_std=args.resume_gripper_std,
                       resume_gripper_mean=args.resume_gripper_mean, capture_pre_step=args.capture_pre_step,
                       learning_rate_override=args.learning_rate)
      run_train(args.task, cfg, run, runner_cls_override=runner)
    finally:
      import sys
      import wandb
      if wandb.run is not None:
        wandb.finish(exit_code=1 if sys.exc_info()[0] else 0)


if __name__ == "__main__":
  main()
