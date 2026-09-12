"""Strict first-episode evaluation of one RSL-RL task checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
import numpy as np
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.benchmark import active_cl_tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.scripts.train import TrainConfig
from rl_recipes import apply_recipe


def snapshot_randomized_model(cfg, model):
  """Retain initial model draws separately from trajectories with later resets."""
  fields = sorted({event.params['field'] for event in cfg.events.values()
                   if event.domain_randomization})
  return {f'initial_model_{name}':getattr(model,name).detach().cpu().clone().numpy()
          for name in fields}


def evaluate(task: str, checkpoint: Path, episodes: int, seed: int, trace_dir: Path | None = None, diagnostics: bool = False) -> dict:
  training_cfg = TrainConfig.from_task(task)
  manifest_path = checkpoint.parent / "manifest.json"
  manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
  if manifest and manifest["task"] != task:
    raise ValueError("Checkpoint manifest belongs to another task")
  recipe = manifest.get("recipe", "baseline")
  apply_recipe(training_cfg, recipe)
  cfg = training_cfg.env
  cfg.sim.elliptic_hessian_compat = manifest.get("elliptic_hessian_compat", False)
  cfg.sim.free_body_implicitfast_compat = manifest.get("free_body_implicitfast_compat", False)
  cfg.scene.num_envs = episodes
  cfg.seed = seed
  agent_path = checkpoint.parent / "params/agent.yaml"
  agent = yaml.load(agent_path.read_text(), Loader=yaml.FullLoader) if agent_path.exists() else asdict(training_cfg.agent)
  # Installed RSL-RL pops class names before run_train writes agent.yaml.
  agent["policy"].setdefault("class_name", training_cfg.agent.policy.class_name)
  agent["algorithm"].setdefault("class_name", training_cfg.agent.algorithm.class_name)
  agent["seed"] = seed
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent["clip_actions"])
  try:
    command_names = env.command_manager.active_terms
    if len(command_names) != 1:
      raise RuntimeError(f"Expected one success command, found {command_names}")
    command = env.command_manager.get_term(command_names[0])
    if not hasattr(command, "compute_success"):
      raise RuntimeError("Command has no strict success predicate")

    # env.step resets finished instances before returning. Read the actual
    # predicate after physics/reward computation, immediately before reset.
    original_reset = env._reset_idx
    terminal_success = torch.zeros(episodes, dtype=torch.bool, device=device)
    fields = ("qpos", "qvel", "mocap_pos", "mocap_quat")
    terminal_states = {}
    if trace_dir is not None:
      trace_dir.mkdir(parents=True, exist_ok=False)
      terminal_states = {name: torch.zeros_like(getattr(env.sim.data,name)).cpu() for name in fields}

    def snapshot():
      return {name:getattr(env.sim.data,name).detach().cpu().clone() for name in fields}

    def capture_reset(env_ids):
      command._update_metrics()
      terminal_success[env_ids] = command.compute_success()[env_ids].bool()
      if trace_dir is not None:
        ids = env_ids.cpu()
        for name in fields:
          terminal_states[name][ids] = getattr(env.sim.data,name)[env_ids].detach().cpu()
      return original_reset(env_ids)

    env._reset_idx = capture_reset
    runner = OnPolicyRunner(wrapped, agent, None, device=device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy()
    obs = wrapped.get_observations()
    assert obs["policy"].shape == (episodes, 60) and wrapped.num_actions == 8
    model_snapshot = snapshot_randomized_model(cfg,env.sim.model) if trace_dir is not None else {}
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    successes = torch.zeros_like(finished)
    steps = torch.zeros(episodes, dtype=torch.int32, device=device)
    returns = torch.zeros(episodes, device=device)
    reasons = [None] * episodes
    termination_terms = [[] for _ in range(episodes)]
    trace = [snapshot()] if trace_dir is not None else []
    diagnostic_samples = []
    for control_step in range(env.max_episode_length + 1):
      with torch.inference_mode():
        action = policy(obs)
        if not torch.isfinite(action).all():
          raise RuntimeError("Nonfinite deterministic policy action")
        if diagnostics and control_step % 10 == 0:
          from mjlab.tasks.manipulation.mdp.task_geometry import between_fingers, finger_aperture, grasped, touching, tracking_goal, tracking_position
          from contact_grasp import opposed_grasp
          from mjlab.utils.lab_api.math import quat_apply
          active = ~finished
          grip = command.robot.data.site_pos_w[:,command.robot_cfg.site_ids].squeeze(1)
          pos = tracking_position(command.object)
          quat = command.robot.data.site_quat_w[:,command.robot_cfg.site_ids].squeeze(1)
          approach_axis = quat_apply(quat,torch.tensor([0.,0.,1.],device=device).expand(episodes,3))
          diagnostic_samples.append({
            "control_step": control_step, "active_episodes": int(active.sum()),
            "grasped_fraction": float(grasped(command)[active].float().mean()),
            "enclosed_fraction": float(between_fingers(command)[active].float().mean()),
            "enclosed_grasp_fraction": float((grasped(command)&between_fingers(command))[active].float().mean()),
            "opposed_grasp_fraction": float(opposed_grasp(command)[active].float().mean()),
            "robot_object_contact_fraction": float(touching(command,command.robot,command.object)[active].float().mean()),
            "aperture_mean": float(finger_aperture(command.robot)[active].mean()),
            "gripper_object_distance_mean": float(torch.linalg.vector_norm(grip-pos,dim=-1)[active].mean()),
            "gripper_goal_distance_mean": float(torch.linalg.vector_norm(grip-tracking_goal(command),dim=-1)[active].mean()),
            "gripper_down_alignment_mean": float((-approach_axis[:,2])[active].mean()),
            "object_height_mean": float((pos[:,2]-env.scene.env_origins[:,2])[active].mean()),
            "gripper_action_mean": float(action[active,-1].mean()),
            "absolute_action_max": float(action[active].abs().max()),
            "goal_error_mean": float(command.metrics["goal_error"][active].mean()),
          })
          if hasattr(command,"min_aperture"):
            diagnostic_samples[-1].update({
              "minimum_episode_aperture": float(command.min_aperture[active].min()),
              "valid_no_pinch_fraction": float((command.min_aperture[active]>command.cfg.aperture_min).float().mean()),
              "caged_fraction": float(command.metrics["caged"][active].mean()),
              "caged_progress_mean": float(command.caged_progress[active].mean()),
            })
        obs, reward, done, _ = wrapped.step(action)
      active = ~finished
      steps[active] += 1
      returns[active] += reward[active]
      newly_done = done.bool() & active
      if trace_dir is not None:
        state = snapshot()
        ids = newly_done.nonzero().flatten().cpu()
        for name in fields:
          state[name][ids] = terminal_states[name][ids]
        trace.append(state)
      successes[newly_done] = terminal_success[newly_done]
      for i in newly_done.nonzero(as_tuple=False).flatten().tolist():
        reasons[i] = "timeout" if bool(env.reset_time_outs[i]) else "terminated"
        termination_terms[i] = [name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[i])]
      finished |= newly_done
      if bool(finished.all()):
        break
    if not bool(finished.all()):
      raise RuntimeError(f"{int((~finished).sum())} first episodes did not finish")
    records = [
      {"env_id": i, "success": bool(successes[i]), "steps": int(steps[i]),
       "return": float(returns[i]), "reason": reasons[i], "termination_terms": termination_terms[i]}
      for i in range(episodes)
    ]
    count = int(successes.sum())
    if trace_dir is not None:
      np.savez_compressed(trace_dir / "trace.npz", origins=env.scene.env_origins.cpu().numpy(),
                          **model_snapshot,
                          **{name:torch.stack([state[name] for state in trace]).numpy() for name in fields})
    return {
      "task": task, "checkpoint": str(checkpoint.resolve()),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "seed": seed, "episodes": episodes, "successes": count,
      "success_rate": count / episodes, "deterministic": True,
      "first_episode_only": True, "records": records,
      "protocol": "terminal_first_episode_v1", "interface": "franka_shared_60_v2",
      "recipe": recipe, "episode_length_s": cfg.episode_length_s,
      "elliptic_hessian_compat": cfg.sim.elliptic_hessian_compat,
      "free_body_implicitfast_compat": cfg.sim.free_body_implicitfast_compat,
      "gravity": list(cfg.sim.mujoco.gravity),
      "agent_config_sha256": hashlib.sha256(agent_path.read_bytes()).hexdigest() if agent_path.exists() else None,
      "termination_counts": {name: sum(name in terms for terms in termination_terms) for name in env.termination_manager.active_terms},
      "trace_dir": str(trace_dir.resolve()) if trace_dir is not None else None,
      "trace_initial_model_fields": sorted(model_snapshot),
      "diagnostic_samples": diagnostic_samples,
    }
  finally:
    wrapped.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", required=True, choices=active_cl_tasks())
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--episodes", type=int, default=128)
  parser.add_argument("--seed", type=int, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--no-wandb", action="store_true")
  parser.add_argument("--trace-dir", type=Path)
  parser.add_argument("--diagnostics", action="store_true", help="Sample first-episode grasp/contact diagnostics for object tasks")
  args = parser.parse_args()
  if args.episodes < 1 or args.output.exists():
    parser.error("Episodes must be positive and output must not already exist")
  result = evaluate(args.task, args.checkpoint, args.episodes, args.seed, args.trace_dir, args.diagnostics)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(f"{result['successes']}/{result['episodes']} = {result['success_rate']:.3f}")
  if not args.no_wandb:
    from publish_evaluation import publish
    publish(args.output)


if __name__ == "__main__":
  main()
