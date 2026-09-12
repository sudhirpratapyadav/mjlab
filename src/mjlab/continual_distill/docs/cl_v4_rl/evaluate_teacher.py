"""Strict first-episode evaluation of one RSL-RL task checkpoint."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.manipulation.benchmark import active_cl_tasks
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg


def evaluate(task: str, checkpoint: Path, episodes: int, seed: int) -> dict:
  cfg = load_env_cfg(task)
  cfg.scene.num_envs = episodes
  cfg.seed = seed
  agent = load_rl_cfg(task)
  agent.seed = seed
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  env = ManagerBasedRlEnv(cfg, device=device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
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

    def capture_reset(env_ids):
      command._update_metrics()
      terminal_success[env_ids] = command.compute_success()[env_ids].bool()
      return original_reset(env_ids)

    env._reset_idx = capture_reset
    runner = OnPolicyRunner(wrapped, asdict(agent), None, device=device)
    runner.load(str(checkpoint), load_optimizer=False, map_location=device)
    policy = runner.get_inference_policy()
    obs = wrapped.get_observations()
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    successes = torch.zeros_like(finished)
    steps = torch.zeros(episodes, dtype=torch.int32, device=device)
    returns = torch.zeros(episodes, device=device)
    reasons = [None] * episodes
    for _ in range(env.max_episode_length + 1):
      with torch.inference_mode():
        action = policy(obs)
        obs, reward, done, _ = wrapped.step(action)
      active = ~finished
      steps[active] += 1
      returns[active] += reward[active]
      newly_done = done.bool() & active
      successes[newly_done] = terminal_success[newly_done]
      for i in newly_done.nonzero(as_tuple=False).flatten().tolist():
        reasons[i] = "timeout" if bool(env.reset_time_outs[i]) else "terminated"
      finished |= newly_done
      if bool(finished.all()):
        break
    if not bool(finished.all()):
      raise RuntimeError(f"{int((~finished).sum())} first episodes did not finish")
    records = [
      {"env_id": i, "success": bool(successes[i]), "steps": int(steps[i]),
       "return": float(returns[i]), "reason": reasons[i]}
      for i in range(episodes)
    ]
    count = int(successes.sum())
    return {
      "task": task, "checkpoint": str(checkpoint.resolve()),
      "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
      "seed": seed, "episodes": episodes, "successes": count,
      "success_rate": count / episodes, "deterministic": True,
      "first_episode_only": True, "records": records,
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
  args = parser.parse_args()
  if args.episodes < 1 or args.output.exists():
    parser.error("Episodes must be positive and output must not already exist")
  result = evaluate(args.task, args.checkpoint, args.episodes, args.seed)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2) + "\n")
  print(f"{result['successes']}/{result['episodes']} = {result['success_rate']:.3f}")


if __name__ == "__main__":
  main()
