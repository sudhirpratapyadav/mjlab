#!/usr/bin/env python3
"""
Test PyTorch policy (RSL-RL checkpoint) in environment with evaluation metrics.

Loads a trained PyTorch policy checkpoint, runs test episodes in parallel environments,
and reports detailed evaluation metrics including success rate, rewards, and episode lengths.

Usage:
    python test_policy.py \
        --checkpoint /path/to/model.pt \
        --task Mjlab-Lift-Cube-Franka \
        --num-envs 16 \
        --num-episodes 10

    This will run 10 episodes per environment = 160 total episodes
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tqdm import tqdm

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def evaluate_policy(
    env,
    policy,
    critic,
    num_episodes_per_env: int,
    episode_length: int,
    num_envs: int,
):
    """Run policy evaluation and collect metrics.

    Args:
        env: mjlab environment (vectorized)
        policy: Trained policy (actor)
        critic: Value function (critic)
        num_episodes_per_env: Number of episodes to run per environment
        episode_length: Maximum steps per episode
        num_envs: Number of parallel environments

    Returns:
        Dictionary with evaluation metrics
    """
    total_episodes = num_episodes_per_env * num_envs

    print(f"\n{'='*80}")
    print(f"Running Evaluation")
    print(f"{'='*80}")
    print(f"  Episodes per env: {num_episodes_per_env}")
    print(f"  Parallel envs: {num_envs}")
    print(f"  Total episodes: {total_episodes}")
    print(f"  Max episode length: {episode_length}")

    # Run exactly num_episodes_per_env episodes per environment
    num_resets = num_episodes_per_env

    # Storage for metrics (pre-allocate for speed)
    all_episode_returns = np.zeros(total_episodes)
    all_episode_lengths = np.zeros(total_episodes, dtype=int)
    all_episode_successes = np.zeros(total_episodes)
    all_episode_rewards = []  # Store per-step rewards for each episode (for plotting)
    all_episode_values = []  # Store per-step value estimates (for plotting)

    episode_idx = 0

    with tqdm(total=num_resets, desc="Evaluating policy", unit="batch") as pbar:
        for batch_idx in range(num_resets):
            # Reset environment for each batch
            obs = env.reset()
            if isinstance(obs, tuple):
                obs = obs[0]

            # Detect device from first step
            device = None
            batch_returns = None
            batch_lengths = None
            batch_dones = None
            batch_reward_trajectories = None

            # Run full episode
            for step in range(episode_length):
                # Get action from policy
                with torch.no_grad():
                    actions = policy(obs)

                # Step environment
                obs, rewards, dones, infos = env.step(actions)

                # Initialize tensors on first step (using correct device)
                if device is None:
                    device = rewards.device
                    batch_returns = torch.zeros(num_envs, device=device)
                    batch_lengths = torch.zeros(num_envs, dtype=torch.int32, device=device)
                    batch_dones = torch.zeros(num_envs, dtype=torch.bool, device=device)
                    batch_reward_trajectories = torch.zeros((num_envs, episode_length), device=device)
                    batch_value_trajectories = torch.zeros((num_envs, episode_length), device=device)

                # Compute value estimates (use critic obs if available, else policy obs)
                with torch.no_grad():
                    critic_obs = obs.get("critic", obs.get("policy", obs))
                    values = critic(critic_obs)

                # Store rewards and values
                batch_reward_trajectories[:, step] = rewards
                batch_value_trajectories[:, step] = values.squeeze(-1)

                # Only accumulate if not already done (torch operations)
                active_mask = ~batch_dones
                batch_returns += rewards * active_mask.float()
                batch_lengths += active_mask.int()

                # Mark newly done episodes (cast dones to bool)
                batch_dones |= dones.bool()

                # If all environments done, exit early
                if batch_dones.all():
                    break

            # Extract success from final info (after episode ends)
            batch_successes = torch.zeros(num_envs, device=device)
            if infos is not None and "log" in infos:
                log_info = infos["log"]
                if isinstance(log_info, dict):
                    for key, value in log_info.items():
                        if "episode_success" in key.lower():
                            if isinstance(value, torch.Tensor):
                                batch_successes = value.to(device)
                            else:
                                batch_successes = torch.full((num_envs,), float(value), device=device)
                            break

            # Convert to numpy only once per batch (move to CPU first if needed)
            batch_returns_np = batch_returns.cpu().numpy()
            batch_lengths_np = batch_lengths.cpu().numpy()
            batch_successes_np = batch_successes.cpu().numpy()
            batch_rewards_np = batch_reward_trajectories.cpu().numpy()
            batch_values_np = batch_value_trajectories.cpu().numpy()

            # Store batch results
            all_episode_returns[episode_idx:episode_idx + num_envs] = batch_returns_np
            all_episode_lengths[episode_idx:episode_idx + num_envs] = batch_lengths_np
            all_episode_successes[episode_idx:episode_idx + num_envs] = batch_successes_np

            # Store reward and value trajectories (trim to actual episode length)
            for env_i in range(num_envs):
                actual_length = batch_lengths_np[env_i]
                all_episode_rewards.append(batch_rewards_np[env_i, :actual_length])
                all_episode_values.append(batch_values_np[env_i, :actual_length])

            episode_idx += num_envs

            pbar.update(1)

    return {
        'episode_returns': all_episode_returns,
        'episode_lengths': all_episode_lengths,
        'episode_successes': all_episode_successes,
        'episode_rewards': all_episode_rewards,  # List of arrays (one per episode)
        'episode_values': all_episode_values,  # List of arrays (one per episode)
        'num_episodes': total_episodes,
    }


def plot_reward_trajectories(metrics: dict, save_path: Path):
    """Plot reward vs step for all episodes.

    Args:
        metrics: Dictionary with evaluation metrics
        save_path: Path to save the plot
    """
    episode_rewards = metrics['episode_rewards']
    num_episodes = len(episode_rewards)

    plt.figure(figsize=(12, 6))

    # Plot each episode's reward trajectory (no legend)
    for rewards in episode_rewards:
        steps = np.arange(len(rewards))
        plt.plot(steps, rewards, alpha=0.3, linewidth=0.5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title(f'Reward Trajectories ({num_episodes} episodes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*80}")
    print("Reward Trajectory Plot Saved")
    print(f"{'='*80}")
    print(f"Location: {save_path}")
    print(f"{'='*80}\n")


def plot_value_trajectories(metrics: dict, save_path: Path):
    """Plot value estimates vs step for all episodes.

    Args:
        metrics: Dictionary with evaluation metrics
        save_path: Path to save the plot
    """
    episode_values = metrics['episode_values']
    num_episodes = len(episode_values)

    plt.figure(figsize=(12, 6))

    # Plot each episode's value trajectory (no legend)
    for values in episode_values:
        steps = np.arange(len(values))
        plt.plot(steps, values, alpha=0.3, linewidth=0.5)

    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Value Estimate V(s)', fontsize=12)
    plt.title(f'Value Function Trajectories ({num_episodes} episodes)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n{'='*80}")
    print("Value Trajectory Plot Saved")
    print(f"{'='*80}")
    print(f"Location: {save_path}")
    print(f"{'='*80}\n")


def print_evaluation_summary(metrics: dict):
    """Print formatted evaluation summary.

    Args:
        metrics: Dictionary with evaluation metrics
    """
    returns = metrics['episode_returns']
    lengths = metrics['episode_lengths']
    successes = metrics['episode_successes']
    num_episodes = metrics['num_episodes']

    print(f"\n{'='*80}")
    print("Evaluation Summary")
    print(f"{'='*80}")

    # Episode-level metrics
    print(f"\nEpisode Metrics (n={num_episodes}):")
    print(f"  Return:        {returns.mean():8.3f} ± {returns.std():6.3f}  "
          f"[min: {returns.min():7.3f}, max: {returns.max():7.3f}]")
    print(f"  Length:        {lengths.mean():8.1f} ± {lengths.std():6.1f}  "
          f"[min: {lengths.min():3d}, max: {lengths.max():3d}]")

    # Success rate
    success_rate = successes.mean()
    num_successes = int(successes.sum())
    print(f"  Success Rate:  {success_rate:8.3f} ({num_successes}/{num_episodes} episodes)")

    # Percentiles
    print(f"\nReturn Percentiles:")
    for percentile in [0, 25, 50, 75, 100]:
        val = np.percentile(returns, percentile)
        print(f"  {percentile:3d}th:         {val:8.3f}")

    print(f"{'='*80}\n")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test PyTorch policy with evaluation metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to PyTorch checkpoint (.pt file)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="mjlab task ID (e.g., Mjlab-Lift-Cube-Franka)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes per environment (total = num_episodes * num_envs)",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=16,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=None,
        help="Maximum steps per episode (default: from env config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda:0 if available)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save results as numpy .npz file in plots/ directory",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    configure_torch_backends()

    # Validate checkpoint
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)

    print(f"{'='*80}")
    print("PyTorch Policy Evaluation")
    print(f"{'='*80}")
    print(f"Task:            {args.task}")
    print(f"Checkpoint:      {checkpoint_path.name}")
    print(f"Episodes/env:    {args.num_episodes}")
    print(f"Num envs:        {args.num_envs}")
    print(f"Total episodes:  {args.num_episodes * args.num_envs}")

    # Setup device
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device:     {device}")

    # Load environment config
    env_cfg = load_env_cfg(args.task, play=False)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed

    # Load agent config
    agent_cfg = load_rl_cfg(args.task)

    # Create environment
    print(f"\nLoading environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Get episode length
    episode_length = args.episode_length or env.unwrapped.max_episode_length
    print(f"Episode length: {episode_length}")

    # Print environment info
    obs_sample = env.reset()
    # Handle both old and new gym API (reset returns obs or (obs, info))
    if isinstance(obs_sample, tuple):
        obs_sample = obs_sample[0]
    obs_dim = obs_sample.shape[-1]
    action_dim = env.unwrapped.action_space.shape[-1]
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {action_dim}")

    # Load policy
    print(f"\nLoading policy from checkpoint...")
    agent_cfg.logger = "tensorboard"  # Disable wandb
    runner_cls = load_runner_cls(args.task) or OnPolicyRunner
    log_dir = checkpoint_path.parent
    runner = runner_cls(env, asdict(agent_cfg), log_dir=str(log_dir), device=device)
    runner.load(str(checkpoint_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Get critic for value estimates
    # In RSL-RL, the actor-critic model is at runner.alg.policy
    actor_critic = runner.alg.policy
    critic = actor_critic.critic
    print(f"✓ Policy and critic loaded successfully")

    # Run evaluation
    metrics = evaluate_policy(
        env=env,
        policy=policy,
        critic=critic,
        num_episodes_per_env=args.num_episodes,
        episode_length=episode_length,
        num_envs=args.num_envs,
    )

    # Print results
    print_evaluation_summary(metrics)

    # Generate and save plots in plots/ directory
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    # Reward trajectories plot
    reward_plot_filename = f"{args.task.replace('-', '_')}_{checkpoint_path.stem}_reward_trajectories.png"
    reward_plot_path = plots_dir / reward_plot_filename
    plot_reward_trajectories(metrics, reward_plot_path)

    # Value trajectories plot
    value_plot_filename = f"{args.task.replace('-', '_')}_{checkpoint_path.stem}_value_trajectories.png"
    value_plot_path = plots_dir / value_plot_filename
    plot_value_trajectories(metrics, value_plot_path)

    # Save results if requested
    if args.save_results:
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        save_filename = f"{args.task.replace('-', '_')}_{checkpoint_path.stem}_results.npz"
        save_path = plots_dir / save_filename

        # Prepare data to save
        save_data = {
            'episode_returns': metrics['episode_returns'],
            'episode_lengths': metrics['episode_lengths'],
            'episode_successes': metrics['episode_successes'],
            'checkpoint_path': str(checkpoint_path),
            'task': args.task,
            'num_episodes': metrics['num_episodes'],
            'num_episodes_per_env': args.num_episodes,
            'num_envs': args.num_envs,
            'episode_length': episode_length,
            'seed': args.seed,
        }

        np.savez(save_path, **save_data)

        # Print detailed save summary
        print(f"\n{'='*80}")
        print("Results Saved")
        print(f"{'='*80}")
        print(f"File: {save_path}")
        print(f"\nSaved arrays:")
        print(f"  episode_returns:    {save_data['episode_returns'].shape}  (mean={save_data['episode_returns'].mean():.3f})")
        print(f"  episode_lengths:    {save_data['episode_lengths'].shape}  (mean={save_data['episode_lengths'].mean():.1f})")
        print(f"  episode_successes:  {save_data['episode_successes'].shape}  (rate={save_data['episode_successes'].mean():.3f})")
        print(f"\nMetadata:")
        print(f"  checkpoint:         {checkpoint_path.name}")
        print(f"  task:               {args.task}")
        print(f"  num_episodes:       {save_data['num_episodes']} (total)")
        print(f"  episodes_per_env:   {save_data['num_episodes_per_env']}")
        print(f"  num_envs:           {save_data['num_envs']}")
        print(f"  episode_length:     {save_data['episode_length']}")
        print(f"  seed:               {save_data['seed']}")
        print(f"\nTo load in Python:")
        print(f"  data = np.load('{save_path}')")
        print(f"  returns = data['episode_returns']")
        print(f"  successes = data['episode_successes']")
        print(f"{'='*80}\n")

    # Cleanup
    env.close()
    print("Done!")


if __name__ == "__main__":
    main()
