"""PPO runner configs for the floating LEAP hand (Class-C) tasks."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
)


def _leap_ppo_runner_cfg(experiment_name: str) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    wandb_project="continual_mjlab",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )


def leap_reach_target_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _leap_ppo_runner_cfg("leap_reach_target")


def leap_lift_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _leap_ppo_runner_cfg("leap_lift_cube")


def leap_stack_cube_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _leap_ppo_runner_cfg("leap_stack_cube")


def leap_peg_insertion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _leap_ppo_runner_cfg("leap_peg_insertion")


def leap_lift_sphere_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _leap_ppo_runner_cfg("leap_lift_sphere")
