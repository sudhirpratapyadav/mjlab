import pytest
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


@pytest.mark.parametrize("task", sorted(CLASSICAL_POLICIES))
def test_task_rates_are_200hz_physics_50hz_control(task):
  cfg = load_env_cfg(task, play=False)
  assert cfg.sim.mujoco.timestep == 0.005
  assert cfg.decimation == 4
  assert cfg.sim.mujoco.timestep * cfg.decimation == 0.02
  assert cfg.sim.mujoco.integrator == "implicitfast"


@pytest.mark.parametrize("physics_dt,decimation", [(0.005, 4), (0.001, 20)])
def test_actual_simulation_clock_advances_20ms_per_control(physics_dt, decimation):
  cfg = load_env_cfg("Mjlab-Reach-Target-Franka", test=True)
  cfg.scene.num_envs = 1
  cfg.sim.mujoco.timestep = physics_dt
  cfg.decimation = decimation
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    before = env.sim.data.time.detach().clone()
    counter = env._sim_step_counter
    for _ in range(10):
      env.step(torch.zeros((1, env.action_manager.total_action_dim)))
    assert env._sim_step_counter - counter == 10 * decimation
    torch.testing.assert_close(
      env.sim.data.time.detach() - before,
      torch.full_like(before, 0.2),
      atol=1e-5,
      rtol=0,
    )
    assert torch.isfinite(env.sim.data.qpos).all()
  finally:
    env.close()
