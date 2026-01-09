from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import franka_lift_cube_env_cfg, franka_open_door_env_cfg, franka_open_drawer_env_cfg, franka_push_button_env_cfg
from .rl_cfg import franka_lift_cube_ppo_runner_cfg, franka_open_door_ppo_runner_cfg, franka_open_drawer_ppo_runner_cfg, franka_push_button_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Franka",
  env_cfg=franka_lift_cube_env_cfg(),
  play_env_cfg=franka_lift_cube_env_cfg(play=True),
  test_env_cfg=franka_lift_cube_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=franka_lift_cube_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Open-Door-Franka",
  env_cfg=franka_open_door_env_cfg(),
  play_env_cfg=franka_open_door_env_cfg(play=True),
  test_env_cfg=franka_open_door_env_cfg(test=True),
  rl_cfg=franka_open_door_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Open-Drawer-Franka",
  env_cfg=franka_open_drawer_env_cfg(),
  play_env_cfg=franka_open_drawer_env_cfg(play=True),
  test_env_cfg=franka_open_drawer_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=franka_open_drawer_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Push-Button-Franka",
  env_cfg=franka_push_button_env_cfg(),
  play_env_cfg=franka_push_button_env_cfg(play=True),
  test_env_cfg=franka_push_button_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=franka_push_button_ppo_runner_cfg(),
)
