from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import yam_lift_cube_env_cfg, yam_lift_cylinder_env_cfg, yam_push_cuboid_env_cfg, yam_push_disc_env_cfg
from .rl_cfg import yam_lift_cube_ppo_runner_cfg, yam_lift_cylinder_ppo_runner_cfg, yam_push_cuboid_ppo_runner_cfg, yam_push_disc_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam",
  env_cfg=yam_lift_cube_env_cfg(),
  play_env_cfg=yam_lift_cube_env_cfg(play=True),
  test_env_cfg=yam_lift_cube_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=yam_lift_cube_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cylinder-Yam",
  env_cfg=yam_lift_cylinder_env_cfg(),
  play_env_cfg=yam_lift_cylinder_env_cfg(play=True),
  test_env_cfg=yam_lift_cylinder_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=yam_lift_cylinder_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Push-Cuboid-Yam",
  env_cfg=yam_push_cuboid_env_cfg(),
  play_env_cfg=yam_push_cuboid_env_cfg(play=True),
  test_env_cfg=yam_push_cuboid_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=yam_push_cuboid_ppo_runner_cfg(),
)

register_mjlab_task(
  task_id="Mjlab-Push-Disc-Yam",
  env_cfg=yam_push_disc_env_cfg(),
  play_env_cfg=yam_push_disc_env_cfg(play=True),
  test_env_cfg=yam_push_disc_env_cfg(play=True),  # TODO: Add test parameter
  rl_cfg=yam_push_disc_ppo_runner_cfg(),
)
