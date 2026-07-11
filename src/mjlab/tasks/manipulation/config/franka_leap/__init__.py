"""Registration of Class-B (Franka arm + LEAP hand) manipulation tasks."""

from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  franka_leap_lift_cube_env_cfg,
  franka_leap_reach_target_env_cfg,
  franka_leap_stack_cube_env_cfg,
)
from .rl_cfg import (
  franka_leap_lift_cube_ppo_runner_cfg,
  franka_leap_reach_target_ppo_runner_cfg,
  franka_leap_stack_cube_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-Reach-Target-Franka-Leap",
  env_cfg=franka_leap_reach_target_env_cfg(),
  play_env_cfg=franka_leap_reach_target_env_cfg(play=True),
  test_env_cfg=franka_leap_reach_target_env_cfg(test=True),
  rl_cfg=franka_leap_reach_target_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_HAND,
    skill=SkillFamily.REACH,
    fragility=Fragility.PLANAR,
    source="native",
    contact_rich=False,
    notes="Franka arm + LEAP hand reaching to a 3D target. First Class-B (arm + "
    "dexterous hand) task; reuses the arm reach base.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Franka-Leap",
  env_cfg=franka_leap_lift_cube_env_cfg(),
  play_env_cfg=franka_leap_lift_cube_env_cfg(play=True),
  test_env_cfg=franka_leap_lift_cube_env_cfg(test=True),
  rl_cfg=franka_leap_lift_cube_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_HAND,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Franka arm + LEAP hand grasping and lifting a cube (dexterous grasp on a "
    "full arm).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Stack-Cube-Franka-Leap",
  env_cfg=franka_leap_stack_cube_env_cfg(),
  play_env_cfg=franka_leap_stack_cube_env_cfg(play=True),
  test_env_cfg=franka_leap_stack_cube_env_cfg(test=True),
  rl_cfg=franka_leap_stack_cube_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_HAND,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Franka arm + LEAP hand stacking a cube on a cuboid base.",
  ),
)
