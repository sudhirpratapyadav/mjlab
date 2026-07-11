"""Registration of Class-C (floating LEAP hand) manipulation tasks."""

from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import leap_lift_cube_env_cfg, leap_reach_target_env_cfg
from .rl_cfg import leap_lift_cube_ppo_runner_cfg, leap_reach_target_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Reach-Target-Leap",
  env_cfg=leap_reach_target_env_cfg(),
  play_env_cfg=leap_reach_target_env_cfg(play=True),
  test_env_cfg=leap_reach_target_env_cfg(test=True),
  rl_cfg=leap_reach_target_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.FLOATING_HAND,
    skill=SkillFamily.REACH,
    fragility=Fragility.PLANAR,
    source="native",
    contact_rich=False,
    notes="Floating LEAP hand reaching its palm to a 3D target (actuated 6-DoF base). "
    "First Class-C task; reuses the arm reach base with the hand embodiment.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Leap",
  env_cfg=leap_lift_cube_env_cfg(),
  play_env_cfg=leap_lift_cube_env_cfg(play=True),
  test_env_cfg=leap_lift_cube_env_cfg(test=True),
  rl_cfg=leap_lift_cube_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.FLOATING_HAND,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Floating LEAP hand grasping and lifting a cube with its fingers "
    "(dexterous grasp). Reuses the arm lift base with the hand embodiment.",
  ),
)
