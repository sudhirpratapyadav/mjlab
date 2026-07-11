from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import franka_lift_cube_env_cfg, franka_lift_cylinder_env_cfg, franka_open_door_env_cfg, franka_open_drawer_env_cfg, franka_push_button_env_cfg, franka_push_cuboid_env_cfg, franka_push_disc_env_cfg, franka_reach_target_env_cfg, franka_stack_cube_env_cfg
from .rl_cfg import franka_lift_cube_ppo_runner_cfg, franka_lift_cylinder_ppo_runner_cfg, franka_open_door_ppo_runner_cfg, franka_open_drawer_ppo_runner_cfg, franka_push_button_ppo_runner_cfg, franka_push_cuboid_ppo_runner_cfg, franka_push_disc_ppo_runner_cfg, franka_reach_target_ppo_runner_cfg, franka_stack_cube_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Franka",
  env_cfg=franka_lift_cube_env_cfg(),
  play_env_cfg=franka_lift_cube_env_cfg(play=True),
  test_env_cfg=franka_lift_cube_env_cfg(test=True),
  rl_cfg=franka_lift_cube_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Grasp-and-lift; the canonical fragile task in FINDINGS.md (catastrophic "
    "forgetting under regularization-only CL).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cylinder-Franka",
  env_cfg=franka_lift_cylinder_env_cfg(),
  play_env_cfg=franka_lift_cylinder_env_cfg(play=True),
  test_env_cfg=franka_lift_cylinder_env_cfg(test=True),
  rl_cfg=franka_lift_cylinder_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Grasp-and-lift a cylinder; rolls and lacks flat faces, so grasp alignment "
    "is less forgiving than the cube (grasp-geometry generalization).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Stack-Cube-Franka",
  env_cfg=franka_stack_cube_env_cfg(),
  play_env_cfg=franka_stack_cube_env_cfg(play=True),
  test_env_cfg=franka_stack_cube_env_cfg(test=True),
  rl_cfg=franka_stack_cube_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Stack a cube on a cuboid base: grasp + precise place + release. Dynamic "
    "goal (tracks base). Higher-fragility pick_place than lift. STRUCTURALLY validated "
    "(builds/steps); NOT yet train-validated — did not solve in 400 PPO iters "
    "(episode_success ~0.01). Multi-stage task likely needs longer training (lift-cube "
    "uses 5000) and/or reward tuning; deferred (see docs/benchmark/LOG.md 2026-07-11).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Reach-Target-Franka",
  env_cfg=franka_reach_target_env_cfg(),
  play_env_cfg=franka_reach_target_env_cfg(play=True),
  test_env_cfg=franka_reach_target_env_cfg(test=True),
  rl_cfg=franka_reach_target_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.REACH,
    fragility=Fragility.PLANAR,
    source="native",
    contact_rich=False,
    notes="Move the gripper to a sampled 3D target; dense monotonic reward, no "
    "contact. Anchors the low-fragility end of the axis (new 'reach' skill family).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Open-Door-Franka",
  env_cfg=franka_open_door_env_cfg(),
  play_env_cfg=franka_open_door_env_cfg(play=True),
  test_env_cfg=franka_open_door_env_cfg(test=True),
  rl_cfg=franka_open_door_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=True,
  ),
)

register_mjlab_task(
  task_id="Mjlab-Open-Drawer-Franka",
  env_cfg=franka_open_drawer_env_cfg(),
  play_env_cfg=franka_open_drawer_env_cfg(play=True),
  test_env_cfg=franka_open_drawer_env_cfg(test=True),
  rl_cfg=franka_open_drawer_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=True,
  ),
)

register_mjlab_task(
  task_id="Mjlab-Push-Button-Franka",
  env_cfg=franka_push_button_env_cfg(),
  play_env_cfg=franka_push_button_env_cfg(play=True),
  test_env_cfg=franka_push_button_env_cfg(test=True),
  rl_cfg=franka_push_button_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.PLANAR,
    source="native",
    notes="Press a button; forgiving, retained near-1.0 through sequences.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Push-Cuboid-Franka",
  env_cfg=franka_push_cuboid_env_cfg(),
  play_env_cfg=franka_push_cuboid_env_cfg(play=True),
  test_env_cfg=franka_push_cuboid_env_cfg(test=True),
  rl_cfg=franka_push_cuboid_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PLANAR_PUSH,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    notes="Task-0 fragility case in FINDINGS.md (retains ~0.6-0.84).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Push-Disc-Franka",
  env_cfg=franka_push_disc_env_cfg(),
  play_env_cfg=franka_push_disc_env_cfg(play=True),
  test_env_cfg=franka_push_disc_env_cfg(test=True),
  rl_cfg=franka_push_disc_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PLANAR_PUSH,
    fragility=Fragility.MILD_CONTACT,
    source="native",
  ),
)
