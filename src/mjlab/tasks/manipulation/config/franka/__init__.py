from mjlab.tasks.manipulation.taxonomy import (
  Embodiment,
  Fragility,
  SkillFamily,
  TaskTaxonomy,
)
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import (
  franka_turn_lever_env_cfg,
  franka_rotate_valve_env_cfg,
  franka_flip_switch_env_cfg,
  franka_slide_window_env_cfg,
  franka_open_lid_env_cfg,
  franka_place_in_container_env_cfg,
  franka_reorient_object_env_cfg,
  franka_tool_pull_env_cfg,
)
from .rl_cfg import (
  franka_turn_lever_ppo_runner_cfg,
  franka_rotate_valve_ppo_runner_cfg,
  franka_flip_switch_ppo_runner_cfg,
  franka_slide_window_ppo_runner_cfg,
  franka_open_lid_ppo_runner_cfg,
  franka_place_in_container_ppo_runner_cfg,
  franka_reorient_object_ppo_runner_cfg,
  franka_tool_pull_ppo_runner_cfg,
)
from .env_cfgs import franka_lift_cube_env_cfg, franka_lift_cylinder_env_cfg, franka_lift_ellipsoid_env_cfg, franka_lift_sphere_env_cfg, franka_open_door_env_cfg, franka_open_drawer_env_cfg, franka_peg_insertion_env_cfg, franka_push_button_env_cfg, franka_push_cuboid_env_cfg, franka_push_disc_env_cfg, franka_reach_target_env_cfg, franka_stack_cube_env_cfg
from .rl_cfg import franka_lift_cube_ppo_runner_cfg, franka_lift_cylinder_ppo_runner_cfg, franka_lift_ellipsoid_ppo_runner_cfg, franka_lift_sphere_ppo_runner_cfg, franka_open_door_ppo_runner_cfg, franka_open_drawer_ppo_runner_cfg, franka_peg_insertion_ppo_runner_cfg, franka_push_button_ppo_runner_cfg, franka_push_cuboid_ppo_runner_cfg, franka_push_disc_ppo_runner_cfg, franka_reach_target_ppo_runner_cfg, franka_stack_cube_ppo_runner_cfg

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
    "goal (tracks the base object). Higher-fragility pick_place than lift.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Sphere-Franka",
  env_cfg=franka_lift_sphere_env_cfg(),
  play_env_cfg=franka_lift_sphere_env_cfg(play=True),
  test_env_cfg=franka_lift_sphere_env_cfg(test=True),
  rl_cfg=franka_lift_sphere_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Grasp-and-lift a sphere; rolls, no flat faces — hardest free-object grasp "
    "geometry (grasp-geometry generalization).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Lift-Ellipsoid-Franka",
  env_cfg=franka_lift_ellipsoid_env_cfg(),
  play_env_cfg=franka_lift_ellipsoid_env_cfg(play=True),
  test_env_cfg=franka_lift_ellipsoid_env_cfg(test=True),
  rl_cfg=franka_lift_ellipsoid_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Grasp-and-lift an elongated ellipsoid; grasp success depends on approach "
    "orientation (grasp across the short axis).",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Peg-Insertion-Franka",
  env_cfg=franka_peg_insertion_env_cfg(),
  play_env_cfg=franka_peg_insertion_env_cfg(play=True),
  test_env_cfg=franka_peg_insertion_env_cfg(test=True),
  rl_cfg=franka_peg_insertion_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.INSERTION,
    fragility=Fragility.DEXTEROUS,
    source="native",
    contact_rich=True,
    notes="Insert a peg into a hole board: grasp + tight alignment into a ~3cm hole. "
    "Reuses the Stack MDP with tighter xy tolerance. New 'insertion' skill; most "
    "fragile tier (multi-contact precision).",
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


##
# Class A motion-profile expansion (docs/benchmark/CLASS_A_EXPANSION.md).
# Counted as distinct MOTION PROFILES, not object variants: Class A goes 8 -> 16.
##

register_mjlab_task(
  task_id="Mjlab-Turn-Lever-Franka",
  env_cfg=franka_turn_lever_env_cfg(),
  play_env_cfg=franka_turn_lever_env_cfg(play=True),
  test_env_cfg=franka_turn_lever_env_cfg(test=True),
  rl_cfg=franka_turn_lever_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=True,
    notes="Motion profile: WRIST ROTATION about the approach axis. The lever hinge points at the robot, so turning it rotates the wrist rather than pulling with the arm (contrast Open-Door, a whole-arm hinge arc). Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Rotate-Valve-Franka",
  env_cfg=franka_rotate_valve_env_cfg(),
  play_env_cfg=franka_rotate_valve_env_cfg(play=True),
  test_env_cfg=franka_rotate_valve_env_cfg(test=True),
  rl_cfg=franka_rotate_valve_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.DEXTEROUS,
    source="native",
    contact_rich=True,
    notes="Motion profile: MULTI-CYCLE ROTATION WITH REGRASP. The 270deg target exceeds the wrist range from any single grasp, so the policy must turn, release, re-grasp the opposite spoke and continue — the only Class A task needing a regrasp cycle. Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Flip-Switch-Franka",
  env_cfg=franka_flip_switch_env_cfg(),
  play_env_cfg=franka_flip_switch_env_cfg(play=True),
  test_env_cfg=franka_flip_switch_env_cfg(test=True),
  rl_cfg=franka_flip_switch_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=True,
    notes="Motion profile: BALLISTIC COMMIT past a detent. A spring pushes the toggle back until it crosses centre, so quasi-static servoing fails and success is a state flip, not a displacement threshold (contrast Push-Button, a compliant press). Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Slide-Window-Franka",
  env_cfg=franka_slide_window_env_cfg(),
  play_env_cfg=franka_slide_window_env_cfg(play=True),
  test_env_cfg=franka_slide_window_env_cfg(test=True),
  rl_cfg=franka_slide_window_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=False,
    notes="Motion profile: LATERAL FACE-PUSH. Same joint type as Open-Drawer but along the robot's lateral axis, and the drawer's proven fingertip-hook strategy does not transfer (you push a face, not hook a bar). Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Open-Lid-Franka",
  env_cfg=franka_open_lid_env_cfg(),
  play_env_cfg=franka_open_lid_env_cfg(play=True),
  test_env_cfg=franka_open_lid_env_cfg(test=True),
  rl_cfg=franka_open_lid_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.ARTICULATION,
    fragility=Fragility.MILD_CONTACT,
    source="native",
    contact_rich=True,
    notes="Motion profile: VERTICAL ARC UNDER GRAVITY. Gravity opposes the motion throughout and the lid falls shut if released, unlike the door's gravity-neutral vertical-axis swing. NOTE: runs with gravity ENABLED, unlike the other articulation tasks. Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Place-In-Container-Franka",
  env_cfg=franka_place_in_container_env_cfg(),
  play_env_cfg=franka_place_in_container_env_cfg(play=True),
  test_env_cfg=franka_place_in_container_env_cfg(test=True),
  rl_cfg=franka_place_in_container_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Success shape: CONTAINMENT + RELEASE (inside the footprint, below the rim, settled) rather than the stable-contact predicate used by Stack. The bin walls also make the approach non-monotonic — a straight-line reach to the goal hits a wall. Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Reorient-Object-Franka",
  env_cfg=franka_reorient_object_env_cfg(),
  play_env_cfg=franka_reorient_object_env_cfg(play=True),
  test_env_cfg=franka_reorient_object_env_cfg(test=True),
  rl_cfg=franka_reorient_object_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.PICK_PLACE,
    fragility=Fragility.PRECISION_GRASP,
    source="native",
    contact_rich=True,
    notes="Success shape: ORIENTATION, not position — the only Class A task scored on angular alignment (stand a lying cylinder upright). Position enters only as a loose anti-fling drift bound. Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)

register_mjlab_task(
  task_id="Mjlab-Tool-Pull-Franka",
  env_cfg=franka_tool_pull_env_cfg(),
  play_env_cfg=franka_tool_pull_env_cfg(play=True),
  test_env_cfg=franka_tool_pull_env_cfg(test=True),
  rl_cfg=franka_tool_pull_ppo_runner_cfg(),
  taxonomy=TaskTaxonomy(
    embodiment=Embodiment.ARM_GRIPPER,
    skill=SkillFamily.TOOL_USE,
    fragility=Fragility.DEXTEROUS,
    source="native",
    contact_rich=True,
    notes="Motion profile: TWO-STAGE TOOL USE — grasp a stick, then drag a puck that spawns beyond direct reach back into the near zone. Contact that matters happens at the tool tip, not the fingertips. First TOOL_USE task in the benchmark. Structural (smoke) gate only; reward shaping not yet train-validated.",
  ),
)
