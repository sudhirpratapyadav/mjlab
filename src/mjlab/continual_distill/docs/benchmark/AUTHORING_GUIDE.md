# Task Authoring Guide

How to add a new task to the manipulation-diversity benchmark. Reverse-engineered from
the existing tasks and verified end-to-end locally (2026-07-11). Companion to PLAN.md.

## Mental model

mjlab is a manager-based framework. A task is assembled from reusable terms:

```
scene (robot + object entities + sensors)
  + observations (policy/critic groups of ObservationTermCfg)
  + actions       (ActionTermCfg — joint-space today)
  + commands      (CommandTermCfg — spawns goals, tracks success)
  + rewards       (RewardTermCfg)
  + terminations  (TerminationTermCfg)
  + events        (resets, domain randomization)
  = ManagerBasedRlEnvCfg
```

Two layers:
- **Base maker** (`tasks/manipulation/<skill>_env_cfg.py::make_<skill>_env_cfg`) builds
  a robot-agnostic base for a SKILL (lift, push, open-door, ...). Reuse an existing one
  if your task shares the skill; write a new base only for a new skill.
- **Concrete cfg** (`tasks/manipulation/config/<robot>/env_cfgs.py::<robot>_<task>_env_cfg`)
  specializes the base for a robot + specific object(s) + ranges.

The reusable terms live in `tasks/manipulation/mdp/` (observations.py, rewards.py,
terminations.py, commands.py). Check there before writing anything new.

## Recipe A — new task, EXISTING skill + EXISTING assets (cheapest; ~30 min)

This is the `Lift-Cube -> Lift-Cylinder` case (object swap). Steps:

1. **Pick the base maker** for the skill (e.g. `make_lift_object_env_cfg`).
2. **Add a concrete cfg** in `config/franka/env_cfgs.py`. Copy the closest sibling
   (e.g. `franka_lift_cube_env_cfg`) and change:
   - `cfg.scene.entities`: swap the object entity (`get_cube_cfg()` ->
     `get_cylinder_cfg()`) and its mocap goal.
   - `command.asset_name` = new object name.
   - EVERY object-referencing obs/reward/termination term's `object_asset_name` /
     `object_name` param → new object name. (Miss one and it silently reads the wrong
     body.) The object-referencing terms in the lift base are: obs `object_pos`,
     `object_quat`, `object_orientation`, `gripper_to_object`, `object_to_goal`,
     `goal_orientation_diff`; rewards `reach_object`, `move_object_to_goal`;
     termination `object_out_of_bounds`.
   - Keep the Franka specifics: site `"gripper"`, action scale `FRANKA_ACTION_SCALE`,
     fingertip geoms `(left_finger_pad|right_finger_pad)`, collision sensor pattern
     `"link7"`, viewer `body_name="link0"`, `env_spacing=1.5`.
   - Keep the `play` and `test` override blocks verbatim (test = no corruption, drop
     early terminations, `episode_length_s=5.0` for the 150-step eval).
3. **Add an rl_cfg** in `config/franka/rl_cfg.py`. If obs/action shape is unchanged,
   just clone the sibling runner and rename `experiment_name` (see
   `franka_lift_cylinder_ppo_runner_cfg`).
4. **Register + tag** in `config/franka/__init__.py`: import the two new cfg funcs,
   call `register_mjlab_task(task_id=..., env_cfg=..., play_env_cfg=..., test_env_cfg=...,
   rl_cfg=..., taxonomy=TaskTaxonomy(embodiment=, skill=, fragility=, source=,
   contact_rich=, notes=))`. Pick `source` honestly (native / metaworld / maniskill /
   adroit / ...) for the provenance ledger.
5. **Validate**:
   - Structural: `python -m mjlab.scripts.benchmark_smoke --keyword <TaskName>`
     (builds, resets, steps, checks finite obs/reward + shapes).
   - Learnability (for NEW skills / rewards): `python -m mjlab.scripts.benchmark_validate
     --task <TaskID> --iters <N>` — trains briefly and reports env success. Easy tasks
     (reach) solve in ~80-150 iters (~1 min); grasp/place need more. This is the honest
     check that the reward+success predicate actually work.
   - Then `pytest tests/test_task_configs.py tests/test_benchmark_taxonomy.py`.
6. **Regenerate the manifest**:
   `python -c "import mjlab; from mjlab.tasks.manipulation import benchmark;
   benchmark.export_manifest('src/mjlab/continual_distill/docs/benchmark/manifest.json')"`.

## Recipe B — new SKILL (needs new MDP; ~0.5–1 day + training to validate)

E.g. Reach, Stack, Insertion. Beyond Recipe A you must author, in `mdp/`:
- a **command** (spawns/tracks the goal; e.g. a reach target or a stack-on-object goal)
  — model on the existing `LiftingCommand` / `PushingCommand` (resampling, success
  latching, debug viz).
- **reward** shaping (a staged/tanh reward like `staged_manipulation_reward`).
- a **success predicate** and (if needed) a **termination**.
- a **base maker** `make_<skill>_env_cfg` composing them.

**Caveat (important for autonomous work):** reward-shaping correctness for a new skill
CANNOT be verified by the smoke test — it needs an RL training run to confirm the task
is solvable and the success predicate fires correctly. Smoke-test proves it BUILDS;
training proves it WORKS. Author new-skill MDP deliberately, not speculatively in bulk.

## New embodiment (Class B/C — arm+hand, floating hand)

- Add the robot/hand entity under `asset_zoo/robots/` (from MuJoCo Menagerie MJCF).
- Add a sibling `config/<robot>/` dir mirroring `config/franka/` (matches the existing
  `config/franka|yam|kinova` convention — no new top-level concepts).
- Wire the embodiment's site names / fingertip geoms / EE link into the base-maker
  overrides (each embodiment differs: Franka uses `gripper`/`link7`; yam uses
  `grasp_site`/`link_6`).
- Tag with the right `Embodiment`. Decide hand action parameterization first (PLAN Q2).

## Guardrails (don't inflate the count)

- Object-geometry swaps (cube→cylinder→sphere→...) each buy a LITTLE grasp diversity
  but are NOT new skills. A few are fine; don't pad the benchmark with them — that is
  the same scene-diversity inflation we reject in LIBERO. Prefer new SKILLS and new
  FRAGILITY points. Report distinct-skills and instances separately in the manifest.
- Always set `contact_rich` and `fragility` honestly — the benchmark's scientific value
  is the spread across the fragility axis, and wrong tags corrupt CL analyses.
