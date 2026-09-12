# Task geometry review — 2026-09-10

This pass fixes task geometry and rendering before another teacher evaluation.
The existing CL-V3 success counts and published rollout videos were produced with
older collision settings and do not validate this version.

## Changes

- Goals are orange replicas of the actual manipulated part, with alpha 0.28.
  Mesh transforms and tracking-site offsets come from the asset itself. Door,
  drawer, window and lid goals include their moving panels and handles; other
  goals use the actual object, toggle, lever, valve, plug, plate or tool-task puck.
  Reach-Target uses a sphere because it requests a position without an object.
- All ten articulated goals use joint kinematics, including axis, pivot, body/site
  offsets and the randomized mount pose. This replaces approximate Cartesian
  offsets. At the exact requested joint value, source and replica coincide.
  Success tolerances and directional overshoot still permit nearby/further states.
- Stack-Cube and Peg-Insertion now have persistent goal geometry in rendered
  videos. Their goals track the base; the reset pose uses the newly sampled base
  pose, rather than stale kinematics from the previous episode.
- Reorient-Object and Topple-Block show a valid target-axis orientation, resting
  on the floor at the collider-derived height. They no longer show an upright
  placeholder floating at an arbitrary height. Position-only lift, push, place,
  tool-pull and edge-grasp markers follow the object's unconstrained orientation.
- Panda arm, wrist, palm and finger collision meshes now contact task objects and
  terrain. The obsolete coarse hand capsule is disabled in favor of the actual
  hand collision mesh. Robot/object contact uses robot type bit 1 and object bit 2;
  this change does not enable robot self-collision.
- Fixed the switch housing orientation: it sits horizontally below the pivot,
  outside both sides of the toggle sweep. Toggle inertia and joint limits stay
  intact.
- Corrected additional measured overlaps: raised the lid assembly/hinge by 9 mm,
  raised the plug shoulder by 6 mm, moved the flap post back 6 mm (visual and
  collider), and put the window stop at 0.234 m, 1 mm before the sash hits the jamb.
  The window command target remains 0.22 m.
- Added previously absent button housing, collar and rod collisions, with a shaft
  bore through the housing and panel. Added valve bonnet collisions with an open
  bearing bore, and moved the bonnet back 14 mm so it clears the wheel spokes.
- Fixed all-environment mocap writes for batches larger than one; these are needed
  for goals that track a moving object. Fixed the workspace floor audit to copy
  mocap rotations as well as joint positions.

## Evidence

**203 tests passed on CPU**, including the MuJoCo Warp stepping path:

```bash
FORCE_CPU=1 .venv/bin/pytest -q \
  tests/test_manipulation_geometry.py tests/test_workspace_placement.py \
  tests/test_class_a_expansion.py tests/test_class_a_wave1.py \
  tests/test_asset_zoo.py tests/test_entity.py
```

The new geometry tests cover all 29 registered Franka tasks with two environments,
three physics/control steps, transparent non-colliding markers, exact mechanism
FK target agreement, replica/source geometry agreement, physical robot/object
contacts, and 65 positions across every articulated mechanism's joint range.
The existing success-predicate tests remain green; teacher accuracy was not
remeasured. A separate contact audit sampled eight resets per task (232 resets)
and found no robot/object or object/object penetration deeper than 3 mm.

- [All 29 task reset views](all_tasks.jpg)
- [Ten mechanisms at three joint positions](mechanism_sweeps.jpg)
- Individual full-resolution PNGs are alongside the contact sheets.

Recreate the images from the repository root:

```bash
MUJOCO_GL=egl EGL_PLATFORM=surfaceless .venv/bin/python docs/task_geometry_review/render_all_tasks.py
MUJOCO_GL=egl EGL_PLATFORM=surfaceless .venv/bin/python docs/task_geometry_review/render_mechanisms.py
```

These are geometry review images, not teacher success videos. The website's
existing videos have not been replaced by this pass.
