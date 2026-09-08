# CL-V2 — GOAL

Autonomously complete this task-realism upgrade.

Input: the 25 Class-A Franka manipulation tasks registered in `mjlab` (list in
`../cl25/CONTEXT.md`, read-only). All of them are built from untextured primitive
boxes and cylinders.

Goal: Give every one of the 25 tasks a realistic-looking asset with correct kinematics
and physics. Replace the primitives with downloaded or Blender-built assets that look
as close to real objects as possible — not low-poly, not flat-coloured — and make each
task physically and statistically sound.

The result should:
- use a well-chosen, real-scale asset for every task, with the reason recorded
- have verified physics: stable at rest, graspable, correct articulation ranges and units
- have verified initial-condition distributions: poses relative to the robot are
  sampled, reachable, collision-free and inside the measured workspace
- have a sparse, binary success predicate, unreachable at reset and reachable in physics
- render well enough for a video (backgrounds come later)
- be reproducible: one script verifies any task end to end

Use Blender, MuJoCo, trimesh, CoACD, or any other tools you find appropriate. Any asset
license is acceptable. Keep body, site and joint names so existing env cfgs load; task
geometry may be tweaked where it improves the task, as long as the motion profile is
unchanged and the task still earns its name. Classical teacher policies may be changed
to fit the new assets. Decide open questions yourself and record the decision. Do not
stop at the first plausible asset or ask for routine approval. Only declare

DONE after every task passes the physics, init-distribution, success and visual gates,
its classical-teacher success rate is measured at n=128, and a rollout video of the
task with its new real asset is published to the site (added 2026-09-09, user
direction): push the videos of ALL new-asset tasks to `https://cl.untuai.com/v2/<Task-Id>/`
via `ssh untu_vps` (rsync into `~/sudhir/continual_learning/v2/<Task-Id>/`, the same
Caddy-served tree as `phase1/`), with an index page so every task is browsable.

GPU work runs on the shared Slurm account: read `~/use_instructions/README.md` first,
reuse the existing holder job (`squeue ... | grep -i hold`, then
`run_in_holder.sh <JOBID> <GPU> <LOG> <cmd>`), never cancel a job you did not create,
and log the GPU index you took. Track progress in `STATUS.md` and `LOGS.md`, publish
renders to the site, and keep raw downloads and Blender sources outside the repo.
