# Rollout camera framing — root cause + exact patch (W1-b → lead)

The lead asked agents NOT to touch `record_task_videos._autoframe_camera`, because the
lead will fix the framing once and re-render everything. **This file hands over the
diagnosis and the exact patch, verified by re-rendering.** W1-b applied it locally,
confirmed it, and then REVERTED both edits so the tree the lead fixes is clean.

## It is not (only) "fits to the robot"

W1-a has already fixed the half of this the lead's note describes — `_autoframe_camera`
now drops the robot geoms so a 1 m Franka cannot set the frame for a 5 cm object. **That
fix is necessary but not sufficient**, and the two below are independent of it.


`_autoframe_camera` already fits the camera to env-0's whole geom bounding box. Two
things make that box wrong, and both are needed:

1. **Mocap bodies are not synced.** The function builds a fresh `MjData`, copies only
   `qpos`, and calls `mj_forward`. Thirteen asset_zoo objects hang off `mocap="true"`
   roots that carry NO qpos — every articulated mechanism, plus `container`, `ledge`,
   `wall` — so all of them are measured at their MJCF world-frame pose. This is the same
   bug `viewer/offscreen_renderer.py:68-71` already fixes for the render loop, with a
   comment saying exactly why.
2. **It runs before the first reset.** `run_render_phase` calls it immediately after
   `_build_env`, and a mocap asset only takes its sampled per-env pose when the command
   term resamples, i.e. on the first `env.reset()`. So even with (1) applied it reads
   the MJCF pose.

Measured symptom (Throw-To-Bin): the bin is written to x 0.78–0.90 but sits at 0.55 in
`container.xml`, so it was **outside the frame for the entire clip** — the published
video showed a Franka waving at empty floor. With both fixes the basket, the cube and
the goal marker are all in frame (compare
`~/cl_v2_work/W1-b/Mjlab-Throw-To-Bin-Franka/thumb.jpg`, re-rendered after the patch).

## Patch 1 — `src/mjlab/scripts/record_task_videos.py::_autoframe_camera`

`W1-b-rollout-camera-framing.patch` next to this file is a minimal, cleanly-applicable unified diff of BOTH hunks against the current tree (`git apply` or `patch -p1` from the repo root).

```python
  data = mujoco.MjData(model)
  data.qpos[:] = env.sim.data.qpos[0].cpu().numpy()
  # Mocap bodies carry NO qpos. Thirteen asset_zoo objects hang off mocap roots that
  # the command terms write per-env (every articulated mechanism, plus container /
  # ledge / wall), so without syncing them the bounding box below is measured with all
  # of them still at their MJCF world-frame pose.
  if model.nmocap:
    data.mocap_pos[:] = env.sim.data.mocap_pos[0].cpu().numpy()
    data.mocap_quat[:] = env.sim.data.mocap_quat[0].cpu().numpy()
  mujoco.mj_forward(model, data)
```

## Patch 2 — `classical/render_rollout.py::run_render_phase`

```python
  debug_dir = out_dir / "debug_frames"
  try:
    env.reset()                       # <-- ADD: mocap assets only take their sampled
    _autoframe_camera(env, RecordConfig())   #     per-env pose on the first reset
```

## Who this affects

Every task with a mocap-mounted asset whose sampled band differs from its MJCF pose:
Throw-To-Bin (worst — 0.55 vs 0.78-0.90), Place-In-Container, Pivot-Lift (wall),
Edge-Grasp (ledge), and all ten articulated mechanisms. Tasks whose objects are free
bodies (Strike-Slide, Tool-Pull, Reorient-Object, the lift/push family) are already
framed correctly, because free-body poses live in `qpos`.

## What this means for W1-b's published clips

`Mjlab-Place-In-Container-Franka` and `Mjlab-Throw-To-Bin-Franka` were rendered with
both patches applied, so their published `teacher.mp4` / `failure.mp4` / `thumb.jpg`
DO show the basket. The tree no longer contains the patches. If the global re-render
runs without them, those two clips will regress to showing empty floor.
