# Goal — CL-V5: SI continual learning across 15 certified tasks

Train **one shared student policy** across a sequence of the 15 RL teachers that
cleared **>95% validation success** in CL-V4 (`docs/cl_v4_rl/STATUS.md`), using
**Synaptic Intelligence (SI) only** as the anti-forgetting regularizer — this run
does not compare SI/EWC/L2 (that comparison was already settled at N=4/N=6 in
`P1_EXPERIMENTS.md`; SI was chosen here for its higher peak retention).

## The 15 tasks (validation SR from CL-V4)

| task | val SR |
|---|---|
| Axial-Extract | 100% |
| Flip-Switch | 100% |
| Open-Door | 100% |
| Open-Drawer | 100% |
| Open-Lid | 100% |
| Push-Button | 100% |
| Push-Flap | 100% |
| Reach-Target | 100% |
| Slide-Window | 100% |
| Turn-Lever | 100% |
| Rotate-Valve | 99.2% |
| Push-Cuboid | 98.4% |
| Throw-To-Bin | 97.7% |
| Topple-Block | 96.1% |
| Drag-Pull | 95.3% |

Lift-Cube (92.2%) and the remaining 8 CL-V4 tasks are excluded — below the 95%
bar or not yet certified.

## Success definition

Final average success rate across all 15 tasks, measured after the full sequence
finishes, matching the metric used in P0/P1 (`docs/P0_EXPERIMENTS.md`,
`docs/P1_EXPERIMENTS.md`). Report per-task retention alongside the average, not
just the average — P1-6 showed the average can hide a collapsed individual task
(LiftCube fell to 0.016 while the average still looked reasonable).

## What "done" looks like

- Fresh teacher datasets extracted from the actual certified CL-V4 checkpoints
  (not the old `teacher_datasets/` folder, which predates the placement fix and
  the new certified checkpoints for most of these 15 tasks).
- A capacity/LR bracket re-validated at N=15 (P0/P1's 4096-saturates finding was
  only shown up to N=6).
- An SI coefficient bracket re-validated at N=15 (c≈1.0 was optimal at N=4/N=6,
  not guaranteed at N=15).
- At least 2 orderings × 3 seeds trained to completion, logged to the new W&B
  project (see STATUS.md), with a scalability point added to the existing
  N=4 (0.960) → N=6 (0.792) curve.
- Results/videos published to a new page on cl.sudhirpratapyadav.com (see
  PLAN.md) once real runs exist — no placeholder content goes on the live site.
