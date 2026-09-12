# Teacher improvements — September 11, 2026

This pass keeps task definitions, rewards, randomized starts, observation noise and episode budgets fixed. Rates use strict first episodes; policy retries occur inside the same episode. The pass is complete; several teachers still need further improvement.

| Teacher | Original /128 | Retained /128 | Independent confirmation |
|---|---:|---:|---|
| Push-Button | 123 | 126 | 128/128 (baseline 122/128) |
| Lift-Cube | 94 | 125 | — |
| Stack-Cube | 36 | 122 | 123/128 |
| Topple-Block | 105 | 112 | 113/128 (baseline 104/128) |
| Open-Drawer | 79 | 101 | — |
| Cage-Drag | 23 | 95 | — |
| Open-Lid | 119 | 122 | 125/128 (baseline 114/128) |
| Peg-Insertion | 26 | 32 | 33/128 (baseline 21/128) |
| Drag-Pull | 30 | 57 | — |

All nine improvements have success and failure videos published through `untu_vps`. Per-task `videos/*/publication_verification.json` records public HTTPS hashes and full video decode checks. Button, Stack, Topple, Lid and Peg also have independent seed measurements. Lid development-seed initial observations differ at floating-point rounding scale; its independent comparison is bit-identical across all initial state and observation arrays.

Validation: all 40 teacher tests passed after the final code changes. Exact initial state/observation comparisons passed for the other retained pairs. Source-only changes and candidate experiments are preserved in this folder; no task changes were made in this pass.

The full 24-task diagnostic baseline is in [BASELINES.md](BASELINES.md). Current n=128 measurements are under `baseline128/`; all 24 tasks in this pass now have completed n=128 baselines. Reorient was improved and published in the preceding pass (97→100/128). Peg, Tool Pull, Edge Grasp and Pivot Lift remain weak; rejected candidates are not installed.

See [WORK_LOG.md](WORK_LOG.md) for evidence, hypotheses and rejected attempts, and [STATUS.md](../../STATUS.md) for the current scoreboard.

Tool Pull also has a verified orientation-decoding correction: shaft roll no longer reverses its inferred heading. The retained policy still scores 0/32; the original baseline is 0/128. The larger Tool controller redesigns did not improve success and were not retained.

## Current measurements for all tasks

| Task | Baseline | Current |
|---|---:|---:|
| Axial-Extract | 128/128 | 128/128 |
| Cage-Drag | 23/128 | 95/128 |
| Drag-Pull | 30/128 | 57/128 |
| Edge-Grasp | 0/128 | 0/128 |
| Flip-Switch | 128/128 | 128/128 |
| Lift-Cube | 94/128 | 125/128 |
| Open-Door | 6/128 | 6/128 |
| Open-Drawer | 79/128 | 101/128 |
| Open-Lid | 119/128 | 122/128 |
| Peg-Insertion | 26/128 | 32/128 |
| Pivot-Lift | 0/128 | 0/128 |
| Place-In-Container | 118/128 | 115/128 |
| Push-Button | 123/128 | 126/128 |
| Push-Cuboid | 29/128 | 29/128 |
| Push-Flap | 128/128 | 128/128 |
| Reach-Target | 128/128 | 128/128 |
| Rotate-Valve | 63/128 | 63/128 |
| Slide-Window | 108/128 | 108/128 |
| Stack-Cube | 36/128 | 122/128 |
| Strike-Slide | 117/128 | 117/128 |
| Throw-To-Bin | 75/128 | 75/128 |
| Tool-Pull | 0/128 | 0/32 |
| Topple-Block | 105/128 | 112/128 |
| Turn-Lever | 108/128 | 108/128 |
| Reorient-Object (preceding pass) | 97/128 | 100/128 |

Place’s 115/128 current value is an independent seed check of the unchanged policy, compared with118/128 on the development seed; it is not a code regression. The retained Peg implementation also completed a fresh32-episode smoke evaluation at12/32.

For reproduction, `before/` contains the pre-pass teacher sources and `retained_snapshot/` contains the final installed sources. Some experimental wrappers import live policies: after adoption, rerunning those wrappers can apply a change twice. Use the source snapshots or full standalone candidate sources for faithful comparisons. Seeds, initial states, observations, complete trajectories and source fingerprints are saved per measurement.

This pass completed **117 measurements and 8,064 first-episode trials**. All simulation used the existing holder20277 on allocated GPUs1–3. No other job or holder was cancelled.
