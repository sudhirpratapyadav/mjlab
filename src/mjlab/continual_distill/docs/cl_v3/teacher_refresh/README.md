> **Later work:** [All remaining teachers — current improvements and evidence](all_remaining/README.md). The results below describe the preceding Reorient/Peg/Cage pass.

# Teacher improvement pass — 2026-09-11

Published: [CL-V3 gallery](https://cl.sudhirpratapyadav.com/v3/).

**Reorient-Object improved from 97/128 (75.8%) to 100/128 (78.1%)** in a matched GPU evaluation. The fix removes an accidental OPEN action on the CLOSE→LIFT transition. This is a modest measured increase, still below the 90% competence target; it is not a claim of statistical significance. Nine trials gained success and six lost it. Initial qpos arrays are identical between the two batches.

The published success clip is newly successful trial 2; the residual-failure clip is trial 0. Both are rendered from the exact measured MuJoCo states at 1920×1080, 50 fps, with no action replay. All six published files (including the gallery index) were fetched over public HTTPS from the VPS and checked against local SHA-256 hashes. The preceding gallery is preserved at `/v3_before_teacher_improvements_20260911/` on the VPS.

## Retained changes

- Reorient: preserve the closed gripper on the transition to LIFT. A regression test covers both a retained grasp and an empty-grasp retry.
- Cage: behavior remains the original controller; updated its documentation to describe the corrected aperture/enclosure rule.
- Peg: behavior remains the original controller, including the prior task-audit observation-frame adaptation.
- Gallery and scoreboard distinguish fresh measurements from historical results. Previous competence counts no longer masquerade as confirmation under the September 11 task definitions.

**37 regression tests passed** (teacher interfaces, evaluation capture, and gripper transitions). Task definitions, physics, budgets, and reset distributions were not changed during this teacher pass.

## Completed experiments

Rows must be compared within the same device, task, batch size and seed. Every trial is one initial episode, ending at first success, termination or timeout; automatic-reset retries never count. Training observation noise is enabled. Seed: 20260911.

| Task / device | Candidate | Successes | Decision |
|---|---|---:|---|
| Reorient / CPU | Original | 26/32 | CPU baseline |
| Reorient / CPU | Fixed lift waypoint | 24/32 | Rejected |
| Cage / CPU | Original | 8/32 | Original retained |
| Cage / CPU | Centered entry and fixed travel orientation | 5/32 | Rejected |
| Cage / CPU | Higher entry and transport clearance | 5/32 | Rejected |
| Reorient / GPU 3 | Original | 97/128 | Matched confirmation baseline |
| Reorient / GPU 3 | Recovery and recentering | 97/128 | Rejected: no net improvement |
| Reorient / GPU 3 | Fixed arm command during squeeze | 94/128 | Rejected |
| Reorient / GPU 3 | Keep grip closed on CLOSE→LIFT | **100/128** | **Retained and published** |
| Peg / GPU 3 | Original | 8/32 | Original retained |
| Peg / GPU 3 | Hand-frame tip estimator | 8/32 | Rejected: no improvement, more terminations |
| Peg / GPU 3 | Fixed release pose | 6/32 | Rejected |
| Peg / GPU 3 | Peg-orientation feedback and upright release gate | 2/32 | Rejected |
| Peg / GPU 3 | Orientation feedback plus stronger pose priority | 4/32 | Rejected |

Total: **832 completed first-episode trials** across these 14 measurements. Interrupted CPU runs and the superseded OMP=4 GPU baseline are excluded; incomplete folders carry no reported rate.

## Remaining work

Reorient still has 21 terminations and 7 timeouts in the published batch. Contact inspection identifies acquisition-time finger/ground collisions as a major failure mode. Cage needs reliable open-cage transport; the tested alternatives were slower or collided more often. Peg's hand can descend by 3.5–6 cm during release and its actual peg tilt can exceed the bore's tolerance, but the tested remedies did not improve its score. These findings and unsuccessful strategies are preserved to avoid repeating them.

Cage and Peg have only n=32 measurements under the current rules; their old website clips remain explicitly historical. The other 22 teachers await corrected-rule confirmation. This pass does not claim the 25-teacher program is complete.

## Evidence and reproduction

- [Detailed observations and experiment reasoning](WORK_LOG.md)
- [Reorient baseline](gpu_baseline128/Mjlab-Reorient-Object-Franka/result.json)
- [Reorient retained candidate](closed_transition128/Mjlab-Reorient-Object-Franka/result.json)
- [Published metadata](videos/Mjlab-Reorient-Object-Franka/result.json)
- [Public HTTPS verification](publication_verification.json)
- [Scoreboard](../STATUS.md)

`evaluate.py` records policy observations, phase, qpos/qvel/mocap state, metrics and terminal outcomes. `--teacher-file` loads the preserved original teacher for a matched baseline. `render_measured.py` renders selected trials directly from those recorded states. `prepare_publication.py` requires matching task/seed/device/protocol, n=128 and a higher measured score before producing publication metadata. `verify_publication.py` validates the public content and video encoding.

All simulation ran inside existing holder 20277. Confirmation used GPU 3, as assigned in PLAN §6, pinned by UUID and verified with nvidia-smi. OMP/BLAS threads were limited to one for the fast confirmation runs; worker CPU affinities stayed within the holder allocation. No holder or other user's job was cancelled. Source fingerprints include uncommitted changes; original and rejected teacher versions are archived alongside this report.
