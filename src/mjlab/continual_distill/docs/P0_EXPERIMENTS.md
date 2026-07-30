# P0 experiments — floor, ceiling, and the capacity/LR confound

Tracking doc for the three P0 items in `docs/EXPERIMENTS_TODO.md`. These anchor the
`0.960` headline from [SWEEP24_RESULTS.md](SWEEP24_RESULTS.md) on a scale a reviewer
can read: what does the problem look like with no CL mechanism (floor), with no
sequence at all (ceiling), and is the 8192 collapse really about capacity (confound)?

- **wandb project:** `continual_rl_mjlab_p0` (new — kept separate from
  `continual_rl_mjlab_sweep` so the baseline runs don't mix into the sweep's 115)
- **Cluster:** dgx2, holder `19736`, 8× A100-80GB
- **Config:** `config/tasks_p0.yaml` — same four RL teachers as the sweep, 500
  epochs/task, lr `3e-5`, trunk `[4096, 2048, 1024]` unless a row says otherwise
- **Orderings:** best = p06 `PushCuboid → OpenDrawer → OpenDoor → PushButton`;
  worst = p10 `PushButton → OpenDoor → OpenDrawer → PushCuboid`
- **Scheduler:** `slurm/run_p0.sh` — continuous, keeps all 8 GPUs busy and
  dispatches the next queued job the moment one frees (the old wave-based sweep
  scripts idled GPUs waiting for the slowest run in each wave)

> ## ✅ RESOLVED (commit `a0cc9be`) — placement restored, teachers work again
>
> Fixed by option 2: restoring the pre-audit object placement **for these four tasks
> only**, on HEAD. Verified after the fix: PushCuboid **0.8125**, OpenDrawer
> **1.0000** (from 0.008 / 0.000). The other six Class A tasks keep the audited
> workspace placements — they have no frozen teachers to invalidate.
>
> The history below is kept because it documents what was ruled out, and because the
> same trap will recur if the audit ranges are reinstated without retraining.

<details>
<summary>Original diagnosis (click to expand)</summary>

> ## ⚠️ BLOCKED — the RL teachers no longer work in the current environment
>
> The first four P0-1 runs finished at **0.000 success on every task**, which is not
> a forgetting floor (that would leave the *last* task near 1.0). Root cause is not
> the P0 code: the **teachers themselves** now fail. Verified with the stock
> pipeline (SI on, all defaults, none of the P0 flags):
>
> | teacher | July sweep | now (MuJoCo 3.11) |
> |---|---|---|
> | PushCuboid | ~1.00 succ / 6.49 ret | **0.008** succ / 0.81 ret |
> | OpenDrawer | ~0.99 succ / 6.11 ret | **0.000** succ / 3.36 ret |
>
> **The warp upgrade (`fb15756`) is NOT the cause** — it was the initial suspect,
> but a controlled A/B disproved it: rebuilding the exact sweep-era physics stack
> (MuJoCo 3.3.7 / warp 1.11.0.dev / mujoco-warp `46b4421`) in a separate
> `.venv_old` reproduces the *same* failure (PushCuboid 0.008, OpenDrawer 0.000).
> Both stacks fail identically, so the physics is exonerated.
>
> Also ruled out:
> - **Env code** — the four task cfgs are untouched since the sweep, reward terms
>   are byte-identical, and the shared MDP modules (`commands.py`,
>   `observations.py`, `rewards.py`) changed by *pure addition only*
>   (1000 insertions / 0 deletions in commands.py; new reach-task functions in the
>   other two). No existing observation or reward was modified.
> - **Teacher files** — `data.pkl` / `teacher.pkl` are unmodified since January
>   (mtime + md5 verified), so the checkpoints are not corrupt.
> - **Distill code** — no commit touched `continual_distill.py` / `utils.py`
>   between the sweep and this work except the P0 commit itself.
>
> ### ROOT CAUSE: the benchmark workspace commits moved object spawn ranges
>
> Checking out the sweep-era commit `69b2896` into a worktree and re-probing
> **restores the teachers completely**:
>
> | teacher | sweep-era code (`69b2896`) | current code (HEAD) |
> |---|---|---|
> | PushCuboid | **0.81 – 0.84** | 0.008 |
> | OpenDrawer | **1.00** | 0.000 |
>
> (Sweep-era 0.836 matches the July sweep's own first eval of 0.828.) Same teacher
> files, same `.venv_old` physics stack, same probe — only the repo code differs.
>
> The culprit is **`2ab6d11` "benchmark: fix Class A object placement against the
> measured workspace"** (and the related `8e7fd5c` / `babf036`), which rewrote
> `config/franka/env_cfgs.py` (997 lines). PushCuboid's spawn range went from the
> sweep's literal `x=(0.6, 0.8), y=(-0.15, 0.15)` to workspace-derived
> `GRASP_X_RANGE` halves, and the articulated tasks got new `_mech_z()` mount
> heights via overridden `reset_*_position` event `pose_range`s.
>
> **The teachers are not broken — they are being evaluated on a different task
> distribution than they were trained on.** Nothing is corrupt; obs_dim is still 60,
> so this fails silently rather than erroring.

> This affected **every teacher dataset**, so the mix/BC/classical experiments would
> have hit the same wall. The fix in `a0cc9be` unblocks those too.

</details>

## Status

Legend: ⏳ queued · 🔄 running · ✅ complete · ❌ failed

### P0-1 — Naive sequential fine-tuning (the forgetting floor), 6 runs
SI disabled (`--si-coeff 0`), everything else identical to the sweep.

| run | ordering | seed | status | final avg SR |
|---|---|---|---|---|
| `nosi_best_s0` | best | 0 | ✅ | 0.270 |
| `nosi_best_s1` | best | 1 | ✅ | 0.266 |
| `nosi_best_s2` | best | 2 | ✅ | 0.266 |
| `nosi_worst_s0` | worst | 0 | ✅ | 0.504 |
| `nosi_worst_s1` | worst | 1 | ✅ | 0.402 |
| `nosi_worst_s2` | worst | 2 | ✅ | 0.590 |

### ✅ RESULT — catastrophic forgetting confirmed

| ordering | no-SI floor | SI baseline (SWEEP24) | SI gain |
|---|---|---|---|
| best (PC→OW→OD→PB) | **0.267 ± 0.002** | 0.960 | **+0.693** |
| worst (PB→OD→OW→PC) | **0.499 ± 0.094** | 0.659 | +0.160 |

Lands squarely in the predicted 0.25–0.4 band and **anchors the 0.960 headline**:
without SI the same pipeline retains 0.267 on the same ordering. The paper can now
say "0.960 versus 0.267 for sequential fine-tuning" instead of quoting an
unreferenced number.

**Per-task structure — this is textbook forgetting, not noise.** On the best
ordering every seed collapses to the *same* signature: the final task is perfect and
everything before it is gone.

| ordering | PushCuboid | OpenDrawer | OpenDoor | PushButton |
|---|---|---|---|---|
| best (PB last) | 0.05–0.08 | 0.00 | 0.00–0.02 | **1.00** |
| worst (PC last) | **0.83–0.91** | 0.48–0.77 | 0.00 | 0.22–0.77 |

Only the last-trained task survives — exactly "the student can only do whatever it
saw most recently." Seed variance is near zero on the best ordering (±0.002), so the
floor is a property of the method, not of initialisation.

**Two findings worth carrying into the paper:**

1. **The no-SI floor INVERTS the ordering effect.** With SI, best-first scores 0.960
   and worst 0.659. Without SI the ranking flips: 0.267 vs 0.499. The "best" ordering
   is only best *because SI protects the fragile early task* — it ends on PushButton,
   so plain fine-tuning keeps one easy task and loses three. The "worst" ordering ends
   on PushCuboid, the hardest task, which alone scores 0.83–0.91 and drags the average
   up. **Ordering quality is not intrinsic; it is a property of the ordering-plus-
   mechanism pair.** A reviewer asking "is the primacy effect just task difficulty?"
   is answered by this row.
2. **SI's benefit is far larger on the good ordering** (+0.693 vs +0.160). SI does not
   add a constant; it *compounds* with a favourable ordering. This strengthens the
   primacy claim rather than competing with it.

**OpenDoor is 0.00 in every no-SI run, both orderings** — including when trained
third of four. Worth a sentence in the paper: it is the least retention-robust task
here, consistent with it never being the final task in either ordering.

### P0-2 — Joint multitask distillation (the ceiling), 3 runs
All four teachers simultaneously, no sequence, no SI.

| run | seed | status | final avg SR |
|---|---|---|---|
| `joint_s0` | 0 | ⏳ | |
| `joint_s1` | 1 | ⏳ | |
| `joint_s2` | 2 | ⏳ | |

Joint mode is implemented and **validated** (2-task smoke test reached epoch 2/500,
loss 13.38, training normally). The 3 runs are queued behind the 15 above and are
dispatched by re-running the scheduler with `JOINT=1` once slots free.

**Expected:** ~0.95–1.00. If best-ordering 0.960 matches this, that is a strong
result — continual learning at no cost relative to joint training.

**Implementation note:** joint mode did not exist; the trainer was strictly
sequential. Added `--joint-distill` + `train_epoch_joint()` in
`continual_distill.py`. Each epoch interleaves shuffled batches from all four
tasks, each routed to its own output head (the student already has per-task heads
over a shared trunk), optimizer never reset between tasks. Total gradient steps
match the sequential runs for fairness.

### P0-3 — Width 8192 × learning-rate sweep, 9 runs
Best ordering, width `[8192, 4096, 2048]`. Tests whether the 8192 collapse is a
capacity limit or just an LR tuned at 4096.

| run | lr | seed | status | final avg SR |
|---|---|---|---|---|
| (from sweep) | 3e-5 | 0,1,2 | ✅ | 0.656 ± 0.201 |
| `w8192_lr1e5_s0` | 1e-5 | 0 | ✅ | **0.942** |
| `w8192_lr1e5_s1` | 1e-5 | 1 | ✅ | **0.957** |
| `w8192_lr1e5_s2` | 1e-5 | 2 | 🔄 | |
| `w8192_lr5e6_s0/1/2` | 5e-6 | 0,1,2 | 🔄 | |
| `w8192_lr1e6_s0/1/2` | 1e-6 | 0,1,2 | 🔄 | |

### ✅ RESULT — the 8192 "collapse" is an LR artifact, and the LR optimum is 1e-5

All 9 runs in. The sweep is a clean **inverted U**, not a monotone "lower is better":

| width | lr | mean ± std | seeds |
|---|---|---|---|
| 4096 | 3e-5 | 0.960 ± 0.014 | — (SWEEP24 baseline) |
| 8192 | 3e-5 | 0.657 ± 0.242 | 0.76, 0.38, 0.83 |
| 8192 | **1e-5** | **0.946 ± 0.010** | 0.942, 0.957, 0.938 |
| 8192 | 5e-6 | 0.757 ± 0.031 | 0.789, 0.727, 0.754 |
| 8192 | 1e-6 | 0.518 ± 0.113 | 0.598, 0.438 |

**At its correct LR, 8192 matches 4096: 0.946 ± 0.010 vs 0.960 ± 0.014** — a 0.014
gap, well inside seed noise. Variance also collapses by 24× (±0.242 → ±0.010),
so the "bigger nets are less stable" observation was likewise an LR artifact.

**This overturns the §5.G claim in SWEEP24_RESULTS.md**, which states "capacity helps
only up to 4096; going bigger HURTS both mean AND stability" and "**4096 is the sweet
spot; over-parameterization re-introduces instability**". Both sentences are
confounded: every width in that sweep shared the LR tuned at 4096. The correct
statement is **capacity is not the binding constraint — the learning rate must scale
with width.**

Because we swept *past* the optimum, this is stronger than a one-sided recovery: it
locates 1e-5 rather than merely showing "lower helps", and it rules out the obvious
counter-reading that any LR reduction would have worked. Too low genuinely underfits
(1e-6 → 0.518).

**The whole effect is carried by PushCuboid**, the fragile task trained first:

| lr | PushCuboid | (other three tasks) |
|---|---|---|
| 1e-5 | **0.84** | 0.92–1.00 |
| 5e-6 | 0.31 | 0.70–1.00 |
| 1e-6 | 0.07 | 0.62–1.00 |

At every LR the three later tasks stay high; only the earliest-trained task degrades.
So the low-LR failure mode is **not** general underfitting — it is a *consolidation*
failure. Too small a step and SI's surrogate cannot pull the early task's parameters
back, so task 0 is lost while the recent tasks look fine. That is the same primacy
signature P0-1 exposes from the opposite direction, and it ties the capacity story to
the ordering story rather than leaving them as two unrelated observations.

## Results

_Auto-collected by `slurm/collect_p0.py`._

| group | run | status | avg SR | per-task |
|---|---|---|---|---|
| P0-1 | `nosi_best_s0` | ✅ complete | 0.270 | OpenDoor 0.00, OpenDrawer 0.00, PushButton 1.00, PushCuboid 0.08 |
| P0-1 | `nosi_best_s1` | ✅ complete | 0.266 | OpenDoor 0.02, OpenDrawer 0.00, PushButton 1.00, PushCuboid 0.05 |
| P0-1 | `nosi_best_s2` | ✅ complete | 0.266 | OpenDoor 0.02, OpenDrawer 0.00, PushButton 1.00, PushCuboid 0.05 |
| P0-1 | `nosi_worst_s0` | ✅ complete | 0.504 | OpenDoor 0.00, OpenDrawer 0.75, PushButton 0.42, PushCuboid 0.84 |
| P0-1 | `nosi_worst_s1` | ✅ complete | 0.402 | OpenDoor 0.00, OpenDrawer 0.48, PushButton 0.22, PushCuboid 0.91 |
| P0-1 | `nosi_worst_s2` | ✅ complete | 0.590 | OpenDoor 0.00, OpenDrawer 0.77, PushButton 0.77, PushCuboid 0.83 |
| P0-2 | `joint_s0` | 🔄 running | — | — |
| P0-2 | `joint_s1` | 🔄 running | — | — |
| P0-2 | `joint_s2` | 🔄 running | — | — |
| P0-3 | `w8192_lr1e5_s0` | ✅ complete | 0.942 | OpenDoor 1.00, OpenDrawer 0.94, PushButton 0.98, PushCuboid 0.84 |
| P0-3 | `w8192_lr1e5_s1` | ✅ complete | 0.957 | OpenDoor 1.00, OpenDrawer 1.00, PushButton 0.97, PushCuboid 0.86 |
| P0-3 | `w8192_lr1e5_s2` | ✅ complete | 0.938 | OpenDoor 1.00, OpenDrawer 1.00, PushButton 0.92, PushCuboid 0.83 |
| P0-3 | `w8192_lr5e6_s0` | ✅ complete | 0.789 | OpenDoor 1.00, OpenDrawer 0.70, PushButton 1.00, PushCuboid 0.45 |
| P0-3 | `w8192_lr5e6_s1` | ✅ complete | 0.727 | OpenDoor 1.00, OpenDrawer 0.77, PushButton 0.77, PushCuboid 0.38 |
| P0-3 | `w8192_lr5e6_s2` | ✅ complete | — | — |
| P0-3 | `w8192_lr1e6_s0` | ✅ complete | — | — |
| P0-3 | `w8192_lr1e6_s1` | 🔄 running | — | — |
| P0-3 | `w8192_lr1e6_s2` | 🔄 running | — | — |


## Stack A/B comparison — did the upgrade break the teachers?

Measures every RL teacher's standalone competence under both physics stacks,
through an **identical code path**: `config/tasks_teachercheck.yaml` sets
`num_epochs: 0`, so the pipeline performs only its step-0 environment evaluation
(which rolls out the teacher and reports `teacher_success` / `teacher_return`) and
then exits. 128 episodes, seed 0.

| stack | venv | mujoco | mujoco-warp | warp |
|---|---|---|---|---|
| **A (old, = the sweep)** | `.venv_old` | 3.3.7 | git `46b4421` (v0.0.1) | 1.11.0.dev20251124 |
| **B (new, current)** | `.venv` | 3.11.0 | 3.11.0 | 1.15.0 |

`.venv` is untouched; stack A lives in a separate `.venv_old`. Confirmed faithful:
the old-stack run loads the same warp kernel module hashes (`a88f545`, `769a44d`,
`1699532`) as the July sweep log.

Note the four sweep tasks use **none** of the assets `fb15756` reverted
(cylinder/disc/ellipsoid/LEAP), so this isolates the physics stack cleanly.

### Teacher success rate (128 episodes)

| teacher | July sweep (recorded) | stack B (new) | stack A (old) |
|---|---|---|---|
| PushCuboid | ~1.00 (ret 6.49) | **0.008** (ret 1.26) | _pending_ |
| OpenDrawer | ~0.99 (ret 6.11) | **0.000** (ret 3.36) | _pending_ |
| OpenDoor | ~1.00 (ret 7.35) | **0.000** (ret 1.93) | _pending_ |
| PushButton | ~1.00 (ret 7.71) | **0.000** (ret 0.93) | _pending_ |

**Stack B is a total wipeout — all four teachers at ~0.** This is not partial
degradation from slightly different contact dynamics; the policies do not function
at all.

### Outcome

**Stack A also gives ~0 → the physics upgrade is exonerated.** Both stacks fail
identically, so `fb15756` is not the cause. The `.venv_old` build still has value as
a reproduction of the sweep environment, but the deciding variable turned out to be
the repo code (see root cause above), which the sweep-era worktree confirmed.

### How to unblock P0

1. **Run P0 from the sweep-era worktree** (`../mjlab_sweepera` @ `69b2896`) with
   `.venv_old`. Teachers work there and the numbers stay directly comparable to
   SWEEP24's 0.960. The P0 code additions (`--joint-distill`, `tasks_p0.yaml`)
   would need cherry-picking onto that checkout.
2. **Restore the old spawn ranges for the four sweep tasks on HEAD** — keeps the
   benchmark's workspace fix for the new Class A tasks while leaving the four
   continual-distill tasks on the distribution their teachers know.
3. **Retrain the four RL teachers against current placements.** Cleanest long-term,
   but expensive, and SWEEP24's table would have to be regenerated to match.

**Chosen: option 2** (commit `a0cc9be`) — the four continual-distill tasks are meant
to keep working on `main`, so the fix belongs on HEAD rather than in a side worktree.

### Verification after the fix (128 episodes, HEAD + `.venv`)

| teacher | before fix | after fix | sweep-era reference |
|---|---|---|---|
| PushCuboid | 0.008 | **0.79 – 0.81** | 0.82 – 0.84 |
| OpenDrawer | 0.000 | **1.000** | 1.000 |
| OpenDoor | 0.000 | _verifying_ | 0.992 |
| PushButton | 0.000 | _verifying_ | 1.000 |

PushCuboid varies 0.79–0.84 across repeated 128-episode evals on both the fixed HEAD
and the sweep-era checkout — that spread is ordinary evaluation noise for this task,
not a residual gap.

**End-to-end confirmation:** the relaunched `nosi_best_s0` reports
`teacher_success: 0.8281` at its first evaluation — the exact value the July sweep
logged (0.828). The P0 runs are training against working teachers.

### Relaunch bookkeeping

The 18 runs from the pre-fix attempt are archived under
`logs/invalid_p0_preplacementfix/` (with the old `p0_state/`). This matters because
the scheduler treats `"Training complete"` in a run's log as "done and skip" — the
three `joint_*` runs had actually *finished* during the broken window (returning
0.008–0.016) and would otherwise have been silently kept. All 18 now re-run against
the fixed environment.

## Notes / gotchas hit

- **`config/tasks.yaml` cannot run on the cluster.** It still carries the old
  `/media/cvlab/EXTDRIVE/...` workstation dataset paths from the original sweep.
  `tasks_p0.yaml` is the same four teachers with corrected `/ihub/homedirs/...`
  paths. Anything launched from `tasks.yaml` here dies on a missing dataset.
- **`--env-eval-episodes 0` crashes.** `env_cfg.scene.num_envs` is set directly
  from it, so 0 → `ZeroDivisionError` in `terrain_importer.py:230`
  (`num_envs / int(np.sqrt(num_envs))`). Use `--env-eval-every` to make env eval
  rare instead of setting episodes to 0.
- **warp was upgraded 1.11 → 1.15** (commit `fb15756`) *after* the sweep runs
  finished. The P0 runs are the first continual-distill runs on 1.15, so their
  numbers are not strictly wall-clock comparable to the sweep's, though the
  distillation math is unchanged.
- GPU pinning: `slurmstepd` overwrites `CUDA_VISIBLE_DEVICES` passed via
  `--export`, so `_cl_run.sh` re-exports it inside the step from `RUN_GPU`.
  Verify placement with `nvidia-smi --query-compute-apps` — each run must show a
  distinct `gpu_uuid`, per `~/use_instructions/README.md`.
