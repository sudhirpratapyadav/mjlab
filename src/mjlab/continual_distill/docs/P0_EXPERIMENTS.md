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
| `nosi_best_s0` | best | 0 | 🔄 | |
| `nosi_best_s1` | best | 1 | 🔄 | |
| `nosi_best_s2` | best | 2 | ⏳ | |
| `nosi_worst_s0` | worst | 0 | 🔄 | |
| `nosi_worst_s1` | worst | 1 | 🔄 | |
| `nosi_worst_s2` | worst | 2 | ⏳ | |

**Expected:** catastrophic forgetting, final avg ≈ 0.25–0.4. If forgetting does NOT
appear, that is the critical finding — it would mean SI is not what protects the
skills, and the mechanism claim needs revisiting.

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
| `w8192_lr1e5_s0/1/2` | 1e-5 | 0,1,2 | ⏳ | |
| `w8192_lr5e6_s0/1/2` | 5e-6 | 0,1,2 | ⏳ | |
| `w8192_lr1e6_s0/1/2` | 1e-6 | 0,1,2 | ⏳ | |

**Both outcomes publishable:** if a lower LR recovers 8192, the collapse is an
optimizer artifact and §5.G becomes "the LR must scale with width." If it does not,
the over-parameterization claim is much stronger for having ruled out the confound.

## Results

_Auto-collected by `slurm/collect_p0.py`._

| group | run | status | avg SR | per-task |
|---|---|---|---|---|
| P0-1 | `nosi_best_s0` | 🔄 running | — | — |
| P0-1 | `nosi_best_s1` | 🔄 running | — | — |
| P0-1 | `nosi_best_s2` | 🔄 running | — | — |
| P0-1 | `nosi_worst_s0` | 🔄 running | — | — |
| P0-1 | `nosi_worst_s1` | 🔄 running | — | — |
| P0-1 | `nosi_worst_s2` | 🔄 running | — | — |
| P0-2 | `joint_s0` | ⏳ queued | — | — |
| P0-2 | `joint_s1` | ⏳ queued | — | — |
| P0-2 | `joint_s2` | ⏳ queued | — | — |
| P0-3 | `w8192_lr1e5_s0` | 🔄 running | — | — |
| P0-3 | `w8192_lr1e5_s1` | 🔄 running | — | — |
| P0-3 | `w8192_lr1e5_s2` | ⏳ queued | — | — |
| P0-3 | `w8192_lr5e6_s0` | ⏳ queued | — | — |
| P0-3 | `w8192_lr5e6_s1` | ⏳ queued | — | — |
| P0-3 | `w8192_lr5e6_s2` | ⏳ queued | — | — |
| P0-3 | `w8192_lr1e6_s0` | ⏳ queued | — | — |
| P0-3 | `w8192_lr1e6_s1` | ⏳ queued | — | — |
| P0-3 | `w8192_lr1e6_s2` | ⏳ queued | — | — |


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
