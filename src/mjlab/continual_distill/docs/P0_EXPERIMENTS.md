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
