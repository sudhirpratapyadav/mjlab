# Plan — one RL teacher per task

## P0: establish a reproducible baseline

Completed: active 24-task list, 60D/8D contract verification, effective config/source-hash inventory, GPU/allocation check and a two-update PPO/checkpoint smoke.

Before a full pilot:
- Implement or validate the RL checkpoint evaluator with deterministic actions and first-episode accounting. Exercise terminal success, timeout, auto-reset, subset resets and failed episodes; retain raw per-episode outcomes.
- Archive the effective config, current code/patch/assets, seed and GPU UUID. Current worktree is dirty; HEAD alone is not a reproducibility record.
- Select current physics as an explicit baseline or deliberately correct the eight zero-gravity presets and revalidate. Do not silently mix metrics from different physics.
- Keep the approved 60D observations and identical 8D actions. Missing dynamic/history information is a documented limitation, not authorization to restore 143D.

## P1: Reach and Lift pilots

Use one GPU per task and start with 1,024 environments, 500 PPO iterations and one fixed training seed. These are proposed pilot budgets, not launched jobs or a claim that they are sufficient. Default scene count is one, so always override it. Check GPU memory, control/observation finiteness, terminal reasons, action saturation, reward terms, gradient/loss behavior and actual success curves before increasing the budget or environment count. Scale to 2,048/4,096 environments only after measurement supports it.

Run Reach first through the complete launcher/config-saving/checkpoint/evaluation path, then Lift. Train fresh networks under the normalized action contract. Do not infer RL success from classical-teacher rates.

## P2: repair task-specific learning obstacles

- Ten articulation tasks: replace or calibrate the saturated approach signal. At audited reset distances 0.689–0.939 m, `1-tanh(30*d^4)` was effectively zero. Preserve the actual joint-space goal objective. Review constant non-collision bonuses and Door's zero regularizers.
- Cage: retain the common action mapping. With finger target `.02 + .02*a`, preserving the approximately .06949 m aperture requires `a > .737` once tracking settles. A centered initial Gaussian tends to violate the irreversible no-pinch rule. Test an open-gripper initialization/prior or a documented training curriculum, without adding task-specific observation channels or secretly changing action semantics.
- Stack/Place/Peg: final rewards must incentivize release and settled support/containment/seating. Peg's inserted goal reference is corrected, but its approach reward targets the bottom tip instead of a safe grasp point.
- Edge/Pivot: provide useful shaping for the precursor contact sequence; inspect physical progress rather than only reward totals.
- Throw/Strike: tune velocity penalties and sequence shaping against the required fast motion. Evaluate compact-state limitations empirically.
- Reorient/Topple: verify approach signal and settled completion incentive.

Record each reward/config experiment before running it. Preserve strict success criteria; improved shaped return alone is not improvement. No reward/physics modifications were made in this setup pass.

## P3: train and certify all 24

Run at most one initial large job per assigned GPU1–3, leaving memory headroom until throughput is measured. Maintain one independent policy per task; train additional seeds or targeted variants only as needed. Promote checkpoints using validation seeds, then confirm the retained teacher on fresh episodes per GOAL.md. Threshold: strictly >90% on each task, not a suite average. Archive one retained checkpoint/normalizer per task plus its verification and videos.

## Launch template (not yet a completed full pilot)

From the repository root, after readiness items for the chosen task are addressed:

```bash
srun --jobid=20277 --overlap -n1 --cpus-per-task=8 bash -c '
  export FORCE_CPU=0
  export CUDA_VISIBLE_DEVICES=GPU-23a5dcb4-5248-01aa-94a3-d0f998660161
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  taskset -c 2,130 .venv/bin/python src/mjlab/continual_distill/docs/cl_v4_rl/train_teacher.py     --task Mjlab-Reach-Target-Franka --run-id RL-001-reach-s20260912     --num-envs 1024 --iterations 500 --seed 20260912
'
```

The launcher defaults to local TensorBoard, rejects non-active tasks and existing run directories, and preserves an already bound GPU. Its dry-run has been checked. No generic GPU selector is invoked because that selector rejects UUID strings. Recheck actual GPU process placement with nvidia-smi after launch. Never cancel the shared holder or another person's jobs.
