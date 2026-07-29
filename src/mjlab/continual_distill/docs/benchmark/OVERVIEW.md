# Manipulation-Diversity Benchmark — OVERVIEW

Single entry point. Vision, plan, what's done, code paths, how to run.
Companion docs (same folder): `PLAN.md` (roadmap), `STATUS.md` (live state),
`LOG.md` (dated journal), `AUTHORING_GUIDE.md` (how to add tasks),
`manifest.json` (machine-readable task list).

---

## 1. Vision

A large, **vision-free** (state-only, low-dim obs), **manipulation-skill-diverse**
continual-learning benchmark, one simulator (**mjlab / MuJoCo-MJX / warp**), spanning
**3 embodiment classes**:

- **A — arm + 2-finger gripper** (Franka Panda) — action 8-D
- **B — arm + 5-finger hand** (Franka + LEAP) — action 23-D
- **C — floating 5-finger hand** (LEAP, actuated 6-DoF base) — action 22-D

Diversity axis = **manipulation SKILL** (RLBench-style), not scene/spatial/language
(anti-LIBERO). Goal: **as many genuinely distinct vision-free tasks as we reasonably
can** — a real benchmark contribution for the field, not just an eval set.

**The gap we fill:** no existing MuJoCo benchmark is simultaneously vision-free +
skill-diverse + multi-embodiment (incl. dexterous) + CL-oriented, all in one stack.

### Two stages (scope discipline)
- **Stage 1 (THIS effort): BUILD the task suite.** A task is done when structurally
  sound — builds, resets, steps, finite obs/reward, sane shapes, a clear success
  predicate. Acceptance gate = `benchmark-smoke`.
- **Stage 2 (LATER): SOLVE it.** Teachers, RL, continual learning, train-solvability,
  evaluation. NOT in scope now. Train-solvability is NOT a suite-membership gate.

### Design rule: uniform interfaces
obs / action / reward / success kept **similar across tasks**. Joint-space action
everywhere (no EE-delta). Dexterous hands = full joint targets. The SAME reach / lift /
stack / insertion MDP bases serve all 3 embodiments (only entity + EE-site + scale
differ). This is what makes adding tasks cheap.

### Fragility axis (a tag, sampled deliberately)
planar → mild-contact → precision-grasp → dexterous. All 4 tiers present, so downstream
CL comparisons are discriminating (an all-planar suite washes out method signal).

---

## 2. Where it lives (code paths)

Repo: `mjlab` (GitHub `sudhirpratapyadav/mjlab`), branch **`benchmark-manip-diversity`**.
- Local: `/media/cvlab/EXTDRIVE/sudhir/continual_learning/mjlab` (dev, RTX A6000 / sm_86).
- **Cluster: `svs_ald:/ihub/homedirs/svs_ald/sudhir/mjlab`** (same branch; A100 / sm_80).
  Sync = git push local → `git pull` on cluster. `.venv` is uv-managed (use `uv pip`).

```
src/mjlab/
  tasks/manipulation/
    taxonomy.py              # Embodiment / SkillFamily / Fragility enums + TaskTaxonomy
    benchmark.py             # query API (filter/group) + CL orderings + manifest export
    <skill>_env_cfg.py       # SHARED per-skill bases: make_{lift,reach,stack,...}_env_cfg
    mdp/                     # reusable obs/reward/termination/COMMAND terms
      commands.py            #   Lifting/Reaching/Stacking/Pushing/OpenDoor... commands
    config/
      franka/                # Class A tasks (env_cfgs.py + rl_cfg.py + __init__ register)
      leap_hand/             # Class C tasks (floating LEAP)
      franka_leap/           # Class B tasks (Franka + LEAP)
  asset_zoo/
    objects/free/            # cube, cuboid, cylinder, disc, sphere, ellipsoid, peg_in_hole
    objects/articulated/     # door, drawer, button
    robots/
      franka_emika_panda/    # arm
      leap_hand/             # floating dexterous hand (actuated 6-DoF base)
      franka_leap/           # arm+hand (composed via MjSpec: delete gripper, attach hand)
  scripts/
    benchmark_smoke.py       # structural validation (build/reset/step/shapes) — the GATE
    benchmark_validate.py    # optional: brief PPO run, report success (stage-two peek)
  continual_distill/         # the CL CONSUMER (reads env_id + teacher datasets); stage two
    docs/benchmark/          # THIS folder (OVERVIEW/PLAN/STATUS/LOG/AUTHORING_GUIDE/manifest)
tests/
  test_benchmark_taxonomy.py # 8 tests for taxonomy/query/ordering
  test_task_configs.py       # all registered task cfgs build
```

A **task** = registered `env_id` (obs vector + action interface + success predicate) +
optional taxonomy tag. Registration: `register_mjlab_task(...)` in each `config/<robot>/__init__.py`.

---

## 3. Plan (phases)

- **P0** setup + repo audit ✓
- **P1** taxonomy / query / CL-ordering infra + tagged registry ✓
- **P2** authoring loop proven + local validation tooling + guide ✓
- **P3** new skills (reach, stack, insertion) ✓
- **P4** all 3 embodiment classes live ✓  ← **we are here**
- **P5** more breadth per class + 2nd dexterous hand (Shadow/Allegro) + procedural
  scaling (PartNet-Mobility / object libs → 100+ instances) — NEXT
- (stage two) teachers, RL/CL, evaluation

Full detail in `PLAN.md`; dated decisions in `LOG.md`.

---

## 4. Achieved (current state)

**20 tasks · 5 skills · 3 embodiment classes · all 4 fragility tiers.**
All pass `benchmark-smoke` (isolated); 17 unit tests green.

| Embodiment | # | Action | Tasks |
|---|---|---|---|
| arm_gripper (A) | 12 | 8-D | reach; lift ×4 (cube/cylinder/sphere/ellipsoid); stack; peg-insertion; push ×2 (cuboid/disc); articulation ×3 (door/drawer/button) |
| floating_hand (C) | 5 | 22-D | reach; lift ×2 (cube/sphere); stack; peg-insertion |
| arm_hand (B) | 3 | 23-D | reach; lift-cube; stack |

Skills: reach, pick_place, planar_push, articulation, insertion.
Fragility: planar 4, mild_contact 4, precision_grasp 10, dexterous 2.

**Infra built:** taxonomy + query/ordering API + JSON manifest; `benchmark-smoke`
(+`--isolate`) and `benchmark-validate` tools; authoring guide; new assets
(sphere, ellipsoid, peg+hole, LEAP hand, Franka+LEAP).

**Proven integration recipes (reusable):**
- New object: cube-template (xml + constants + `__init__`) → plug into a lift/stack helper.
- New floating hand: Menagerie model → add actuated 6-DoF base + grasp_site → robot cfg.
- New arm+hand: `MjSpec` delete gripper + attach fixed hand at link7.

### Known issues / stage-two TODO
- **mujoco-warp 0.0.1 segfaults on some collision geoms on sm_80 (A100), not sm_86
  (A6000)** — same code, GPU-specific kernel bug. Worked around by geom swaps that keep
  a distinct grasp shape and are warp-safe:
  - LEAP hand: mesh collision disabled (per-phalanx box colliders keep contact).
  - cylinder / disc / ellipsoid: → **capsule** (object + mocap goal).
  Stage-two: revisit with newer mujoco-warp / primitive fingertip colliders.
- **Cluster sweep must use `--isolate`** (subprocess per task): building many warp envs
  in one process corrupts CUDA state → segfault, even when each task passes alone.
- Hand tasks smoke-pass but need training to solve (stage two) — expected for new
  22/23-D embodiments.

---

## 5. How to run

Local (has `uv` / `.venv`):
```bash
# list benchmark tasks
python -c "import mjlab; from mjlab.tasks.manipulation import benchmark; \
  print('\n'.join(sorted(benchmark.all_benchmark_tasks())))"

# structural smoke test (the acceptance gate)
python -m mjlab.scripts.benchmark_smoke --isolate            # all tasks, robust
python -m mjlab.scripts.benchmark_smoke --keyword Leap       # subset

# optional stage-two learnability peek
python -m mjlab.scripts.benchmark_validate --task Mjlab-Reach-Target-Franka --iters 150

# regenerate manifest.json
python -c "import mjlab; from mjlab.tasks.manipulation import benchmark; \
  benchmark.export_manifest('src/mjlab/continual_distill/docs/benchmark/manifest.json')"

# train / play a task (stage two)
uv run train Mjlab-Lift-Cube-Franka --env.scene.num-envs 4096
uv run play  Mjlab-Lift-Cube-Franka
```

**Cluster (`svs_ald`), GPU via holder job** (see `~/use_instructions/README.md`):
```bash
# find/hold GPUs, then run a step inside the holder pinned to one GPU:
squeue -a -o "%A %j %T %N %b" | grep -i hold      # find HOLDER_JOBID
RUN_GPU=0 srun --jobid=<HOLDER_JOBID> --overlap --cpus-per-task=8 --export=ALL,RUN_GPU \
  bash -c 'export CUDA_VISIBLE_DEVICES=$RUN_GPU; cd ~/sudhir/mjlab; \
    PYTHONPATH=src .venv/bin/python -m mjlab.scripts.benchmark_smoke --isolate'
```
Use `--isolate` on cluster: building many warp envs in one process segfaults on A100.

---

## 6. Add a new task (short)
See `AUTHORING_GUIDE.md`. Cheap path: pick a base maker → in `config/<robot>/env_cfgs.py`
swap object entity + names → add `rl_cfg` → `register_mjlab_task(..., taxonomy=...)` →
`benchmark-smoke --keyword <name>` → regenerate manifest. Guardrail: prefer new SKILLS
over object variants; don't inflate count with near-duplicates; tag fragility honestly.
```
