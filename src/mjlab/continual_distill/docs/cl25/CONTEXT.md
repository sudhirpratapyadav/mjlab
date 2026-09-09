# CL-25 — CONTEXT

> Everything an agent needs to start working without reading two months of history.
> If something here contradicts an older doc, THIS file is newer — but say so in
> `LOGS.md` rather than silently diverging.
>
> Written 2026-09-08. Repo `/ihub/homedirs/svs_ald/sudhir/mjlab`, branch
> `benchmark-manip-diversity`.

---

## 1. The 25 tasks, and where each one actually stands

All are Class A: **Franka arm + 2-finger parallel gripper, uniform 8-D joint action**.
This is every registered Franka-2F task except the four pure object swaps
(Lift-Cylinder / Lift-Sphere / Lift-Ellipsoid collapse into Lift-Cube; Push-Disc
collapses into Push-Cuboid) — i.e. exactly the 25 distinct Class-A motion profiles.
Task IDs are the gym IDs; inspect one with
`mjlab.tasks.registry.load_env_cfg(task_id)`.

`SR` = measured classical-teacher success rate from the previous teacher pass
(`docs/benchmark/CLASSICAL_TEACHERS.md`, final table). `n` = episodes it was measured
over. **Every one of these numbers predates the 2026-09-02 placement audit and the
2026-09-08 retag — re-verify before trusting any of them.**

### Group A — classical teacher EXISTS (16 tasks)

| Task ID | skill | tier | SR | n | state |
|---|---|---|---|---|---|
| `Mjlab-Reach-Target-Franka` | reach | 1 | 1.000 | 32 | solid |
| `Mjlab-Lift-Cube-Franka` | pick_place | 3 | 1.000 | 32 | solid |
| `Mjlab-Push-Button-Franka` | articulation | 1 | 1.000 | 32 | solid |
| `Mjlab-Slide-Window-Franka` | articulation | 2 | 1.000 | 32 | solid |
| `Mjlab-Open-Lid-Franka` | articulation | 2 | 1.000 | 32 | solid |
| `Mjlab-Turn-Lever-Franka` | articulation | 2 | 0.812 | 32 | solid |
| `Mjlab-Open-Drawer-Franka` | articulation | 2 | 0.719 | 32 | solid |
| `Mjlab-Flip-Switch-Franka` | articulation | 2 | 0.615 | 96 | usable |
| `Mjlab-Reorient-Object-Franka` | pick_place | 3 | 0.594 | 32 | usable |
| `Mjlab-Rotate-Valve-Franka` | articulation | 4 | 0.500 | 32 | usable |
| `Mjlab-Stack-Cube-Franka` | pick_place | 3 | 0.28–0.375 | 96 | **below bar** — grasp RETENTION: drops the cube in 22/32, and 9 of the 10 runs that kept hold succeeded |
| `Mjlab-Place-In-Container-Franka` | pick_place | 3 | 0.27–0.29 | 96 | **below bar** — same retention failure |
| `Mjlab-Push-Cuboid-Franka` | planar_push | 2 | **0.223** | 128 x2 | **below bar** — current-HEAD baseline (0.219 / 0.227). The older 0.078 AND 0.177 figures were measured under **pre-audit placement** and do not describe this task; see LOGS 2026-09-09. Mechanism is endgame precision, NOT pusher-overtake (disproven, 1/16) |
| `Mjlab-Tool-Pull-Franka` | tool_use | 4 | 0.039 | 128 | **failed, characterised** — pinch ejects the 26 cm shaft axially; reproduced over 7 grasp heights x 3 grasp points x 2 orientations |
| `Mjlab-Peg-Insertion-Franka` | insertion | 4 | 0.031–0.062 | 96 | **failed** — user excluded it from the last improvement pass |
| `Mjlab-Open-Door-Franka` | articulation | 2 | 0.000 | 32 | **failed, characterised** — needs 84.3 of 90 deg with no partial credit in 150 steps while a holding pull advances 1–2 deg per 25 steps |

The five tasks used in the earlier CL experiments (Push-Cuboid, Push-Button, Open-Door,
Open-Drawer, Lift-Cube) are all in this group. They are **in the suite**, not excluded
from it — which is what makes the old 4-task and 5-task sequences nested subsets of
this one, and their P0/P1 results directly comparable.

### Group B — NO teacher yet (9 tasks, all from Wave 1, 2026-08-04)

Wave 1's own design notes already call which route each one needs:

| Task ID | skill | tier | intended route | why |
|---|---|---|---|---|
| `Mjlab-Drag-Pull-Franka` | planar_push | 2 | **scripted** | waypointable: engage far face, retract |
| `Mjlab-Cage-Drag-Franka` | non_prehensile | 2 | **scripted** | straddle + transport, fingers never close |
| `Mjlab-Topple-Block-Franka` | non_prehensile | 2 | **scripted** | poke above CoM |
| `Mjlab-Push-Flap-Franka` | articulation | 2 | **scripted** | hinge-arc face push |
| `Mjlab-Axial-Extract-Franka` | insertion | 3 | **scripted** | pinch, break ~4 N friction, lift straight |
| `Mjlab-Edge-Grasp-Franka` | pick_place | 3 | **scripted** (RL fallback) | drag to overhang, pinch exposed 16 mm |
| `Mjlab-Strike-Slide-Franka` | planar_push | 3 | **RL-dense** | impulse calibration defeats waypoints |
| `Mjlab-Pivot-Lift-Franka` | pick_place | 3 | **RL-dense** | two-phase contact-mode switch |
| `Mjlab-Throw-To-Bin-Franka` | pick_place | 3 | **RL-dense** | release timing |

### What that arithmetic means for Phase 1

- **6 new scripted teachers** to write (Drag-Pull, Cage-Drag, Topple-Block, Push-Flap,
  Axial-Extract, Edge-Grasp).
- **16 existing teachers** to re-verify against current HEAD.
- **3 tasks are RL-by-design** and are NOT Phase-1 work.
- Of the 16 existing, **10 are at 0.5 or above** and **6 are below** (Stack,
  Place-In-Container, Push-Cuboid, Tool-Pull, Peg-Insertion, Open-Door). Three of those
  six are characterised failures that a previous pass already fought and lost.

So Phase 1 covers **22 tasks**, and the realistic ceiling is **~16 teachers above the
bar out of 25, not 25**. That is the honest number to plan against. Do not discover it
at the end.

## 2. Repo orientation

```
src/mjlab/
  tasks/manipulation/
    *_env_cfg.py              one per task family (22 files)
    mdp/commands.py           EVERY command term + success predicate (92 KB)
    mdp/{rewards,observations,terminations,curriculums}.py
    config/franka/__init__.py THE REGISTRY — task IDs + TaskTaxonomy tags
    config/franka/env_cfgs.py per-task env cfg factories (73 KB)
    taxonomy.py               Embodiment / SkillFamily / Fragility enums
    workspace.py              reach-envelope constants, grasp_box(), goal_box()
    benchmark.py              filter/group API, CL orderings, export_manifest()
  continual_distill/
    classical/                SCRIPTED TEACHERS — your Phase-1 workplace
    bc/, bc_teachers/         behaviour cloning — OUT OF SCOPE
    teacher_datasets/         extracted teacher rollouts (573 MB)
    results/                  CL run outputs, 95 dirs, 25 GB, *.pkl checkpoints
    config/tasks_*.yaml       CL sequence definitions
    continual_distill.py      stage-2 entrypoint
    extract_teacher_dataset.py  teacher -> dataset
    docs/benchmark/           benchmark build docs (STATUS/PLAN/LOG/CATALOG/...)
    docs/cl25/                THIS PROGRAM
      PURPOSE.md CONTEXT.md    program-level (this file)
      phase_1/                 phase_1_plan.md + EXPERIMENTS/STATUS/LOGS
  scripts/
    record_task_videos.py, serve_task_videos.py, audit_workspace.py, benchmark_smoke.py
tests/                        pytest suites (44 green over the four task suites)
```

Task videos for all 37 tasks are published at **https://cl.untuai.com** — watch the
clip before writing a teacher for a task you have not seen.

---

## 3. How to run things

**Python.** Always `PYTHONPATH=src .venv/bin/python ...` from the repo root. There is
no installed package; the venv is at `.venv`.

**GPU. Read `~/use_instructions/README.md` before submitting, joining or cancelling
any Slurm job** — that file is authoritative, this is only the summary.
The login node has NO GPU, and `nvidia-smi` there is aliased to `srun` — do
not run it. Work goes through a **holder** job (a long-lived allocation that sleeps
while you run steps inside it):

```bash
# 1. find an existing holder — ALWAYS reuse, never queue your own
squeue -a -o "%A %u %j %T %M %N %b" | grep -i hold

# 2. run inside it, pinned to one GPU
bash ~/use_instructions/run_in_holder.sh <HOLDER_JOBID> <GPU_IDX> <LOGFILE> <command...>

# 3. or a one-off step
srun --jobid=<HOLDER> --overlap -n1 --cpus-per-task=4 <cmd>
```

- **NEVER `scancel` a job you did not create.** A holder at 0% utilisation is not idle
  — cancelling it kills every run inside it, including other people's.
- Check which GPUs are free before claiming one; pick an idle index.
- **Pass `-n1` to `srun`**, or your command runs twice (learned the hard way — a
  thumbnail generator raced with itself).
- The account `svs_ald` is SHARED. `squeue` shows everyone's jobs; job name is the only
  ownership hint.

**NFS gotcha.** After a compute-node job writes to the repo share, `ls -l` on the
**login node** can report stale mtimes for a minute or more. A job's own log saying it
wrote 37 files is more trustworthy than the login node's `stat`. Verify from the node
that did the writing.

---

## 4. The classical-teacher contract

**Base class:** `mjlab/continual_distill/classical/base.py` -> `ClassicalPolicyBase`.
It gives you damped-least-squares IK against the Panda model, forward kinematics, and
orientation-error helpers. You implement the policy for ONE env instance:

```python
class MyTeacher(ClassicalPolicyBase):
  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
      ...  # returns the 8-D action for env i
```

- `__call__(obs)` fans `_act_single` over the batch; `reset(env_ids)` clears per-env
  state (state machines MUST reset here).
- Register the class in `classical/__init__.py` under `CLASSICAL_POLICIES`, keyed by
  task ID — the test harness and dataset extractor both read that registry.
- Follow an existing file for structure. `reach_target.py` is the minimal example
  (one servo law); `rotate_valve.py` is the complex end (multi-phase state machine
  with a ratchet).

**Document the observation layout in the module docstring.** Every existing teacher
does this, and it is the single thing that makes them readable — the obs vector is a
flat concatenation whose slice boundaries differ per task family.

**Test it:**

```bash
PYTHONPATH=src .venv/bin/python -m mjlab.continual_distill.classical.test_classical \
    --task Mjlab-Cage-Drag-Franka --num-envs 16 --num-episodes 2
```

`render_rollout.py` and `debug_rollout.py` in the same directory exist for eyeballing
and instrumenting a failing policy.

---

## 5. Lessons from the previous teacher pass — read these, they cost weeks

From `docs/benchmark/CLASSICAL_TEACHERS.md` (32 KB; skim its final sections):

1. **Instrument before tuning.** Four teachers improved materially in the last pass.
   All four came from finding a MECHANISM BUG; none from tuning constants. In every
   case the pre-existing diagnosis had the right symptom and the wrong cause:
   - Rotate-Valve 0.031 -> 0.500: state-machine deadlock, the handoff gate keyed on
     swept angle so a jammed engagement could never reach it.
   - Reorient 0.125 -> 0.594: one wrong collision constant killing envs during descent.
   - Flip-Switch 0.406 -> 0.615: an UNSIGNED seat gate, equally satisfied 5 cm short of
     the toggle and 5 cm past it.
   - Push-Cuboid 0.104 -> 0.177: the pusher overtaking its workpiece and shoving it
     backwards. **NOTE: superseded.** The 2026-08-01 strategy pass instrumented this and
     found the pusher does NOT overtake (1/16 envs, not the failure mode), and re-measured
     the kept config at 0.078 (128). The live failure is endgame precision. See
     `docs/benchmark/CLASSICAL_TEACHERS.md` "What was actually established".
2. **n=32 does not resolve a weak teacher.** Anything below ~0.3 needs 96+ episodes
   before its number means anything. Always report `n` with the SR.
3. **Some tasks are not scriptable with this gripper, and that is a result.**
   Tool-Pull (0.039) was reproduced across 7 grasp heights x 3 grasp points x 2
   orientation modes. Open-Door (0.000) needs 84.3 of 90 degrees with no partial credit
   in a 150-step budget while a holding pull advances 1–2 deg per 25 steps. Both are
   characterised, not merely low. Aim for that standard.
4. **Grasp retention, not placement, is what breaks the pick_place teachers.** Stack
   drops the cube in 22/32 runs — and 9 of the 10 runs that kept hold succeeded.
5. **Seven measured Push-Cuboid strategy attempts all failed** (2026-08-01). Read that
   section before trying to "just fix" a planar push.

---

## 6. Task-side facts that will bite a teacher author

- **Mechanism tasks run gravity-OFF** (door, drawer, button, lever, valve, switch,
  window, lid, flap, plug); free-object tasks run gravity-ON. Deliberate, inherited,
  and re-enabling gravity later invalidates every teacher trained before the change.
- **Two tasks have goals deliberately OUTSIDE the arm's reach envelope** — Strike-Slide
  (goal x 0.88–1.05) and Throw-To-Bin (bin x 0.78–0.90). This is by design; the object
  gets there ballistically. Do not "fix" it.
- **Four tasks are ungraspable by construction.** Block 0.10x0.14x0.18 is unspannable
  outright (every width exceeds the 0.08 m aperture). Plate (16 mm) and board (20 mm)
  are thin enough to pinch in principle but lie flat with no finger clearance — the
  whole point of Edge-Grasp and Pivot-Lift is manufacturing that clearance.
- **Cage-Drag latches a MINIMUM, not a maximum.** Its `min_aperture` constraint voids
  the episode if the fingers ever close. Every other latch in the repo is a
  success-maximum. A teacher that pinches has already lost, silently.
- **Push-Flap's target hinge rotation is NEGATIVE** (range (-1.4, 0), target -70 deg).
  A positive target would need a handle that does not exist.
- **Ledge, wall and bin are mocap fixtures written per-env** at resample; their MJCF
  pose only ever serves env 0.
- **FK staleness:** compute spawn positions from poses you just wrote, never read back
  through `site_pos_w` in the same step. This caused a real drift bug once already.
- **Placement was audited 2026-09-02** and five tasks moved (Pivot-Lift, Edge-Grasp,
  Topple-Block, Drag-Pull, the four Lift goals). `audit_workspace` reports 0 problems
  across all 29 Class-A tasks. Any teacher tuned before that date is suspect.

---

## 7. Stage 2 — what the teachers feed into

Sequence: **teacher -> `extract_teacher_dataset.py` -> dataset folder ->
`continual_distill.py` with a task YAML**.

Config format (`continual_distill/config/tasks_*.yaml`):

```yaml
tasks:
  ShortName:
    env_id: "Mjlab-Lift-Cube-Franka"
    dataset_folder: "/abs/path/to/teacher_datasets/<run>"
    num_epochs: 200
```

Conventions inherited from the P0/P1 experiments, to keep results comparable:
3 seeds, mean +- std; master trunk `[4096, 2048, 1024]`; 500 distillation epochs per
task; lr `3e-5`; metric = final average success rate plus per-task final SR.
Regularizers available: **SI**, **EWC** (per-sample Fisher, own coefficient), **L2**.

---

## 8. Working agreement for parallel agents

Phase 1 is intended to run as several agents in parallel, one (or a few) tasks each.

- **One agent owns one task file.** `classical/<task>.py` is written by exactly one
  agent. Do not edit another agent's task file.
- **`classical/__init__.py` is shared** — it is the registry every agent must add a
  line to. Append your entry; never reorder or rewrite the file.
- **Append to `LOGS.md`, never rewrite it.** Every entry is dated and names the task.
- **`STATUS.md` is the shared scoreboard.** Update only YOUR task's row.
- **`EXPERIMENTS.md` is the plan.** Read it; do not silently redefine it. If the plan
  is wrong, say so in `LOGS.md` and flag it.
- Report the measured SR **with its episode count**, and say which HEAD you measured on.
- If your task turns out to be unscriptable, write the characterisation. That is a
  completed task, not an abandoned one.

---

## 9. Definition of done for Phase 1

For each of the 22 Phase-1 tasks (6 new scripted + 16 re-verifications):

1. A teacher registered in `CLASSICAL_POLICIES` **or** a written characterisation of
   why scripting cannot do it.
2. A success rate measured on current HEAD, with `n >= 32` (`n >= 96` if SR < 0.3).
3. A row in `STATUS.md` and a dated entry in `LOGS.md`.
4. No change to any task definition (env cfg, command term, success predicate) to make
   the teacher work. A genuine task bug is a benchmark fix — raise it, do not fold it
   into the teacher.
