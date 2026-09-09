# Phase 1 — PLAN

> **Objective: a classical (scripted) teacher at >= 0.90 success rate for every one of
> the 25 Class-A tasks, with a rollout video for each published to
> https://cl.untuai.com.**
>
> Read `../PURPOSE.md` and `../CONTEXT.md` first — this file assumes both.
> `EXPERIMENTS.md` is the run table, `STATUS.md` the scoreboard, `LOGS.md` the journal.

---

## 0. BEFORE YOU START — read this, then the cluster docs

**Read `~/use_instructions/README.md` before submitting, joining or cancelling ANY
Slurm job.** Not optional, and not replaceable by the summary below. Everything in
Phase 1 needs a GPU, so every agent touches this.

The team does **not** use one `sbatch` per experiment. It uses **hold-and-run**:

1. A long-lived **holder** job allocates 8 GPUs on a node and sleeps
   (`~/use_instructions/hold_gpus.sh`).
2. Experiments run as **steps inside that allocation**, each pinned to one GPU:
   ```bash
   bash ~/use_instructions/run_in_holder.sh <HOLDER_JOBID> <GPU_IDX> <LOGFILE> <command...>
   ```

The rules, in order of how much damage breaking them does:

- **NEVER `scancel` a job you did not create.** The account `svs_ald` is SHARED —
  `squeue` shows everyone's jobs mixed together and the job NAME is the only ownership
  hint. A holder sitting at 0% GPU utilisation is **not** necessarily idle: cancelling
  it kills every run inside it, including other people's. Ask the owner first.
- **Always look for an existing holder and reuse it.** Do not queue your own
  allocation per run:
  ```bash
  squeue -a -o "%A %u %j %T %M %N %b" | grep -i hold
  ```
- **Check which GPUs are actually free before claiming one**, from inside the holder:
  ```bash
  srun --jobid=<HOLDER> --overlap -n1 nvidia-smi       --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
  ```
  Pick an idle index. With several agents running in parallel, **say which GPU index
  you took in `LOGS.md`** so the next agent does not land on it.
- **Never run `nvidia-smi` on the login node** — it has no GPU and the command is
  aliased to `srun` there.
- **Pass `-n1` to `srun`**, or your command runs twice.
- If you do submit a holder yourself, give it a recognisable name
  (`hold_dgx2_<name>`) and `scancel` it when you are done — but only yours.
- Inspect what a holder is running with `squeue -s -j <HOLDER_JOBID>`.

Cluster facts: partition `1gpu`, nodes `dgx1` / `dgx2`, 8x A100-80GB each. CUDA 12.4 at
`/usr/local/cuda` on the compute nodes, not on `PATH` by default. Rendering needs
`MUJOCO_GL=egl` and a GPU node — the login node has no EGL device and will fail.

Run python as `PYTHONPATH=src .venv/bin/python` from the repo root.

---

## 1. What the 0.90 bar actually demands

This is a much harder target than "every task has a teacher", and the plan has to say
so up front. Measured state of the 25 from the previous pass:

| band | count | tasks |
|---|---|---|
| **already >= 0.90** | 5 | Reach-Target, Lift-Cube, Push-Button, Slide-Window, Open-Lid (all 1.000) |
| 0.5–0.89 | 5 | Turn-Lever 0.812, Open-Drawer 0.719, Flip-Switch 0.615, Reorient-Object 0.594, Rotate-Valve 0.500 |
| below 0.5 | 6 | Stack-Cube ~0.3, Place-In-Container ~0.28, Push-Cuboid 0.177, Tool-Pull 0.039, Peg-Insertion ~0.05, Open-Door 0.000 |
| no teacher yet | 9 | the Wave-1 tasks |

So **5 of 25 clear the bar today**, and 20 do not. Three of those 20 are characterised
failures that a previous pass already fought and lost with the same gripper and the same
task definitions (Open-Door, Tool-Pull, Peg-Insertion). Three more are RL-by-design.

That does not make the target wrong — it makes it the *point* of the phase. But plan
for it honestly: **the expected outcome is not 25/25.** A task that ends at 0.62 with a
mechanism-level explanation of the residual 38% is a completed Phase-1 task, not a
failure, and it hands Phase 2 a precise brief.

## 2. Lanes

Four lanes, run in parallel. Lane assignment is by *what the work actually is*, not by
task count — the lanes have very different shapes.

### Lane 0 — Infrastructure (ONE agent, runs first, blocks the video deliverable)

Nobody else can publish until this exists. Deliverables:

1. **A teacher-rollout video tool.** `classical/render_rollout.py` is stale: it
   hardcodes 4 policies and emits PNG frames, not mp4. Rewrite it to read
   `CLASSICAL_POLICIES`, take any task ID, run N episodes, and write an mp4 of a
   representative success and (where one exists) a representative failure. Reuse the
   offscreen-render path from `mjlab/scripts/record_task_videos.py` — it already
   handles `MUJOCO_GL=egl`, framing overrides and H.264 encoding.
2. **The site restructure** (section 5), including moving the existing 37 benchmark
   clips under `/benchmark/` and building the new landing page.
3. **A publish script** so 20+ agents are not each inventing an scp command.

### Lane A — Wave-1 scripted teachers (6 tasks, no teacher exists)

Drag-Pull, Cage-Drag, Topple-Block, Push-Flap, Axial-Extract, Edge-Grasp.

Greenfield: write the policy, get it working, push it to 0.90. These are the tasks
Wave 1 explicitly designed as waypointable, so 0.90 is a reasonable target. One agent
per task, or one agent per two related tasks (Drag-Pull + Cage-Drag are both planar
transport; Topple-Block + Push-Flap are both non-prehensile pokes).

### Lane B — Lift to the bar (5 tasks, teacher exists at 0.5–0.89)

Turn-Lever, Open-Drawer, Flip-Switch, Reorient-Object, Rotate-Valve.

The most likely wins in the whole phase. Every one already works most of the time, so
the residual is a *specific* recurring failure — find it, do not tune around it. All
four material gains in the previous pass came from a mechanism bug, none from constants.

### Lane C — The hard six (6 tasks, below 0.5)

Stack-Cube, Place-In-Container, Push-Cuboid, Tool-Pull, Peg-Insertion, Open-Door.

Do NOT start here, and do not assign a fresh agent to Open-Door or Tool-Pull without
first reading their characterisations in `../../benchmark/CLASSICAL_TEACHERS.md`. Two
sub-problems are already identified and are worth attacking directly:

- **Grasp retention** (Stack, Place-In-Container). Stack drops the cube in 22/32 runs,
  and 9 of the 10 runs that kept hold succeeded. Fix retention and both tasks move at
  once — this is the single highest-leverage item in Lane C.
- **Pusher overtake** (Push-Cuboid). Known mechanism, 7 measured strategies already
  failed; read that section before attempt 8.

### Lane D — RL-by-design (3 tasks): attempt, then characterise

Strike-Slide, Pivot-Lift, Throw-To-Bin. Wave 1 called these RL-dense because impulse
calibration, contact-mode switching and release timing defeat waypoint scripting.
The instruction is "classical teachers for all tasks", so **attempt them** — but
time-box it. A crisp measured characterisation of *why* scripting caps out is worth
more than a fourth week of waypoint tuning, and it is exactly the brief Phase 2 needs.

## 3. The per-task loop

Every agent runs the same loop. It is deliberately measurement-first.

1. **Watch the task video** (https://cl.untuai.com) and read its `*_env_cfg.py` and its
   command term in `mdp/commands.py`. Know the success predicate exactly before writing
   a line of policy.
2. **Baseline it.** For an existing teacher, measure on current HEAD before touching
   anything — the placement audit moved five tasks and the numbers on record are all
   pre-audit.
3. **Instrument, then diagnose.** When it fails, find out *what* fails: which phase,
   which env, what the state machine was doing. The rule that held 4/4 times: the
   existing diagnosis names the right symptom and the wrong cause.
4. **Fix the mechanism.** Not the constant.
5. **Re-measure** at the protocol in section 4.
6. **Publish** the rollout video (section 5).
7. **Record**: `STATUS.md` row + dated `LOGS.md` entry, including what did NOT work.

## 4. Measurement protocol

The 0.90 bar needs enough episodes to be a claim rather than a coincidence.

| purpose | episodes | note |
|---|---|---|
| iterating | 32 | fast signal only; never quote as a result |
| claiming a number below 0.3 | 96+ | weak teachers are unresolvable at n=32 |
| **claiming >= 0.90** | **128** | at n=128 a true 0.90 reads 0.90 +- 0.05; at n=32 it reads +- 0.10 and 0.90 is indistinguishable from 0.81 |

Always report `SR (n)` and the HEAD you measured on (`git rev-parse --short HEAD`).
Never merge episodes from different HEADs into one number.

**Do not change the task to reach the bar.** Env cfgs, command terms and success
predicates are frozen for this phase. If you believe a task is genuinely broken (a
placement error, a wrong predicate), that is a benchmark bug: raise it in `LOGS.md`,
flag it as a blocker in `STATUS.md`, and do not fold the fix into your teacher.

## 5. Video deliverable and the site layout

Every task ships a rollout video. The site is being reorganised as part of this phase
because a flat directory of 37 mp4s will not survive another 25-50 clips.

**Target layout on the VPS** (`untu_vps:~/sudhir/continual_learning/`, served by Caddy
at https://cl.untuai.com — the vhost already exists, no server work needed):

```
/                     landing page: what this is, links to both galleries
/benchmark/           the 37 random-agent task clips (MOVED here from the root)
    index.html  *.mp4  thumbs/  manifest.json
/phase1/              classical-teacher rollouts
    index.html        per-task cards: SR, n, HEAD, date, pass/fail against the 0.90 bar
    manifest.json
    <Task-Id>/
        teacher.mp4   a representative SUCCESSFUL rollout
        failure.mp4   a representative FAILURE (required for any task below 0.90 —
                      the failure clip is the evidence for the characterisation)
        thumb.jpg
```

Rules:

- **Lane 0 owns the layout and the landing page.** Task agents only add their own
  `phase1/<Task-Id>/` directory. Nobody edits another task's directory.
- The `phase1/index.html` card must show the number **with its `n`** — a video without
  its measured SR invites exactly the "looks great" conclusion this program exists to
  avoid.
- Publishing is `scp` to `untu_vps:~/sudhir/continual_learning/...`. Caddy serves it
  immediately; there is no build or deploy step.
- Moving the existing clips into `/benchmark/` changes their URLs. Nothing external
  depends on them, but do it in one pass and update the landing page in the same pass.

## 6. Exit criteria

Phase 1 is done when:

- [ ] All 25 tasks have a resolved status: `>= 0.90`, or a below-bar number **with a
      mechanism-level characterisation and a failure video**
- [ ] Every claimed `>= 0.90` was measured at n = 128 on a named HEAD
- [ ] Every task has a rollout video published under `/phase1/`
- [ ] The site restructure is live and the landing page lists both galleries
- [ ] `STATUS.md` summary counts match the per-task rows (they are checked, not assumed)
- [ ] Tasks that cannot reach 0.90 are handed to Phase 2 with a written brief saying
      what specifically defeats scripting

## 7. Known traps

Repeated from `../CONTEXT.md` because they are the ones that will actually bite:

- `srun` without `-n1` runs your command **twice**.
- Login-node `ls -l` reports **stale mtimes** for a minute after a compute-node write.
- Cage-Drag latches a **minimum** aperture — a teacher that pinches has already lost,
  silently, with no error.
- Push-Flap's hinge target is **negative**.
- Mechanism tasks run **gravity-off**; free-object tasks run gravity-on.
- Strike-Slide and Throw-To-Bin have goals **deliberately outside the reach envelope**.
- Never `scancel` a holder you did not create.
