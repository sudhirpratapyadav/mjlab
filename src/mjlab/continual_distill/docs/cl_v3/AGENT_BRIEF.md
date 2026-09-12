# CL-V3 — AGENT BRIEF (read fully before touching anything)

## Current scope — 2026-09-12

**24 active tasks**, using the common 60D observations and normalized 8D actions.
Tool-Pull is **deferred**, not deleted: it needs both the stick and puck poses, while
this observation layout represents only one object and its goal. Keep its task,
assets, registration, teacher code and historical results for future work.
Revisit it when an agreed observation design supports both objects.

Use `mjlab.tasks.manipulation.benchmark.active_cl_tasks()` for new suite runs.
The reviewable list is [active_tasks.json](active_tasks.json). The full registry
continues to contain Tool-Pull and the four object variants outside this CL suite.
Older 25-task scoreboards and results below are historical; Tool-Pull is excluded
from current RL/CL training, evaluation aggregates and completion requirements.


You are one of the wave agents raising classical teachers to ~100 % on the 25 Class-A
Franka tasks. Repo `/ihub/homedirs/svs_ald/sudhir/mjlab`, branch `benchmark-manip-diversity`.
**Do not commit or run git write commands** — the lead commits. Read, in order:
`GOAL.md`, `PLAN.md` (§3 init spec, §4 strategy hypotheses for YOUR group, §5 protocol),
`STATUS.md` (your rows), `../cl25/phase_1/STATUS.md` ("Handover to Phase 2" + your tasks'
rows), your tasks' entries in `../cl25/phase_1/LOGS.md` (grep the task name — the failure
analyses and the graveyard of things already measured worse), `../cl_v2/HANDOVER.md`
(what changed mechanically with the real assets), `../cl_v2/CODE_MAP.md` §3–§4 (task
table, teacher constants). Then the teacher file `classical/<task>.py`, `classical/base.py`,
the task's `*_env_cfg.py`, its command term in `mdp/commands.py`, and its Franka override
in `config/franka/env_cfgs.py`.

## Cluster rules (non-negotiable)

- Read `~/use_instructions/README.md`. Holder **20277** (`hold_dgx_amit`, dgx1) is NOT
  ours: reuse it, never `scancel` anything. Use ONLY the GPU index assigned to you in
  `PLAN.md` §6. GPUs 4–7 are someone's training; GPU 0 is reserved for others.
- Every GPU command:
  ```bash
  srun --jobid=20277 --overlap --cpus-per-task=4 bash -c "export CUDA_VISIBLE_DEVICES=<GPU> MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0; cd /ihub/homedirs/svs_ald/sudhir/mjlab; PYTHONPATH=src .venv/bin/python -m <module> <args>"
  ```
  Never type `nvidia-smi` bare on the login node (it is aliased to an interactive srun
  and queues a GPU job). Long runs: `bash ~/use_instructions/run_in_holder.sh 20277 <GPU> <log> <cmd...>`.
- The login node has no GL: renders and every SR measurement go through `srun`. CPU-only
  checks (`verify_task.py`, pytest with `FORCE_CPU=1`, `audit_workspace`) run on the login node.
- `/tmp` is invisible from dgx1. GPU-side files live under `~/cl_v3_work/<your-agent>/`.
- Python: `PYTHONPATH=src .venv/bin/python ...` (never bare `python`, never `pip install`).

## The loop per task (measurement-first; PLAN.md §2 gates)

1. **Read the failure history** (phase-1 STATUS row + LOGS entries + v2 HANDOVER). Know
   the success predicate exactly (command term) and the episode budget.
2. **Confirm the init spec is frozen** (gate I `ok` in your row, set by W0). If not, wait
   for W0 or, if W0 has handed the task to you, implement PLAN §3 for it and run
   `verify_task.py --num-resets 1000` + `audit_workspace --keyword` + the placement tests.
3. **Baseline at n = 128** on the frozen spec (gate B) — unless W0 already posted it.
4. **Analyse failures** (gate A): `classical/diagnose.py --task <ID> --num-envs 32 --out
   docs/cl_v3/diag/<Task>/` gives per-env per-step phase, gripper aperture, object pose,
   contacts and the final outcome; render a failure clip with `render_rollout` if you need
   to SEE it (Read the thumb / frames). Write down, per failing env, the phase and the
   physical event. Sort failures into classes with counts.
5. **Reason about the strategy** (gate S) BEFORE editing: how would the task actually be
   done with this gripper and this object? Consider a different grasp (finger in a gap or
   hole, hook behind a handle, pinch on the thin axis), a different contact (closed
   gripper as a paddle, two-point contact), a different sequence (staged approach,
   re-grasp, unconditional per-phase timeouts), a different hand pose. Write the
   reasoning and the expected failure modes in your log, then implement.
6. **Iterate at n = 32**, re-analyse, repeat. Every phase needs an unconditional
   step-count escape (phase-1 M3). Do not re-run experiments the phase-1 graveyard
   already measured worse unless the geometry changed.
7. **Claim at n = 128** (gate T). Between 0.90 and 0.97: classify the residual and say
   whether another pass would close it; go for it if the mechanism is clear.
8. **Publish** (gate V) when >= 0.90: `render_rollout --task <ID> --out ~/cl_v3_work/<agent>/<ID>`
   then `docs/cl_v3/publish_v3.sh <ID> <dir>` with a `task.json`:
   ```json
   {"sr_v2": 0.523, "init_spec": "cube yaw ±π, base yaw ±π, robot ±10°",
    "strategy": "Lift's grasp phase reused; hand yaw aligned to cube face; close at mid-height; aperture check",
    "gates": {"I":"ok","B":"ok","A":"ok","S":"ok","T":"ok","V":"ok"}, "notes": "..."}
   ```
   `result.json` comes from `render_rollout` (sr, n, head).
9. **Record**: `update_status.py` row (gates, Baseline SR, Current SR, Strategy, Notes) +
   dated entry in `docs/cl_v3/logs/<your-agent>.md` (template in `LOGS.md`), including
   what did NOT work and the numbers.

## Task changes — the rule

Tasks and success metrics are frozen. A change is allowed ONLY when the task as
specified is physically impossible or clearly mis-specified, and it needs:
`update_status.py --decision "D<n> | <what is impossible and the measurement that shows it> | <what you changed>"`
BEFORE the change, plus the row note. Never change a task because it raises the score.
Examples of legitimate grounds: an episode budget shorter than the fastest feasible
motion (measure it), a target beyond the achievable launch speed (measure it), a
predicate that cannot be satisfied under the physics. Examples that are NOT grounds:
"the tolerance is tight", "the object is slippery", "the handle is small".

## Tools

| Tool | What |
|---|---|
| `python -m mjlab.continual_distill.classical.test_classical --task <ID> --num-envs 32 --num-episodes 4` (GPU) | the n = 128 measurement |
| `python -m mjlab.continual_distill.classical.diagnose --task <ID> --num-envs 32 --out docs/cl_v3/diag/<Task>` (GPU) | per-env failure dump + summary (W0 writes it; until then use `debug_rollout.py` or instrument your teacher) |
| `python -m mjlab.continual_distill.classical.render_rollout --task <ID> --out <dir> --batch-size 32` (GPU) | teacher.mp4 / failure.mp4 / thumb.jpg / result.json |
| `python -m mjlab.scripts.verify_task --task <ID> --num-resets 1000 --out docs/cl_v3/runs/<Task>` (CPU) | init distribution + success-at-reset check |
| `python -m mjlab.scripts.audit_workspace --keyword <Task>` ; `FORCE_CPU=1 .venv/bin/python -m pytest tests/test_workspace_placement.py tests/test_classical_teachers.py -k <task>` | regression checks; run the teacher tests after ANY teacher/registry edit |
| `docs/cl_v3/publish_v3.sh <ID> <dir>` | publish |
| `python docs/cl_v3/update_status.py ...` | the ONLY way to edit STATUS.md |

## Shared files

`env_cfgs.py`, `commands.py`, `workspace.py`, `classical/*.py` are edited by several
agents: edit only your task's function / class / constants, re-read the region right
before each edit, keep edits small, never reformat. Shared helpers you add to `base.py`
must be additive and default-off. Never edit `LOGS.md`, `STATUS.md` (except through the
tool), `PLAN.md`, `GOAL.md`, or another agent's rows/logs.

## Traps (phase-1 + v2)

- Relative observations only (`gripper_to_object` 40:43, `object_to_goal` 43:46);
  absolute terms carry the per-env origin. `tests/test_classical_teachers.py` enforces it.
- The right `DEFAULT_QPOS` per task (NEUTRAL for mechanisms, HOME for free objects).
- MJCF angles are degrees unless `<compiler angle="radian"/>`; read `jnt_range` from the compiled model.
- Mechanism placement is set by the reset EVENT in `env_cfgs.py`, not the command `pose_range` (dead config).
- Fingertip friction is randomized 0.3–1.5 per env at startup; check whether your failing envs are the low-friction ones.
- The Franka hand capsule fouls on any collidable face within ±4 cm of the site xy.
- "Height above centre" constants must be swept, not scaled.
- Success is latched; "succeeded then knocked out" is not a failure mode.
- Warm-started IK can plateau short (M4): reset the posture bias at phase changes if you see it.
