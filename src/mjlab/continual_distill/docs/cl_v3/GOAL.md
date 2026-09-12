# CL-V3 — GOAL

**Next stage:** [CL-V4 RL teachers](../cl_v4_rl/README.md): one independent RL teacher per active task, each above 90% success.

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


> Read this first, then `PLAN.md`, then `STATUS.md` (scoreboard) and `AGENT_BRIEF.md`
> if you are a wave agent. `LOGS.md` is the append-only trail. Predecessors:
> `../cl25/phase_1/` (classical teachers, 0.90 bar, the M1–M4 mechanism handover) and
> `../cl_v2/` (realistic assets, all 25 tasks; `HANDOVER.md` lists mechanics changes).

## The one-sentence goal

Every one of the 24 active Class-A Franka tasks has a classical teacher that solves it at
**~100 % (hard floor 0.90) at n = 128**, on the v2 realistic assets, over an initial
distribution of object / mechanism / robot poses that is **wide enough to be honest**
and is frozen in writing before the teacher is measured.

## Why

The next stage (continual distillation over the 24 active tasks) is only as good as its teachers:
distilling from a 0.3 teacher measures the teacher, not the CL method. cl_v2 ends with
8 of 25 teachers at or above 0.90, 5 at exactly 0.000. Phase-1 found that the failures
are not 17 separate problems but a handful of *strategies* that cannot work (a pinch
that squeezes the object out sideways, a position servo that cannot sustain a push, a
stroke that lands 10 % of the time and relies on retries). The fix is a better way of
doing the task, not a better constant.

## Standing rules (from the owner, 2026-09-09)

1. **Strategy first.** Before touching a constant, watch the failure, instrument it, and
   ask how the task would actually be done — a different grasp (a finger in the gap or
   hole, hooking behind a handle instead of pinching it), a different contact (closed
   gripper as a paddle, two-point contact), a different sequence (re-grasp, staged
   approach), a different pose of the hand. Write the reasoning in `LOGS.md` before
   the code.
2. **Tasks and success metrics are frozen** — EXCEPT when the task as specified is
   physically impossible or was clearly mis-specified (a wrong predicate, a target
   nobody could reach, an episode budget shorter than the fastest feasible motion,
   a spawn band the arm cannot serve). Such a change needs a **decision row** in
   `STATUS.md` with the measurement that proves impossibility, and it is never made
   "because it raises the score". If in doubt, leave the task alone and record the gap.
3. **Init distribution has real coverage.** Position bands stay workspace-derived; add
   orientation variation wherever the object's symmetry does not make it meaningless;
   mechanisms get a yaw and height band; the robot starts with joint noise on every
   task. The spec per task is in `PLAN.md` §3 and is frozen per task BEFORE its teacher
   is iterated; SR is only ever quoted on the frozen spec.
4. **Measure before claiming.** n = 32 to iterate, n = 128 to claim, HEAD named. A
   failure analysis (which phase, which envs, what happened) accompanies every number
   below 0.90.
5. **Publish when > 0.90**: teacher rollout video to https://cl.sudhirpratapyadav.com/v3/<Task-Id>/
   (`publish_v3.sh`), which is a new gallery with its own card on the home page.
6. **Cluster etiquette**: `~/use_instructions/README.md`. Reuse holder 20277, never cancel
   anything, use only your assigned GPU index, log it.

## Done means

All 24 active rows in `STATUS.md` read `>= 0.90 @ n=128` on their frozen init spec, with a
published success video, OR carry a decision row explaining exactly what is physically
impossible and what was changed (or not) about the task. Summary counts are recomputed
from the rows.
