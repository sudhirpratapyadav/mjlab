# CL-25 — PURPOSE

> Why this program of experiments exists, what it must produce, and what would
> count as it having failed. Read this first; read `CONTEXT.md` second.
> `EXPERIMENTS.md` / `LOGS.md` / `STATUS.md` are the working files.

## The one-sentence goal

Extend the existing two-stage continual-learning setup (teacher -> distillation +
regularization) from the **4-5 tasks it has run on so far** to the **full 25-task
Class-A suite**, so that every claim about forgetting, capacity and ordering is made
over a task suite wide enough to be believed.

## The problem this is solving

Every continual-learning result in this repo to date — the forgetting floor (0.267
best / 0.499 worst ordering), the joint ceiling (0.958), the width-8192 LR artifact,
EWC-beats-SI on the worst ordering, the 24-permutation ordering sweep — was measured on
**one sequence of four tasks**: Push-Cuboid, Push-Button, Open-Door, Open-Drawer, with
Lift-Cube as a fifth. Four tasks drawn from three skill families.

That is too narrow to support the conclusions being drawn from it. With four tasks:

- **Ordering effects are undersampled.** 24 permutations exhaust the space; with 20
  tasks the orderings are a sample, and "best vs worst ordering" becomes a
  distribution rather than two points.
- **Skill-family transfer is unmeasurable.** Three families cannot separate
  "forgetting" from "interference between related skills". Twenty-five tasks across seven
  families can.
- **Fragility is confounded with task identity.** The current sequence has one task per
  interesting tier, so "fragility-3 tasks are forgotten worse" is a claim about
  Lift-Cube, not about fragility.
- **Capacity claims do not scale-test.** A trunk that holds four tasks tells you
  nothing about where the capacity wall actually is.

The benchmark work of the last two months built the tasks needed to fix this: Class A
is now 29 registered IDs / 25 distinct motion profiles across 7 skill families and all
4 fragility tiers. **This program is the payoff for that work.**

## Scope

**In scope — all 25 tasks.** Every Class-A task (Franka + 2-finger gripper, uniform
8-D action), counted once per distinct MOTION PROFILE. The exact list, with per-task
teacher status, is in `CONTEXT.md`.

The 25 are exactly: all 29 registered Franka-2F IDs, minus the 4 that are pure object
swaps of a profile already in the set — Lift-Cylinder, Lift-Sphere and Lift-Ellipsoid
(same profile as Lift-Cube) and Push-Disc (same profile as Push-Cuboid). Nothing else
is removed. This is the standing counting rule: a task counts once per distinct motion
profile, and swapping the object does not make a new task.

**The 5 tasks used in previous experiments stay IN** — Push-Cuboid, Push-Button,
Open-Door, Open-Drawer and Lift-Cube are part of the suite, not excluded from it. That
is deliberate and useful: it makes the old 4-task and 5-task sequences **nested subsets
of the new suite**, so the P0/P1 results (forgetting floor 0.267/0.499, joint ceiling
0.958, EWC-beats-SI on the worst ordering) remain directly comparable rather than
becoming a separate island of numbers. Their teachers already exist, so they are also
the cheapest tasks in the program.

**In scope — teachers.** Classical (scripted diff-IK waypoint) teachers and RL-dense
(PPO) teachers, in that priority order.

**Out of scope for now — BC teachers.** Behaviour cloning is deliberately excluded
from this program. There is one BC teacher in the repo (Open-Drawer); it is not being
extended. If a task turns out to need BC, record it as a finding and move on rather
than building the BC path.

**Out of scope — other embodiments.** Class B (Franka+LEAP) and Class C (floating
LEAP) are untouched. This is a Class-A program.

## The two stages, unchanged

The pipeline is the one already in the repo, applied to more tasks — this program does
NOT redesign it:

1. **Stage 1 — teacher per task.** A policy that solves one task well enough to
   generate a dataset worth imitating. Classical first (cheap, no GPU training);
   RL-dense where scripting provably cannot work.
2. **Stage 2 — sequential distillation into one student, with a regularizer.**
   Distill teacher datasets into a single trunk, task after task, with SI (and the
   EWC/L2 comparators that already exist), measuring what is retained.

## What this program must produce

1. **A teacher per task, or a recorded reason there is none.** Twenty-five tasks; for each,
   either a teacher above the competence bar with its measured success rate, or a
   characterised failure explaining what defeats scripting/RL. A characterised failure
   is a legitimate output, not a gap — the existing Tool-Pull and Open-Door writeups
   are the model.
2. **A competence-gated task set for stage 2.** The subset of the 25 whose teachers
   clear the bar. Distilling from a 0.04-success teacher measures the teacher, not the
   continual-learning method, and would silently corrupt every downstream number.
3. **Continual-distillation results over that set** — forgetting, ordering sensitivity,
   capacity, SI vs EWC vs L2 — at a suite width where the conclusions mean something.

## What "done" looks like, and what failure looks like

**Done:** every one of the 25 tasks has a resolved teacher status; the gated subset is
named and justified; stage-2 sequences run over it with the same measurement
discipline as the P0/P1 experiments (3 seeds, mean +- std, per-task final SR).

**Failed:** a big number is reported over 25 tasks that is actually carried by teacher
quality, ordering luck, or a task set quietly trimmed to whatever worked. The specific
failure mode to guard against is **reporting suite width as if it were suite
difficulty** — 25 tasks with 15 good teachers is a 15-task result with 10 open
questions, and must be written that way.

## Standing rules for this program

- **Measure before claiming.** Every number in `STATUS.md` is one someone ran, with the
  episode count attached. Numbers copied from a plan are marked as targets, not results.
- **Instrument before tuning.** The single most reproducible lesson from the previous
  teacher pass: four material improvements out of four came from finding a mechanism
  bug, none from tuning constants. The prior diagnosis named the right symptom and the
  wrong cause every time.
- **Negative results are output.** Seven measured Push-Cuboid strategy attempts that
  all failed are in the record because they were expensive to obtain and stop the next
  agent repeating them. Do the same here.
- **Never change a task to make a teacher work.** Task definitions are frozen for this
  program; a teacher that needs the task relaxed is a teacher that has failed. Report
  it. (The one exception is a genuine task bug — a placement error, a wrong predicate —
  which is a benchmark fix and belongs in the benchmark docs, not here.)
