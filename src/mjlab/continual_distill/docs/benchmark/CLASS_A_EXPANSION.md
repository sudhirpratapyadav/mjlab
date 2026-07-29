# Class A Expansion — motion-profile task scale-out

> Autonomous work log for growing Class A (Franka arm + 2-finger gripper, action=8).
> Companion to PLAN.md / STATUS.md / LOG.md. Decisions are recorded here rather than
> escalated — the user's mandate (2026-07-30) is: work autonomously, take the decision,
> log it.

## Mandate (user, 2026-07-30)

> "i want to increase the num of tasks in class A ... i count them as 8 (3x
> articulations, reach, lift, stack, peg, push), i dont want to differentiate between
> lift a vs b ... improve these set of tasks not object change. Articulation i counted
> different because of motion profile kind of"
>
> "do all except multi object like sweep"
>
> "do everything autonomously without my help, no asking from me, no stopping"

### The counting rule (THIS is the accounting standard for Class A)

A task counts as distinct iff its **motion profile** is distinct — the control problem
the policy must solve. Object-geometry swaps do NOT count.

- lift-cube / lift-cylinder / lift-sphere / lift-ellipsoid = **1** task (lift)
- push-cuboid / push-disc = **1** task (push)
- open-door / open-drawer / push-button = **3** tasks (hinge arc / linear slide /
  normal-force press are genuinely different control problems)

So the **pre-expansion Class A baseline is 8 motion profiles**, not the 12 registered
task IDs. The manifest keeps reporting 12 registered IDs; this doc tracks the honest
motion-profile count. Both numbers are real and they answer different questions —
see "Accounting" below.

## Scope decision

From the candidate list, the user selected everything **except multi-object tasks**.
That excludes **sweep** (N cubes into a zone) explicitly, and by the same rule excludes
**unstack / de-palletize** (requires a pre-built multi-cube stack and its success
predicate is "don't disturb the others" — inherently multi-object).

Also dropped, with reasons:
- **insert-peg-at-angle / multi-hole** — under the user's own counting rule this is a
  peg variant, not a new motion profile. Including it would be exactly the object-swap
  padding the mandate rejects.
- **wipe / trace-surface** — needs a contact-force success predicate that does not
  exist in `mdp/`, and is arguably a force-control task rather than a manipulation one.
  Deferred, not rejected.

### Selected: 8 new motion profiles (Class A 8 -> 16)

Articulation (3 -> 7) — new joint types/axes, reuses the proven door/drawer recipe:

| # | Task | Joint | Why a distinct motion profile |
|---|---|---|---|
| A1 | Turn-Lever | hinge about the **approach** axis | wrist rotation, not an arm pull |
| A2 | Rotate-Valve | hinge, multi-turn | sustained rotation past wrist range -> regrasp |
| A3 | Flip-Switch | small hinge + detent | ballistic commit, not servoing |
| A4 | Slide-Window | slide along **lateral** axis | push a face; drawer's hook trick fails |

Manipulation (5 -> 9) — new reward/success shapes:

| # | Task | New reward shape | Why distinct |
|---|---|---|---|
| B1 | Open-Lid | hinge w/ vertical arc | gravity opposes throughout |
| B2 | Place-In-Container | **containment** predicate | release timing; not stable-contact |
| B3 | Reorient-Object | **orientation** predicate | success is rotational, not positional |
| B4 | Tool-Pull | two-stage, tool-mediated | grasp stick -> drag out-of-reach object |

B1 (Open-Lid) is articulation-family by skill tag but is listed here because it needs a
new asset rather than a joint-axis edit of an existing one.

## Accounting (honest reporting — anti-LIBERO guardrail)

Two numbers, never conflated:
- **Registered task IDs**: what `manifest.json` counts (includes object variants).
- **Distinct motion profiles**: the scientific claim. Class A: 8 -> 16.

Risk acknowledged: lever / valve / switch are all "actuate a small articulated part"
and a reviewer could compress them the way we compress lift variants. The defence is
that their motion profiles differ (wrist rotation about approach axis / sustained
multi-turn with regrasp / ballistic commit past a detent). To make that defensible
rather than asserted, each new task carries a `notes` field naming its motion profile
explicitly, and this doc records the argument. If a profile turns out NOT to be
distinct in practice, it gets merged in the accounting — not quietly kept for count.

## Method

Per AUTHORING_GUIDE Recipe B (new skill = new MDP). Each task needs:
1. asset (`asset_zoo/objects/articulated/<name>/` — xml + constants + `__init__`)
2. command term in `mdp/commands.py` (goal, latched success, metrics)
3. base maker `tasks/manipulation/<task>_env_cfg.py`
4. concrete cfg + rl_cfg in `config/franka/`
5. register + tag in `config/franka/__init__.py`
6. `benchmark_smoke --keyword <Name>` as the acceptance gate

**Stage-one gate only.** Per PLAN.md commitment #4, structural soundness
(benchmark-smoke) is the acceptance criterion; train-solvability is stage two. The
AUTHORING_GUIDE caveat is noted and respected: smoke proves it BUILDS, training proves
it WORKS, and reward correctness for these new skills is NOT verified by smoke. Every
new task is tagged with that caveat in its `notes` until a training run says otherwise.

## Progress — COMPLETE (2026-07-30)

- [x] Scope + counting rule decided and logged (this doc)
- [x] A1 Turn-Lever          `Mjlab-Turn-Lever-Franka`
- [x] A2 Rotate-Valve        `Mjlab-Rotate-Valve-Franka`
- [x] A3 Flip-Switch         `Mjlab-Flip-Switch-Franka`
- [x] A4 Slide-Window        `Mjlab-Slide-Window-Franka`
- [x] B1 Open-Lid            `Mjlab-Open-Lid-Franka`
- [x] B2 Place-In-Container  `Mjlab-Place-In-Container-Franka`
- [x] B3 Reorient-Object     `Mjlab-Reorient-Object-Franka`
- [x] B4 Tool-Pull           `Mjlab-Tool-Pull-Franka`
- [x] manifest regenerated, STATUS.md + LOG.md updated

### Results

**Class A: 8 -> 16 distinct motion profiles** (12 -> 20 registered task IDs).
Benchmark total: 20 -> 28 registered IDs, 5 -> 6 skill families (TOOL_USE is new).

| Check | Result |
|---|---|
| `benchmark-smoke --keyword Franka` | **23/23 PASS** (CPU), all 8 new tasks included |
| Obs / action dims | 60-D / 8-D on all 8 — uniform Class A interface preserved |
| `tests/test_class_a_expansion.py` | 11/11 (new) |
| `tests/test_task_configs.py` + `test_benchmark_taxonomy.py` | 17/17 (unchanged) |

Artifacts per task: asset (`asset_zoo/objects/{articulated,free}/<name>/`), command term
in `mdp/commands.py`, base maker `tasks/manipulation/<task>_env_cfg.py`, concrete cfg +
rl_cfg in `config/franka/`, registration + taxonomy tag.

### Success predicates are tested, not assumed

`benchmark-smoke` proves an env builds and steps; it does NOT prove the success
predicate can ever fire. A predicate that never fires yields a task that is silently
UNSOLVABLE while passing every structural check. `tests/test_class_a_expansion.py`
therefore drives each mechanism/object into a success state and asserts the predicate
latches, with a near-miss negative control so we know it discriminates.

**Flip-Switch's detent was monostable and had to be redesigned.** The first version
used a joint spring (`stiffness` + `springref` at the OFF stop). A single linear spring
has exactly ONE rest pose, so the toggle fell back to OFF even when released past
centre — verified empirically. That silently reduces the task to "hold against a
spring" and destroys the ballistic-commit profile that justifies counting it as a
distinct motion profile at all. It is now an **over-centre weighted lever** (inverted
pendulum): the centre of mass sits directly ABOVE the pivot, so the equilibrium at
centre is unstable and the toggle accelerates to whichever stop it leans toward.
A second iteration was needed because the initial weight was offset in x, which adds a
constant gravity torque that also destroys bistability — the COM must be on the pivot
axis. Flip-Switch therefore runs with **gravity ENABLED**, and a test asserts the snap
in both directions.

That caught a real bug: **Tool-Pull's goal sat 0.3m above the puck's resting height**
(`goal_offset` z=0.31 vs the puck's z=0.012, against a 0.07 threshold), so no amount of
correct dragging could ever satisfy it. Fixed to z=0.012, with a regression test
pinning the goal to the puck's rest height.

## Decisions log (append-only)

**2026-07-30 — object spawn belongs to the COMMAND, not an event.** The lift base has
no `reset_object_position` event; `LiftingCommand._resample_command` writes the object
pose from `object_pose_range`. The three new manipulation cfgs initially patched a
nonexistent event (a `KeyError` at construction). Fixed by giving the new commands their
own `_ObjectSpawnRangeCfg` + `_spawn_object` helper, which also adds roll/pitch (the
reorient task must spawn the cylinder LYING DOWN, which the lift range's yaw-only
sampling cannot express).

**2026-07-30 — `Scene` has no `.get()`.** Only `__getitem__`. The optional `mocap_goal`
lookup is a guarded `try/except KeyError`, so a task without a goal marker degrades to
no-viz instead of crashing.

**2026-07-30 — Tool-Pull episode is 12s and Rotate-Valve 8s**, against the 3s default of
the articulation template. Both are multi-stage (acquire tool then drag; turn, regrasp,
turn) and cannot finish in 150 steps. Flagged because episode length interacts with the
CL protocol — if the sequence runner assumes a uniform episode length, these two need
attention.

**2026-07-30 — gravity is ENABLED for Flip-Switch, Open-Lid and the manipulation
tasks; DISABLED for lever/valve/window.** The three original articulation cfgs
(door/drawer/button) disable gravity, and lever/valve/window inherit that: their
mechanisms hang off a mocap base, so gravity only loads the arm. Flip-Switch and
Open-Lid override it because their premises ARE gravity (over-centre detent; lid falls
shut). Tool-Pull / Place-In-Container / Reorient use the lift base, which never
disabled gravity.

**2026-07-30 — gravity is DISABLED in the articulation env cfgs.** `open_drawer_env_cfg`
(and door/button) set `gravity=(0,0,0)` with the comment "DISABLED for debugging". New
articulation tasks inherit whatever their template used, EXCEPT Open-Lid and Tool-Pull,
whose entire premise is gravity (lid falls shut; object must be dragged). Those two get
real gravity. Flagged as a pre-existing inconsistency worth resolving separately — not
silently "fixed" here, since re-enabling gravity on door/drawer/button would invalidate
the existing teacher datasets and the 0.93 CL baseline.
