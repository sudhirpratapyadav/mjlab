# mjlab Manipulation-Diversity Benchmark — Build Plan / Design Doc

Status: DRAFT v1 (2026-07-11). Author-facing design doc; supersedes the "next steps"
in FINDINGS.md for the benchmark-scaling effort. Reads on top of BENCHMARK_SURVEY.html
(source evidence) and FINDINGS.md / GRASP_FIX_EXPERIMENTS.md (the CL method the
benchmark exists to serve).

---

## 0. Decision record (what this doc is committed to)

Fixed by the user on 2026-07-11:

1. **Goal**: enable *better continual-learning research on contact-rich manipulation,
   vision-free*. The bottleneck is **number of tasks × manipulation diversity**, not
   the CL method — delta_value / SI / etc. are probes. A large, diverse task set is
   the deliverable that unblocks the research.
2. **One simulator only: mjlab (MuJoCo / MJX / mujoco-warp).** Port everything in
   ourselves. No second sim stack. This doubles as **a new benchmark for the field.**
3. **Diversity axis = MANIPULATION SKILL, not scene/spatial/language.** Explicit
   preference for RLBench-style skill breadth over LIBERO-style spatial/object/goal
   variation. LIBERO-type "130 tasks" are mostly the same few skills over different
   scenes — we do NOT want that kind of count.
4. **Structure the benchmark by EMBODIMENT CLASS** (this is the headline organizing
   idea, and a genuine gap in the field):
   - **A. Arm + 2-finger gripper** (our current Franka Panda setup).
   - **B. Arm + 5-finger hand** (dexterous; cf. CHORD-style suites, ~1800 tasks).
   - **C. Floating 5-finger hand, no arm** (Adroit/in-hand lineage).
5. **Teachers are OUT OF SCOPE for now.** Do not design the RL/scripted/BC teacher
   strategy yet. Focus this phase entirely on the **sim benchmark**: embodiments,
   scenes, tasks, obs/action interfaces, skill taxonomy, and the porting pipeline.
   (Teacher strategy gets its own doc once the env set exists.)

Everything below serves those five commitments.

---

## 1. Why this benchmark doesn't exist yet (the gap we fill)

From the survey, no existing benchmark satisfies all of {MuJoCo-native GPU-parallel,
vision-free / state-solvable, manipulation-skill-diverse, multi-embodiment incl.
dexterous, CL-oriented}:

- **Meta-World MT50 / CompoSuite / Continual World** — MuJoCo, state-based, CL-proven,
  but 2-finger only and skill-narrow (MT50 ≈ 30-35 distinct skills; CompoSuite's 256
  tasks are 4 skills × contexts). No dexterous track.
- **RLBench (~100 tasks)** — the best *skill taxonomy*, but CoppeliaSim, non-portable
  binary assets, restrictive license, sparse-only. Usable only as a design checklist.
- **Dexterous suites** (Adroit=4, Bi-DexHands=~20 IsaacGym, DexMimicGen=9 bimanual,
  DexJoCo=11 IL-only, ShadowHand=~9 in-hand, TCDM=50 traj-mimicry) — each is small,
  single-embodiment, wrong engine, or IL/VLA-only. None is state-based RL-ready, none
  sits in one stack with a gripper suite.
- **LIBERO / RoboCasa** — large counts but scene/spatial diversity, sparse rewards,
  CPU robosuite.

**Our differentiator**: one MuJoCo/MJX stack, vision-free, spanning three embodiment
classes with a *skill-first* (not scene-first) taxonomy, built for continual learning.
That is a novel benchmark contribution independent of the CL method results.

---

## 2. Architecture: what "a task" is, and what stays fixed

The existing 2-stage pipeline already makes the benchmark's integration surface tiny
and embodiment-agnostic (verified against `config/tasks_5task.yaml`,
`continual_distill.py`, `extract_teacher_dataset.py`):

> **A task = an mjlab env that (a) emits a low-dim obs vector and (b) accepts a
> continuous action vector, plus a success predicate.** The distill/CL stage consumes
> only `(obs, action_target)` arrays + an `env_id` for eval. It is already
> **task-count-agnostic and obs/action-dimension-agnostic** (student input/output dims
> are read from the dataset; per-task heads handle heterogeneity).

Implications that shape the whole build:

- **Heterogeneous obs/action dims across tasks are already supported** by per-task
  heads. So embodiment B (22-D action) and C (24-30 DoF hand) can coexist with A
  (8-D) in one suite *if* we keep per-embodiment student heads. (Whether one shared
  trunk across embodiments is even desirable is itself a research question the
  benchmark should let us ask — do NOT hard-assume a single shared policy across
  embodiment classes.)
- **We do not need teachers to define the benchmark.** An env + success predicate +
  a documented "state-solvable" claim IS the task. Teacher datasets come later.
- The scaling cost is **authoring mjlab envs**, i.e. (scene MJCF, reset distribution,
  obs function, action interface, success predicate, optional dense reward). Nothing
  in stage 2 changes as we go from 5 to 100 tasks.

### 2.1 Standardized interfaces (author once, reuse everywhere)

To keep 100 tasks maintainable, fix a small number of shared control/obs conventions
so per-task work is only "scene + success + reset":

- **Action interface per embodiment class** (one each, not per task):
  - A: EE Cartesian delta (dx,dy,dz[,drpy]) + gripper — mirrors Meta-World's uniform
    `Box(-1,1,4)` and our current Franka setup. This uniformity is what makes MT50
    portable at ~0.5-1 day/task; adopt the same discipline.
  - B: EE Cartesian delta (arm) + hand joint targets (or a low-D synergy/eigengrasp
    action to keep dims sane). Decide the hand action parameterization ONCE.
  - C: floating-base 6-DoF wrist pose delta + hand joints.
- **Obs vector convention**: proprio + task-relative object/goal deltas (the
  FINDINGS "use RELATIVE obs only" lesson — world-frame obs carry a per-env
  scene-origin offset and break policies/experts). Bake relative-obs helpers into a
  shared task base so every new task inherits it.
- **Success predicate convention**: latch live per-env success with `np.maximum`
  every step (the FINDINGS eval-bug fix) + per-env-instance RNG isolation for reset
  poses (the LiftCube start-pose artifact). These MUST be in the shared base or every
  new task reintroduces the bugs we already paid for.

Writing this shared base FIRST is the single highest-leverage step — it turns each
task into a small diff and prevents re-debugging solved problems 95 more times.

---

## 3. Skill taxonomy (the diversity backbone)

Count is not the goal; *skill coverage* is. Use RLBench's taxonomy as the checklist
(its one legal, portable contribution) and tag every task by skill family AND by our
own **fragility axis** (from FINDINGS: fragility is what makes CL comparisons
discriminating — an all-planar suite washes out the method signal).

**Skill families** (target ≥1 embodiment each; contact-rich ones are the priority):

| Family | Examples | Fragility | Contact |
|---|---|---|---|
| Reach / position | reach, push-to-goal | low | none |
| Planar push / slide | push cuboid, sweep, slide block | low | pushing |
| Articulation | open door/drawer/cabinet, turn faucet, flip switch, press button | low-mid | contact |
| Pick-and-place | pick, lift, place, stack, bin-pick | **high** (grasp) | grasp |
| Insertion / assembly | peg-in-hole, nut-on-bolt, plug, gear-mesh | **high** | precise |
| Tool use | hammer, sweep-with-tool, scoop, reach-with-stick | high | multi-contact |
| In-hand / reorient | rotate block/pen/egg, regrasp | **very high** | dexterous |
| Deformable / misc | rope, cloth (STRETCH; MJX cost — defer) | high | continuous |

**Fragility axis** (deliberately sample all four — this is the scientific payload):
1. Planar / forgiving (object keeps sliding toward goal) — SI-protectable.
2. Mild-contact (PushCuboid class).
3. Precision-grasp (LiftCube class — all-or-nothing at the grasp instant).
4. Multi-contact / dexterous (insertion, in-hand — hardest to retain).

A publishable benchmark should have a roughly balanced spread across 1-4 within each
embodiment class, so CL methods can be scored *per fragility tier*, not just on a mean.

---

## 4. Source strategy per embodiment class (all verified in the survey)

We import **task DESIGNS and ASSETS**, re-authoring envs in mjlab. We never depend on
another runtime. Licenses matter because we're publishing a benchmark — prefer MIT /
Apache-2.0 assets; avoid NVIDIA-non-commercial (MimicGen/DexMimicGen code) and
RLBench (non-transferable).

### Class A — Arm + 2-finger gripper (the backbone; get to ~40-50 here first)

| Source | Contributes | License | Port path |
|---|---|---|---|
| **Our current 5** | Push/Button/Door/Drawer/Lift | ours | native (done) |
| **MuJoCo Playground manip** | ~5 new native envs (Robotiq push, etc.) | Apache-2.0 | native, ~0 port |
| **Meta-World MT50** (designs+MJCF) | ~30-35 distinct skills, dense rewards, uniform EE action | MIT | re-author on Franka EE; 0.5-1 d/task after pipeline |
| **ManiSkill2/3 recipes** | ~15-20 primitive-geom tasks (PegInsert, StackCube, PlugCharger, PushT…) w/ open dense-reward code | Apache-2.0 | re-implement task+reward, **zero asset dependency** — cheapest contact-rich tasks |
| **robosuite / robomimic** | NutAssembly, PickPlace variants, Stack, Door | MIT | MJCF re-implement in manager API |
| **dm_control manipulation** | snap-brick assembly generator, stack/place | Apache-2.0 | assets loadable; Jaco→Franka remount |
| **Franka Kitchen** | 7 composable subgoals (Franka-native) | MIT | native-ish MJCF |
| **PartNet-Mobility** | **scaling lever**: 2,347 articulated objects → hundreds of open/close/turn/press instances | asset (registration) | build URDF→MJCF + CoACD convex-decomp converter ONCE, then procedural |
| **MolmoSpaces / MS-Bench** | 48k manipulable MJCF objects + 42M precomputed grasps, Franka FR3, MuJoCo-native | Apache-2.0 | tabletop carve-out; procedural pick/place/insert generator |

RLBench = taxonomy checklist only (§3). CompoSuite = optional compositional stress
grid (64 Panda tasks = 4 skills × contexts) if we want a controlled context-shift
axis; low priority given it's skill-narrow.

### Class B — Arm + 5-finger hand (dexterous arm; the novel middle tier)

The field has almost nothing state-based + RL-ready here — this is where the
benchmark is most novel. Embodiment from **MuJoCo Menagerie** (Franka + Allegro /
Shadow / Leap hand MJCF, all permissive). Task designs/assets scavenged from:

| Source | Contributes | Caveat |
|---|---|---|
| **DexJoCo** (11 Allegro tasks: Hanoi, assembly, fold, microwave, tongs…) | task designs + MJCF scenes | IL-only, no rewards; state-RL solvability UNPROVEN — treat as design source, validate solvability ourselves |
| **Bi-DexHands** (~20 task designs + reward formulas: cap, scissors, switch, cup, stack, reorient) | reward formulas reusable; single-hand-ify the bimanual ones | IsaacGym engine (excluded); re-author only |
| **TCDM** (~60 GRAB/YCB object MJCF) | clean object library for dex pick/use tasks | traj-mimicry paradigm not reused; assets only |
| **DexMimicGen / MimicGen** | object assets (needle/ring, three-piece assembly, coffee) | NVIDIA non-commercial CODE — use only clean-room re-authored assets or skip |

Decision needed later (teacher phase): hand action parameterization (full joint vs
eigengrasp/synergy). Flag now because it affects obs/action dims for every B task.

### Class C — Floating 5-finger hand, no arm (in-hand / dexterous core)

Embodiment: floating Shadow/Allegro hand (Menagerie). Sources:

| Source | Contributes | License |
|---|---|---|
| **Adroit / DAPG** (door, hammer, pen, relocate) | 4 canonical state-based RL tasks, MuJoCo-native, maintained | MIT (Gymnasium-Robotics) — near-native port, best starting point |
| **ShadowHand reorient** (block/egg/pen × rotate/full) | ~9 in-hand reorient tasks, state-based | MIT |
| **TCDM** | object library for floating-hand pick/use | archived; assets clean |
| **MyoSuite** | REJECT (muscle-actuated, incompatible action space) | — |

Adroit's 4 + ShadowHand's ~9 give a ~13-task class-C core immediately (near-native
MuJoCo), enough to make the multi-embodiment claim real from day one.

---

## 5. Realistic task budget (how the 50-100 is actually reached)

| Tier | Source | Tasks | Confidence |
|---|---|---|---|
| Existing native | our 5 + Playground | ~10 | done / trivial |
| Class-A backbone | Meta-World MT50 re-author | +30-35 distinct | high (mechanical) |
| Class-A contact | ManiSkill recipes + robosuite + dm_control | +15-20 | high (open rewards) |
| Class-A generated | PartNet-Mobility + MolmoSpaces procedural | +50-100s (instances) | med (build converter) |
| Class-C dexterous | Adroit + ShadowHand | +13 | high (near-native) |
| Class-B dex-arm | DexJoCo/Bi-DexHands re-author | +10-20 | med (solvability unproven) |

**≥50 distinct skills is reachable without the procedural lever; 100+ needs it.**
The procedural articulation/grasp generators (PartNet, MolmoSpaces) are what push past
50 and give near-unlimited within-skill instances for CL "many tasks" studies — but
those are *instance* diversity, so keep them tagged separately from *skill* diversity
so the benchmark can report both honestly.

---

## 6. Staged milestones

**M0 — Shared task base + interfaces (highest leverage; ~1-2 wk).**
Author the mjlab shared manipulation-task base encoding §2.1: relative-obs helper,
EE-delta action interface (class A), latched-success + RNG-isolated reset predicate,
skill/fragility tags in task metadata, dataset/eval hooks matching stage 2. Validate
by re-expressing our existing 5 tasks on top of it with identical results (regression
gate: reproduce the 0.93 4-task baseline).

**M1 — Class-A EE pipeline + 10-task set (de-risk the port).**
Port 5 Meta-World tasks (spanning fragility tiers: reach, push, pick-place, peg-insert,
door) onto Franka EE control via the shared base. Confirm Sawyer→Franka EE retarget
works and tasks are state-solvable. Combine with existing 5 → first 10-task
skill-diverse set. Order using Continual World CW10 ideas.

**M2 — Class-A backbone to ~40 (scale the mechanical port).**
Complete Meta-World MT50 re-author + ManiSkill contact-rich recipes + a few robosuite
/ dm_control assembly tasks. This alone is a strong standalone gripper CL benchmark.

**M3 — Multi-embodiment (the novelty).**
Bring up Class C (Adroit + ShadowHand, near-native) and a Class-B beachhead (2-3
DexJoCo/Bi-DexHands re-authored, solvability-validated). Now the benchmark spans all
three embodiment classes — the headline contribution.

**M4 — Procedural scaling to 100+ (the "many tasks" lever).**
Build the PartNet-Mobility URDF→MJCF+CoACD converter and the MolmoSpaces grasp-based
pick/place generator. Instantiate hundreds of articulation/grasp task instances,
tagged as instance-diversity.

**M5 — Benchmark packaging.** Task registry + metadata (skill, fragility, embodiment,
license, solvability evidence), standard CL orderings, eval protocol, docs. Then hand
off to the teacher-strategy doc + the actual CL research runs.

---

## 7. Open questions to resolve before/within each milestone

1. **Shared trunk across embodiment classes — yes or research question?** Per-task
   heads already handle dim heterogeneity; but is a single trunk over A+B+C even
   desired, or do we distill per-embodiment students and study cross-embodiment
   transfer separately? (Recommend: per-embodiment students first; cross-embodiment
   transfer is its own paper.) — affects M3 design.
2. **Hand action parameterization for B/C** (full joints vs synergy/eigengrasp) —
   fixes obs/action dims for all dexterous tasks. Decide before M3.
3. **Instance vs skill diversity accounting** — how the benchmark reports "N tasks"
   honestly (distinct skills vs procedural instances). Decide before M4/M5.
4. **State-solvability validation for Class B** — DexJoCo/Bi-DexHands solvability from
   state via RL is unproven; needs a small validation spike before committing budget.
5. **Contact cost in MJX/warp for dexterous + assembly** — 24-30 DoF hand-object and
   snap-brick (~49 geoms) contact is heavy for GPU-parallel stepping; may cap how many
   dexterous envs run at scale. Benchmark early.
6. **License hygiene for a published benchmark** — keep an asset-provenance ledger;
   exclude NVIDIA-NC-derived and RLBench assets entirely.

---

## 8. What NOT to do (guardrails from the survey + our own findings)

- Don't adopt LIBERO/RoboCasa scene-diversity counts as "tasks" — wrong diversity axis.
- Don't add a second simulator or keep any CPU-robosuite runtime dependency — port
  designs/assets only.
- Don't build an all-planar suite — it makes CL results non-discriminating (FINDINGS:
  planar tasks are "bulletproof 1.0", the method signal lives on contact/precision).
- Don't re-author RLBench assets (license) — use its list as a taxonomy checklist only.
- Don't reintroduce the solved eval bugs — success latching + reset-RNG isolation live
  in the shared base, not per task.
- Don't design teachers yet (explicit user scope call).
```
