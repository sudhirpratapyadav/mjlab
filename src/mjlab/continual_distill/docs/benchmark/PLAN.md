# Benchmark Build — PLAN

> Durable roadmap for building the mjlab manipulation-diversity benchmark.
> Companion files: `STATUS.md` (mutable current state), `LOG.md` (append-only journal).
> Design rationale lives in `../BENCHMARK_BUILD_PLAN.md`; source evidence in
> `../BENCHMARK_SURVEY.html`; the CL method this serves in `../FINDINGS.md`.
>
> This PLAN is the *what and in what order*. The user will refine taste on it later;
> keep it editable but treat phase boundaries and the commit discipline as stable.

## North star (one sentence)

A single-simulator (mjlab / MuJoCo-MJX), **vision-free**, **manipulation-skill-diverse**
continual-learning benchmark spanning **three embodiment classes** (arm+2-finger,
arm+5-finger, floating 5-finger hand), reaching **as many genuinely distinct tasks as
we reasonably can** (target ≥50 distinct skills, 100+ with procedural instances) — a
real benchmark contribution, not just a paper's eval set.

## Fixed commitments (from the user, 2026-07-11 — do not drift)

1. One simulator: mjlab. Port task *designs + assets* in; never depend on another runtime.
2. Diversity = **manipulation skill** (RLBench-style), not scene/spatial/language (anti-LIBERO).
3. Structure by embodiment class: A=arm+2-finger, B=arm+5-finger, C=floating hand.
4. **This effort BUILDS THE TASK SUITE; it does NOT solve it.** Teachers / RL / CL /
   train-solvability / evaluation are all STAGE TWO (a later, separate effort). A task
   is DONE for stage one when it is well-defined and structurally sound: builds, resets,
   steps, finite obs/reward, sane shapes, a clear success predicate. Whether PPO solves
   it is NOT a stage-one gate — `benchmark-smoke` is the acceptance check;
   `benchmark-validate` (training) is an optional informational tool, not a gate.
5. **Keep obs / action / reward / success spaces SANE and SIMILAR across tasks.**
   Uniform interfaces beat per-task cleverness:
   - Action: JOINT-SPACE everywhere (Franka 8-D). NO EE-delta term for now (stage two).
   - Obs: consistent term layout across tasks as far as possible.
   - Reward: consistent structure (staged reach→manipulate + shared regularization).
   - Success: consistent latched predicate (`torch.maximum` on per-env success).
   Dexterous hand (B/C) = FULL JOINT TARGETS (~16-22 dims), same reward/success patterns.
6. Diversity should still span the fragility axis (planar → mild → precision-grasp →
   dexterous) so the suite is scientifically useful downstream — but fragility is a
   TAG we assign, not something we train-verify here.
7. Work autonomously in phases; milestone-level commits only (not per-fix); minimal-
   but-well-divided folder structure (framework is already well-factored → additive).

## Architectural ground truth (verified 2026-07-11)

- mjlab is ALREADY a manager-based "shared base + per-task diff" framework. Tasks live
  in `src/mjlab/tasks/manipulation/` as `*_env_cfg.py` composing reusable `mdp/` terms
  (observations, rewards, terminations, commands, events), registered via
  `tasks/registry.py::register_mjlab_task`. Our 6 existing Franka tasks (Lift, Push
  ×3, Open ×2) follow this. **We extend this pattern; we do not invent a new one.**
- `continual_distill/` is a CONSUMER of registered envs (reads `env_id` + teacher
  datasets). It is already task-count- and obs/action-dim-agnostic (per-task heads).
  Nothing in the distill/CL stage changes as task count grows.
- Actions today: joint-space only (`JointPositionActionCfg` etc). **No EE/Cartesian-
  delta action term exists yet.** Meta-World portability wants uniform EE-delta control
  → building an EE-delta action term is a real sub-task (Phase 2), not a blocker for
  the existing joint-space tasks.
- Known-solved bugs that MUST stay solved as we scale (from FINDINGS.md): success
  latching (`np.maximum` live per-env success), reset-RNG isolation for start pose.
  These live in the eval/consumer path — audit + lock into shared infra in Phase 1.

## Embodiment classes & source strategy (all sources verified in the survey)

**Class A — arm + 2-finger gripper (backbone; reach ~40-50 here first):**
existing 6 + MuJoCo Playground natives → Meta-World MT50 re-author (MIT, uniform EE) →
ManiSkill dense-reward recipes (Apache, zero-asset contact-rich) + robosuite / dm_control
assembly → procedural scaling via PartNet-Mobility (2,347 articulated objects) +
MolmoSpaces (48k objects, 42M grasps, MuJoCo-native FR3).

**Class B — arm + 5-finger hand (novel middle tier):** Menagerie Franka+Allegro/Shadow/
Leap embodiment; task designs from DexJoCo (11) + Bi-DexHands (~20 designs/reward
formulas) + TCDM object library. State-RL solvability UNPROVEN → validation spike required.

**Class C — floating 5-finger hand (in-hand core):** Adroit/DAPG (4, near-native MIT) +
ShadowHand reorient (~9) → ~13-task core available almost immediately. TCDM objects for
floating pick/use. (MyoSuite REJECTED — muscle actuation incompatible.)

License hygiene (we publish): prefer MIT/Apache; exclude NVIDIA-non-commercial
(MimicGen/DexMimicGen code) and RLBench assets. Keep an asset-provenance ledger.

## Phases (each executed autonomously, milestone commit at end of each)

**Phase 0 — Setup & tracking (this phase).**
Branch `benchmark-manip-diversity` off `continual_distill`; tracking docs; audit the
current manipulation framework + continual_distill consumer; produce the concrete
restructuring proposal in STATUS.md. Commit: "benchmark: phase 0 — tracking + repo audit".

**Phase 1 — Repo restructure + shared task infra + regression gate.**
Reorganize into the long-term structure (proposed in STATUS after the audit). Factor a
clean shared manipulation-task base + skill/fragility/embodiment metadata into the
registry. Audit + lock the solved eval bugs into shared infra. **Regression gate:
reproduce the existing 4-task 0.93 baseline unchanged** before proceeding. Commit:
milestone.

**Phase 2 — Class-A EE control + 10-task skill-diverse set.**
Build the EE-delta action term. Port ~5 Meta-World tasks spanning fragility tiers onto
Franka via the shared base; validate state-solvability. Combine with existing 6 → first
~10-15 task set, tagged by skill/fragility. Establish the task-authoring recipe (the
"0.5-1 day/task" loop). Commit: milestone.

**Phase 3 — Class-A backbone to ~40.**
Complete Meta-World MT50 re-author + ManiSkill contact-rich recipes + select robosuite/
dm_control assembly tasks. Standalone strong gripper CL benchmark. Commit: milestone.

**Phase 4 — Multi-embodiment (the novelty).**
Class C core (Adroit + ShadowHand, near-native). Class B beachhead (2-3 dex-arm tasks,
solvability-validated). Benchmark now spans all three embodiment classes. Commit: milestone.

**Phase 5 — Procedural scaling to 100+.**
PartNet-Mobility URDF→MJCF+CoACD converter; MolmoSpaces grasp-based generator. Hundreds
of articulation/grasp instances, tagged as instance-diversity (kept distinct from skill
diversity in accounting). Commit: milestone.

**Phase 6 — Benchmark packaging.**
Task registry + metadata (skill, fragility, embodiment, license, solvability evidence),
standard CL orderings, eval protocol, README/docs. Hand-off point to the teacher-strategy
doc and the actual CL research runs. Commit: milestone.

## Open architecture questions (resolve within the phase noted; default given)

1. **Shared trunk across embodiment classes vs per-embodiment students?**
   Default: per-embodiment students first (per-task heads already handle dim
   heterogeneity); cross-embodiment transfer is a separate study. — decide by Phase 4.
2. **Hand action parameterization (full joints vs synergy/eigengrasp)** — fixes dims
   for every B/C task. — decide by Phase 4.
3. **Instance vs skill diversity accounting** — how "N tasks" is reported honestly. —
   decide by Phase 5.
4. **EE-delta action design** (IK-based vs mocap-weld vs impedance) — Phase 2.
5. **State-solvability of Class B** — validation spike before committing Phase-4 budget.
6. **MJX/warp contact cost** for dexterous + assembly at scale — benchmark early in
   Phases 2/4.

## Guardrails (from survey + our findings)

- No second simulator; no CPU-robosuite runtime dependency.
- Don't count scene variations as tasks (anti-LIBERO). Report distinct skills honestly.
- Don't build an all-planar suite (kills CL signal).
- Don't reintroduce the solved eval bugs — they live in shared infra.
- Don't re-author RLBench assets (license) — taxonomy checklist only.
- Don't design teachers yet.
- Milestone commits only; restructure before building; minimal well-divided folders.

## Definition of "reasonably done" (scope ceiling — don't over-extend)

Done = all three embodiment classes represented AND ≥50 distinct skills across A/B/C
AND procedural lever operational for 100+ instances AND packaged with metadata + CL
orderings + eval protocol. Stretch goals (deformables, bimanual, cross-embodiment
transfer studies) are explicitly OUT unless they fall out cheaply.
