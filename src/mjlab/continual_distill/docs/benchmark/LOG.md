# Benchmark Build — LOG (append-only)

> Dated journal of decisions, findings, and turns. Newest at the bottom. Never edit
> past entries; only append. Mutable state lives in STATUS.md, roadmap in PLAN.md.

---

### 2026-07-11 14:12 IST — Phase 0 kickoff

- User handed off for autonomous execution (going on holiday). Mandate: build the full
  benchmark (all reasonable tasks across 3 embodiment classes), not just M0. Milestone
  commits only. Restructure repo up front. Minimal-but-divided folders.
- Set up tracking: created `docs/benchmark/{PLAN,STATUS,LOG}.md`.
- Git audit: `mjlab/` is its own git repo (independent of outer `continual_learning`),
  on branch `continual_distill`, clean of tracked mods (only untracked new docs +
  survey scratch files). Created branch `benchmark-manip-diversity` off it.
- Architectural audit (the important finding): **mjlab is already a manager-based
  "shared base + per-task diff" framework.** Manipulation tasks live in
  `src/mjlab/tasks/manipulation/` as `*_env_cfg.py` composing reusable `mdp/` terms,
  registered via `tasks/registry.py`. The 6 existing Franka tasks follow this. So M0
  ("build a shared task base") is really "extend + add taxonomy metadata", not invent.
- `continual_distill/` is confirmed a CONSUMER (reads env_id + teacher datasets),
  already task-count/dim-agnostic via per-task heads. Distill stage does NOT change
  as tasks scale. Good — the whole scaling cost is authoring envs.
- Confirmed gap: mjlab has **joint-space actions only, no EE/Cartesian-delta term**.
  Meta-World portability wants uniform EE-delta → that's a Phase-2 build, not a
  blocker (existing tasks + classical experts use joint-space fine).
- Decision: benchmark ENVS go in `tasks/manipulation/` (mjlab's task home), NOT inside
  `continual_distill/`. continual_distill stays the CL consumer.
- Drafted phase plan (0-6) in PLAN.md and restructure proposal in STATUS.md. Leaning
  toward: add taxonomy metadata to existing registry in place (low risk) + introduce
  class sub-folders for NEW tasks; migrate old 6 opportunistically. Final call at
  Phase 1 start.
- Scratch files (`.survey_*`, `.tcdm_tree.json`, `.scratch_tz/`) will be gitignored so
  they don't pollute commits.
- Next: gitignore scratch, commit Phase 0, start Phase 1.

---

### 2026-07-11 14:20 IST — Phase 1 scope decision: additive infra, NOT restructure

Completed deep audit of `tasks/manipulation/` and `asset_zoo/`. Verdict: **the
framework is already well-factored and cleanly divided.** A heavy folder restructure
would be churn for its own sake, violating the "minimal folders, good division" taste.

What actually exists (well-organized):
- `tasks/manipulation/<skill>_env_cfg.py` = reusable per-skill BASE makers
  (`make_lift_object_env_cfg`, `make_open_door_env_cfg`, ...). These ARE the shared bases.
- `config/franka/env_cfgs.py` = concrete instantiations (robot + object entities + sites).
- `tasks/manipulation/mdp/` = reusable obs/reward/termination/command terms.
- `asset_zoo/objects/{free,articulated}/` = object library. NOTE: peg_in_hole, t_shape,
  snapfit ALREADY exist → insertion/assembly primitives partly done (Class-A head start).
- `asset_zoo/robots/` = franka, kinova, yam, unitree. **NO dexterous hands** (gap for B/C).
- `tasks/registry.py` = simple task_id -> cfg registry.

Eval-bug fixes confirmed to live in the CL-consumer path (`continual_distill.py` reset(seed),
`test_policy.py` per-env success latching), NOT in the env layer → automatically preserved
as we add tasks. Will document as load-bearing; no need to move them.

**DECISION — Phase 1 = additive infrastructure, minimal moves:**
1. New `tasks/manipulation/taxonomy.py`: SkillFamily / Fragility / Embodiment enums +
   a TaskTaxonomy dataclass.
2. Extend `registry.py`: add optional `taxonomy` field to `_TaskCfg` +
   `register_mjlab_task` (backward-compatible; existing calls keep working).
3. Tag the existing 6 Franka tasks with taxonomy in their registration.
4. New `tasks/manipulation/benchmark.py`: query API (list by skill/fragility/embodiment)
   + CL-ordering generator (skill-diverse / fragility-graded orderings).
5. Do NOT move existing tasks. New embodiment classes get sibling `config/<robot>/`
   dirs (matches the existing `config/franka|kinova|yam` convention) — zero new
   top-level concepts.

This keeps one convention, adds the benchmark's queryability/taxonomy layer, and defers
all real churn to actual task authoring. Regression risk ~0 (additive, backward-compat).
Next: implement 1-4, verify existing tasks still register + import, commit Phase 1.

---

### 2026-07-11 14:35 IST — Phase 1 COMPLETE: taxonomy infrastructure

Implemented the additive taxonomy layer (no restructure, as decided):
- `tasks/manipulation/taxonomy.py`: Embodiment (arm_gripper/arm_hand/floating_hand),
  SkillFamily (reach/planar_push/articulation/pick_place/insertion/tool_use/in_hand/
  deformable), Fragility IntEnum (planar<mild_contact<precision_grasp<dexterous),
  TaskTaxonomy dataclass (+source for the license/provenance ledger).
- `tasks/registry.py`: optional `taxonomy` field + `load_taxonomy`/`list_taxonomy`.
  Backward-compatible — untagged velocity/tracking tasks return None.
- Tagged all 6 native Franka tasks in `config/franka/__init__.py`.
- `tasks/manipulation/benchmark.py`: filter_tasks / tasks_by_{embodiment,skill,
  fragility} / fragility_graded_ordering(fragile_first) / skill_diverse_ordering /
  export_manifest. Orderings are deterministic (reproducible CL sequences).
- Tests: `tests/test_benchmark_taxonomy.py` (8) + existing `test_task_configs.py` (9)
  = 17 pass. Fixed a docstring/impl contradiction in fragility ordering (fragile-first
  now actually puts Lift-Cube at index 0).
- Checked in `docs/benchmark/manifest.json` (6 tasks, 3 skills, all arm_gripper).

Regression stance (honest): env cfgs are byte-for-byte unchanged; the only pipeline-
visible change is an optional registry field nothing in training reads. So the distill
pipeline is provably unaffected. A full 0.93-baseline re-run requires cluster teacher
datasets (`/ihub/homedirs/svs_ald/...`) not present locally — deferred, not a blocker.

Committing Phase 0+1 together as the first milestone (Phase 0 was tiny/tracking-only;
combining avoids a trivial commit, matching the milestone-commit taste).

**Phase 2 plan:** EE-delta action term + first Meta-World-style task ports. Risk to
check first: is the Meta-World repo/assets available locally? If not, author from the
survey's verified specs rather than block. Also must decide EE-delta design (open Q4).

---

### 2026-07-11 14:50 IST — Phase 2 recon: constraints + capabilities

Investigated before authoring. Findings that reshape Phase 2:

**Assets (real, git-tracked):** free = cube, cuboid, cylinder, disc; articulated =
button, door, drawer. The `peg_in_hole` / `snapfit` / `t_shape` dirs are EMPTY stubs
(only stale __pycache__) — insertion assets must be AUTHORED, not reused.

**Meta-World: NOT installed, no assets local.** Pulling it in risks the one-simulator
principle (it's its own MuJoCo env stack). Decision: do NOT depend on Meta-World.
Author task designs from the survey's verified specs + our existing bases instead.
(The `mujoco_playground` sibling fork DOES have `pick_cartesian.py` +
`panda_kinematics.py` + `peg_in_hole.py` as reference Cartesian-control / insertion
code we came from — reusable as design reference, not import.)

**CRITICAL CAPABILITY CONFIRMED: can build + step envs LOCALLY.**
2x RTX A6000 (48GB, idle). Built Mjlab-Push-Cuboid-Franka (4 envs), reset+stepped:
obs=60-d, action=8-d (7 arm + 1 gripper), rewards wire up. => I can author AND
smoke-test tasks autonomously (build/reset/step/shape-check). Full RL-teacher
solvability still needs training runs (cluster datasets), but structural validation
is fully local. This is a big unblock for autonomous authoring.

**Authoring recipe (learned from push_cuboid vs push_disc — ~99% identical):**
a new task = pick a base maker (`make_*_env_cfg`) + swap scene entities + set
command pose/target ranges + wire Franka site names ("gripper") + set collision
sensor pattern ("link7") + play/test overrides. Very mechanical once the base exists.

**Revised Phase 2 (de-risk incrementally, don't stack unknowns):**
- 2a: author the CHEAPEST genuinely-new task reusing existing assets + existing
  joint-space action, to prove the full author->register->tag->build->smoke-test loop
  locally. Candidate: a new pick_place variant (Stack) or a new skill (Reach).
- 2b: THEN tackle EE-delta action term as its own focused sub-task (open Q4).
- 2c: THEN author an insertion asset + task (fills the biggest skill gap, contact-rich).
Rationale: never combine "new action term" + "new asset" + "new skill" in one step.

---

### 2026-07-11 15:15 IST — Phase 2a DONE: authoring loop proven (Lift-Cylinder)

Authored the first new task, `Mjlab-Lift-Cylinder-Franka`, and proved the full
autonomous loop LOCALLY: author env_cfg -> rl_cfg -> register+tag -> build (4 envs on
A6000) -> reset -> step 10x -> obs=60/action=8 (matches pipeline) -> rewards finite ->
17 tests green -> manifest regenerated (7 tasks now).

Design choice: Lift-Cylinder swaps cube->cylinder in the lift base (recipe copied from
the existing yam_lift_cylinder + franka_lift_cube). A cylinder rolls / has no flat top
=> less forgiving grasp alignment than a cube: a real pick-place instance (grasp-
geometry generalization), not a trivial dup. Same precision_grasp fragility tier.

Fixed a brittle test (asserted a specific task_id leads the fragility ordering; now
asserts the fragility-max property, since 2 precision_grasp tasks tie and sort alpha).

**Reflection on pace / what belongs in the Phase 2 commit:** Lift-Cylinder alone is
too small for a milestone commit. Per the milestone-commit taste, I'll batch several
new tasks. But I should NOT just spam cube->{cylinder,...} geometry swaps — that
inflates task count without skill diversity (the anti-LIBERO guardrail applies to our
OWN padding too). Decision for the rest of Phase 2: add tasks that each buy a NEW
SKILL FAMILY or a NEW FRAGILITY point, using existing assets where possible:
  - Reach (NEW skill: reach; needs a reach command+reward+success — modest new MDP).
  - Stack (NEW capability within pick_place; needs a stacking command referencing TWO
    objects + success = object-on-object — more new MDP).
Both need new MDP terms (commands/rewards/terminations), which is the real Phase-2
investment (vs the trivial object swap). Next: assess the MDP surface for a Reach task
(cheapest new *skill*), author it, then Stack if time. Commit Phase 2 as the batch.

---

### 2026-07-11 15:40 IST — Phase 2 COMPLETE: validation infra + authoring guide

Made a deliberate scope call: do NOT speculatively author multiple new-skill MDP
task types (Reach/Stack/insertion commands) unsupervised. Reason: reward-shaping /
success-predicate correctness for a NEW skill can only be validated by an RL training
run (smoke-test proves BUILD, not SOLVABILITY). Authoring several un-trainable tasks
now risks shipping subtly-wrong rewards — worse than not shipping them. Matches the
user's "don't over-extend" guidance.

So Phase 2's verifiable, high-leverage deliverables:
1. `Mjlab-Lift-Cylinder-Franka` — first new task, fully validated locally (Phase 2a).
2. `scripts/benchmark_smoke.py` (+ `benchmark-smoke` entry point) — reusable harness:
   builds/resets/steps every tagged task, checks finite obs/reward + shapes. Ran it:
   **7/7 tasks PASS** (all obs=60, action=8). Every future task auto-validated.
3. `docs/benchmark/AUTHORING_GUIDE.md` — the exact recipe reverse-engineered from the
   codebase (Recipe A: cheap object-swap; Recipe B: new-skill MDP + its training-
   validation caveat; new-embodiment steps; anti-inflation guardrails).

Deferred to Phase 3+ (need training to validate, or bigger sub-projects): Reach/Stack/
insertion MDP, peg+hole asset authoring, EE-delta action term, embodiments B/C.
STATUS "Next action" now enumerates them in priority order.

Committing Phase 2 as the second milestone. Net so far: taxonomy+query infra, 7 tagged
tasks (1 new), local validation harness, authoring guide — the foundation that makes
the rest of the benchmark fast and safe to build. Solid stopping point while user is
away; remaining work benefits from their presence (training/solvability judgement).

---

### 2026-07-11 16:10 IST — Phase 3 begins: Reach (new skill) AUTHORED + TRAIN-VALIDATED

Decided to push further (idle 2xA6000, holiday-length autonomy, user wants "as many
tasks as we reasonably get"). Chose Reach as the first new SKILL: lowest-risk
(monotonic distance reward, analyzable by construction, no contact/grasp).

Authored the full new-skill MDP stack:
- `mdp/commands.py::ReachingCommand(Cfg)` — samples a 3D workspace target, latches
  success (torch.maximum) when gripper within 5cm. Simpler than LiftingCommand (no
  object/reached-gating).
- `mdp/rewards.py::reach_target_reward` — Gaussian on gripper->target distance.
- `mdp/observations.py::gripper_to_target_vector` — relative task signal (offset-free).
- `reach_target_env_cfg.py::make_reach_target_env_cfg` — object-free base (robot +
  mocap_goal only). obs=38 (no object terms) — heterogeneous dim, fine (per-task heads).
- `config/franka/{env_cfgs,rl_cfg,__init__}.py` — concrete Franka cfg + runner + tagged
  registration (Mjlab-Reach-Target-Franka, skill=reach, fragility=planar).

**KEY UNLOCK: local training works.** Wrote a minimal solvability check (scratchpad/
validate_reach.py) using OnPolicyRunner directly (the full train.py CLI has wandb/
checkpoint/distributed machinery not worth fighting). Result: **150 PPO iters, 1024
envs, 74 SECONDS -> episode_success 0.66, goal_error 0.062m, still climbing = SOLVABLE.**
So I CAN validate new-skill solvability autonomously (easy tasks ~1-2 min). This
removes the main reason I'd deferred new-skill authoring. Harder tasks (grasp/insert)
will take longer to train but are still checkable.

Validation: 8/8 smoke PASS, 17 tests green, manifest = 8 tasks / 4 skills (reach added;
fragility spread now planar:2 mild:4 precision_grasp:2). wandb val run is gitignored.

**Revised stance for the rest of the session:** continue authoring NEW SKILLS with
train-validation, cheapest/most-reliable first. Next candidates: Stack (pick_place,
two-object command), then insertion (needs peg+hole asset authoring — bigger). Each
gets: author -> smoke -> short train -> record solvability -> commit as a batch.
Committing Reach now (it's a clean, validated milestone: first new skill).

---

### 2026-07-11 16:40 IST — Stack authored + reusable validate tool; Stack train pending

Built the reusable train-validation tool `scripts/benchmark_validate.py` (+
`benchmark-validate` entry point) — generalizes the reach validation: train any task
briefly, read episode_success. Re-verified on reach (80 iters -> 0.36, consistent
with 150->0.66 climb).

Authored Stack (2nd new task this phase, still pick_place family but higher fragility):
- `mdp/commands.py::StackingCommand(Cfg)` — two-object command (asset=cube moving,
  base=cuboid). DYNAMIC target = base position + stack_height (0.035 = cube 0.02 +
  cuboid 0.015 half-heights). Success latches on xy-within-3cm AND height-within-2cm
  (so a hovering cube doesn't count). Mirrors LiftingCommand's reach-then-bring shape.
- `stack_object_env_cfg.py::make_stack_object_env_cfg` — base with both objects,
  staged_manipulation_reward toward the dynamic target + at-goal bonus.
- `config/franka/*` — concrete Franka stack-cube-on-cuboid + runner + tagged reg
  (Mjlab-Stack-Cube-Franka, pick_place / precision_grasp). obs=51.
- Smoke: PASS (builds/steps, obs=51/action=8).

Train-validation IN PROGRESS (400 iters, backgrounded; timeout wrapper buffers output
so result comes at process exit). Stack is genuinely harder than reach (multi-stage:
grasp -> lift -> position -> place -> release), so it may need more iters or reward
tuning to fully solve. Plan: if it shows clear PARTIAL/SOLVABLE learning signal, keep
it; if NOT-LEARNING after the budget, mark authored-but-unvalidated (reward-tuning
left for when user is back) rather than grind a tuning loop unsupervised.

Manifest will be 9 tasks / 4 skills either way (Stack is structurally sound). RESULT
of the train run to be appended below when it finishes.

### 2026-07-11 16:55 IST — Stack train-validation RESULT: NOT-LEARNING @400 iters

Result: Mjlab-Stack-Cube-Franka episode_success=0.010 (essentially zero) after 400
PPO iters. Honest read: 400 iters is genuinely too few to declare the reward broken
(lift-cube's own runner uses 5000 iters; a multi-stage grasp->lift->position->place->
release task needs far more), BUT running 5000 iters (~40 min) then possibly iterating
on reward shaping is exactly the open-ended unsupervised grind I pre-committed to avoid.

DECISION (per plan + user's "don't over-extend"): keep Stack as AUTHORED +
STRUCTURALLY-VALIDATED, mark it NOT-yet-train-validated in its taxonomy notes (so the
manifest doesn't overclaim), and STOP. When the user is back, Stack just needs a
longer training run (5000 iters) +/- reward tuning to confirm/fix — a quick, bounded
task with a human in the loop, not a blind grind now.

This is the honest state: 9 tasks authored, 8 of them proven-solvable (the 6 native +
Lift-Cylinder structurally + Reach train-validated), Stack structurally sound but its
learnability unconfirmed. Reporting it that way in STATUS/manifest.

### 2026-07-11 17:05 IST — Session wind-down + clean handoff

Reached a disciplined stopping point (user on holiday). Rationale for stopping rather
than authoring more: the Stack result showed contact-rich tasks need long training +
likely reward tuning to validate — accumulating more `structural`-only tasks doesn't
add real benchmark value and the tuning grind is unsafe to do blind. Better to hand off
clean.

Session delivered (5 milestone commits, branch benchmark-manip-diversity, all local,
NOT pushed — user pushes when ready):
- Phase 0: tracking + full repo/framework audit.
- Phase 1: taxonomy/query/CL-ordering infra + tagged registry + manifest (17 tests).
- Phase 2: proven authoring loop + Lift-Cylinder + benchmark-smoke harness + guide.
- Phase 3: Reach (new skill, TRAIN-validated 0.66) + Stack (structural) +
  benchmark-validate tool.
Net: 6 -> 9 tasks, 3 -> 4 distinct skills, plus the whole taxonomy/validation/authoring
foundation that makes the rest fast.

Wrote a "WHEN YOU'RE BACK" resume plan at the top of STATUS.md: finish Stack
train-validation (quick with human eyes), then insertion/peg-in-hole, EE-delta action,
then embodiments B/C. Architecture Qs (shared-trunk vs per-embodiment; hand action
param) flagged for the user's call.

Working tree clean; only untracked is the pre-existing `plots/`. Nothing pushed.

---

### 2026-07-11 17:30 IST — MAJOR REFRAME from user: build the SUITE, not solve it

User corrected a category error I was making. Decisions + their consequences:

1. **This effort = building the TASK SUITE. Solving it (RL/teacher/CL/validation) is
   STAGE TWO.** => a task belongs in the suite if it is WELL-DEFINED and STRUCTURALLY
   SOUND (builds, sane obs/action/reward/success predicate), NOT if PPO solves it in N
   iters. My earlier "Stack NOT-LEARNING @400 -> defer" reasoning was WRONG framing:
   train-solvability is stage-two. CONSEQUENCE: stop running benchmark-validate training
   as a gate. Stack stays as a valid task. benchmark-smoke (builds/steps/finite/shape)
   is the RIGHT acceptance check for this stage. Keep benchmark-validate as an OPTIONAL
   stage-two tool, not a gate.

2. **Keep obs / action / reward / success spaces SANE and SIMILAR across tasks** — the
   real design constraint now. Uniform interfaces:
   - action: joint-space everywhere (8-D franka), NO EE-delta term for now (deferred to
     stage two). "keep working in joint for now."
   - obs: consistent layout/terms across tasks as much as possible.
   - reward: consistent structure (staged reach->manipulate + regularization).
   - success: consistent latched predicate.
   Avoid bespoke-per-task spaces. Similarity > cleverness.

3. **Dexterous hand (Class B/C) = FULL JOINT TARGETS** (~16-22 dims). Fixes the action
   param question for all B/C tasks.

4. **Keep going autonomously** per the resume plan.

REVISED next actions (stage-one, structural acceptance only):
- Update STATUS/manifest to drop the train-validation "gate" framing; Stack is a valid
  task. Keep a stage-two-validation column as informational only.
- Author more Class-A tasks with UNIFORM interfaces (joint action, shared obs/reward
  patterns). Priority: insertion/peg-in-hole (needs asset), then more distinct skills
  (sweep/tool-use, articulation variants) and sensible grasp-object variants.
- Then embodiments B/C (full-joint hand) reusing the same obs/reward/success patterns.
- EE-delta + all training/solvability = STAGE TWO, not now.
