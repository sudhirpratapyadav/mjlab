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
