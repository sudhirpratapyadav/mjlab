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
