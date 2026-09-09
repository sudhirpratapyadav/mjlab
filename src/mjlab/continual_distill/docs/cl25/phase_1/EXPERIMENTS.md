# Phase 1 — EXPERIMENTS

> The run table for Phase 1 only: classical teachers at >= 0.90 SR for all 25 tasks.
> The plan behind it is `phase_1_plan.md`; results land in `STATUS.md`, narrative in
> `LOGS.md`. Phases 2 (RL teachers) and 3 (distillation + SI) get their own folders.
>
> **Empty on purpose — filled by the agent that owns each row.**

Measurement protocol (from `phase_1_plan.md` section 4): iterate at n=32, claim a
sub-0.3 number at n>=96, **claim >= 0.90 only at n=128**. Always report `SR (n)` and
the HEAD.

---

## Wave schedule and agent assignment

> Added 2026-09-08. Orchestrated by the lead agent; each row below is dispatched as one
> subagent. **Compute budget: holder `20277` (`hold_dgx_amit`, dgx1, 8x A100). It is the
> team's own holder — reuse it, never `scancel` it. GPUs 0,1,2,3 were idle on 2026-09-08
> and are ours to take; GPUs 4-7 were at 80-98% util and carry someone's live run — do
> NOT touch them.** Four concurrent GPU agents, no more. Re-check free indices at the
> start of every wave; record the index you took in `LOGS.md`.

| Wave | Agent | Scope | GPU | Blocks |
|---|---|---|---|---|
| 0 | W0 | L0-1 render tool, L0-2 site restructure, L0-3 publish script | 0 | every video |
| 1 | W1-a | E-1..E-5 confirm at n=128 | 0 | — |
| 1 | W1-b | A-1 Drag-Pull, A-2 Cage-Drag | 1 | — |
| 1 | W1-c | A-3 Topple-Block, A-4 Push-Flap | 2 | — |
| 1 | W1-d | A-5 Axial-Extract, A-6 Edge-Grasp | 3 | — |
| 2 | W2-a | B-1 Turn-Lever, B-2 Open-Drawer, B-3 Flip-Switch | 0 | — |
| 2 | W2-b | B-4 Reorient-Object, B-5 Rotate-Valve | 1 | — |
| 2 | W2-c | C-1 Stack-Cube + C-2 Place-In-Container (shared grasp-retention fix) | 2 | — |
| 2 | W2-d | C-3 Push-Cuboid (attempt 8) | 3 | — |
| 3 | W3-a | C-4 Tool-Pull, C-6 Open-Door — re-measure + refresh characterisation | 0 | — |
| 3 | W3-b | C-5 Peg-Insertion | 1 | — |
| 3 | W3-c | D-1..D-3 time-boxed attempt, then characterise | 2 | — |
| 3 | W3-d | Video render + publish sweep for every resolved task | 3 | needs W0 |
| 4 | lead | STATUS reconciliation, exit-criteria check, Phase-2 handover brief | — | needs all |

**Wave gate:** a wave starts only when every agent in the previous wave has written its
`STATUS.md` row and its `LOGS.md` entry. W3-d additionally needs L0-1 and L0-2 done.

**Realistic outcome to plan against:** ~16-18 of 25 above the bar, the remainder
characterised with a failure video. Not 25/25 — stated now, not discovered at the end.

## Lane 0 — Infrastructure (blocks the video deliverable)

| ID | Item | What it unblocks | Owner | State | Done |
|---|---|---|---|---|---|
| L0-1 | Rewrite `classical/render_rollout.py` -> any task, mp4 out, success + failure clip | every task's video | W0 | done | verified on Reach-Target (success only), Open-Door (failure only), Open-Drawer (both) |
| L0-2 | Site restructure: `/benchmark/` + `/phase1/` + landing page | publishing | W0 | done | live at cl.untuai.com, all URLs verified 200 |
| L0-3 | Publish script (scp wrapper) | 20+ agents not reinventing it | W0 | done | `classical/publish_rollout.sh`, uses rsync, ssh verified |

## Lane A — Wave-1 scripted teachers (6, greenfield)

| ID | Task | Owner | State | SR (n) | HEAD | Video |
|---|---|---|---|---|---|---|
| A-1 | Drag-Pull | W1-b | **below bar** | 0.477 (128) | `1127d12` | published |
| A-2 | Cage-Drag | W1-b | **below bar** | 0.141 (128) | `1127d12` | published |
| A-3 | Topple-Block | W1-c | **above bar** | 0.938 (128) | `1127d12` | published |
| A-4 | Push-Flap | W1-c + lead | **above bar** | 1.000 (128) | `1127d12` +asset fix | published |
| A-5 | Axial-Extract | W1-d | **below bar** | 0.781 (128) | `1127d12` | published |
| A-6 | Edge-Grasp | W1-d | **characterised failure** | 0.000 (96) | `1127d12` | published |

## Lane B — Lift to the bar (5, currently 0.5–0.89)

| ID | Task | prior SR (n) | Owner | State | SR (n) | HEAD | Video |
|---|---|---|---|---|---|---|---|
| B-1 | Turn-Lever | 0.812 (32) | W2-a | not started | | | |
| B-2 | Open-Drawer | 0.719 (32) | W2-a | baselined 0.711 (128) | | | published |
| B-3 | Flip-Switch | 0.615 (96) | W2-a | not started | | | |
| B-4 | Reorient-Object | 0.594 (32) | W2-b | not started | | | |
| B-5 | Rotate-Valve | 0.500 (32) | W2-b | not started | | | |

## Lane C — The hard six (below 0.5)

| ID | Task | prior SR (n) | known mechanism | Owner | State | SR (n) | Video |
|---|---|---|---|---|---|---|---|
| C-1 | Stack-Cube | 0.28–0.375 (96) | grasp retention | W2-c | not started | | |
| C-2 | Place-In-Container | 0.27–0.29 (96) | grasp retention | W2-c | not started | | |
| C-3 | Push-Cuboid | 0.078 (128, stale) | endgame precision; overtake/direction-instability/endgame-overtake DISPROVEN; cross-track only partial; 0.078 was measured under a pre-`1127d12` placement current HEAD no longer uses | W2-d | **characterised, below bar** | 0.219 / 0.227 (128 x2) | pending |
| C-4 | Tool-Pull | 0.039 (128) | squeeze-phase ejection, lateral >= axial (refined, was "axial ejection") | W3-a | **characterised, below bar** | 0.055 (128) | pending |
| C-5 | Peg-Insertion | 0.031–0.062 (96) | endgame precision at 3 mm | W3-b | not started | | |
| C-6 | Open-Door | 0.000 (32) | 84.3/90 deg, no partial credit, 150 steps | lead | **characterised failure** | 0.000 (128) | published |

## Lane D — RL-by-design: attempt, then characterise (3)

| ID | Task | what defeats waypoints | Owner | State | SR (n) | Video |
|---|---|---|---|---|---|---|
| D-1 | Strike-Slide | impulse calibration | W3-c | not started | | |
| D-2 | Pivot-Lift | two-phase contact-mode switch | W3-c | not started | | |
| D-3 | Throw-To-Bin | release timing | W3-c | not started | | |

## Already at the bar — confirm only (5)

Re-measure on current HEAD at n=128 and publish a video; no development expected.

| ID | Task | prior SR (n) | Owner | State | SR (n) | Video |
|---|---|---|---|---|---|---|
| E-1 | Reach-Target | 1.000 (32) | W1-a/lead | **done** | 1.000 (128) | pending |
| E-2 | Lift-Cube | 1.000 (32) | W1-a/lead | **done** | 1.000 (128) | pending |
| E-3 | Push-Button | 1.000 (32) | W1-a/lead | **done** | 0.984 (128) | pending |
| E-4 | Slide-Window | 1.000 (32) | lead | **done** | 1.000 (128) | pending |
| E-5 | Open-Lid | 1.000 (32) | lead | **done** | 1.000 (128) | pending |

## Open questions

| # | Question | Why it matters | Decided? |
|---|---|---|---|
| Q1 | Competence bar for a classical teacher | — | **yes: >= 0.90 at n=128** |
| Q2 | What happens to tasks that cannot reach 0.90 — RL teacher, or out of the suite? | 3 of the likely shortfalls (Push-Cuboid, Open-Door, Peg-Insertion) are in the prior CL sequence, so dropping them breaks comparability with P0/P1 | no — Phase 2 |
| Q3 | How are orderings sampled once 24 permutations no longer exhaust the space? | ordering claims depend on it | no — Phase 3 |
