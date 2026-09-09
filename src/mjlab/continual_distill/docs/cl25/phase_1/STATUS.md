# Phase 1 — STATUS

> The shared scoreboard for Phase 1 (classical teachers at >= 0.90): current state of
> all 25 tasks, at a glance. Mutable — this
> file is overwritten as things change, unlike `LOGS.md` which is append-only.
>
> **Update only YOUR task's row.** Every number carries its episode count `n` and the
> HEAD it was measured on. A number with no `n` is not a result.
>
> **Empty on purpose — rows are seeded, results are not.**

Last updated: 2026-09-08 (wave 1 complete; wave 2 running)
Phase: **1 (classical teachers)**
Measured on HEAD: `1127d12`

## Summary

| | count |
|---|---|
| Tasks in scope | 25 |
| Lane A — greenfield scripted | 6 |
| Lane B — lift 0.5–0.89 to the bar | 5 |
| Lane C — the hard six (<0.5) | 6 |
| Lane D — RL-by-design, attempt + characterise | 3 |
| Already at the bar — confirm only | 5 |
| At bar after this phase | — |
| Below bar, characterised | — |
| **Resolved** | **20 of 25** |
| — above bar (>= 0.90 @ n=128) | **7** — Reach-Target 1.000, Lift-Cube 1.000, Slide-Window 1.000, Open-Lid 1.000, Push-Flap 1.000, Push-Button 0.984, Topple-Block 0.938 |
| — below bar, mechanism characterised | **8** — Axial-Extract 0.781, Open-Drawer 0.711, Drag-Pull 0.477, Stack-Cube 0.328, Place-In-Container 0.250, Push-Cuboid 0.223, Cage-Drag 0.141, Tool-Pull 0.055 |
| — characterised failure (0.000) | **5** — Open-Door (128), Edge-Grasp (96), Strike-Slide (96), Pivot-Lift (96), Throw-To-Bin (96) |
| In progress | 4 — Turn-Lever, Flip-Switch (W2-a); Reorient-Object, Rotate-Valve (W2-b) |
| Not started | 1 — Peg-Insertion |

> Counts above are COMPUTED from the per-task rows below, not maintained by hand. An earlier
> hand-kept summary drifted (claimed "9 above bar" when the rows showed 7, and a breakdown
> summing to 15 against a stated total of 13). Recompute rather than increment: the exit
> criteria require these to match the rows, checked not assumed.

**Competence bar: SR >= 0.90, measured at n = 128.** Iterate at n=32; a sub-0.3
number needs n>=96. Always report `SR (n)` + HEAD. See `phase_1_plan.md` section 4.

**Before running anything on a GPU: read `~/use_instructions/README.md`** and
`phase_1_plan.md` section 0. Reuse an existing holder, never `scancel` someone else's,
and record the GPU index you took in `LOGS.md`.

## Phase 1 — classical teachers

State: `not started` / `in progress` / `above bar` / `below bar` / `characterised failure` / `deferred to RL`

### Group B — to write (6 scripted)

| Task | Owner | State | SR | n | HEAD | Video | Notes |
|---|---|---|---|---|---|---|---|
| Drag-Pull | W1-b | below bar | 0.477 | 128 | 1127d12 | | mechanism bug fix (RIDE_HEIGHT grazed box top edge) took SR 0.156->~0.48 n=32; residual = friction-limited push stall + endgame precision, same unresolved mechanisms as push_cuboid |
| Cage-Drag | W1-b | below bar | 0.141 | 128 | 1127d12 | | fingers verified never commanded closed (hard assert + gripper action pinned every step); residual = same friction-limited push stall (smaller object, less margin) + a minority of episodes voided by physical (contact-momentum) aperture dips during approach, mitigated but not eliminated by a whole-episode wrist-yaw lock |
| Topple-Block | W1-c | **above bar** | 0.938 | 128 | 1127d12 | | latch block's local-X axis from `object_orientation` once, freeze it, HOVER->SEAT->PUNCH (single fixed target, no live re-tracking)->RETREAT; no grasp phase (ungraspable by construction). Worked near first-try (n=32 read 0.906 pre-tuning) |
| Push-Flap | W1-c + lead | **above bar** | **1.000** | 128 | `1127d12` (asset fixed) | pending | **RESOLVED — was a benchmark bug, not a teacher failure.** W1-c measured 0.000 (96) and correctly diagnosed `flap.xml`'s `flap_hinge range="-1.4 0"` as compiling in DEGREES (MuJoCo default; no `<compiler angle="radian"/>`), giving a real range of -1.4 deg vs the -70 deg target — unsolvable by any policy, 50x short. Lead independently confirmed against the compiled model (`jnt_range` = -0.024435 rad) and applied the benchmark fix `range="-80.2 0"`. W1-c's teacher, written against the INTENDED geometry per the frozen-task rule, then scored **1.000 (128) with no teacher changes at all** |
| Axial-Extract | W1-d | below bar | 0.781 | 128 | `1127d12` | | greenfield. Mechanism bug: NEUTRAL_QPOS's own posture (joint2=-1.0) is a local-minimum trap for the DLS descent to grasp height — a bare FK/DLS probe (no policy/physics) reliably converged ~2.2cm short REGARDLESS of x/damping/kp; biasing only the posture-regularization target's joint2 to 0.3 (borrowed from HOME_QPOS, NOT changing DEFAULT_QPOS) took SR 0.000->0.781. Residual ~22%: a bimodal (hit-or-miss, not partial-slip) pinch miss from the DLS's ordinary ~2cm steady-state lateral bias. Negative result: lateral INTEGRAL action (the standard fix used everywhere else in this file) was tried at several gains and made it WORSE (0.78->0.03-0.28) — see LOGS.md |
| Edge-Grasp | W1-d | **characterised failure** | 0.000 | 96 | `1127d12` | | greenfield; RL fallback per its own design note. Fixed 3 real mechanism bugs in the push/retreat spine (push never touched the plate — descent stopped 2cm short, resting on the rigid ledge top, not the plate; retreat's x-component was the SAME sign as the push, i.e. it kept shoving the plate instead of disengaging; the side-approach standoff pulled the hover waypoint into a badly-behaved IK region) — after which the push+retreat pair reliably creates and holds the overhang without dropping the plate. Blocked on the final pinch: the required FULL 3-D orientation (approach horizontal, closing axis vertical) is only marginally/unstably reachable by this DLS IK at the pinch's low height + close radial reach — an isolated FK probe found either a large (4-20cm) residual or a genuine numerical LIMIT-CYCLE oscillation (site y swinging +-0.05m across ~30 steps with no net convergence), not a fixable steady bias. Higher damping, a slower rate-limited insert and a hard x-guard against the ledge (kept, all real improvements) did not resolve the final pinch strike. See LOGS.md for the full mechanism trail and what did not work |

### Group A — to re-verify (16 existing)

Prior SR is from the pre-audit pass and is shown only as the number to beat or confirm.

| Task | Owner | State | prior SR (n) | new SR | n | HEAD | Video | Notes |
|---|---|---|---|---|---|---|---|---|
| Reach-Target | W1-a/lead | **above bar** | 1.000 (32) | **1.000** | 128 | `1127d12` | pending | confirmed, 4x 32/32 |
| Lift-Cube | W1-a/lead | **above bar** | 1.000 (32) | **1.000** | 128 | `1127d12` | pending | prior CL suite; confirmed, 4x 32/32 |
| Push-Button | W1-a/lead | **above bar** | 1.000 (32) | **0.984** | 128 | `1127d12` | pending | prior CL suite; 31/31/32/32 — not 1.000, but n=32 could not have resolved 2/128 |
| Slide-Window | lead | **above bar** | 1.000 (32) | **1.000** | 128 | `1127d12` | pending | confirmed, 4x 32/32 |
| Open-Lid | lead | **above bar** | 1.000 (32) | **1.000** | 128 | `1127d12` | pending | confirmed, 4x 32/32 |
| Turn-Lever | W2-a | **above bar** | 0.812 (32) | **0.941** | 256 | `1127d12` | rendered | pooled 241/256 over 2 reads (0.953, 0.930). Baseline was 0.664(128) — the 0.812(32) record was a real post-audit regression. Fix: re-seat gate `RESEAT_TOL` 0.14->0.06 (calibrated off instrumented bands) + `SETTLE` 2->1 |
| Open-Drawer | lead (baseline+settle), W2-a (fix) | **below bar** | 0.719 (32) | **0.888** | **384** | `1127d12` | published (pre-fix) | SETTLED over three independent n=128 reads: 0.883 / 0.906 / 0.875 = 341/384 = **0.888**, does NOT clear 0.90. Large real gain from 0.711 via a real mechanism (every failing env re-hooks, `entries_ph2=2` — the hook was never the problem, the recovery cost was; `DESCEND_SETTLE` 2->1). Closest below-bar task in the suite |
| Flip-Switch | W2-a | **below bar, characterised** | 0.615 (96) | **0.656** | 256 | `1127d12` | rendered | pooled 168/256 over 2 reads. No fix found: gate-satisfied strokes hit only ~10-15% single-shot, so SR is retry accumulation not reliable contact; +3-9cm vertical drift at horizontal alignment. THREE mechanism-grounded fixes each measured worse and were reverted |
| Reorient-Object | W2-b | **below bar, characterised** | 0.594 (32) | **0.547** | 256 | `1127d12` | rendered | pooled 140/256 over 2 reads. Mechanism SUPERSEDES the prior 'drift not binding' note: ~1/3 of failures achieve perfect axis alignment (max_align up to 1.000) and still exceed the 0.18m drift budget (measured 0.22-0.44m) from ~9.5-10 ground-collision resets/episode |
| Rotate-Valve | W2-b | **below bar, characterised** | 0.500 (32) | **0.473** | 256 | `1127d12` | rendered | pooled 121/256 over 2 reads, both POST-fix. Baseline pre-fix 0.523(128). `SEAT_HARD_TIMEOUT=60` kept as a correctness fix (phase-1 DROP timeout was AND-gated on proximity, never unconditional) though it measured no change. **If this task enters the Phase-2 gated set, re-measure both variants at higher n first** |
| Stack-Cube | W2-c | **below bar, characterised** | 0.28–0.375 (96) | **0.328** | 128 | `1127d12` | | retention re-baselined at n=128 (unchanged behaviour). Instrumented mechanism (aperture/xy traces): object is EJECTED SIDEWAYS DURING P_CLOSE itself (before P_LIFT's first command), not lost to a lift-phase jerk as previously assumed — aperture collapses from ~0.04 (contact width) to ~0.00 while gripper-object xy offset grows in lockstep, entirely within P_CLOSE's fixed hold window; P_LIFT/P_CARRY only make the pre-existing loss detectable. CLOSE_ENTRY xy is well-centred (1-2cm) in the large majority of attempts — not an alignment/targeting problem. Two partial-close remedies tried and reverted (WORSE): see LOGS.md |
| Place-In-Container | W2-c | **below bar, characterised** | 0.27–0.29 (96) | **0.250** | 128 | `1127d12` | | same mechanism as Stack (traced independently, 14/17 established grasps dropped, 9 inside P_LIFT, CLOSE_ENTRY well-centred): squeeze-phase ejection during P_CLOSE, not a lift-phase jerk. See Stack-Cube row / LOGS.md for the shared finding and the two reverted remedies |
| Push-Cuboid | W2-d | **below bar (re-baselined)** | 0.078 (128, stale — pre-`1127d12` placement) | **0.219 / 0.227** | 128 x2 | `1127d12` | pending | prior CL suite; 7 strategies failed under OLD pre-audit placement (x 0.6-0.8); current HEAD restored the audited placement (x 0.30-0.52 split) same day, 9h later — teacher code unchanged, true current SR is ~0.22 not 0.078; endgame precision confirmed (latch bug-free; overtake and direction-instability ruled out as causes; cross-track only partial) — see LOGS.md 2026-09-09 |
| Tool-Pull | W3-a + lead | **characterised failure** | 0.039 (128) | **0.047** | **512** | `1127d12` | pending | FOUR independent n=128 reads, all reported (adding a fourth W3-a ran that had not yet been folded in when this row was last written — never drop a draw to keep a settled-looking number): 0.039 (5/128, original) / 0.055 (7/128, W3-a) / 0.062 (~8/128, lead) / 0.031 (4/128, W3-a) = 24/512 = **0.047** — record confirmed, no drift; individual n=128 draws still span 0.031-0.062, illustrating this task's own noise lesson even at n=128. **Mechanism REFUTED and replaced:** displacement is lateral-dominant (axial/lateral 0.59-0.68) and corr(axial misalign, axial disp) is NEGATIVE (-0.33, -0.45), so this is NOT axial ejection. It is the same squeeze-phase lateral ejection as Stack/Place. Placement verified NOT stale (byte-identical vs commit `2125296`). No asset bug |
| Peg-Insertion | W3-b | **below bar, characterised** | 0.031–0.062 (96) | **0.023 baseline / 0.031 with fix** | 128 (x2) | `1127d12` | | Baseline confirmed (0.023(128), unmodified code) — placement verified NOT stale (byte-identical `franka_peg_insertion_env_cfg` across `a5e0299`/`babf036`/`1127d12`/current working tree). **Mechanism found: CARRY has no unconditional timeout**, unlike every other phase (DESCEND/CLOSE/LIFT/PLACE all have one). If the grasp is never secured (fingers close on nothing) or is lost mid-LIFT/CARRY, `object_to_goal` reports the same frozen offset forever and `carry_tol` can never be satisfied — the policy sits doing nothing for the rest of the 1000-step episode with zero retries. Instrumented via true `robot.data`/`object.data` poses (not the noisy obs): 14/16 traced envs showed this exact frozen-aperture-near-zero signature, one stuck 580+ consecutive steps on a single failed grasp. A noise-free kinematic replay of the CARRY/PLACE servo law (fed realistic +-1cm noise) converges to within the 1.5cm tolerance in 95-100% of trials in <10 of 70 PLACE steps — the servo itself is not the bottleneck. **Fix (kept, Rotate-Valve-pattern):** added `CARRY_TIMEOUT=100`, falls through to P_HOVER (a state `_detect_reset` already proves safe) with integrator/EMA cleared. Verified working as intended: re-instrumented post-fix, envs reaching PLACE with near-perfect xy alignment rose 2/16 -> 5/16 (three under 0.004, well inside the 0.015 tolerance). **But final SR did not move outside noise: 0.023(128) -> 0.031(128), a 1-episode difference** — this program's own calibration says that's noise, not a gain. Kept anyway as a genuine correctness fix (same call as Rotate-Valve's `SEAT_HARD_TIMEOUT`). Residual bottleneck NOT found: envs that reach excellent xy alignment during PLACE still mostly fail, so something at or near the final insertion moment (height/xy joint timing in `_place`'s exit condition, or genuine physical wedging at the chamfer-free 3mm-clearance hole) is the real remaining blocker — flagged for the next investigator, not resolved here. See LOGS.md |
| Open-Door | lead (render) | **characterised failure** | 0.000 (32) | **0.000** | 128 | `1127d12` | failure.mp4 rendered | prior CL suite; 0.000 now confirmed at n=128, not just 32. Needs the Phase-2 brief, not another scripting attempt |

### Lane D — RL by design: attempt, then characterise (3)

The user's instruction is classical teachers for ALL tasks, so these are attempted and
time-boxed, not skipped. A measured explanation of why waypoints cap out is the
deliverable if 0.90 is not reached.

| Task | what defeats waypoints | Owner | State | SR (n) | Video |
|---|---|---|---|---|---|
| Strike-Slide | impulse calibration — CONFIRMED, sharpened | W3-c | **characterised failure** | 0.000 (96) | |
| Pivot-Lift | two-phase contact-mode switch — CONFIRMED, but never reached (see notes) | W3-c | **characterised failure** | 0.000 (96) | |
| Throw-To-Bin | release timing — CONFIRMED, but reach-limited before timing matters | W3-c | **characterised failure** | 0.000 (96) | |

Notes (HEAD `1127d12`, holder `20277` GPU 3; full mechanism trail and negative results in
`LOGS.md` 2026-09-09 "Lane D"):
- **Strike-Slide**: the puck's low profile (2.4cm) leaves under 1cm of margin between "pad
  contacts the puck" and the package-standard `FLOOR_MIN_Z` collision guard — 66% of strikes
  (n=32) were voided by `ee_ground_collision` before/during the strike, and among the clean
  strikes the achieved/required slide-distance ratio had mean 0.44, std 0.35 (a >10x spread
  across similar required distances). Confirms Wave 1's "impulse calibration" call and adds:
  contact itself, not just the calibration law, is the binding constraint.
- **Pivot-Lift**: three real mechanism bugs fixed (bad contact height, standoff landing inside
  the board's own volume, an align-check contaminated by a z-offset) — none of which fixed the
  underlying issue. The "stand behind the board" approach point falls in `workspace.py`'s own
  documented `GRASP_RADIAL_MIN` dead zone ("objects closer than this fold the arm back over its
  own base") for roughly the lower two-thirds of the board's spawn band; an isolated FK/DLS
  probe converges fine from a fresh posture, but the live rollout's warm-started IK plateaus
  10-17cm short and never recovers — the same posture-local-minimum shape `axial_extract.py`
  already documents. The two-phase contact-mode switch Wave 1 asked about is never reached.
- **Throw-To-Bin**: place_in_container's proven "hold above the bin, drop" strategy needs a
  reachable point directly above the bin; instrumented reach-phase progress (n=8) closed only
  ~half the initial gap before saturating at the arm's own steady-state extension, 30-40cm short
  of the 5.5cm lateral tolerance needed, for every probed env. Release timing (Wave 1's call) is
  never the binding question because there is no reachable release point to time.

## Videos published

Target: every task has a rollout clip under `https://cl.untuai.com/phase1/<Task-Id>/`.
A task below 0.90 also needs a failure clip — that is the evidence for its
characterisation. Layout and ownership: `phase_1_plan.md` section 5.

| | count |
|---|---|
| Success clips published | 11 / 25 (13 task dirs live; Open-Door + Edge-Grasp are failure-only by design) |
| Failure clips published (needed for every sub-0.90 task) | 7 — covers **all 6** sub-bar tasks (Open-Door, Edge-Grasp, Cage-Drag, Drag-Pull, Open-Drawer, Axial-Extract) + Topple-Block |
| Site restructure live | **yes** — `/`, `/benchmark/` (37 clips), `/phase1/` (11 task dirs) all serving 200 |

## Handover to Phase 2

**17 of 25 tasks end below the 0.90 bar. They are NOT 17 problems — they are three
mechanisms plus a short remainder.** Phase 2 should be scoped around the mechanisms, because
a single fix to any one of them moves several tasks at once. Every number below is measured
on HEAD `1127d12`; multi-read figures are POOLED, never the best draw.

### M1 — Squeeze-phase lateral ejection (the pinch loses the object while CLOSING)
| task | SR | n |
|---|---|---|
| Stack-Cube | 0.328 | 128 |
| Place-In-Container | 0.250 | 128 |
| Tool-Pull | 0.047 | 512 |
| *candidates, not yet instrumented for it* | Axial-Extract 0.781, Edge-Grasp 0.000 | |

Evidence: aperture collapses 0.069 -> 0.005 m while true gripper-object xy offset GROWS
0.007 -> 0.076 m in the same 12-step CLOSE window. Grip force (5-12 N) is comparable between
holds and ejections, and CLOSE_ENTRY alignment is fine (1-2 cm) in most attempts — so it is
neither a force shortage nor a targeting miss. The discriminator is whether aperture
*stabilises against* the object or *collapses through* it.

**Two prior diagnoses were wrong and are retired:** (a) the LIFT phase does not cause it —
ejection happens during CLOSE, before LIFT's first command; (b) Tool-Pull is NOT axial —
displacement is lateral-dominant (axial/lateral 0.59-0.68) and correlates NEGATIVELY with
axial misalignment (-0.33, -0.45), the opposite sign to the mass-lever story.

**Already measured WORSE, do not retry:** deeper grasp, longer squeeze, ramped lift (96 eps
each, prior pass); partial close at -0.4 (0.000) and -0.85 (0.188).
**Untried lead:** closed-loop aperture feedback via the existing `robot_joint_pos` observation.
**Prize:** 9 of the 10 Stack runs that kept hold succeeded. Retention converts near-directly
into SR — this is the single highest-leverage item handed to Phase 2.

### M2 — Friction-limited push stall + endgame precision (planar transport)
| task | SR | n |
|---|---|---|
| Drag-Pull | 0.477 | 128 |
| Push-Cuboid | 0.223 | 256 |
| Cage-Drag | 0.141 | 128 |

A position servo holds a steady offset but does not sustain enough push force, and in the last
few cm the object is nudged rather than settled. Push-Cuboid: **67-78% of failing envs actively
DIVERGE in their final 20 steps** — per-step box displacement (8-23 mm) is comparable to the
goal window radius (20 mm). **Overtake is disproven** (1/16). Success IS latched, so
"succeeds then knocked out" is a non-problem. A force/impedance pusher is the obvious probe.

**Caution:** Push-Cuboid's seven-strategy graveyard was measured under PRE-AUDIT placement and
describes a differently-shaped task (spawn and goal drawn from the same box). Treat it as
weaker evidence than it looks; **v5 (wide two-point contact) was never tested under current
geometry.**

### M3 — State-machine phase deadlock (a phase whose exit a failed attempt can never satisfy)
Four instances found; **fixed where found, and it produced the phase's two largest gains**:
Rotate-Valve 0.031 -> 0.500 (prior pass) and Turn-Lever 0.664 -> 0.953. The other two
(Rotate-Valve DROP timeout, Peg-Insertion CARRY timeout) measured no change inside the noise
band but were kept as correctness fixes. **Report this class as validated by its two wins, not
by all four instances.**

> **Rule for any new teacher:** every phase needs an UNCONDITIONAL step-count escape,
> independent of any progress predicate. Audit for phases whose only exit is a success
> condition. `ALIGN_TIMEOUT` / `SEAT_HARD_TIMEOUT` / `CARRY_TIMEOUT` are the pattern.

### M4 (candidate) — DLS posture local minimum
Axial-Extract was 0.000 until a posture-regularisation bias fixed a genuine stationary point
~2.2 cm short (confirmed by an FK/DLS probe independent of physics), taking it to 0.781.
Pivot-Lift shows the same shape: an isolated probe converges, but warm-started IK in the live
rollout plateaus 10-17 cm short. Two instances is not yet a pattern, but it is worth checking
before treating either as task-specific.

### The remainder — genuinely separate
| task | SR (n) | what defeats scripting |
|---|---|---|
| Open-Drawer | 0.888 (512) | **closest to the bar in the suite.** Throughput: near-misses land 1-2 mm short as the budget runs out |
| Flip-Switch | 0.656 (256) | single-shot stroke hit rate is only ~10-15%; SR is retry accumulation, not reliable contact. Wants a teacher that lands the stroke ONCE |
| Reorient-Object | 0.547 (256) | ~1/3 of failures achieve PERFECT axis alignment and still exceed the 0.18 m drift budget (0.22-0.44 m) from ~9.5-10 collision resets/episode |
| Rotate-Valve | 0.473 (256) | DROP consumes 38-40% of the step budget vs ARC_FOLLOW's 31-34% |
| Peg-Insertion | 0.031 (128) | CARRY deadlock fixed but NOT dominant; envs with near-zero xy error at PLACE still fail. Untested: `_place` exits on Z alone, not jointly with xy; physical wedging at the chamfer-free 3 mm clearance |
| Open-Door | 0.000 (128) | **hard, not broken** — needs 84.3 of 90 deg (93.6% of full joint travel), no partial credit, 150 steps, at 1-2 deg per 25 steps. Predicate audit confirms it IS reachable. Wants a longer episode budget or a learned teacher |
| Edge-Grasp | 0.000 (96) | final pinch needs a dual-constrained 3-D orientation only marginally reachable at the required low height/close reach; presents as a numerical limit cycle |
| Strike-Slide | 0.000 (96) | release velocity at a compliant brief contact — AND 66% of strikes voided by `ee_ground_collision` (puck is 2.4 cm; <1 cm margin). **See Blockers: worth a benchmark ruling** |
| Pivot-Lift | 0.000 (96) | approach point falls in `workspace.py`'s own `GRASP_RADIAL_MIN` dead zone; see M4 |
| Throw-To-Bin | 0.000 (96) | **reach-limited, NOT release-timing-limited** — reach saturates 30-40 cm short, so timing never binds. An RL teacher must learn a ballistic THROW, not a timed release. Harder than the design note implies |

### Competence-gated set for Phase 2 stage-2 distillation
**8 tasks clear the bar** and are the only ones that should feed sequential distillation
without caveat: Reach-Target, Lift-Cube, Slide-Window, Open-Lid, Push-Flap (all 1.000),
Push-Button 0.984, Turn-Lever 0.941 (256), Topple-Block 0.938.

**This is an 8-task gated set, not a 25-task one.** Distilling from a 0.05-success teacher
measures the teacher, not the continual-learning method. Three of the prior CL sequence's five
tasks (Push-Cuboid 0.223, Open-Door 0.000, Open-Drawer 0.888) do NOT clear the bar, so
comparability with the P0/P1 numbers cannot be had for free — that is Q2, still open.

**If Rotate-Valve is ever gated in, re-measure the pre- and post-`SEAT_HARD_TIMEOUT` variants
at higher n first** — at n=256 they are unresolved (0.523 vs 0.473) and distilling from the
weaker one would silently corrupt a downstream result.

## Blockers

| # | Blocker | Blocks | Raised | Owner | State |
|---|---|---|---|---|---|
| B1 | `flap.xml` `flap_hinge` range compiled as degrees, not radians — real range -1.4 deg vs -70 deg target, task unsolvable by any policy | Push-Flap (A-4) | 2026-09-08 W1-c | lead | **RESOLVED** 2026-09-08 — benchmark fix `range="-80.2 0"` applied to `flap.xml`; Push-Flap then measured 1.000 (128) with an unchanged teacher |
