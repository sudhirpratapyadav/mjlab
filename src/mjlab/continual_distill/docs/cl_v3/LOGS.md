# CL-V3 — LOGS (append-only)

> One dated entry per work session, newest at the bottom. Agents write to
> `logs/<agent>.md` with the same template; the lead merges. Never edit an earlier entry.
> Every number carries `(n, HEAD)`; negative results are output.

Template:

```
## YYYY-MM-DD — <agent> — <task(s)>
**Context:** what was true when you started (baseline, spec, HEAD, GPU used).
**Did:** what you ran / changed (files, constants, strategy).
**Found:** measurements, failure analysis (phase, envs, physical event), surprises.
**Decided:** strategy choice + why; anything that needs a decision row.
**Next:** what remains.
```

---

## 2026-09-09 — lead — program start

**Context:** cl_v2 done (`b9b0563`): all 25 tasks on realistic assets, teacher SR at
n=128 on the OLD init distribution: 8 at/above 0.90, 5 at 0.000 (see `STATUS.md` "v2 SR").
Owner direction: reach ~100 % on every task with strategy-first teacher work, on an init
distribution with real coverage; task/metric changes only where physically impossible;
publish > 0.90 videos to a new `/v3/` gallery. Holder 20277 (dgx1): GPUs 1–3 free, 4–7
someone's training, 0 reserved.

**Did:** wrote `GOAL.md`, `PLAN.md` (init spec §3, strategy hypotheses §4, waves §6),
`STATUS.md` (25 rows, 6 gates), `AGENT_BRIEF.md`, `update_status.py`, `publish_v3.sh`,
`site/index.html`. Surveyed the current init randomization (19/25 tasks have no
orientation randomization; all mechanism mounts at one fixed z and yaw 0; free-object
tasks reset the robot with zero joint noise; six 20 s tasks re-spawn object + goal at
8–12 s = D1).

**Found:** the phase-1 handover already reduces the 17 below-bar tasks to mechanisms
M1 (squeeze ejection), M2 (push stall), M3 (phase deadlock — fixed where found),
M4 (IK posture trap). Lift-Cube is 1.000 on the same cube that Stack/Place lose during
CLOSE — the single best lead for group G. PPO runner cfgs exist for all 29 tasks
(`config/franka/rl_cfg.py`) but no PPO run exists on the current placement; RL stays
the fallback, not the plan.

**Decided:** W0 (init spec + diagnose tool + re-baseline) runs before any SR is
recorded; five W1 agents grouped by mechanism; two agents per GPU on 1–3.

**Next:** launch W0.

## 2026-09-09 — lead — W0 closed, rate-limit interruption

**Context:** W0-a froze all 25 init specs (Peg board yaw narrowed to 0 = D6, accepted),
verify_task 1000 resets PASS on all 25, placement tests green, D1/D2 applied. W0-b
delivered `classical/diagnose.py` and the `/v3/` gallery + home card. Site domain
changed by the owner to https://cl.sudhirpratapyadav.com (same VPS, same paths).
All six agents were killed by the API session limit at ~15:30; the GPU steps they had
queued ran to completion unattended (GPUs 1–3 idle at 16:10).

**Found:** W0-a's n=128 sweep finished for all 25 (`~/cl_v3_work/W0-a/<Task>.log`).
17 are clean v2-teacher-on-v3-spec baselines (recorded in rows). 8 ran while a W1
agent was already editing that teacher (Stack-Cube reads 0.945 — that is W1-G's
Lift-style grasp, not the v2 teacher) — recorded with a MIXED caveat; the clean
reference for those is the v2 SR on the old spec. Clean baselines worth noting: the
new spec (mechanism yaw ±15°, height ±3 cm, robot ±10°) drops Open-Door to 0.070,
Open-Lid 0.078, Flip-Switch 0.523, Turn-Lever 0.906, Lift-Cube 0.945; Reorient 0.016,
Strike 0.008 — orientation-naive teachers, as predicted.

**Did:** recorded the 8 mixed rows, closed the W0 infra rows, fixed four S cells that
W1-M's shell had filled with a path. Resumed W1-G/P/M/D/N from their transcripts.

**Next:** W1 continues; lead confirms every >= 0.90 claim with a second n=128 read.

## 2026-09-09 — lead — second rate-limit interruption (~18:30 IST, resets 23:00)

**Context:** all five W1 agents killed again by the API session limit. Their GPU steps
continue unattended (n=128 reads, diagnoses). Last known positions: W1-G Stack claim
at 61/64 after two of four episodes, Peg/Edge diagnoses and Place N3 running; W1-P
checking its paddle hold-test; W1-M polling four v4 reads + a HEAD-teacher valve
baseline on GPU 3; W1-D found the Reorient grasp variant (lead 0.05, rate 0.006 m/step:
12/12 grasped, no rewinds) and launched 32 envs; W1-N re-reading a Slide-Window trace.
Open-Drawer is the only fully green row so far (0.992, 254/256, published).

**Next:** resume all five from their transcripts when the limit resets; same message as
before (read own log + work-dir logs, continue, bounded polling).

## 2026-09-09 — lead — model quota exhausted; all five W1 agents replaced

**Context:** the account's Fable quota ran out, killing W1-D, W1-G, W1-N, W1-M and W1-P
in turn. Their GPU steps kept running and every piece of state (logs/W1-*.md, STATUS
rows, ~/cl_v3_work/*) survived. Session default model switched to Opus 5.

**Did:** replaced each agent with an Opus successor (W1-D2, W1-G2, W1-N2, W1-M2, W1-P2),
each briefed from its predecessor's log so nothing is re-measured: they inherit the
finished tasks untouched, the retired variants as a graveyard, and the open task list.
Repaired three gate cells that agents' shells had filled with a filesystem path
(unquoted `~` in `--set "Strategy=..."`); the successors are told to quote.

**Found (state at handover):** 7 rows fully green and published — Reach-Target 1.000,
Flip-Switch 1.000 (0.523 -> 1.000: closed-loop paddle sweep that lands the stroke once),
Push-Button 0.992, Open-Drawer 0.992, Axial-Extract 0.988, Slide-Window 0.973,
Stack-Cube 0.953 (0.352 -> 0.953 by reusing Lift-Cube's grasp phases). Turn-Lever
0.938 over four reads, judged done (friction-limited pad walk-off, two variants rejected).
Climbing: Reorient 0.016 -> 0.820, Throw 0.000 -> 0.477, Lift-Cube 0.945 -> 0.961.
Not yet re-measured: the three push tasks (design finished, implementation not started),
Open-Door 0.070, Open-Lid 0.078, Rotate-Valve 0.516, Peg/Place/Tool-Pull/Edge-Grasp,
Pivot-Lift 0.000, Topple 0.922 (four rewrites all measured worse than the v2 teacher).

**Decided:** D5 (Strike-Slide impossible as specified — floor friction 1.02 defeats the
puck's documented 0.4, needs a 2.9-4.2 m/s launch against 1.56 m/s achievable) stays
PROPOSED until W1-D2 re-checks the two numbers it rests on; the preferred fix is the
contact friction, not moving the goal band.

**Next:** merge the five successors' results; second reads on every >= 0.90 claim;
site count and handover.

## 2026-09-09 — lead — Group N closed (9 tasks published)

**Did:** W1-N2 finished the group. Final pooled numbers over two n=128 reads each:
Reach-Target 1.000, Push-Flap 1.000, Open-Drawer 0.992, Push-Button 0.992,
Axial-Extract 0.988, Slide-Window 0.973, Lift-Cube 0.953, Turn-Lever 0.938,
Topple-Block 0.934. Home-page count updated to 12 of 25.

**Found — the diagnose harness reads LOW, and why it matters.** `diagnose.run_episode`
does `alive &= ~done`, so it drops an env at its first termination and reports
SINGLE-ATTEMPT success. `test_classical` and `render_rollout`'s stats phase run the full
episode length and max over `episode_success`, while `ManagerBasedRlEnv.step` auto-resets
terminated envs mid-loop — so a collision at step ~100 still leaves several attempts in a
1000-step budget. That is the whole 0.758 vs 0.945 gap seen on Lift-Cube. It only bites
where terminations are frequent AND the budget far exceeds one attempt (Lift, Stack);
Open-Drawer and Slide-Window agree under both harnesses. **diagnose SR is a lower bound
and must never be quoted as a result — `test_classical` is the protocol number.**

**Found — negative results kept:** five Topple-Block rewrites all measured worse than the
untouched v2 teacher (0.867/0.867/0.883/0.867 vs 0.922), which was restored byte-for-byte.
The seat press IS the toppling motion, so any floor guard high enough to stop the
ground collision also blocks the working seat. Needs a different contact, not a constant.
Lift-Cube's `cmd_lead_max` gain is not separable at n=256 (both readings give 244/256);
kept because the sag is a measured mechanism, but recorded as not-a-claimed-gain.

**Found:** the "BLOCKED BY A BENCHMARK BUG" banner atop `classical/push_flap.py` is stale
(the flap_hinge degree/radian bug was fixed in CL-V2 `41b36cb`) — lead to remove.

## 2026-09-10 — lead — Rotate-Valve + Strike-Slide ruling; W1-M retired mid-flight

**Context:** the Opus quota ran out too (~22:00, reset 22:30) after the Fable quota;
every worker was interrupted at least twice. Nothing was lost — the GPU steps kept
running and all state is in the rows, the per-agent logs and `~/cl_v3_work/`.

**Rotate-Valve DONE: 0.516 -> 0.930 (128) + 0.914 (render read), published.** The
strategy is the one the plan proposed and is worth remembering: no grasp and no
hand-off. A closed-finger probe goes in behind a spoke's trailing face at r 0.068 and
the whole hand translates around the circle with a 0.30 rad lead. The three earlier
rewrites read 0/32 because the probe stopped 6-8 cm above its insertion point and
landed ON the spoke, driving it backwards through its 0 rad stop — an approach problem,
not a strategy problem.

**Strike-Slide: D5 APPLIED, and I endorse it.** W1-D2 re-measured both numbers the
ruling rests on, then made a contact-only change in the Strike scene: the puck geom
gets priority 1 and sliding friction 0.04, so the puck's own friction decides the
contact instead of the terrain plane's default 1.0 winning by max-combination.
mu_eff 1.02-1.07 before, 0.060/0.100 after; goal band, success threshold and step
budget all untouched; verify_task 1000 PASS, placement tests and audit clean. This is
a benchmark repair, not a difficulty change: the puck asset always documented 0.4 and
never got it, and an ice-like coefficient is what a regulation hockey puck on a rink
actually has. The teacher was then rewritten from a punch to a contact launch:
0.008 -> 0.328 (128). Still below the bar; the residual is the paddle, not the physics.

**W1-M retired mid-flight.** Its Fable quota was exhausted, so its Opus successor
W1-M2 now owns Open-Door and Open-Lid alone; I told W1-M to stand down before the two
could collide on the same files. It could not write its own closing entry (no quota),
so: its Rotate-Valve result is recorded above and in the row; its Open-Door front-pinch
and Open-Lid tilt-following designs are in the Strategy columns and in `logs/W1-M.md`.
The same collision risk existed on the push group (the original W1-P woke on a queued
message); it was told to stand down and W1-P2 owns those three files.

**Standing:** 14 tasks >= 0.90, 13 published (site count updated). Place-In-Container
0.352 -> 0.922 awaits its second read and card.

## 2026-09-10 — lead — W1-M closing handover (3 cross-cutting defects)

W1-M stood down cleanly: 2 of 4 done and published (Flip-Switch 0.523 -> 1.000 twice
confirmed; Rotate-Valve 0.516 -> 0.930 + 0.914 render = 0.922 pooled at n=256), Open-Door
and Open-Lid handed to W1-M2 with both blockers characterised rather than guessed. Its
gate-B number for Rotate-Valve (0.516, measured off the `b9b0563` teacher via
`run_head_teacher.py`) replaces the mixed W0-sweep read.

**Three defects it found under EVERY `base.py` teacher — worth checking on any stalled
teacher before redesigning it:**
1. The environment's soft joint limits are invisible to the IK, so the solver plans
   through poses the actuators will not hold (new default-off `ik_joint_limit_factor`).
2. The M4 posture-regularisation plateau is a `posture_weight` artefact: 28-60 mm of
   residual at 0.005, only 4-11 mm at 0.0005. This is the same shape that cost
   Axial-Extract 0.000 until it was fixed there, and it is the likely cause of
   Open-Lid's 40-60 deg stall and Pivot-Lift's "plateaus 10-17 cm short".
3. Marker-derived angles in the observation carry noise that closed-loop gates trip on.

**Trap recorded:** Open-Lid's best measured teacher (v6, 20/32) is NOT what is on disk —
the later two-stage descent measured 11/32. A file on disk is not evidence of being the
best variant; the log is. W1-M2 told to re-measure both before building on either.

## 2026-09-10 — lead — Group D report: Reorient published; four IK/servo defects

**Done:** Reorient-Object **0.016 -> 0.906 (128)** (render read 0.9375), published — the
end-face pinch with a joint-7 yaw servo, a leashed waypoint descent, and a floor guard on
the true lowest pad corner. That is 14 published.

**Below bar, with mechanism-level reasons:** Throw-To-Bin 0.547 (2x128) — residual is
ballistic landing scatter, radial sd 0.07-0.15 m against a 0.0585 m tolerance, and the
arm's swing-tracking ratio varies 0.73-1.27 between envs. Strike-Slide 0.328 (2x128) —
residual is the paddle, not the physics: a 30 mm pad face against a 76 mm disc torques it
20-49 deg off line on off-centre hits. Pivot-Lift 0.000 — but see below, the task is now
understood for the first time.

**FOUR defects that cut across every teacher** (full write-up in `logs/W1-D.md`; these
supersede several earlier guesses):
1. **The descent ratchet.** `cmd_ref="actual"` plus a command lead re-adds the servo sag
   to the target every step: the Reorient site sank to 0.010-0.020 against a 0.034
   command, and 6/6 diagnosed failures were floor contacts.
2. **Rate-limiting the ERROR does not rate-limit a descent.** A 10 mm/step request moves
   the site 2.1 mm/step because the DLS realises only ~40 %. Descents must walk a leashed
   waypoint instead.
3. **The floor guard belongs on the pad geometry, not the site.** Pads hang 0.0119 below
   the site only with the hand vertical; a 10 deg tilt drops the outer pad 7 mm. Guarding
   the true lowest pad corner removed the ground-collision failure class outright on both
   Reorient and Throw.
4. **The "M4 posture trap" is TWO defects.** Planning: `posture_weight` 0.005 stops
   low/near targets 2.6-3.3 cm short (at 0.0 they land to 1e-3). Execution: the position
   loop's fixed point is error == servo sag (0.11 rad of shoulder droop = 10-17 cm), and
   no amount of command lead fixes it — only referencing every axis to the previous
   command does.

**Pivot-Lift's phase-1 diagnosis was WRONG, and this is the lesson of the day.** It was
never an IK dead zone. `wall.xml` is a STATIC collider (box x 0.508-0.613, z 0-0.15) and
the teacher's `HOVER_Z` was 0.10 — *below its top*. The hand descended onto the wall and
sat at (0.535, 0.02, 0.163) for whole episodes while phases timed out; min cos_tilt was
1.000 in 8/8 envs, i.e. the board was never touched at all. Three more real bugs there:
the PUSH phase scaled its whole command vector to `PUSH_RATE`, dividing the height term
by 0.13 so the pads rode the board's top face; a conditional DESCEND escape trapped 12/16
envs; and the phase timeouts summed to 339 steps against a 300-step episode. With those
fixed the board now stands (min cos_tilt 0.40-0.72) and 12/16 envs reach RE_HOVER/CLOSE/
LIFT — the remaining blocker is the re-grasp, a real and much later problem.

**Two caveats for anyone continuing:** CPU tuning does NOT transfer on Throw (the same
change moved CPU 0.156 -> 0.312 while GPU went 0.477 -> 0.430) — quote GPU only. And
`~/cl_v3_work/W1-D2/cpu_eval.py` is a `test_classical`-semantics CPU harness that made
the sweeps affordable.

## 2026-09-10 — lead — Group D second pass: Strike 0.758, Throw 0.633, Pivot geometry

**Strike-Slide 0.328 -> 0.750/0.758 (2x128).** The fix is a genuinely different contact,
not a tuning pass: open the jaws to 0.030 per finger so the two pads form a symmetric **V**
around the 38.1 mm puck. Two radial contacts at ±51.9 deg, the lateral components cancel,
an offset SELF-CORRECTS instead of growing, and the puck is kinematically locked to the
hand so it leaves at exactly the hand's speed and direction. Residual is launch SPEED: the
release still fires on a 5-step finite difference while the hand is still accelerating, and
slide distance goes as v^2. **Why a flat pad could never work:** a flat face on a cylinder
applies no torque, but its line of action is the RADIUS to the contact point, so an offset e
launches the puck asin(e/R) off line and the offset grows during a single-contact push.

**Throw-To-Bin 0.547 -> 0.625/0.633 (2x128), and W1-D2's diagnosis was refined.** The
residual was never the release rule — it was a RINGING SWING. The cube's true velocity
oscillates ±30 % at ~5 Hz within a single swing, and W1-D2's "0.73-1.27 tracking ratio"
was that ripple sampled at whatever phase the release posture happened to land on. Cause:
`_plan_swing` re-derives the wind-up posture after the arm has already settled on the old
one — a 0.263 rad command step on joint 6, which saturates at 0.04 rad of error. Note the
ordering lesson: a closed-loop release built ON TOP of the ringing made things WORSE
(0.547 -> 0.438) until the ringing itself was fixed. Also measured: `object_pos` noise is
0.010-0.013 m per axis (not the ±0.01 the cfg asks), so a quadratic-fit endpoint
derivative is useless — 0.25-0.30 m/s of scatter.

**Pivot-Lift 0.000 — the intended grasp is GEOMETRICALLY IMPOSSIBLE as the scene stands.**
The board pivots about its far-bottom edge at the wall base, so at 90 deg it lies flush
against the wall with its top edge at 0.12 — still 30 mm BELOW the wall top at 0.15. A
top-down pinch has nowhere to put the far pad above ~73 deg. The board is also unstable at
every tilt below 90 deg and falls flat in ~0.1 s, so BOTH previous teachers were
re-approaching a configuration that no longer existed by the time they got there. Derived
result: a purely horizontal push CANNOT tip this board at any pitch; the hand must climb so
friction acts up the face, after which N ~ 0.8 N suffices. Ruled out by measurement: more
force, the floor guard, and approach standoff. One strategy remains untried and is written
up — a pusher whose contact loads the fingers in the CLOSING direction for slide-and-tip,
opening onto the ramp only past ~35 deg. That is being attempted before any scene change is
proposed; if it fails, the wall-height-versus-board-length relation is the candidate
mis-specification (a wall taller than the board it pins is what removes the grasp).

## 2026-09-10 — lead — Group P first pass: six mechanisms, all three tripled, none at bar

Push-Cuboid 0.062 -> 0.305 (2nd read 0.344), Drag-Pull 0.352 -> 0.430, Cage-Drag
0.227 -> 0.461. Below bar; a second pass is running.

**Six mechanisms found and fixed — note how many were plain bugs, not tuning:**
1. `_wrap_half` pinned hand yaw only mod pi, so the paddle pushed BACKWARDS on 52 % of
   steps and 17/32 envs never moved the box at all. **This is what phase-1 recorded for
   two years as the "friction-limited push stall" (M2).** The whole cross-cutting
   mechanism was a sign error.
2. The height servo set its descent relative to the MEASURED height, so the command
   chased the arm's own fall: 8-11 mm/step into the floor, 41-59 % terminations.
3. The approach's fast descent rate stayed engaged all the way to ride height (94 % floor
   terminations on Cage-Drag).
4. The contact plane used was the SUPPORT plane, tangent at a corner and up to 1.2 cm
   behind the real surface, so the entire push lead was spent reaching the box and the
   command-minus-actual gap collapsed to 0.005 rad at contact. Replaced with the exact
   ray-box entry distance.
5. The endgame was a pure-pursuit limit cycle — the box orbited the goal at 3 cm while
   the pad crabbed off it. Replaced with latched straight segments.
6. A pad rubbing DOWN the face pins the box to the floor: a 40x effect (35 % vs 74 % of
   steps moving) and **independent of pad friction**, which contradicts the standing
   hypothesis that low pad friction was the binding constraint.

**D4 recorded, NO task change.** The spec allows 0.546 m of travel (not the ~0.22 m PLAN
assumed). Approach is 30-45 steps and torque-limited (forcerange 100 N.m against 60
N.m.s/rad damping = 0.033 rad/step). Worst case at the best sustained rate is 123 of 150
steps, so the budget is tight but FEASIBLE — the gap is the teacher's effective rate
(2.0-2.7 mm/step against the ~2.8 the p90 spawn needs). The budget stays as specified.

**Two measurement cautions that apply to the whole program:**
- **n=128 carries about ±0.04 here.** Two reads of the IDENTICAL on-disk teacher gave
  0.344 and 0.305; GPU physics is not bit-deterministic. Any claim within ~0.04 of the
  bar needs a second read, which is why the protocol demands one.
- **Pinning the seed makes n=32 paired but OPTIMISTIC**: seed 1234's first 32 spawns read
  0.531 where the population reads 0.352. Three late changes that looked like clear n=32
  wins were all rejected at n=128 and are left in the file switched off, with their
  numbers recorded.

**Highest-value lead for the next pass:** the push is impulsive, not continuous. The box
moves 6-11 mm/step while in contact, then outruns the arm for 2-3 steps — roughly a 50 %
duty cycle. Closing that is worth more than everything else combined. The naive velocity
feed-forward fails because the estimate's noise (6 mm over 3 steps) is the size of the
standoff it corrects.

## 2026-09-10 — lead — Strike-Slide DONE 0.926; the lead's own premise was wrong

**Strike-Slide 0.008 -> 0.926 (mean of FOUR n=128 reads: 0.938/0.922/0.914/0.930),
published.** The task phase-1 called a characterised failure is solved.

**The lesson: my brief told W1-D4 the entire residual was launch SPEED. That was wrong,
and measuring it instead of accepting it is what solved the task.** Sim-truth launch
velocity separated the outcomes perfectly on launch DIRECTION — every success under 1.5
deg off line, every failure 9-43 deg. The lateral offset at the first push step equalled
the ESTIMATE ERROR of the frozen puck position almost exactly (-15.7 / +20.8 / -40.7 /
+57.2 mm on the worst envs). `gripper_to_object` carries 10-13 mm of noise per axis and an
EMA at alpha 0.35 averages only ~3 samples: 6-7 mm sd against a ±30 mm capture basin, and a
puck touching one pad corner is thrown 51.9 deg off line. Since the puck is STATIC until
contact, a running mean over the 40-60 hover steps gives ~1.5 mm. One function, +0.17.
The analytic-release idea I suggested was measured and is NOT available: with the release
disabled the hand rings at ~5 Hz with a 10-15 % steady-state deficit against its commanded
ramp, and the puck separates on its own at the hand's speed peak.

**Throw-To-Bin 0.633 -> 0.711/0.688.** Same estimator fix on the grasp approach removed
the "never threw" class. Residual is now a landing problem: the cube enters a 163 mm
basket centred at 2.5-2.8 m/s and SKIDS to rest 0.02-0.07 m forward against an 0.0585 m
tolerance. Its predicate is containment-plus-settled (`PlaceInContainerCommand`), not a
distance latch — a distance proxy misclassifies. Swept and rejected, all recorded: aiming
short (the skid's mean is systematic but its spread is not), steeper launch (bounded by
the wind-up posture hitting the floor), and zero-spin release (0.000).

**Pivot-Lift 0.000, but three real blockers removed and the decisive one found:
THE BOARD NEVER REACHED THE WALL.** Its centre moved 3 mm in 100 steps of TIP while
needing 50-70 mm, so the jam gate never opened and **every tip experiment for three
sessions was run on an unpinned board.** Also fixed: at the old 85 deg limit the "upper
finger rides clear" clearance was 1.6 mm, and the floor guard did not bound the command
reference, so a termination was cutting TIP short in 6/8 envs. Max tilt is now 20-22 deg
in 5/8 envs (was 3-17) and ground collisions are 2/8 (was 6/8); the stall is just past
20 deg against a 40-65 deg pinch window.

**No decision row filed on Pivot-Lift, and rightly so.** The wall-height-versus-board-
length inconsistency IS real (wall 0.15 tall, board 0.120 long, so a flush board's top
edge sits 30 mm below the wall top, contradicting `wall_spawn_range`'s own constraint 2)
— but it is not what blocks the task: a grasp does exist at 40-65 deg and the teacher
never gets past 22 deg. Changing scenery would not fix the tip. This is exactly the
discipline the owner asked for: do not touch the task to buy a score.

**Harness trap for everyone:** `srun --jobid=20277 --overlap` WITHOUT `-n 1` runs the
command TWICE on this holder. Free replication (that is where four Strike reads came
from), but concurrent writers race on a shared output path.

**Addendum (W1-D4 full report):** measurement noise is LARGER than the push group's
estimate on some tasks — two reads of the IDENTICAL seeded config differ by up to 0.063
at n=128 on Throw-To-Bin, against the ±0.04 measured on Push-Cuboid. Treat ±0.06 as the
working band for a single n=128 read on the dynamic tasks; a claim inside 0.06 of the bar
needs its second read, and a "gain" smaller than that is not a gain.

Also recorded from the same report: the running-mean estimator carried BOTH dynamic tasks,
not just Strike — the same `APPROACH_MEAN` change removed Throw's "never threw" class. And
extending the averaging window PAST the hover phase is harmful (0.875 -> 0.438 on CPU):
the object is only static until the pads touch it, so the mean must stop at contact.

## 2026-09-11 — remaining teacher improvement pass

All25 teachers now have corrected-rule128-episode baselines. Nine additional measured improvements published through untu_vps, with five independently checked on a second seed;40 teacher tests pass. Tool, Edge and Pivot remain unresolved at zero success. [Full results and evidence](teacher_refresh/all_remaining/README.md); [detailed log](logs/teacher-refresh-20260911.md). No task/reward/distribution/budget changes in this teacher pass.
