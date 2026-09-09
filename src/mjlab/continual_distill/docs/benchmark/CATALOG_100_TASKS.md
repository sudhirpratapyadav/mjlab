# 100-Task Catalog (distinct motion profiles)

## 1. Counting rule and honest count

**Rule.** A task counts once per distinct MOTION PROFILE — a distinct contact/kinematic strategy or phase structure. Object-geometry swaps collapse (lift-cube = lift-cylinder = lift-sphere = ONE). What counts as distinct: different contact strategy (drawer hook-pull vs window face-push), different phase structure (single twist vs multi-turn regrasp), different closure topology (hook vs pinch), or a different embodiment executing a genuinely different strategy (finger gaiting). Embodiment ports of the same strategy are variants: included in notes, never counted.

**Arithmetic.**

| bucket | count |
|---|---|
| Existing built profiles (A:16, B:3, C:4) | 23 |
| Verified new profiles (A:29, B:4, C:7, bimanual:3) | 43 |
| **Total distinct profiles** | **66** |

**100 is NOT reached. Gap: 34.** We do not pad with object swaps to close it. Highest-yield closers, in order:

1. **Rescue from the kill list:** only `inhand-slide-palm-fingertip-shift` is conditionally rescuable (+1) — it needs a slip-expressibility spike (grip-force modulation via joint-position actions on LEAP in MJWarp) plus a gaiting-forbidding predicate. `screwdriver-drive-screw` and `articulate-free-hinge-object` are refuted, not deferred.
2. **Under-mined Class B** (7 profiles today): DexJoCo (11 designs) + Bi-DexHands (~20 reward formulas) + TCDM object-use verbs. After profile-collapse, realistic yield ~8–12 — gated on the repo's own state-RL solvability spike.
3. **Under-mined Class C** (11 today): Adroit/DAPG core (~13 tasks, near-native MJCF), MyoSuite verbs, ShadowHand suite. Realistic yield ~5–8 after collapsing reorient variants into repose/rotate.
4. **Bimanual expansion** (3 today): once the 2-arm scene class exists, robosuite/Bi-DexHands two-effector catalogs yield ~4–6 more genuinely coordinated profiles (the plumbing is the cost; marginal tasks are cheap after it).
5. **New predicate families as seeds:** wipe-force-trace unlocks a force-control family; wire-loop-traverse unlocks a path-constraint-with-fail-latch family. Each plausibly seeds 2–4 distinct profiles.
6. **PartNet-Mobility import** yields mostly instance diversity (does not count), but novel joint *mechanisms* (telescoping, latch-then-open two-stage, double-hinge) could add a few real profiles.
7. A second adversarial mining round concentrated on B/C/bimanual/dynamic tasks is the honest path to ~100; a 2nd dexterous hand (Shadow/Allegro) adds embodiment variety but almost nothing to the distinct count under the rule.

**Large-N claims elsewhere (for related-work framing).** NVIDIA CHORD (arXiv 2607.00033, Jun 2026) reports 4,739 tasks (1,831 evaluated, 82.12% avg success) for bimanual Sharpa hands in Isaac Lab. There, a "task" = ONE human demonstration clip (hand keypoints + object 6-DoF trajectory from ARCTIC/TACO/HOT3D/OakInk2/DexYCB/GRAB/H2O + in-house video); success = tracking the reference object trajectory within 15 cm / 40°, one policy trained per clip with demo-derived contact-wrench reward. Under this catalog's counting rule those are instance/trajectory tasks (object swaps + trajectory swaps of a handful of verbs), collapsing to a few dozen motion profiles at most — cite it as the instance-scale regime, orthogonal to profile-scale. Its transferable idea is the TEACHER FACTORY: per-clip RL teachers auto-generated from demonstrations, no per-task reward engineering — the same role our scripted/RL teacher routes play, industrialized. Porting CHORD tasks here is L-cost per family (PhysX→MJWarp retune, mocap retargeting to LEAP, mixed upstream dataset licenses incl. non-commercial GRAB/ARCTIC), and each ported family counts as ~1 profile.

## 2. The catalog

Cost key: S = reuses existing base cfg + asset pattern; M = new asset or new command term; L = new asset AND new predicate machinery, or new embodiment/scene plumbing. Within class: [built] first, then cheapest-first.

**For `[built]` rows the REGISTRY is the source of truth, not this table.** The skill/fragility columns here were written before implementation; where building a task revised the judgement, the registered `TaskTaxonomy` wins and this table is corrected to match it. Reconciled 2026-09-08: T18 strike-to-slide 2→3, T19 cage-and-drag 1→2 and T25 throw-to-bin 2→3 (all three are one-shot — a mis-timed impulse, release or pinch is unrecoverable within the episode, which is what fragility 3 means), and T22 axial-extraction was retagged in the registry articulation→insertion to match the family this table always gave it.

### Class A — Franka arm + parallel gripper (T01–T45)

| # | task | motion profile | skill family | fragility | success predicate sketch | teacher route | cost | source |
|---|---|---|---|---|---|---|---|---|
| T01 | Reach [built] | EE to sampled 3D target, no contact | reach | 1 | EE within tol of target; latched | existing | — | repo |
| T02 | Push [built] | Planar servo-push of free object to goal (cuboid/disc = one) | planar_push | 2 | obj xy at goal; latched | existing | — | repo |
| T03 | Lift [built] | Grasp-and-raise free object (4 geometry IDs = one) | pick_place | 3 | obj z above threshold, grasped; latched | existing | — | repo |
| T04 | Stack [built] | Grasp + precise place + release on base object | pick_place | 3 | obj seated on base, released; latched | existing | — | repo |
| T05 | Peg-Insertion [built] | Grasp + tight alignment into ~3cm hole | insertion | 4 | peg depth in hole within xy tol; latched | existing | — | repo |
| T06 | Open-Door [built] | Whole-arm hinge arc (vertical axis, gravity off) | articulation | 2 | hinge angle past threshold; latched | existing | — | repo |
| T07 | Open-Drawer [built] | Prismatic pull via fingertip hook | articulation | 2 | slide joint past threshold; latched | existing | — | repo |
| T08 | Push-Button [built] | Normal-force compliant press | articulation | 1 | button joint depressed; latched | existing | — | repo |
| T09 | Turn-Lever [built] | Wrist rotation about approach axis | articulation | 2 | lever angle past threshold; latched | existing | — | repo |
| T10 | Rotate-Valve [built] | Multi-cycle rotation with regrasp (270°, 8s) | articulation | 4 | cumulative valve angle ≥ 270°; latched | existing | — | repo |
| T11 | Flip-Switch [built] | Ballistic commit past over-centre detent (gravity on) | articulation | 2 | bistable state flip; latched | existing | — | repo |
| T12 | Slide-Window [built] | Lateral face-push along robot lateral axis | articulation | 2 | slide joint past threshold; latched | existing | — | repo |
| T13 | Open-Lid [built] | Vertical arc against gravity (falls shut if released) | articulation | 2 | lid angle held past threshold; latched | existing | — | repo |
| T14 | Place-In-Container [built] | Top-down containment + release over rim | pick_place | 3 | inside footprint, below rim, settled, released; latched | existing | — | repo |
| T15 | Reorient-Object [built] | Stand lying cylinder upright (orientation goal, grasped) | pick_place | 3 | axis alignment within angular tol; latched | existing | — | repo |
| T16 | Tool-Pull [built] | Grasp stick, rake out-of-reach puck into near zone (12s) | tool_use | 4 | puck in near zone; latched | existing | — | repo |
| T17 | planar-drag-pull [built] | Grasp/hook far side of free object, sustained drag toward base (engagement inverts vs push) | planar_push | 2 | obj xy in near-zone goal; latched | scripted | S | Meta-World pull/push-back |
| T18 | strike-to-slide [built] | Single calibrated impulse; object slides ballistically to goal outside reach envelope | planar_push | 3 | obj at goal, at rest, goal outside reach radius; latched | rl-dense | S | FetchSlide; MS3 RollBall |
| T19 | cage-and-drag [built] | Straddle with open gripper (never closes), form-closure transport any direction | non_prehensile | 2 | obj in goal AND min aperture over episode > threshold; latched | scripted | S | caging literature (Rodriguez & Mason) |
| T20 | topple-tumble-to-face [built] | Poke above CoM to tip past tipping point; designated face lands down, no grasp | non_prehensile | 2 | dot(face_normal, −z) > 0.9, at rest; latched | scripted | S | Ruggiero et al. taxonomy |
| T21 | nonprehensile-hinge-arc-push [built] | No-grasp fingertip/palm contact drives revolute member along arc (faucet/dial/close-flap; ~15 sources collapsed) | articulation | 2 | hinge angle past threshold in commanded direction, no contact at latch; latched | scripted | S | MW faucet/dial/close family; RLBench close_* |
| T22 | axial-extraction [built] | Friction-breakaway force ramp + guided withdrawal along socket axis, base stays put | insertion | 3 | axial displacement ≥ free, obj clear, base displaced < eps, grasped; latched | scripted | S | MW peg-unplug/disassemble; AutoMate-Disassembly |
| T23 | slide-to-edge-overhang-grasp [built] | Drag thin flat object to table edge, pinch top+bottom at overhang (table as fixture) | pick_place | 3 | both-face contacts + lifted clear; floor contact latches FAIL | scripted (RL fallback) | S | edge-grasp literature (Chavan-Dafle) |
| T24 | pivot-against-wall-grasp [built] | Push ungraspable flat object INTO wall to pivot onto edge, then grasp exposed face | pick_place | 3 | both finger pads in contact AND height > thresh; latched | rl-dense | S | Zhou & Held CoRL'22 |
| T25 | throw-to-bin [built] | Accelerate + timed release; ballistic arc into bin outside reach envelope | pick_place | 3 | obj at rest in bin, bin outside reach radius; latched | rl-dense | S | TossingBot; dm_control throw |
| T26 | push-to-pose | Non-prehensile push of asymmetric T-block to position AND yaw via contact-face switching | planar_push | 2 | xy + yaw within tol, at rest; latched | rl-dense | M | ManiSkill3 PushT |
| T27 | lift-large-nonprehensile | Wedge/scoop box wider than aperture using palm+forearm (closure impossible by construction) | non_prehensile | 2 | box z > table+0.1, near goal, aperture < width; latched | rl-dense | M | dm_control lift_large_box |
| T28 | crank-handle-continuous | Continuous circular Cartesian path spins axle multi-rev via free-spinning knob, no regrasp | articulation | 2 | cumulative axle angle > 4π; latched | scripted | M | NIST-board crank |
| T29 | nut-assembly-ring-over-peg | Inverse-polarity insertion: align HOLE of offset-grasped annular part over fixed peg, seat | insertion | 3 | ring center within radial tol of peg axis, seated height; latched | scripted | M | MW assembly; robosuite NutAssembly |
| T30 | insert-and-twist-lock | Insert peg/key, then rotate about insertion axis to lock angle against detent | insertion | 3 | depth > d AND lock angle > θ; latched | scripted | M | FMB/NIST bayonet; BEHAVIOR lock |
| T31 | gear-mesh | Axial placement onto shaft + rotational micro-search until teeth mesh | insertion | 3 | axial depth ≥ d AND relative angle within tooth pitch; latched | scripted (Factory RL fallback) | M | Isaac Factory GearMesh; NIST |
| T32 | snap-fit | Align, advance, force-ramp past elastic detent which clicks back | insertion | 3 | depth ≥ d AND latch returned past detent, simultaneous; latched | rl-dense (scripted fallback) | M | novel (FORGE-adjacent) |
| T33 | drop-through-slot | Reorient thin disc edgewise to orientation-keyed aperture, align, release to free-fall through | insertion | 3 | obj below slot plane, inside receptacle; latched | scripted | M | novel (coin-in-slot) |
| T34 | shelf-place | Near-horizontal via-point approach under overhang, release in ceiling-bounded cavity, clean retreat | pick_place | 3 | static in shelf AABB, no gripper contact, gripper cleared; latched | scripted | M | MW shelf-place; CompoSuite |
| T35 | hang-on-hook | Align aperture over hook, lower to engage, release to passive hanging equilibrium | pick_place | 3 | hook through aperture, hook-only contact, open gripper, settled; latched | scripted | M | RLBench hang family; robosuite ToolHang |
| T36 | hammer-drive-nail | Grasp hammer, drive friction-loaded prismatic nail to depth (push-through → strikes) | tool_use | 2 | nail slide displacement > depth AND hammer grasped; latched | scripted (RL for impact) | M | MW hammer; Adroit hammer |
| T37 | stick-insert-pull | Tool-tip insertion into loop, then coupled two-body drag to goal | tool_use | 3 | target obj within goal radius; latched | rl-dense (staged) | M | Meta-World stick-pull |
| T38 | squeeze-trigger-within-grasp | Extra finger closure actuates sprung trigger inside a maintained off-table grasp | articulation | 3 | trigger angle > θ WHILE lifted; latched | scripted | M | novel (spray/drill); Bi-DexHands Scissors |
| T39 | tray-balance-transport | Transport grasped tray with free ROLLING cargo; tilt regulation keeps cargo aboard | non_prehensile | 3 | tray at goal pose, cargo on tray; cargo-table contact latches FAIL | rl-dense | M | waiter-task literature |
| T40 | put-item-in-drawer-chain | Ordered open→place-inside→close chain; three contact episodes, forced release+regrasp | articulation | 3 | staged latches in order: item inside THEN drawer closed | scripted | M | RLBench put_item_in_drawer; LIBERO |
| T41 | unscrew-jar-lid | Helical rotation+translation coupling through wrist-limit regrasp cycles, terminal detach+lift-clear | articulation | 3 | unscrew angle ≥ free AND lid clear of jar top; latched | scripted (Factory RL for screw-in) | L | RLBench open_jar; Isaac Factory NutThread |
| T42 | wire-loop-traverse | Carry ring along bent wire end-to-end, continuous wrist reorientation, zero contact allowed | insertion | 3 | loop in end zone AND wire contact count == 0; any contact latches FAIL | scripted | L | RLBench beat_the_buzz |
| T43 | scoop-with-spatula | Slide thin blade under ungraspable flat object at shallow wedge angle, lift balanced on blade | tool_use | 3 | blade-only contact, height ≥ h, CoM in blade polygon for T; latched | rl-dense (scripted warm start) | L | RLBench scoop_with_spatula |
| T44 | pour-beads | Level transport of cup with bulk bead cargo, controlled tilt over receiver, re-level | tool_use | 2 | ≥K of N beads in receiver, cup grasped; bead-on-table latches FAIL | scripted | L | RLBench pour; BEHAVIOR pour — MANDATE FLAG (multi-object sign-off) |
| T45 | wipe-force-trace | Sweep coverage path with normal force held in a band over K marked sites | tool_use | 2 | all K sites visited with force in [fmin,fmax]; per-site latch array | scripted (RL also easy) | L | robosuite Wipe; RLBench wipe_desk |

### Class B — Franka arm + LEAP hand (T46–T52)

| # | task | motion profile | skill family | fragility | success predicate sketch | teacher route | cost | source |
|---|---|---|---|---|---|---|---|---|
| T46 | Reach [built] | Arm+hand EE to 3D target, 23-D actions | reach | 1 | EE within tol; latched | existing | — | repo |
| T47 | Lift [built] | Dexterous grasp-and-lift on full arm | pick_place | 3 | obj raised, held; latched | existing | — | repo |
| T48 | Stack [built] | Dexterous stack on cuboid base | pick_place | 3 | obj seated on base, released; latched | existing | — | repo |
| T49 | hook-grasp-carry | Fingers through handle, curl to hook — no opposition; pendulum-load regulation in transport | pick_place | 2 | sustained finger-handle contact AND bucket at goal; latched | scripted | M | Agarwal arXiv:2312.02975; Feix taxonomy |
| T50 | prehensile-push-regrasp | Press held object against table so external contact slides it to target in-hand pose | in_hand | 3 | obj pose in hand frame within tol, grasp never lost; latched | scripted | M | Chavan-Dafle prehensile pushing |
| T51 | lift-reorient-in-air | Pick then 6-DoF in-air pose servo via coordinated wrist+finger adjustment under moving base | in_hand | 4 | pos+rot within tol off-table, held T; latched | rl-dense (DexPBT recipe) | M | Isaac Dexsuite; DexPBT — embodiment-variant FLAG |
| T52 | unscrew-cap-multiturn | Fingertip gaiting turns threaded cap multi-rev with per-turn regrasps; screw joint couples lift to spin | articulation | 4 | cap joint > N turns AND cap separated; latched | rl-dense | L | Twisting Lids; Bi-DexHands BottleCap |

### Class C — floating LEAP hand (T53–T63)

| # | task | motion profile | skill family | fragility | success predicate sketch | teacher route | cost | source |
|---|---|---|---|---|---|---|---|---|
| T53 | Reach [built] | Floating palm to 3D target, 22-D actions | reach | 1 | palm within tol; latched | existing | — | repo |
| T54 | Lift [built] | Floating-hand dexterous grasp-and-lift (cube/sphere = one) | pick_place | 3 | obj raised, held; latched | existing | — | repo |
| T55 | Stack [built] | Floating-hand stack on cuboid base | pick_place | 3 | obj seated, released; latched | existing | — | repo |
| T56 | Peg-Insertion [built] | Floating-hand peg into hole board | insertion | 4 | peg depth within tol; latched | existing | — | repo |
| T57 | palm-roll-object-on-table | Flat palm presses ball to table, rolls it to goal by translating under regulated normal force — no grasp | non_prehensile | 3 | obj at goal, at rest, never grasped; latched | rl-dense | S | rolling-under-palm literature |
| T58 | repose-cube-leap | Goal-conditioned SO(3) reorientation via finger gaiting, palm-up | in_hand | 4 | quat dist < tol sustained T, above drop plane; latched | rl-dense | M | Isaac Repose-Cube; OpenAI 1808.00177 |
| T59 | inhand-rotate-continuous-fingertip | Cyclic rotate-regrasp gait (limit cycle) about fixed hand axis, revolution count not goal pose | in_hand | 4 | accumulated rotation > 2π without drop; latched per rev | rl-dense (HORA; mjlab port exists) | M | Qi HORA; Msornerrrr mjlab port |
| T60 | extrinsic-tabletop-reorient | Fingertip press/tilt/pivot of on-table cube to goal pose, table as second contact, never grasped | non_prehensile | 3 | quat err < 0.4 rad, xy < 5cm, table contact; latched | rl-dense | M | ManiSkill3 TriFingerRotateCube |
| T61 | catch-tossed-object | Predict, intercept, time finger closure on randomized ballistic object | pick_place | 4 | secured (contacts + near-zero rel vel) above floor after flight; latched | rl-dense | M | Dynamic Handover 2309.05655 |
| T62 | key-turn-pinch-gaited | Fingertip-only rotary torque with thumb-index regrasps on small key, hand base static | articulation | 4 | key hinge > π, key still in lock; latched | rl-dense | M | MyoSuite Key Turn |
| T63 | finger-sequence-press | Individual fingers strike spring keys in prescribed spatial order — individuation + sequencing | articulation | 2 | key joints depressed in order; per-key latch, full sequence | scripted (RL proven) | M | RoboPianist — embodiment-variant FLAG vs push-button |

### Bimanual — 2× Franka + gripper, NEW CLASS (T64–T66)

PLAN gate applies: bimanual is a stretch class, in only if the 2-arm scene plumbing (16-D action) is built.

| # | task | motion profile | skill family | fragility | success predicate sketch | teacher route | cost | source |
|---|---|---|---|---|---|---|---|---|
| T64 | two-arm-lift | Simultaneous closed-chain lift of one large body; internal-force + tilt regulation | bimanual | 3 | height > h, tilt < θ, both contacts, at goal; latched | scripted (mirrored dual diff-IK) | L | robosuite TwoArmLift; MS3 TwoRobotPickCube |
| T65 | bimanual-handover | Grasp-exchange: A presents, dual-contact phase, A releases, B places beyond A's reach | bimanual | 3 | B contact, A absent, never touched table since lift, goal outside A reach; latched | scripted (phased handshake) | L | robosuite TwoArmHandover; Aloha |
| T66 | two-arm-peg-in-hole | Relative-frame insertion: both arms servo peg-to-board pose, neither target world-fixed | bimanual | 3 | in board frame: depth ≥ d, radial < tol; latched | scripted (attached-object start) | L | robosuite TwoArmPegInHole; Aloha |

## 3. Phased build plan

Recipe-B cost basis: each profile is ~0.5–1 day authoring + a validation training run; every task ships with the success-predicate unit test (drive mechanism to success, assert latch, near-miss negative control) that caught Flip-Switch and Tool-Pull.

**Wave 1 — S items (10 tasks: T17–T25, T57).** Cumulative distinct total: **33**.
**STATUS (2026-08-04): the 9 Class-A items T17–T25 are BUILT and registered** (Mjlab-{Drag-Pull,Strike-Slide,Cage-Drag,Topple-Block,Push-Flap,Axial-Extract,Edge-Grasp,Pivot-Lift,Throw-To-Bin}-Franka; Class A 16 -> 25 profiles, registry 28 -> 37 IDs). Gates passed: CPU success-predicate tests with negative controls (tests/test_class_a_wave1.py, 13 tests), reset+step finite-obs sanity, config/taxonomy suites (30 tests). Outstanding: cluster GPU benchmark-smoke --isolate, benchmark_validate learnability runs, and teachers. T57 (palm-roll, Class C) not yet built. See CLASS_A_WAVE1.md.
New machinery: episode-long constraint latches (min-aperture / never-closed / base-stationarity / floor-contact-FAIL), out-of-reach-goal check, face-down orientation predicate, static wall/edge scene props. No new assets beyond trivial props; heavy reuse of push/insertion/lift bases. Teachers: 6 scripted, 4 RL-dense.
Unblocks: Class A grows 16→25 profiles; 30-task continual-learning sequences become possible with Wave 1 alone; first non_prehensile family members and first dynamic (release-timing) task enter the suite.

**Wave 2 — M items (24 tasks: T26–T40, T49–T51, T58–T63).** Cumulative: **57**.
New machinery: new free/articulated assets (T-block, oversized box, crank, ring+peg, detent lock, gears, sprung latch, slot, shelf, hook, hammer+nail, stick+loop, trigger body, tray, key row, lock+key, handled bucket); SO(3)/6-DoF goal-pose command term (shared by T51/T58/T60, fills the empty IN_HAND family); staged ordered-latch predicate (T40, T63); ballistic-launch command (T61).
Unblocks: 50-task CL sequences; family-balanced and fragility-graded orderings; the entire B/C dexterous CL program (gated on the B-class state-RL validation spike); long-episode protocol handling (T40 joins valve-8s/tool-pull-12s class).

**Wave 3 — L items + new class (9 tasks: T41–T45, T52, T64–T66).** Cumulative: **66**.
New machinery: helical joint (hinge+slide equality coupling) with mid-episode detach validated against CUDA-graph capture (test like test_sm80_graph_capture); contact-force + per-site coverage predicate (the deferred wipe machinery); zero-contact fail-latch with generous margins at 4096 envs (T42); K-of-N bulk-cargo counting + settle init (T44, needs multi-object mandate sign-off); thin-blade collision tuning (T43); bimanual scene class — 2-robot scene, 16-D action, dual diff-IK (T64–T66).
Unblocks: full 66-profile suite; force-control and path-constraint predicate families (seed tasks for future mining); cross-class and cross-embodiment transfer studies; hardest fragility-4 sequences.

## 4. Teacher strategy at scale

**Classical-scripted (diff-IK waypoints).** Covers most of Wave 1 (T17, T19–T23), the Wave-2 assembly/placement/chain block (T28–T31, T33–T36, T38, T40, T49–T50, T63), Wave-3 scripted items (T41, T42, T44, T45), and all three bimanual tasks (T64–T66, mirrored/phased dual scripts). Viable wherever the phase structure is waypointable and tolerances sit above the measured precision floor. **Known failure mode — endgame precision:** measured scripted ceilings are peg-insertion 0.03–0.06 at 3mm clearance and rotate-valve 0.03–0.08, with 2–3.6cm DLS lateral bias and ~1cm obs noise. At-risk tasks: T29–T33 (all sub-cm alignment), T42 (clearance margin), T66 (relative-frame tolerance). Mitigation: author with forgiving MW/robosuite-scale clearances first; Factory-style RL-dense keypoint reward is the proven fallback for T31/T41.

**RL-dense (PPO).** Required where impulse calibration, contact-mode switching, or finger gaiting defeats waypoints: T18, T24–T27, T32, T37, T39, T43, T51–T52, T57–T62. **Known failure modes — throughput and verification:** each RL teacher costs a full training run (smoke cannot verify reward correctness for a new skill — "author deliberately, not speculatively in bulk"), so RL-taught tasks are rate-limited by GPU budget, not authoring time; batch them behind benchmark_validate short-PPO gates. B-class state-only RL solvability is UNPROVEN in this repo — run the validation spike before committing Wave-2 B budget. C-class recipes (repose/HORA) are literature-proven and a community mjlab LEAP port exists for T59.

**Hybrid (scripted warm start → RL).** T32 (compliant push fallback), T37 (staged dense reward), T43 (shallow-approach warm start), T61 (planar sliding-trap precursor first).

**Cross-cutting feasibility flags.** Joint-position action rate limits may cap EE tip speed — verify before building T25 and the ballistic variant of T36. CUDA-graph capture forbids mid-episode model mutation — T41/T52 detach events must pass a graph-capture test before the assets are trusted. Contact-heavy predicates at 4096 envs need margined geometry (T42 false positives, T33 wedging, T43 tunneling).

## 5. Rejected candidates (appendix)

| name | reason killed | rescuable? |
|---|---|---|
| screwdriver-drive-screw | Real tip-in-slot torque needs ~mm features, below the stack's measured precision floor; the equality-coupling escape hatch deletes cam-out/tip-alignment, degenerating to unscrew-jar-lid's motion at a tool offset. Infeasible as specified, non-distinct as buildable; both inherited scripted lineages are measured failures. | No |
| articulate-free-hinge-object | Dilemma: realistic-friction regime is a duplicate of nonprehensile-hinge-arc-push (free base = dynamics variant); low-friction active-bracing regime has no executable scripted teacher and no sketched dense reward for the brace/nudge ratchet. Feasible regime is a dup; distinct regime is unteachable. | No |
| inhand-slide-palm-fingertip-shift | No large-scale sim-RL demonstration; slip requires grip-force modulation whose expressibility via joint-position actions on LEAP/MJWarp is unverified; predicate cannot forbid gaiting, so PPO would solve it as repose-gaiting, voiding the distinctness defence. | Conditionally: run the slip-expressibility spike + design a gait-forbidding predicate; if both pass, +1 profile |
