# CL25 RL readiness audit — 2026-09-11

**Current scope: 24 active tasks; Tool-Pull is deferred.** See [active_tasks.json](../active_tasks.json). This 25-task audit remains a historical record.

**Update 2026-09-12:** observations now use the user-requested [working 60D layout](../shared_60/README.md), with normalized 8D actions. The provisional 143D expansion was superseded. Reward/physics findings remain open; omitted dynamics/tool/history information remains a known limitation of the selected compact layout.

**All 25 registered environments passed the short numerical checks. Full-scale PPO training is not ready to launch unchanged:** action scaling, saturated approach rewards, incomplete success alignment, and observation/physics choices need attention first. These are readiness findings, not measured PPO learning failures.

## Scope and evidence

- Reviewed effective registered training configurations, reward/success implementations, observation functions, action processing, and PPO defaults.
- Ran all 25 environments on dgx1 GPU2 through allocation 20277: four environments each, reset and 12 standard-normal action steps, **1,200 environment transitions**. All observations/rewards were finite and dimensions matched. Control period was 0.02 s for every task.
- Probed another 40 articulation resets (10 tasks × four environments) to measure manipulation-site distance and the exact configured approach reward.
- No PPO optimization, convergence evaluation, or new rendered rollout was performed in this audit. Production task/teacher code was not changed. Earlier teacher success measurements are not evidence of PPO learnability.
- Raw evidence: [audit.json](audit.json), [reach_probe.json](reach_probe.json), [effective_physics.json](effective_physics.json). Reproduction scripts: [audit.py](audit.py), [reach_probe.py](reach_probe.py). `mean_weighted_reward_rates` are weighted terms **before multiplication by the 0.02 s control period**; the recorded total reward range is per step.

## Shared action contract

Every task uses eight absolute position targets: seven arm joints and one coupled finger actuator. The effective mapping is `target = default_position + 0.04 * action`. This is **not incremental joint control**, despite the delta-control comment beside `FRANKA_ACTION_SCALE`.

With PPO's initial standard deviation of 1, arm exploration has only 0.04 rad (2.3 degrees) standard deviation around the default pose. Initial teacher commands already require maximum absolute arm actions between roughly 3.7 and 11.1 across the tasks. The interface is internally consistent but poorly conditioned for learning large reaches from scratch. Neither the action term nor the PPO wrapper currently clips raw actions. Adding a conventional raw clip of ±1 would restrict arm targets to ±0.04 rad and make the problem worse.

The gripper mixes meters with arm radians under the same scale: its default is 0.04 m, so raw -1 closes it, raw 0 is already fully open, and raw +1 requests 0.08 m beyond the actuator's 0.04 m control maximum. Positive commands therefore waste exploration in saturation.

Before training, introduce a versioned RL action interface with separate arm/gripper mappings and deliberate reachable bounds, or explicitly designed incremental control. Keep existing teacher/BC action labels compatible through an adapter. For Cage-Drag, a fixed-open/masked gripper is a strong candidate because closing it irreversibly invalidates the episode.

Source: `src/mjlab/envs/mdp/actions/joint_actions.py`, `src/mjlab/asset_zoo/robots/franka_emika_panda/franka_constants.py` and Panda actuator XML.

## Observations

Actual actor and critic dimensions are **60 for 21 tasks, 51 for Stack/Peg, 38 for Reach, and 69 for Tool**. These are valid for separate PPO policies; do not assume a universal 60D schema. Peg has a stale 60D comment. Actor corruption is enabled; critic corruption is disabled. Both use running observation normalization and feedforward networks.

Shared issues:

- Object/gripper positions, and Stack/Peg base positions, use global scene coordinates. Relative vectors are correct, but global positions expose the vectorized environment grid. Use environment-local or robot-base coordinates before scaling to many environments.
- Object/tool linear and angular velocities and mechanism joint velocities are absent. Single-frame feedforward policies must infer dynamic phases without them. Add relevant velocities and contact information, or observation history/recurrent state, especially for Throw, Strike, Reorient and settling/insertion.
- History-dependent validity/progress such as Cage's minimum aperture, Tool's direct-contact/tool-use history, and Pivot's pivot history is not observed. The critic also lacks these flags. Supply relevant state to the critic and an appropriate actor memory/state interface, or terminate irrecoverably invalid episodes explicitly.
- Peg's position noise can reach ±1 cm while seating tolerance is 3 mm and bore clearance is only a few millimeters. Use precision-appropriate channels or temporal filtering/history. Board yaw is currently fixed, which is consistent with the missing board-orientation channel; add that channel before randomizing board yaw.

Source: `src/mjlab/tasks/manipulation/mdp/observations.py`, individual environment observation groups, and the effective configurations in `audit.json`.

## Reward findings

### Articulation approach signal is saturated

The ten articulation tasks use `1 - tanh(30 * distance**4)` for approach. Measured initial distances were **0.689–0.939 m**. The reset reward was exactly zero for most samples, with the remaining Door/Drawer samples only about 3e-7–3e-6. During the random-action smoke, all ten tasks had zero mean approach reward. Legacy `reaching_max_dist`/`bringing_max_dist` keyword arguments are accepted but ignored by the replacement articulation reward.

The precise manipulation term correctly uses actual joint displacement/angle, including the valve's non-periodic travel. Keep this. Replace the approach term with a distance scale that provides useful signal across actual reset poses. Nine of these tasks also pay a +0.25 non-collision reward rate while stationary; reconsider this constant bonus when early progress reward is effectively absent. Door has that term and all regularizers weighted zero.

### Dense goal proximity does not imply success

`object_at_goal_reward` and `staged_manipulation_reward` do not generally enforce release, low speed, support contact, or containment. Stack, Peg and Place can receive high goal reward while still held, even though their success predicates reject that state. The approach term also encourages keeping the hand near the object after placement. Retain useful dense approach/transport shaping, but add completion stages and an exact-success bonus based on the existing strict predicate.

Peg's goal reference is now consistent: inserted root height is 0.050 m, board root height 0.015 m, hence a 0.035 m root offset; the command converts this into the peg tracking-site convention for rewards/observations. The remaining problem is that the shared approach reward targets the **bottom-tip tracking site**, near the ground, instead of a safe grasp point higher on the shaft. Separate grasp approach from insertion-tip alignment, and reward alignment, insertion depth, release and settled seating. Success already checks uprightness and bore fit; the generic goal bonus does not reproduce all those checks.

Throw inherits Place's goal reward and containment success predicate. There is **no separate flight-history predicate** in its command: the remote bin geometry motivates throwing. Current shaping does not explicitly teach launch/release/flight, and the approach term favors staying close to the object. Add dynamic observations and phase-appropriate shaping; if an actual airborne throw is a benchmark requirement, encode and test it explicitly.

### Special constraints are meaningful but hard to discover

- Cage reward correctly rejects pinch history and requires enclosure for transport. Success additionally requires accumulated caged transport. Random gripper exploration often invalidates the whole episode before the arm approaches. Preserve the no-pinch semantics while improving action masking, progress shaping and invalid-episode handling.
- Tool reward correctly requires tool-mediated transport and rejects direct robot/puck contact. Its held-tool bonus can reward holding without useful hooking/pulling. Add tool-tip engagement/progress signals and observable validity state.
- Edge and Pivot require genuine grasp; Pivot also requires prior wall-assisted tilt. These gates are appropriate but need useful shaping for the preceding edge exposure/pivot/contact sequence.
- Reorient/Topple use the correct body-axis objective and drift constraint. Their orientation reward does not fully encode the settled-state success requirement.

Source: `src/mjlab/tasks/manipulation/mdp/rewards.py`, `commands.py`, and `task_geometry.py`.

## Per-task assessment

Every row also inherits the shared action/observation findings. “Pilot” means a candidate after shared preparation, not a certified converged policy.

| Task | Obs | Episode (s) | Task-specific assessment before RL |
|---|---:|---:|---|
| Reach-Target | 38 | 20 | Best first pilot; direct reach objective is coherent. Fix action reachability/exploration and local coordinates. |
| Lift-Cube | 60 | 20 | Next pilot; goal reward is grasp-gated. Balance weak early approach against regularization; expose grasp/motion state. |
| Push-Cuboid | 60 | 3 | Position shaping is useful; check goal settling/ground completion incentives and adequacy of the short horizon in a pilot. |
| Drag-Pull | 60 | 3 | Add approach/contact/drag progress if the position objective stalls; expose object velocity and tune horizon. |
| Strike-Slide | 60 | 4 | Needs dynamic observations and impact-to-slide shaping; review velocity penalty against the required strike speed. |
| Stack-Cube | 51 | 20 | Add release, support-contact and settled-success reward; high proximity reward alone permits held-at-goal behavior. |
| Peg-Insertion | 51 | 20 | Goal offset is consistent; fix grasp-point shaping, precision observations, insertion/release/settle stages. |
| Place-In-Container | 60 | 20 | Align final reward with actual released, settled containment and container contact. |
| Throw-To-Bin | 60 | 5 | Place-style reward lacks launch/release/flight shaping; add velocities and review speed penalty. Flight history is not separately enforced. |
| Cage-Drag | 60 | 4 | Keep no-pinch/enclosure checks; mask open gripper or handle invalid episodes, expose validity and reward caged progress. |
| Tool-Pull | 69 | 12 | Add hook engagement and tool-mediated progress; avoid holding-tool reward plateau; expose history/dynamics. |
| Edge-Grasp | 60 | 6 | Shape edge exposure/contact/capture before the grasp-gated lift objective. |
| Pivot-Lift | 60 | 6 | Shape wall approach, pivot and capture; expose pivot state. |
| Reorient-Object | 60 | 20 | Axis objective is correct; add angular velocity and settled completion reward. |
| Topple-Block | 60 | 4 | Axis objective is correct; add controlled toppling/settling signals and check approach saturation. |
| Open-Door | 60 | 3 | Fix saturated approach; deliberate decision needed on zero gravity and disabled regularizers. |
| Open-Drawer | 60 | 3 | Fix saturated approach and idle bonus; freeze intended gravity setting (currently zero). |
| Push-Button | 60 | 3 | Fix saturated approach and idle bonus; currently zero gravity. |
| Push-Flap | 60 | 3 | Fix saturated approach and idle bonus; currently zero gravity. |
| Turn-Lever | 60 | 3 | Fix saturated approach and idle bonus; currently zero gravity. |
| Rotate-Valve | 60 | 8 | Keep joint-space travel reward; fix approach and add useful velocity/progress state for regrasping; zero gravity. |
| Slide-Window | 60 | 3 | Fix saturated approach and idle bonus; currently zero gravity. |
| Axial-Extract | 60 | 4 | Fix saturated approach and idle bonus; currently zero gravity. |
| Flip-Switch | 60 | 3 | Fix saturated approach and idle bonus; gravity is enabled. |
| Open-Lid | 60 | 5 | Fix saturated approach and idle bonus; expose hinge motion; gravity is enabled. |

## Physics and PPO setup

Effective gravity is zero for **Door, Drawer, Button, Flap, Lever, Valve, Window and Axial**. Lid, Flip and all free-object tasks have -9.81 m/s² gravity. Some zero-gravity source comments call this debugging; others describe inherited behavior. This is not a numerical failure, but the intended physics must be decided and frozen before expensive training. Enabling gravity would change the benchmark and require renewed teacher/geometry evaluation.

The common PPO defaults are feedforward 512/256/128 networks, normalized actor/critic observations, initial action noise 1, gamma 0.99, lambda 0.95, and 24 steps per rollout. All registered scene defaults are **one environment**, so parallel training needs an explicit environment-count override. At 50 Hz, gamma 0.99 has an effective discount time of about two seconds; a reward 20 seconds away is discounted by roughly 4e-5. Long sequential tasks warrant a deliberate discount/horizon choice after shaping is fixed.

For 21 tasks, the joint-velocity penalty rises from -0.01 to -0.1 at environment step 24,000 and -1 at 36,000 (about iterations 1,000 and 1,500 with 24-step rollouts). Reach, Stack and Peg have no such curriculum; Door remains zero. The velocity threshold is 0.5 rad/s. Tune this against each task's required motion, especially Throw/Strike. Early random action-rate penalties also exceeded the small approach rewards on examples such as Lift/Stack/Peg. These observations justify calibration; they do not prove that the current PPO hyperparameters cannot learn.

## Recommended next phase

1. Version the RL action/observation interface and freeze intended physics. Preserve the existing teacher/BC contract with explicit conversion.
2. Repair approach saturation, precision grasp targeting, and final success incentives; add task-specific state/phase shaping where listed.
3. Run small Reach/Lift PPO pilots, then mechanism and contact-sequence pilots. Log reward components, action/actuator saturation, contacts, strict first-episode success, and learning curves; inspect both successful and high-reward failed rollouts.
4. Scale to all 25 only after pilot policies improve strict success across held-out seeds without exploiting the shaped reward.
