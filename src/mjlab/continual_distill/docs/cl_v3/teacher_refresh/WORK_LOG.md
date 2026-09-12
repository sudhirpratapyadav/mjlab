# Teacher improvement pass — 2026-09-11

Authorized: improve teachers and publish measured improvements to cl.sudhirpratapyadav.com via untu_vps. Task definitions, physics, reset distributions and budgets remain at the September 11 audited revision. CPU simulation in existing holder 20277; no holder cancellation or GPU allocation changes.

## Reorient-Object hypothesis (before editing)

The saved diagnostic rollout reaches 0.0344 rad axis error and low object speed, but fails because its XY drift is 0.181365 m, just beyond the 0.18 m limit: initial object XY (0.44447,-0.10592), final (0.27276,-0.04753). The teacher's lift commands a fresh relative +0.18 m every step with zero lateral correction; the arm drifts inward under the posture objective. Rotation then freezes the already displaced site. Strategy: latch a fixed lift waypoint at the actual grasp, anchor XY during lifting and rotation, and rotate only after reaching clearance. This preserves the task's in-place requirement rather than changing its tolerance. Existing grasp failure modes remain to be measured. Matched-seed baseline and candidate runs use first episodes with training observation corruption enabled.

## Cage-Drag hypothesis (before editing)

The prior native rollout reaches 23.8 mm goal error and has 24.5 mm of caged progress, but the minimum physical aperture falls to 23.1 mm despite permanently open actions. Its final rendered hand is tilted away from the cube. The shared pushing teacher descends during lateral approach, accepts a timed-out descent as a completed cage, and changes face/yaw during transport. These contacts can load the outsides of the open pads and close the compliant fingers. Candidate strategy: align the full hand frame above the cube, descend centered before transport, keep the closing axis along a fixed travel direction, and carry the open cage along a bounded path at a fixed height. No changes to aperture threshold or reward. Risks: narrower capture margin for diagonal cubes, insufficient transport time, and unobserved contact loss.

Reorient first candidate is weaker during the matched 32-trial run (20 successes by step 400 versus baseline 25). Fixing the lift changes its clearance and reachable rotation poses. A second, narrower strategy preserves the original successful grasp/lift/rotation path and adds an observation-driven recentering motion only in SETTLE. `object_to_goal` already points back to the spawn anchor. If its XY length exceeds 12 cm (6 cm inside the task boundary), move the held upright bottle back gently, at most 3 mm per control waypoint. This specifically repairs upright-but-displaced outcomes without perturbing successful acquisition.

## Peg-Insertion estimator finding (before editing)

The inherited `gto_estimator="mean"` claims to average a held object's position in the hand frame, but actually averages the world-axis `gripper_to_object` vector. That vector rotates when the hand rotates; its old samples become geometrically inconsistent during peg alignment. It also continues treating the peg as rigidly held after RELEASE, when it should fall independently. Candidate: keep the pre-grasp static-object mean, transform held tip observations into the live hand frame before averaging, and return to a live filtered observation after release. A bounded averaging window lets it follow grip slip rather than retaining a stale estimate indefinitely. Restrict this change to the peg teacher until measured; do not change other transport teachers implicitly.

Cage centered-entry candidate: 5/32. Failure classification: 15 ground/failure terminations (3 ALIGN, 1 LOWER, 11 TRANSPORT); 12 transport timeouts. Most retained the aperture; seven still violated it. Three collisions occurred during the low traverse before entry. Next candidate performs the orientation at 18 cm rather than 10 cm, and transports at a 40 mm site height. At the latter the pad's bottom is ~28 mm, leaving 17 mm of overlap with the 45.2 mm cube, while adding 7 mm of ground-clearance margin. This addresses the observed collisions without changing dynamics or task thresholds.

Reorient baseline completed: 26/32. Of six failures, two terminate in CLOSE/DESCEND; four time out. Three timeout cases remain in SETTLE with a lying bottle and gripper/object distances 0.18–0.27 m. The teacher only checks grasp retention at the end of LIFT; a drop during ROTATE becomes a permanent empty-hand hold. Add the same retention check during ROTATE/SETTLE, recovering with a safe open-hand lift then the existing acquisition sequence. These are retries within the original episode, never automatic-reset retries. The recenter-only candidate is retained separately to distinguish the hypotheses.

Fixed-lift Reorient candidate completed at 24/32 versus baseline 26/32: rejected; saved as `reorient_object.fixed_lift.py`. The active candidate preserves the original lift and adds recovery/recentering. Recovery uses distance >8 cm alone; aperture can shrink during a retained barrel contact and is not decisive evidence of a drop.

Confirmation runs use GPU 3 (UUID GPU-fd08a1c1-caff-efc7-e4e6-e1117772f9fc), assigned to W2 confirmation in PLAN §6; verified both processes on that UUID. The slow incomplete CPU n128 baseline was explicitly stopped (only this pass's process, not the holder) once the GPU replacement ran. Its incomplete folder is not a reported measurement. Slurm overlap initially bound every step to the same two physical CPU cores; the two GPU workers were pinned to separate remaining cores inside this holder's allocated eight logical CPUs, avoiding the CPU diagnostic workers.

Cage baseline completed at 8/32; centered-entry first candidate 5/32 is rejected. The raised-clearance candidate is still pending. The GPU Reorient candidate completed at 97/128; matched GPU baseline is pending. CPU/GPU rates are not compared. OMP=4 oversubscribed tiny IK linear solves and severely slowed the baseline; confirmation now uses OMP=1. Incomplete CPU Peg and Reorient ablation runs were stopped and are not reported as rates. Peg is being rerun as a matched GPU pair.

Raised-clearance Cage candidate also completed at 5/32 versus baseline 8/32: rejected. Restored the original Cage teacher; neither experimental rewrite will be published. Both candidates and all completed traces remain under this review directory. Cage still needs a better transport strategy; no competence claim is made.

Matched GPU confirmations: Reorient original 97/128, recovery/recenter 97/128 (5 gained and 5 lost); Peg original 8/32, hand-frame estimator 8/32 with three extra terminations. Neither is a measured improvement. Both behavioral changes were removed from the active teachers and archived. No site update has been sent.

## Reorient acquisition control (next hypothesis)

Native terminal contacts identify finger-pad/terrain collisions, especially CLOSE (17/28 candidate terminations). Just before collision, true lowest-pad heights are 0.3–1.9 mm and the joints are still moving. The controller continues replanning noisy arm/yaw targets during the squeeze, even though the object should remain stationary between the pads. Next strategy: once CLOSE starts squeezing, hold one fixed arm command until the squeeze ends; only the fingers move. This separates acquisition from transport and removes observation-driven arm motion while closing. Preserve the original lift/rotation to isolate the change.

## Peg release failure — measured causal motion

The baseline's peg is often nearly aligned when RELEASE starts, then gets knocked over. Native FK from its actual qpos shows the hand sinking while it opens: env 0 site z 0.101 → 0.040 m over 16 steps, peg tilt 6.5° → 111.7°; env 1 site 0.110 → 0.067 m, tilt 5.1° → 21.6°; env 18 site 0.106 → 0.071 m while a briefly near-seated peg goes from 3.6° to 14°. RELEASE returns zero positional error referenced to the *actual* arm each step, repeatedly accepting gravity sag. This makes the gripper descend onto the peg after dropping it. Strategy: latch a fixed release-site position, hold that absolute pose with command-referenced feedback until the existing opening interval ends, then retreat. Keep the original estimator, approach and insertion strategy to isolate this mechanism.

## Reorient CLOSE→LIFT bug

A direct state-machine probe confirms that the action on successful completion of CLOSE is **OPEN (+1)**. `_goto(P_LIFT)` clears `_closing`, then the same call derives the gripper action from the cleared flag. This opens a just-acquired grasp for one control interval before lifting. Fix: return CLOSED explicitly on that transition. The quiet-close experiment is kept separate; start again from the original controller to measure this discrete correctness fix alone.

Quiet-close Reorient 94/128 versus 97/128: rejected. Peg release-hold 6/32 versus 8/32: rejected. Holding avoids the downward arm drift, but does not fix the peg's insertion orientation.

## Peg orientation feedback

At release, native FK shows the hand consistently tilted 4.5–4.9° despite its nominal vertical target, while the peg is typically tilted 5–12° (sometimes much more after slipping). The corrected straight bore requires <5°. The controller only sets a nominal wrist orientation, assuming a perfectly aligned rigid grasp. Next strategy: use the observed peg quaternion and robot FK to command the *peg* upright and square to the board, compensating the measured hand-to-peg orientation. Smooth the resulting rotation targets; latch the last alignment during release. Require a near-upright peg at the release gate instead of dropping a tilted post simply because its estimated XY is close.

The peg orientation controller also needs the insertion constraint to dominate the IK posture preference. The baseline's ~4.8° wrist tilt is already at the task's 5° limit before any grasp slip. During CARRY/PLACE/RELEASE only, reduce the posture regularizer from 0.005 to 0.0003 (the existing precision-push controller's value), preserving the original grasp/lift behavior. This removes the home-posture bias that competes with tight position/orientation alignment; it is not a change to task tolerance.

Peg orientation feedback completed at 2/32; feedback plus reduced posture bias completed at 4/32, both below baseline 8/32. Rejected both and restored the original peg teacher. The geometry diagnosis remains useful, but these controllers do not yet solve it better.

Reorient CLOSE→LIFT fix completed at **100/128**, versus matched baseline **97/128** (identical initial qpos). Nine trials gained success, six lost it; this is a modest measured increase, not evidence of 90% competence or statistical significance. The transition regression test proves the removed open pulse directly. Selected gained environment 2 for the success video and failed environment 0 for the residual clip, both taken from this exact measured batch. Only this teacher improvement is being published.
