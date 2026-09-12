# Remaining teacher improvement pass — 2026-09-11

All 24 remaining teachers receive matched seed 20260911 GPU n=32 diagnostic baselines with strict first episodes. Sources archived in before/. Tasks, distributions, rewards, budgets are unchanged. GPU allocation: baseline grasp group GPU1/core1, planar GPU2/core2, mechanisms GPU3/core3. No other jobs touched.

## Push Button: contact pressure hypothesis
Baseline 30/32. Both failures reach phase PRESS but finish with essentially zero button displacement. Native MuJoCo replay at steps 75,140,150 shows actual finger-pad/cap contacts throughout, rather than missing approach. The closed pads oscillate on the cap, with speed up to 1 rad/s late in the episode. Candidate retains the approach/contact geometry and uses bounded command lead during press to sustain load against the spring instead of re-anchoring the command on the deflected joints every step. Test isolated teacher override before adoption.

## Lift Cube: brake before floor-level grasp
Baseline 23/32, all nine failed endpoints are terrain terminations in DESCEND/CLOSE. Failure montages show several did lift earlier, then lost the cube and hit the floor on re-approach. Existing grasp height is derived from object-minus-hand under a floor-resting assumption and has no physical hand clearance guard. Candidate uses FK height for final descent/close, reduces descent below 12 cm, and bounds vertical error against the lowest hand geometry. This addresses clearance and stopping distance; no goal/predicate changes.

Button pressure candidate rejected: 27/32 versus 30/32. Next hypothesis: actual contacts are sustained yet button moves upward in one failed trial; remove the forward approach tilt during PRESS so the pads load the top face vertically instead of sliding down the cap side. Retain original travel controller.

## Topple Block: fixed waypoint is accidentally a repeated displacement
Baseline 23/32; several failed episodes terminate during PUNCH, including env12 with axis alignment .9992. Code calls _anchor a fixed point but stores the relative face-to-site vector and returns that unchanged every step. Base IK adds that vector to the current site each time, so the target keeps running away. Candidate stores the target in the robot-base frame via FK and subtracts current FK on every punch. This is a reference-frame bug, not a new success tolerance.

## Door and Drawer: remove seating obstructions
Door baseline 3/32, 22 timeout in SEAT. Native replay env1/3/4 has the bar contacting an unnamed collision geom on the robot while the hand stays short. Controller docs assume an 80 mm open aperture with 30 mm lateral clearance, but GRIPPER_OPEN=0 commands only the midpoint opening. Candidate commands +1 on approach/seat so the finger bodies can pass around the 20 mm bar before closing.
Drawer baseline 21/32, 7 seat timeouts. Contact replay shows panel/cabinet collisions while attempting to insert a closed fingertip into a gap narrower than the fingertip. Test the existing front-pinch slider controller (horizontal bar, close vertically), approaching the exposed bar without needing the slot. This is an isolated alternative, not adopted until matched improvement.

Door full-open and Drawer front-pinch alternatives rejected: 2/32 versus3/32 and1/32 versus21/32. Original sources retained.

## Tool Pull: descent controller turns off gravity support
All32 terminate in DESCEND at steps51–63 in representative traces, before any tool-grasp contact. Renders show the hand simply sinks past the shaft onto terrain. Code explicitly removes bounded command lead in DESCEND; this re-anchors joint targets to gravity-deflected joints. Test retained0.12lead (as successful Lift brake candidate), raw FK height instead of delayed EMA, slower final descent and12mm geometric clearance. No direct-puck fallback result can count under the audited task.

Button vertical press confirmed123/128→126/128, same seed/batch/device/protocol; adopted (format/comment-only difference from evaluated candidate). This is a modest paired sample increase, not a significance claim. Success/failure batch clips rendering for publication.

## Model-based gravity compensation experiment
Tool braking still0/32 with near-identical early terrain terminations. Drawer/door seating also stalls despite sustained commanded error and physical obstructions. The shared controller returns joint-position commands without explicit gravity support; its bounded command lead is not an inverse-dynamics correction. Test an isolated mixin that computes gravity torque on the existing private arm model using only observed joint positions, then adds torque/Kp to the position command. This is the standard position-servo gravity offset using the XML actuator gains; no simulator state or privileged object signals are used. Evaluate each affected task separately and retain only demonstrated improvements.

Lift brake confirmed94/128→125/128 (73.4%→97.7%), identical seed/batch/protocol; adopted on LiftCube only, other shape classes unchanged. Full retained source archived in lift_brake_full.py. Compare initial states before publication.

## Stack: staged transport and controlled release
Baseline7/32. Failure render/contact logs show plate/base contact, cube rolling off during placement, then hand/terrain contact during PLACE/RELEASE. Original RELEASE returns zero displacement, which re-anchors onto each sagged observed pose. Candidate reuses the now-confirmed safe LiftCube acquisition, estimates the stationary goal in robot-base coordinates before lift and the retained grasp in the hand frame, transports above the base before lowering at a bounded rate, then holds the last joint command while opening. It does not use privileged contact state or alter stacking success.

Topple fixed anchor confirmed105/128→112/128, stillbelow90%. Next candidate ends the punch once observed body-X reaches0.8 vertical alignment: past the tipping point, continued pushing is unnecessary and can prevent settling or drive excess drift. The fixed target correction remains.
Gravity support improved Drawer21/32→28/32; others so far tie/worsen and will not be adopted. Tool now actually grasps27/32 and makes tool/puckcontact7/32 but remains0success; its old direct-puck fallback invalidates11episodes and wastes the remainder. Next Tool strategy must retain/reacquire the tool instead.

Tool regrasp candidate: gravity support preserves terrain clearance and gets27/32 tool grasps, but most slip out and switch to forbidden direct pushing. Reuse gravity support, lower the pinch from30 to26mm (pad bottom nearer shaft equator), and replace entry to direct-push phases with another tool attempt. Still observation-only, same600step budget.

Further Tool trace evidence: env9 retains21–22mm aperture through ADVANCE/SWEEP, but puckgoalerror stays~.26m and motion phases hit their timeouts. The12mm Cartesian increment competes with the default posture bias known to produce20–30mm stationary error. Reduce posture regularization only during sliding (0.0003, used elsewhere for millimetric positioning), preserving acquisition posture.
Topple early-settle21/32 versus26/32 fixed-anchor: rejected. Retain only reference-frame correction (confirmed112/128).

## Peg: align the held object, insert before opening
Baseline9/32,22failures finishDONE with a peg lying across the board (previous measured release/tilt diagnosis). New candidate adapts the promising staged Stack controller to peg-tip observations and78mm grasp height. It estimates peg-to-hand rotation during lift, then commands the hand orientation that makes the PEG upright and square to the bore, instead of assuming vertical wrist means vertical peg. Place targets tip6mm above the floor (within the straight bore), then holds the last joint command while releasing. No task changes; same noisy observations.

Drawer gravity support confirmed79/128→101/128 (61.7%→78.9%), stillbelow90. Adopted as an opt-in base flag enabled only on Drawer. Base default isFalse; previous gravity wrappers must be reproduced against before/ sources to avoid adding support twice. Cage gravity diagnostic4/32→15/32 merits a matched128comparison next.

Stack staged confirmed36/128→101/128 (28.1%→78.9%), identical initial states/observations. Adopted as staged_stack.py; the existing GraspTransportPolicy stays available and unchanged for Place/Peg. Stillbelow90, residual25timeouts/2terminations must be inspected before calling it solved.

Planar push gravity alone0/32Pushand2/32Drag: existing height integral and upward feed-forward now add on top of explicit gravity torque, raising pads away from the30mm carton. Test support with those redundant height compensations disabled, keeping ride height and planar strategy.
Peg staged candidate0/32 all early DESCEND ground contacts: reject this orientation branch. Nearest equivalent square-face yaw is tested next; the first version forced one exact yaw even when the90-degree-equivalent face was nearer the current wrist.

Cage gravity support confirmed23/128→68/128 (18.0%→53.1%), identical initial states/observations. Adopted as per-task gravity_compensation=True. Original cage aperture/transport success rules and OPEN command stay intact. Stillbelow90.

Planar support without height bias: Push4/32 (reject), Drag17/32 versus9/32 (confirm). Push failure contact states show large carton/ground penetration while loaded by pads; next candidate inclines the pushing face15degrees upward so its normal has an upward component and unloads floor friction. It preserves original gravity/height controller and uses actual orientation in the existing pad-clearance calculation. This is a contact-direction change, not a friction/task change.

Stack residual classification corrected: all27 failures of staged128 are timeouts in RELEASE, zero terminations. Native end states and metrics show the cube on the floor after a failed drop; the controller then parks open forever. Candidate adds observation-based recovery after80settling steps, using20smoothed goal-error samples to distinguish a failed placement, and reuses the safe pickup. Goal/grasp estimates reset for the new attempt. No new episodes or reset retries; same1000step budget.

Strike baseline26/32 with3ground terminations during PUSH. _guard first enforces minimum upward clearance, then norm-scales the whole vector; that can shrink the very upward correction that makes the guard safe. Candidate reapplies the vertical floor constraint after norm limiting. Same strike geometry and launch law.

Drag support confirmed30/128→57/128, stillbelow90; adopted per-task gravity support with learned Z bias and upward feed-forward disabled.
Cage fixed-heading alternative is revisited only after new evidence: explicit gravity support improves the original23→68/128 and removes its ground-terminals, but residuals include pinched aperture after reorientation/recaging. Centered entry with fixed heading previously failed without this support. Test it with explicit gravity support and actual-pose reference, keeping its earlier geometry unchanged.

OpenLid independent seed20260912 scored114/128 after119/128 on seed20260911. Do not call reliably above90 yet. Baseline failures include missed seating on the narrow stem. Candidate filters knob position in the robot-base frame during approach/close, rather than filtering a relative vector that changes as the hand moves. Retains original lifting controller once the lid moves.
Cage centered+support25/32 versus15/32 support on the original planar controller; confirming128.

- Larger confirmations: Stack recovery 122/128 (original 36, staged 101); Cage centered/open with gravity support 95/128 (original 23, planar gravity 68). Both exact initial state/observation comparisons passed. Adopted both. 39 teacher tests passed. Lid world-position filter 122/128 versus 119; independent seed running. Place independent confirmation 115/128 versus 118 on original seed. Drag 57/128 versus 30 published.
- Peg next hypothesis: reduce posture competition during orientation alignment, wait for wrist alignment before descent, and add gravity support; previous staged attempts terminated during acquisition. Task and episode budget unchanged.

Correction to the Door hypothesis above: action zero already reaches the configured maximum opening after the actuator offset and clipping; it does not establish a half-open physical gap. The rejected +1 experiment provided no improvement.

Lid world-position filter confirmed on both independent batches: seed 20260911 baseline119/128 to122/128; seed20260912 baseline114/128 to125/128. Adopted. This corrects filter lag from moving the reference frame; no mechanism or task changes.

Cage residuals at95/128: most misses have valid aperture but are just outside the enclosure near the goal; six halt with under5mm caged progress because HOLD parks over the current cube instead of the goal. Test final hand-centering at the goal inside35mm, preserving open aperture and all success predicates.
Stack independent confirmation123/128 (seed20260912), after122/128 on development seed.

Rejected Cage final-goal centering22/32 versus25/32 and Peg supported acquisition0/32. Peg stayed empty through repeated acquisitions; no claim of improvement. Edge next test filters object position in the robot-base frame, with and without gravity support, motivated by the same moving-reference filter defect confirmed in Lid.

Precision qualification: the Lid development-seed comparison has bit-identical qpos/qvel/mocap positions, but goal-marker quaternion differs by at most1.2e-7 and observations by3.6e-7. It is not bit-identical across all saved arrays. The independent Lid114→125 comparison is bit-identical for all initial state and observation arrays and independently supports adoption.

Edge world-position filters both remain0/32; rejected. Peg next isolates only the verified Lift descent brake and real hand-clearance guard on the existing peg teacher, retaining its yaw, goal estimation, grasp checks and transport.

Window: SEAT accumulates compensation needed to reach the bar, but _go clears it on entering CLOSE even though CLOSE adds that same integral to its hold target. Test preserving this compensation only across SEAT→CLOSE to avoid moving off the handle as the pads close.

Window preserved-seat compensation26/32 equals baseline; rejected. Peg brake10/32 versus9/32 merits128 confirmation. Valve next: estimate the fixed hinge center in the robot-base frame instead of combining a lagged moving-tip vector with the current raw spoke angle, which describes inconsistent physical points during rotation.

Pivot next: the controller overwrites wrist joint7 after solving IK and applying the pad guard, changing the contact pose outside that solve. Test solving approach and closing axes jointly as a full frame and removing the post-solve wrist override; retain contact plan and phase thresholds.

Valve fixed-center16/32 equals baseline; Pivot full-frame0/32 remains zero. Reject both. Place independent failures include two terrain terminations in RELEASE: its zero Cartesian displacement re-anchors on the sagged actual joints instead of holding station. Test holding the last arm command during the existing60-step release window, the same principle that helped Stack.

Tool handle geometry: the shaft spans x±0.13m, object_site is the intended handle at x=-0.09m, but PINCH_U=+0.02 puts the hand only35mm behind the puck at U_HOOK=.055. This explains direct robot/puck contacts despite disabling direct fallback. Test PINCH_U=-.09 (145mm separation) on the supported regrasp controller, keeping required tool contact and direct-contact invalidation unchanged.

Tool handle grasp still0/32, but direct-contact invalidations drop26→9 versus the center-grasp regrasp candidate.15 ground terminations now dominate. Next restores the original30mm grasp height (from26mm) and uses current FK height instead of its lagged EMA during height control.

Tool handle plus current-height control0/32: rejected for production despite fewer collisions. Door next tests the simpler front-pinch controller already effective on Window, centered on the actual bar at object_site-15mm with70-degree downward tilt, instead of the repeated staged seat law. Original128 baseline6/128.

Tool orientation bug verified from saved MuJoCo observations: handle_height32 env27 step400 has tool rotation row1=(.25,-.96,-.01), controller yaw2.89rad versus shaft-axis yaw0.25rad reconstructed from rows1,2. The cylinder rolled about its shaft without reversing heading; the yaw-only estimator incorrectly reverses the tool plan. Test the true shaft axis (R00,R10), recovering row0 by row1×row2, on the supported handle-height candidate.
Door simpler front pinch0/32: rejected.

Tool shaft-axis candidate remains0/32, so no new Tool success-rate claim or publication. Retained only the independently verified orientation-decoding bug fix in the original controller (recover true shaft +x axis after arbitrary roll), with a roll-invariance regression test. The larger handle/gravity/regrasp redesign remains experimental.

Peg brake confirmed twice: seed20260911 original26/128→32/128 (22gained,16lost); independent seed20260912 original21/128→33/128. Adopted the narrow FK descent brake and lowest-hand guard on the original peg controller. Still only25–26%, not solved; goal tip reference and insertion success are unchanged.

Full remaining-task128 baseline sweep completed: Peg26, Tool0, Edge0, Pivot0, Door6, Window108, Valve63, Lever108, Push29, Strike and Throw recorded in baseline128. All24 baseline128 results now exist; Reorient already had128 in the preceding pass.40 teacher tests passed after final Peg and Tool code changes. Nine task improvements are published; no claim that all25 teachers are solved.

Final installed-policy smoke: Tool orientation correction0/32; Peg brake12/32. The original candidate10/32 and installed12/32 were run after different preceding tasks; the larger paired comparisons are the adoption evidence. Baseline sweep finishes Strike117/128 and Throw75/128.
Small-gain audit: project LOGS reports up to6percentage-point run variability on dynamic tasks. Add independent matched Button and Topple comparisons rather than treating their development-seed gains as reliable alone.

Independent small-gain checks confirmed: Button122/128→128/128 and Topple104/128→113/128, seed20260912. Both retained; published metadata includes these confirmations.
