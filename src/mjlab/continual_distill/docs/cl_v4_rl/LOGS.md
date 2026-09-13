# Logs — RL teacher stage

Append-only record. Keep planned actions separate from completed work and measured outcomes.

## 2026-09-12 — Stage creation and readiness check

User set the scope to 24 tasks, deferring Tool-Pull without deleting it, and specified **one RL teacher per task with success >90%**. Created this stage folder with goal, context, plan, task/status inventory and experiment ledger.

Completed a read-only inventory of all current active registered configs. Confirmed the selected 60D observation/8D normalized action contract and current PPO settings. Saved package versions, source hashes, git HEAD and dirty paths. Retained prior historical 25-task results without rewriting them as 24-task measurements.

At 14:30:05 UTC, holder20277 on dgx1 was running; all eight GPUs showed 0 MiB / 0% with no compute processes, but only assigned GPUs1–3 are available for this work. Slurm expiry: 2026-09-28T08:58:54. Shared disk: approximately 1.6 TB free, 95% used. Saved raw snapshots.

Ran PREFLIGHT-PPO-001 on assigned GPU2 via UUID pinning inside the holder. Reach, 64 envs, two PPO updates (3,072 transitions). Network parameters changed; losses stayed finite. Saved a checkpoint and reloaded matching deterministic inference outputs and optimizer state. This is a plumbing test, not a trained teacher or success measurement. Evidence: `evidence/ppo_smoke.json` and `.log`; checkpoint under `runs/PREFLIGHT-PPO-001/`.

Found a launcher issue: the generic GPU selector parses `CUDA_VISIBLE_DEVICES` entries as integers and rejects GPU UUIDs. Recorded the failure in `evidence/launcher_probe.json`; created a stage-local already-pinned launcher using `run_train` directly and checked its dry-run. Its full pilot/config-saving path remains pending.

Readiness conclusion: infrastructure supports a small Reach pilot after evaluator setup. A blanket 24-task training launch is premature: articulation approach rewards saturate, completion shaping has gaps, Cage's initial gripper policy tends to invalidate episodes, and eight physics presets have zero gravity. No reward/success/physics changes or full teacher jobs were made/launched in this setup pass. **0/24 certified RL teachers.**

## 2026-09-12 — Checkpoint/worktree handoff

User requested saving current work before handing off to a new agent in an isolated worktree. Preparing branch `checkpoint/cl24-rl-handoff-20260912` with source, changed assets, tests, plans and compact evidence. Large generated outputs remain in the original checkout; LOCAL_ARTIFACTS.json records the non-ignored exclusions. HANDOFF.md specifies a new worktree and explicit PYTHONPATH to prevent the shared editable environment from importing the wrong checkout.

## 2026-09-12 — Isolated RL continuation

Created worktree `mjlab-rl-teachers-24-codex` on branch `exp/rl-teachers-24-codex` at 4149157; verified shared `.venv` imports this worktree under explicit PYTHONPATH. Holder20277 and process placement were rechecked. User subsequently authorized GPUs1–7, leaving GPU0 unused.

Added strict checkpoint evaluator that captures the task command's actual success predicate at the environment's internal reset boundary, then counts only each environment's first episode. Reach smoke checkpoint scored 0/8 seed20260913; this was evaluator plumbing, not teacher evaluation. A fresh Reach pilot (RL-001) trained 1,024 environments for 500 PPO iterations on GPU2. `model_499.pt` scored 128/128 seed20260914 and 128/128 seed20260915; raw per-episode results and checkpoint hash are in evidence. Both batches ended by timeout. This is the first measured teacher passing the proposed rate gate; targeted evaluator edge-case tests remain pending.

Created W&B project `mjlab-cl24-rl-teachers-20260912` under the currently authenticated `domimagi-iitj` entity. Reach TensorBoard history was synced, and its strict evaluation plus checkpoint were uploaded. Fresh Lift, Push-Cuboid, Drag-Pull, Reorient and Topple pilots launched on GPUs1,3,4,5,6 respectively with W&B logging. User requested a Sudhir W&B destination instead. The exact target entity/project and current account's write access remain unverified; no global login or settings were changed. Subsequent launcher invocations now require an explicit per-run `--wandb-entity` to avoid further uploads to the shared default account.

User provided target profile `sudhirpratapyadav`. The currently authenticated API identity cannot create the experiment project in that entity (HTTP 404). User questioned continued running; immediately stopped only this experiment's Slurm steps 20277.1601/1604/1605/1606. Verified holder batch remains and all GPUs have zero compute processes. The stage launcher now defaults only this project's W&B entity to `sudhirpratapyadav` and supports offline logging; no global W&B change. Do not resume training or upload until destination access is resolved. Push-Cuboid independently failed at PPO iteration 268 (invalid action std); its model_200 checkpoint scored 1/128 strict validation. Lift/Drag-Pull/Reorient/Topple retain partial checkpoints.

### W&B resolved with the supplied experiment credential

Authenticated as `sudhirpratapyadav`; API reports the default entity as `sudhirpratapyadav-indian-institute-of-technology-jodhpur`. The earlier short profile namespace was not the actual team destination. Created `mjlab-cl24-rl-teachers-20260912` there via SDK initialization and verified a logged scalar through the API (probe c72lpymk). Snapshots confirm the shared `.netrc` and W&B settings did not change. Credential is stored privately outside the repository; process-local configuration selects that key/entity/project and clears the inherited `WANDB_USERNAME` override used by RSL-RL.

Copied all six original pilot histories, retained checkpoints, effective YAMLs, saved source diffs, and available strict evaluation JSONs into Sudhir's project. Remote summaries verified; `evidence/wandb_pilot_imports.json` maps all imported run IDs and checkpoint hashes. Original records remain intact. The pause condition is resolved. Prepared continuations of the four interrupted pilots from their own finite checkpoints, restoring optimizer and normalizers while restarting simulator/RNG with the recorded seed. No observation/action/reward/success/physics changes. Push-Cuboid remains under diagnosis and is not automatically restarted.

Resumed after the successful imports and a fresh idle-GPU/holder check. Lift: step20277.1615, GPU1, W&B10krztet. Drag-Pull: step20277.1618, GPU4, W&Bgg9be6ai. Reorient: step20277.1617, GPU5. Topple: step20277.1616, GPU6. Full URLs/IDs are in `evidence/wandb_resumed_runs.json`; all four remote runs report running and contain loss metrics. GPU process placement verified (~2.5GB each); GPU0 unused. Code provenance is commit adb9bb0 plus each run's archived source patch and manifest. Resume starts iteration labels at the saved label, as implemented by the installed RSL-RL version.

## 2026-09-12 — Strict evaluation, recipes and two retained teachers

Completed 44 focused checks, including adversarial execution of the real evaluation loop (terminal-before-reset capture, staggered subset termination, retries excluded), recipe invariants and approved interface checks. Preserved registered physics explicitly, including eight zero-gravity tasks. Added optional mechanism approach_scale (default preserves baseline), stage-local task reward recipes, log-std/fixed-LR PPO variants, and first-nonfinite diagnostics. Six-task short rollout preflight and a real guarded3-update Button PPO run passed. Source commit f39765d precedes the six training launches; manifests in evidence/training_wave2.json record exact source, seeds, steps and GPU UUIDs.

Completed baseline continuations: Lift0/128, Drag100/128, Reorient0/128, Topple123/128 validation; Topple confirmation118/128. Fresh Lift/Reorient grasp-shaping variants, Push stable-PPO retry, Button/Drawer mechanism variants and a longer Drag continuation now run on GPUs1,5,3,2,7,4 respectively. No old teacher weights were used; continuations restore only this experiment's own checkpoints.

Reach is fully certified at128/128 +128/128; Topple at123/128 +118/128. Retained checkpoint/optimizer/normalizers/configs and reviewed actual-rollout videos are uploaded as certified-teacher W&B artifacts. Topple's highest-return failure is physically tipped but moving above the unchanged settling thresholds. Repeated traced Topple validation again scored123/128 but8 per-environment outcomes changed; recorded explicitly rather than claiming bitwise replay. Reach video batch128/128; no measured failure clip exists. See task-review and task-certificate JSON evidence.

Button model700 strict validation128/128; confirmation seed20260915 launched. No confirmation-seed tuning. Last GPU snapshot:GPU0 andGPU6 zero MiB before evaluation, six training jobs placed on assigned GPUs. W&B API identity and destination reverified; import path reverified from our worktree. Holder expiry2026-09-28T08:58:54.

### Button and Drawer certification; Push/Drag outcomes

Button model700 and Drawer model1000 each passed validation128/128 and independent confirmation128/128. Reviewed recorded reset/terminal frames: button physically depresses (joint0 to -0.052029m); drawer physically extends (joint0 to -0.251889m). No failed episodes occurred in those batches. Certified artifacts and normalizers uploaded; certificates in evidence. Stopped only their own training steps20277.1632 and20277.1635 after retention. **4/24 certified.**

Push stable V2 completed1,000 updates without the original numerical failure; strict model999 validation111/128. Prepared an additional1,000-update continuation with unchanged recipe. Drag R2 stopped after iteration1380 due to nonfinite qpos/qvel in environment563 while policy parameters remained finite. Retained full diagnostic state and compact evidence/RL-004-R2-numerical-failure.json. An attempted final-model evaluation found the expected final checkpoint absent and produced no result; evaluated actual model1300 instead:122/128 validation. Independent confirmation launched. This is a simulator-state failure, distinct from the original Push scalar-std failure.

Finite-state preflight passed for all eight remaining mechanism tasks with the validated mechanism_v1 recipe. Fresh Flip/Door/Flap runs are recorded before launch. Drawer rendering was successful; Button's first render invocation used an incorrect CLI flag and was retried with the documented positional argument, with both logs preserved.

### Fifth certificate and next training wave

Drag model1300 confirmation scored120/128 after validation122/128. Reviewed recorded success and highest-return failure: terminal position errors0.008500m and0.030054m respectively, correctly separated by the unchanged0.03m threshold. Retained teacher/normalizers/video artifact uploaded; **5/24 certified**. Published a24-task W&B scoreboard and the full Drag simulator-failure diagnostic artifact.

Launched fresh Flip (GPU2), Door (GPU7), Flap (GPU4), Lever (GPU6), and Push continuation (GPU3). Exact manifests/source/Slurm IDs are copied to evidence/training_wave3.json. Worktree sourcea5b5eee plus archived patch; Lever's patch includes its pre-recorded budget. Reorient V2 completed1,500 updates with zero training success and full action saturation; strict final-checkpoint evaluation with recorded physical trace is running on GPU5. Lift V2 is still training. GPU0 remains0MiB.

Reorient V2 final model1499 strict validation completed: **0/128** on seed20260914. Uploaded measured result to W&B and retained full physical trace under runs/diagnostics/Reorient-V2-final. No confirmation batch warranted. Highest-return failure rendering is the next diagnostic.

## 2026-09-12 — Bounded-policy and grasp-readiness repair

Previous goal turn made progress: retained the fifth teacher, started the next mechanism pilots and measured Reorient V2 failure. Revalidated six live training steps and GPU0 at0MiB before further work. Reviewed Reorient V2's actual highest-return failure: floor collision in8 control steps. All policy dimensions had saturated; extending that checkpoint is not a learning strategy.

Added a process-local BoundedActorCritic with tanh Gaussian means, small output initialization around each task's configured robot pose, log std and gripper std0.03. The native normalized8D action mapping is unchanged, including all learnable gripper outputs. New recipes use LR0.0001, zero entropy coefficient and gradient norm0.5. Lift V3 retains the V2 grasp reward to isolate PPO/policy changes. Reorient V3 adds end-face alignment and actual-grasp-gated lift/upright shaping. Cage receives a valid open prior and dense approach/caged-transport shaping without changing its episode-long no-pinch rule. All benchmark invariants are preserved;48 tests passed, including optimization/checkpoint reload and bounded means under extreme observations.

Physical preflight:64 environments ×200 steps for Cage, Reorient and Lift, all finite. Cage's minimum aperture stayed above the registered threshold under initial stochastic exploration. Results uploaded to W&B; evidence/bounded_preflight.json. Small guarded PPO updates precede expensive launches.

### Seven certificates and bounded-policy training wave

Flip model800 and Door model900 each measured128/128 validation plus128/128 independent confirmation. Reviewed initial/final physical states; switch turns from -0.785398rad to0.787500rad, and the door visibly opens to its target. Uploaded retained teacher artifacts and stopped only own training steps1655/1656. **7/24 certified.**

Bounded-policy guarded PPO and strict reload succeeded (0/16 from only3 updates is plumbing, not teacher performance). Launched fresh Cage onGPU5, Reorient V3 onGPU2 and Lift V3 onGPU1. Exact manifests are in training_wave4.json. Lift V2 final strict result was0/128 and remains rejected. Push continuation finished with model1998; its strict gate runs onGPU3. Flap model900 strict gate runs onGPU7. Prepared Window/Lid/Valve/Axial budgets before launch; Edge/Pivot/Stack/Place/Peg/Strike/Throw still require task-specific readiness.

### Ten certificates; all seven allowed GPUs active

Push final model1998 measured126/128 validation and125/128 confirmation; Flap model900 and Lever model1000 each measured128/128 twice. Reviewed actual success and failure states and retained all three artifacts. Push's endpoint is still moving, which is allowed by its unchanged position-only predicate; the highest-return failure ends0.023802m away, correctly outside the0.02m threshold. **10/24 certified.** Stopped only own Flap step1659 and Lever step1662 after artifacts were retained.

Started Window/Lid/Axial/Valve on GPUs7/4/3/6. Exact source/config/step manifests are in training_wave5.json. Together with bounded Lift/Reorient/Cage on GPUs1/2/5, all seven authorized GPUs have independent training. GPU0 was rechecked at0MiB. Lever's first evaluation invocation preceded creation of model1000 and correctly stopped before rollout; retried only after the checkpoint existed, preserving both logs.

Early bounded-policy diagnostics around iteration400: Lift/Reorient approach within about0.04m and action saturation falls to0.006/0.001; Cage caged fraction about0.877 but negligible transport and zero strict training at_goal yet. This is progress toward contact, not a success claim. Remaining readiness review confirms seven tasks need targeted completion/precursor/dynamic shaping.

## 2026-09-12 — Remaining readiness and early mechanism candidates

Previous goal turn made progress: retained ten teachers and launched seven current runs. Revalidated all seven live Slurm steps, GPU placement and GPU0 at0MiB. Window and Axial training logs reached100%; verified finite saved model300 and model200 respectively. Pausing only their own training steps1681/1685 at these retained candidates to run strict evaluation on their assigned GPUs. If either fails the rate gate, continue from its own checkpoint or record a targeted retry; no certification inferred from training logs.

### Twelve certificates; completion recipe ready for PPO

Window model300 and Axial model200 each passed128/128 validation and128/128 independent confirmation. Reviewed physical panel translation and plug extraction from its socket. Checkpoints/normalizers/configs/videos uploaded as certified-teacher artifacts. **12/24 certified.** Their training steps were deliberately paused after checking finite saved candidates; neither needs further training after certification.

Implemented completion_v1 for Stack/Place/Peg: broad approach plus actual grasp/lift/transport, then dominant native released/supported/settled completion bonus. Peg grasp target uses the upper collider body rather than insertion tip (measured initial target height0.070m). The exact native predicate supplies the terminal bonus, including full containment or square-bore fit; no predicate/physics/interface changes. Completion actors initialize gripper mean0.5/std0.1 to permit useful closure exploration.53 CPU tests passed, including incentive comparisons and geometry invariants; three32-env ×100-step physics preflights are finite and uploaded to W&B.

Added optional evaluator diagnostics sampled only from still-active first episodes. Lift V3 model900 debug seed20260913 scored0/16; all16 lasted the full horizon. No two-pad grasp occurred, despite endpoint gripper/object distance0.00670m. Mean aperture0.06665m, gripper action0.6736: it reaches the object but has not closed enough. This is direct contact evidence, not a rate claim; continue its recorded budget while checking Reorient similarly.

### Closure-reward discontinuity found before full completion pilots

Reorient model1000 debug batch:0/16, no two-pad grasp, mean final aperture0.07639m and gripper distance0.04473m. Inspection found a hard preferred-aperture switch at distance0.035m. With an aligned hand and aperture0.075m, approaching from0.036m to0.034m drops the original approach reward from1.4816 to0.9112 (about38.5%). This is a plausible cause of hovering, not a claim that all learning failure is explained. The draft completion reward had the same issue.

Preserved old recipe behavior and added reorient_v3/completion_v2 with a monotone proximity-weighted closure incentive. Added regression checks over the full approach path for multiple apertures. Previous physics/PPO preflights remain recorded under old source/recipe; repeat the changed reward's finite-state/PPO checks before full pilots. Planned a continuation of Reorient from its own finite checkpoint with only the reward recipe changed.

### Smooth closure validated; first full completion pilots

58 focused tests passed after preserving old recipe behavior and adding completion_v2/reorient_v3. The revised three-task physics preflight is finite; three guarded PPO updates succeeded for both the revised Stack recipe and Reorient recipe. New evidence remains separate from the earlier hard-switch preflight.

Stopped only Reorient V3 step1675 after verifying finite model1500; its recorded mean policy has no grasps in the debug batch and the reward cliff makes continued training under that recipe unattractive. Prepared a1,500-additional-update continuation from its own model1500 with reorient_v3, preserving policy/optimizer/normalizers and the benchmark. Stack and Place full pilots use completion_v2. Cage and Lid finished their budgets with zero training success; final checkpoints will be evaluated and diagnosed without assuming certification.

## 2026-09-12 — Thirteen RL teachers and diagnosed retries

Previous goal turn verified live Stack/Place/Peg and W&B. Valve model1999 strict127/128 +128/128 certified after actual success/failure review: red handle rotates toward vertical marker and stays there through8s; sole validation failure ends with fingers on ground at2.74s, correctly rejected. Teacher/normalizers/configs/videos uploaded.13/24 independently trained PPO RL teachers certified.

Lift V3 final strict0/128; no sampled two-pad grasp. Final aperture0.055137m, distance0.007280m, gripper action0.329568. Recorded failure reaches the cube but remains on floor through20s. Next intervention targets stronger closure and an explicitly recorded gripper-only exploration reset; not yet launched.

Cage first pilot strict0/128,97.7% final caging but negligible transport. Lid first pilot strict0/128, alltimeouts. Both revisions passed3 resumed guarded PPO updates and61 CPU tests. Launched Cage cage_v2 GPU5 step1716 and Lid lid_v1 GPU6 step1717. Exact new manifests in training_wave6.json.

Reorient R4 stopped automatically at1970: nonfinite qpos12 entries/qvel15 only in environment64, policy parameters finite. Last saved model1900 under16-episode debug evaluation seed20260913 onGPU2. Numerical diagnostic archived; no benchmark physics changes. Stack/Place/Peg remain live; GPU0 unused.

### Gripper exploration, saturation and remaining-task readiness

Added explicit --resume-gripper-std to the stage runner, with positive/finite validation and manifest provenance. It changes only gripper exploration and its Adam moments after loading; tests verify exact preservation of deterministic actions, all other weights/normalizers/std and optimizer state. Lift closure weight0.5→2; Reorient smooth closure weight1→3 and true-contact base bonus2→4 so actual grasp dominates noncontact closure. Both resumed3-update preflights passed; finite checkpoint stds verified.

Lift R4 onGPU1 failed at2009 with finite policy parameters. Added opt-in --capture-pre-step to save qpos/qvel/ctrl/warmstart/mocap/actions immediately before a numerical failure. A same-seed/config2048-env30-update replay completed without reproducing the failure; finite model2028 retained. The underlying numerical cause is unresolved; no resets masking invalid state or solver/physics changes introduced. Reorient R5 remains active onGPU2.

Lid R2 showed unbounded mean12644 and100% action saturation with zero training success. Stopped only ownstep1717; source evidence/Lid-R2-saturation.json. Fresh bounded lid_v2 with gripper mean0.5/std0.15 passed3 PPO updates and starts2000-update2048-env pilot onGPU6.

Implemented edge_v1/pivot_v1/strike_v1/throw_v1 training recipes. Edge shapes near-rim exposure then side pinch/lift; Pivot shapes ramp/wall-contact tilt and native completion history; Strike uses approximate sliding endpoint with registered mu0.04; Throw uses descending rim-plane crossing and actual release. Native predicates remain authoritative.70 focused CPU tests passed, including undershoot/overshoot/sideways and unreachable-rim prediction cases. Four32-env100-step physical preflights finite; uploaded W&B run5g2qp9h4. Short PPO preflights now run onGPU1. First full Strike pilot takesGPU1 after completion-task debug checks; Lift/Edge/Pivot/Throw use slots as current pilots complete.

### Top-press reward loophole confirmed; corrected completion pilots launched

Remaining-task PPO preflights completed: Edge/Pivot/Strike/Throw each3 finite updates and a finite saved checkpoint. First full Strike strike_v1 launched onGPU1,2048 environments ×3000 updates.

Debug seed20260913 evaluations: Stackmodel1500, Placemodel1200 and Pegmodel800 each0/16. All show high native two-pad contact but nearly closed aperture and no completed lift/placement. Stack re-evaluation with enclosure diagnostics and actual recorded frames confirms both pads pressing the cube top into the floor. Final active-lane sample:93.3% two-pad contact,0% enclosure, aperture0.001164m, cube height0.014724m. Inspected frames0000/0250/0499. The native task correctly rejects this; the defect is in training grasp credit, not certification.

Added completion_v3: actual grasp reward now requires both native two-pad contact and existing between_fingers geometry. Old recipe behavior is preserved. An adversarial geometry test rejects top pressing, accepts an enclosed bilateral pinch, and rejects enclosure without contact.74 tests passed. Retained finite old checkpoints and stopped only ownsteps1704/1705/1714. All three fresh64-env3-update PPO preflights and checkpoints finite. Launched fresh Stack V2 onGPU3, Peg V2 onGPU4 and Place V2 onGPU7 with pre-recorded budgets. Exact seven current manifests in training_wave8.json. Native observations/actions/physics/success are unchanged. Edge/Pivot/Throw remain preflighted but need first full pilots; review their contact shaping before launch in light of this finding.

W&B scoreboard remains13/24; all simulator numerical-failure artifacts now publish through the generic inventory, including Lift R4 and Reorient R4. Debug evaluations, preflights and reviewed Stack failure video are online. No numerical failure has been hidden or counted as success.

## 2026-09-12 — Evaluate bounded Lid and continue remaining pilots

Previous goal turn made progress: Valve became the13th certified independent RL teacher, remaining recipes passed physics/PPO checks, and a confirmed top-press reward loophole led to completion_v3 fresh pilots. Revalidated all seven actual training processes and GPU0 at0MiB.

Fresh bounded Lid V3 now reports100% training at_goal arounditeration780. Verified finite savedmodel700. Pause only ownstep1728 at this retained candidate, then run both strict held-out gates with actual-state rendering. Training logs alone do not certify it; resume its own finite checkpoint if the gate fails.

### Fourteen certified; remaining grasp-credit revisions pass PPO

Lid model700 passed128/128 validation and128/128 confirmation. Reviewed initial, middle and final actual-state frames: wooden lid rotates upright, box interior exposed, and stays open through5s. Retained model/optimizer/normalizers/configs/video artifact; certification runp5lo2knx.14/24 independent PPO RL teachers certified.

Added enclosure-gated edge_v2/pivot_v2/throw_v2 plus prepared lift_v4/reorient_v5, preserving older recipe behavior.79 focused tests passed. Edge/Pivot/Throw each completed3 finite guarded PPO updates and checkpoint save. Added per-update training diagnostics for gripper std/mean, two-pad contact, enclosure, enclosed grasp, aperture and object height; Throw preflight exercised the new metrics online. Full Edge pilot now takesGPU6 after Lid certification;3000 updates at2048 environments, fresh independent policy.

### Cage R2 strict failure diagnosed; contact-target retry prepared

Cage completed its budget, finite finalmodel3498. Strict128-episode validation measured0/128. Final samples show100% no-pinch validity and enclosure, but0% actual robot-object contact; aperture0.07997m, hand distance0.02074m and goal error0.15692m. Reviewed actual failure frames0000/0050/0099: fingers settle open around a stationary cube. The earlier hypothesis that caging transport simply needed more training is not supported by this final result.

Prepared cage_v3: target the trailing inner pad at the cube rear face using actual pad/collider geometry, align open fingers with the goal direction, reward side contact and transport without a binary geometric-cage gate on all shaping. The native no-pinch history mask and strict completion remain unchanged.81 CPU tests passed, including correct rear-face targeting in both directions. Three resumed64-env PPO updates and finalcheckpoint3500 finite; evidence/cage_contact_preflight.json. Full retry awaits a slot while Pivot gets its first full pilot onGPU5.

## 2026-09-13 — Capture-width shaping and the final first pilot

Reorient R5 completed its full budget without numerical failure, but finalmodel3899 strict0/128. Reviewed actual failure frames0000/0250/0499: the hand closes empty then hovers above the unmoved object. Final active sample shows0 contact/enclosure, mean aperture0.0000375m, mean gripper action-0.999663; saved gripper std0.06444. Thus increasing exploration alone did not solve capture and the mean became saturated closed.

Added geometry-aware grasp-width rewards: measured pad gap should match the object's projected collider span plus a smooth3–23mm approach clearance. At actual contact the fit score stays near its maximum; empty closed fingers receive little credit. Versions lift_v5/reorient_v6/throw_v3 preserve earlier recipes and native success. Fresh gripper mean0.5/std0.15 avoids a saturated initial prior.85 full focused tests passed before the final initialization adjustment;46 relevant recipe/geometry/policy tests passed after it. Three fresh64-env3-update PPO cases and checkpoints are finite; capture_width_preflight.json.

A future resume of Reorient should reset its saturated gripper output row, preserving learned arm outputs, rather than only reset exploration std. That output reset is not implemented yet. Lift's prior numerical cause remains unresolved; finite replaymodel2028 is retained and pre-step failure capture is available.

Throw's first full independent pilot uses throw_v3 onGPU2,2048 environments ×3000 updates, with pre-step capture enabled for any numerical failure. All24 active tasks have now received a full RL pilot once this launch is verified; this does not mean24 teachers are certified. Certification remains14/24. Lift/Reorient/Cage retries await slots while the other seven object tasks train.

### Stack numerical stop; preserve approach and reset only gripper output

Live process check found Stack V2 stopped atiteration777: one nonfinite simulator lane261, policy parameters finite, pre-step capture was disabled for that run. Retained finite model700. Debug16-episode evaluation0/16:100% final enclosure but no actual contact, aperture0.07669m and gripper mean0.90457. The top-press credit is gone; useful closure still has not been learned.

Implemented --resume-gripper-mean for bounded actors. It zeroes only the eighth output row and initializes its bias to the requested bounded mean, clearing only those Adam moments. Tests verify exact preservation of seven deterministic arm outputs, other weights, critic, normalizers, std and unrelated optimizer state. This supersedes the earlier note that output reset was not implemented.

Added completion_v4 capture-width shaping.88 focused tests pass. Resumed64-env3-update preflight from Stackmodel700 with gripper mean0.5/std0.15 succeeded; finite model702, final sampled robot-object contact15.625% (not a success claim), aperture0.06368m. Full Stack R3 resumes ownmodel700 for1800 additional updates onGPU3 with pre-step capture. The numerical cause remains unresolved; no physical settings or native success predicates changed.

### Coverage audit

All24 active tasks have RL training evidence. The strict launch-manifest filter covers22; early Reach/Topple pilots predate complete manifests, so their already-certified PPO checkpoints/configs establish the remaining two. Evidence/rl_training_coverage.json makes this distinction explicit. This coverage audit does not change certification:14/24 meet the full two-batch/video/artifact gate.

### Wave12 — strict rejection evidence and prepared retries launched

Previous goal turn classified as progress: it changed authoritative training/evaluation state and provided contact-failure evidence. This continuation also made concrete progress. Verified holder20277 expires2026-09-28T08:58:54 (scheduler time), GPU0 at0MiB and one assigned process perGPU1–7 before scheduling; worktree import verified. Filesystem still about1.6TB free.

Strike completed3000 updates and synced final2999. Strict validation9/128 (7.03125%); no confirmation run. Actual success/failure initial/mid/final frames reviewed. Selected success error0.03864m; selected failure0.08192m. Across128,110 undershot by>5cm longitudinally,8 overshot by>5cm and18 had>5cm lateral error (categories overlap). Median terminal error0.55464m. This teacher is rejected; source-state trajectory analysis retained in Strike-RL023-review.json, videos W&Bc7qxtmyc.

Verified Place/Peg live PID/task identity and finite model1500, then stopped only steps1738/1739. Strict128-episode evaluations each0/128; actual failure videos reviewed. Place:97.66% final enclosure,0% two-pad contact, aperture0.05669m; static cube. Peg: shaft knocked horizontal then pressed,0% final enclosure,15.25% two-pad contact;118timeouts,7ground collisions,3out-of-bounds. Videos online in xxu543n9/lcjtk7nk; reviews retained. Their unchanged recipes are not being blindly extended.

Cage R3 started onGPU1 from own3498 with the already-preflighted cage_v3 contact target,1024 ×2000 additional updates, capture enabled. Lift/Reorient resumed64-env3-update checks completed with finite models2030/3901 and online sync. Full Lift R5 onGPU4 from own diagnosticreplay2028 and Reorient R6 onGPU7 fromown3899 use width-shaped rewards, explicit gripper-only mean0.5/std0.15 resets,2048 ×2000 additional updates and preceding-state capture. Seven arm output rows, native success/physics and60D/8D unchanged.

W&B scoreboard publishing now also retains the six experiment documents, review JSONs, training-wave manifests, preflight reports and coverage audit in an experiment-records artifact. Certification remains14/24; all24 have RL coverage, which does not meet the full teacher gate.

Wave12 placement was reverified with nvidia-smi: one actual training process on eachGPU1–7, GPU0 at0MiB. Lift/Reorient passed100 additional updates without a numerical failure; this is a finite-training check, not success certification. W&B scoreboard runa240y562 synced14/24 plus the experiment-records artifact; a remote API read verified52 files including all three new reviews, wave12 inventory and capture-resume preflight (wave12_wandb_records.json).

### Wave13 — reproduce numerical failure and repair episode reset bookkeeping

Previous goal turn was progress: strict rejection evidence and three launched retries. This turn produced a reproducible simulator bug and a targeted reset fix. Stack R3 stopped at1478 onlane1322 with finite policy. Its pre-step qvel/ctrl were reset to zero but qacc_warmstart retained prior accelerations. Recorded-state replay failed at substep1 for both one lane and all2048; CPU remained finite without warnings. Clearing only warm-start acceleration passes all4 native5ms substeps. Simulation.reset_solver_state now selectively clears reset worlds before reset-pose forward dynamics. No timestep, solver settings, geometry, initialization distributions, action meaning or success predicates changed.

Production-method replay passes all2048 lanes, and100 resumed PPO updates under the same completion_v4 recipe remain finite (model1499). Fresh-state probes across all24 tasks find zero cache before initial reset, so clearing is a no-op there.91 focused reset/reward/strict/interface checks passed, then3 additional completion_v5 invariant cases;7 targeted history/reset/bounded checks passed after capture changes. Source commits0f1c412 and1990076.

Pivot1500 and StackR3/1400 each strict0/128, actual recorded failure frames reviewed. Pivot presses the board flat; CPU recomputed terminal contacts show124 two-pad worlds but0 opposing-normal pairs. Peg similarly18 two-pad worlds and0 opposed. These audits do not replace original GPU contacts or native success. Stack encloses the cube but leaves a gap (100% final enclosure,0% contact). Stack/Place now use completion_v5, smoothly targeting width-4mm near capture instead of width+3mm, preserving width-aware opening farther away. Native contact stiffness and geometry unchanged. Their3-update PPO checks passed; full StackR4/PlaceR3 retries launched with gripper std0.15 and preserved learned means.

Reorient R6 stopped at4609 with a nonfinite rollout output after native reset had already replaced bad qpos/qvel. Replay identifies lane853 failing on substep4; the saved preceding object already moves at~78m/s and~9100rad/s. Zero warm start does not fix it; CPU stays finite but also has extreme velocities. This is a distinct unresolved instability. Added8-state pre-step history and explicit nonfinite reward/observation IDs; adversarial test verifies failure evidence survives native resets. R7 resumes finite4600 for1500 additional updates with current reset fix/history, no output/std reset.

Cage R3 stopped at4649 onlane447, with the same newly reset zero-velocity/nonzero-warmstart signature as Stack. Its native-cache replay reproduces failure at substep1. Conditional continuation from finite4600 uses the cache fix after a cold-cache regression. Older live Throw/Lift/Edge processes retain their original loaded source until completion or the next resume. Certification remains14/24; full goal active.

Cage cold-cache replay passed all1024 lanes/all4substeps with no CPU warning. R4 launched from4600 for1000 additional updates onGPU1, no policy/std/reward changes. Wave13 inventory records which older live processes still lack the newly committed reset fix. Numerical_reset_audit.json connects the reproduced failures, selective fix, all24 cold initialization probes, tests and100-update PPO regression; it explicitly leaves the separate Reorient instability unresolved.

Wave13 live placement reverified: one training process perGPU1–7; GPU0 at0MiB. Holder still expires2026-09-28T08:58:54. Scoreboard runkw50ospi synced14/24 and experiment-records v1; remote API verified69 record files, including captured-state replays, all24 cold-reset checks,100-update regression and wave13 inventory. All seven current runs were finite at the last check. Edge is near its3000-update budget; evaluate its retained final checkpoint after actual process exit.

### Wave14 — fix a training grasp classifier that rejects opposed contacts

Previous goal turn was progress: reproduced/reset-fixed numerical failure, new capture evidence and resumed trials. Edge completed3000 updates normally and synced final2999. Strict validation0/128, all timeouts; actual failure initially slides the plate toward an overhang, then the open hand hovers behind it (final contact0%, hand-object distance0.20964m). Reviewed initial/mid/final frames; video W&Bbsqq9a1o.

CPU collision-geometry sweep on128 saved Stack terminal poses found74 poses with opposing inward pad contacts within[-3,+1]mm after closure. Whole-object enclosure rejected the first qualifying contact in64/74; median extra closure8mm per finger. This exposed a real mismatch between conservative projected width/full-box enclosure and local pad contact. A production GPU forward query on those64 configurations reports64 native two-pad contacts,64 opposing grasps,0 enclosed grasps. These counterfactual states are neither policy trajectories nor integrated stable grasps; do not count them as RL successes. Audit and witness evidence retained.

Added contact_grasp.py: valid actual pad/object contacts must point into the gap (normal cosine>0.5 each side), with native1mm contact tolerance and contact-count/world/object masks. The shaping gap uses a ray through pad center intersected with the local collider box; full projection is a miss fallback, and curved meshes remain approximations. lift_v6/completion_v6 use these training-only changes. Existing native success, geometry, physics and60D/8D unchanged; old recipe behavior retained.95 focused tests passed, including reverse geom order, stale contacts, other-world/object contacts, same-face pressing and rotated-box width. Source0484710.

Verified Lift R5 PID/task and finite3500, stopped only step1766. Strict3500 validation0/128: approach/enclosure without lift or opposed contact. Actual frames reviewed, video v7ngp8ra. Lift/Peg64-env3-update PPO checks finite and online. LiftR6 resumes own3500/std0.15 onGPU4 for2000 additional updates; PegR3 resumes own1500 with gripper mean0.5/std0.15 onGPU6 for2500 additional updates. Learned arm outputs preserved; reset fix and8-state capture enabled. New diagnostics report opposing-contact fraction. Certification remains14/24; full goal active.

### Wave14 continuation and public RL gallery

The preceding status-only turn made no experiment progress. This continuation revalidated Slurm/GPU state and made concrete progress: Stack R5 and Place R4 launched as preregistered after both3-update preflights passed. Source0484710, completion_v6, own2200/2300, gripper std0.15, preserve means,2048 ×2000 additional updates. Actual assigned GPU3/5 placement verified; GPU0 remains0MiB. Holder expiry remains2026-09-28T08:58:54 scheduler time.

Cage R4 completed its1000-update continuation normally; final5599 strict2/128. Reviewed actual success/failure frames: short open-hand displacement then static enclosure; final pad/object contact1.57%, no-pinch97.64%, mean goal error0.139m. No confirmation; not certified. Lift R6/model4100 strict0/128 but97.6% final opposing grasp contact, mean object height0.0278m. Actual failure shows grasp/rotation without goal lift. Throw/model2500 strict0/128, actual failure approaches and hovers without launch; full run continues. Reorient R7 completed1500 additional updates finite; final6099 strict0/128, actual open enclosure/no contact, all timeouts. These measured failures inform the next targeted retries; no success/physics/interface changes.

User explicitly requested a new current-RL card on cl.sudhirpratapyadav.com via untu_vps, with a summary and one policy video per task. Added /v4-rl/ as the newest home-page card and latest link. Build source selects the14 certified checkpoints and10 latest fully recorded validation checkpoints, showing success examples for certified policies and labeled failures for unfinished tasks. Public output contains only summary fields and24 videos/posters; full records remain in W&B. Prior home page backed up outside the served directory. Browser checks cover all24 video playback,14/10 filters, search and mobile overflow; desktop/mobile screenshots visually inspected. PUBLICATION.md documents reproducible updates. Full24-teacher goal remains active.

Public HTTPS verification passed after the final Reorient/Lift media refresh:24/24 videos actually play in Chromium, home-page newest-card navigation works,14/10 filters/search pass,390px mobile has no horizontal overflow. All50 public HTML/JSON/video/poster files return200 and their SHA256s match the local generated artifacts. Evidence/public_gallery_verification.json retained.

Wave14 scoreboard iq07o1jx synced14/24 and experiment-records; remote API verified82 files including PUBLICATION.md, public gallery checks, contact audits, current training inventory and new reviews (wave14_wandb_records.json). Work committed as c18710d; user publication request fulfilled, full teacher goal still active. Five PPO runs remain live onGPUs2–6; GPUs1/7 available for targeted retries, GPU0 verified0MiB.

### Wave15 — contact retries and native integration compatibility

Previous goal turn made progress: strict evaluations, Stack/Place retries, current public gallery and W&B records. New cage_v4 increases only the reward contact-drive target1→12mm after2/128 hover/stall. reorient_v7/throw_v4 use the proven opposing-normal/local-width capture correction.100 focused recipe/interface checks passed; all three64-env3-update resumed PPO preflights/checkpoints finite and online. Cage R5 startedGPU1 from5599 ×2000, Reorient R8 GPU7 from6099 ×2000/std0.15, Throw R2 GPU2 fromfinal2999 ×2500/std0.15. Throw final2999 strict0/128, actual failure still open enclosure; reviewed and retained.

Peg R3 stopped2329 (lane184); finite2300 strict0/128, actual shaft topple/press reviewed. The8-state history exposes free-spin growth145→771808rad/s. GPU recorded-state replay reproduces it; CPU MuJoCo3.11.1 from the oldest state stays finite and decays to38.53rad/s. CPU alternative integrator/timestep cases remain finite. The installed Warp implicitfast omits native MuJoCo's leaf-free-body gyroscopic derivative solve. A first prototype copied the installed full-implicit inverse-bias sign and made growth worse; rejected evidence retained. Correct positive bias derivative in M+h*d(bias)/dv brings GPU38.52rad/s with final qpos difference~7e-5 from CPU.

Reorient R8 then stopped6589 with nonfinite reward lane35 after reset; its8-state history shows171→10847rad/s growth. Corrected replay ends84.79rad/s versus native CPU83.29; contact differences remain, not bitwise equivalence. Both captured histories pass28 substeps in2048 replicated GPU worlds through production CUDA graphs. Sparse/dense64-world100-step regressions include off-center rotated inertia, zero/low/high spins and an independent actuated hinge; errors stay below1e-3qpos/1e-2qvel, hinge error below1e-5.98 focused tests passed before an additional5 saved-backend/evaluation checks. Core source f821aa0.

The correction is opt-in via --free-body-gyro, persisted in each manifest/config and inherited on resume; evaluation restores it. Existing certified teachers and live runs retain the legacy default. No state clipping, dropped invalid lanes, model parameter, timestep, native predicate or60D/8D changes. This is an explicit backend compatibility correction; gyro_compatibility_audit.json lists evidence and limits. Peg fresh and Reorient own6500 each passed3 PPO updates with the correction;2048-world100-update stress runs are now active before full training. Peg is starting a fresh PPO lineage because repeated resumed policies learn shaft toppling/pressing.

Lift R6 completed2000 additional updates to5499, acquiring strong opposing contact and increased object height; strict final evaluation is running onGPU4. Training contact/height diagnostics alone remain insufficient for certification. Full goal remains14/24 certified.

Both optional-gyro100-update2048-environment PPO stress checks completed finite and synced (gyro_stress_preflight.json). PegR4 resumes the fresh stressmodel99 for3000 additional updates; ReorientR9 resumesstress6599 for2000, both inherit the manifest setting. Lift5499 strict0/128, actual reviewed lift/transport with mean goal error0.09496m and height0.19646m; R7 continues5499 unchanged. Place R4 regressed after3400: sampled grasps~88% before,0% after3500, open gripper, joint-limit penalties and no approach. Model/normalizer tensors remain finite. Verified PID2320755/task and stopped onlystep1808; finite3400 strict evaluation running onGPU5.

Place3400 strict0/128, actual initial/mid/final failure frames reviewed: cube lifted beside container, no release/containment. Added explicit process-local optimizer LR override after Adam restore, inherited from manifest; preserves moments and aligns PPO scheduler/logger LR. Six focused bounded-policy, failure-capture and strict-evaluation tests passed. Preregistered PlaceR5 from3400 at5e-5 after a64-env3-update PPO preflight; no reward/output/std/backend changes.

Place lower-LR preflight completed online (p0ncrxmz), finite3402 with saved Adam LR5e-5; place_optimizer_preflight.json records the check. Full R5 started onGPU5 from original3400 for2000 additional updates. Verified PID2327764 onassignedUUID. All seven assigned GPUs have one PPO process; GPU0 free. Updated website Lift5499, Place3400, Peg2300 and Throw2999 videos;24 playback/filter/search/mobile browser checks passed.

Wave15 public verification:50/50 served files HTTP200 with SHA256 matching generated local summary/media;24/24 videos play in Chromium, home newest-card navigation, filters, search and mobile overflow pass. Python verification used the host CA bundle at /etc/pki/tls/certs/ca-bundle.crt; TLS verification remained enabled.

Wave15 W&B scoreboard p7a1sv8z reports14/24. Remote API verified101 files in the current CL24-experiment-records artifact, including all new reviews, optional-backend audit, seven-run inventory, Place LR preflight and public50-file/24-video verification (wave15_wandb_records.json). The scoreboard process completed successfully after syncing numerical failure artifacts. Holder20277 expiry reverified2026-09-28T08:58:54, GPU0 at0MiB, filesystem~1.6TB free.

### Wave16

Previous goal turn made progress: Place LR retry, reviewed four final policies, updated public media, verified50 served files/24 playable videos, synced101 W&B records and committeda46f8af. Revalidated liveSlurm; StackR5 exited normally and CageR5 reachedfinal7598 then synced/exited. Strict evaluations: Stack4199 0/128 (126timeouts,2bounds), final opposing grasp65.08%; Cage7598 16/128 (127timeouts,1ground), improved from2/128. Actual frames reviewed and evidence retained. Preregister bounded unchanged CageR6×1500 andStackR6×1000; no confirmation for failed gates. Full goal remains14/24.

CageR6 andStackR6 launched as registered, sourceed4b8b3, exactsteps1852/1853 andPIDs2328726/2328742 verified onGPUs1/3. The other five training processes remain live, with no numerical_failure.json in any current run. Wave16 inventory records all seven manifests. Cage/Stack new strict results and reviewed policy videos now published to /v4-rl/; certification remains14/24.

Wave16 public refresh verified:24 playable policy videos, home newest-card link,14/10 filters, task search and mobile overflow all pass;50/50 HTTPS files match local hashes.

Strike recorded-state speed audit separates the110 longitudinal undershoots:58/128 episodes have maximum puck speed below0.05m/s,25 between0.05 and0.4m/s,45 at least0.4m/s; all9 successes are in the last group. Speed is a launch proxy, not proof of contact or a new success test. This changes the next diagnosis toward approach/contact coverage before launch-speed tuning; strike_launch_audit.json retained and separately published to W&B.

Wave16 scoreboardrp1l6kwj completed and synced14/24; remote API verified104 experiment-record files, including new reviews, inventory and public verification (wave16_wandb_records.json). Strike launch audit separately synced as rund6da3q80/artifactCL24-strike-launch-audit. Final GPU check shows one training process on eachGPU1–7 and0MiB onGPU0. No additional certifications; goal remains active.

### Wave17

Revalidated all seven Slurm training steps and GPU0 at0MiB. New audit_strike_contact.py replays only FK/contact geometry through128 recorded first episodes. Fixed repeated NPZ decompression in the analysis loop before completing the audit; no GPU/training process was interrupted.58 negligible launches have0 sampled contact states and median81.85mm vertical miss at closest target; weak launches61.19mm, substantial24.00mm. CPU contacts can miss between-frame contact and are diagnostic only. Added strike_v2 precise approach +12mm reward contact drive;57 focused tests pass and benchmark/60D/8D remain unchanged. Preregistered own-policy3-update preflight followed by2000-update continuation in the next suitable free slot. All seven current long runs remain active.

Edge FK audit across128 saved episodes: all reach exposure>=0.8, but median closest existing pinch-target distance0.20312m (horizontal0.20291m), vertical closing-axis alignment0.4003, hand height0.09751m and plate height0.10843m. This is a side-approach/wrist transition problem after exposure, not a lack of plate exposure. Sampled every10th control state plus terminal, no integration or new success evaluation. Retained edge_pinch_audit.json; future reward design should check a reachable full side-wrist orientation, not only abs(vertical closing axis). No Edge recipe change or training launch yet.

Wave17 sourcea74b981 retained the reward/audit changes. Scoreboardjbxzqp99 completed and synced14/24; remote API verified109 experiment-record files including both pose audits, Strike readiness and wave17 inventory. Native puck XML confirms cylinder radius0.0381m/half-height0.0127m; paddle gap check uses the existing Panda pad dimensions. Current long runs remain live, GPU0 stays0MiB. No new certification and no Strike PPO claim before its pending preflight.

### Wave18

Revalidated seven live training steps/GPU0 free. probe_edge_wrist.py checks32 recorded exposed endpoints using bounded IK and CPU collision queries. Each of0,+70,-70degree headings passes32/32 endpoint gates; this does not establish transition feasibility or success. Added edge_v3 full side-wrist alignment,5mm pinch target offset and proven opposed/local-width capture shaping;62 focused tests pass, no native/60D/8D changes. Preregistered own-policy64×3 PPO preflight then2048×2000 continuation. LiftR7 now records occasional training success; strict7100 evaluation launched alongside its single full training process onGPU4 to test deterministic first-episode behavior.

Lift7100 strict0/128:110 positions inside5cm, no settled terminal object; median speeds0.2663m/s/1.866rad/s. Video frames show grasped transport near goal with changing cube orientation; settling fails. Constant-control GPU/CPU comparisons use matched newly sampled per-world fingertip friction (original trace friction unavailable); strong closure drops31/31 on both, gentle2mm closure retains12 GPU/14 CPU grasps and8 GPU native endpoint passes. Interventions are not RL success measurements. Added lift_v7 mild loaded-closure and near-goal settling bonus;64 focused tests pass, registered physics/actions/success unchanged. Preregistered3-update preflight beforeR8 continuation.

CageR6 final9097 strict37/128 (previous16), StackR6 final5198 strict0/128 with95.8% final opposing contacts but little lift; LiftR7 final7498 strict0/128. All three budgets completed normally, evaluations finished. Strike/Edge3-update PPO preflights passed finite, online (aptzcu71/q4mb9cbo), and fullR2 runs launched onGPUs1/3. Edge initial launch rejected a mistyped checkpoint directory before creating a run; corrected to its authoritative RL-021-edge-enclosed/model2999. No policy failure or extra training occurred in that rejected invocation.

Lift R8 and Cage R7 launched after finite online preflights; new Lift diagnostics track actual object speeds and settled opposing grasps. Reorient R9 completed its full gyro-enabled budget without numerical failure, final8598 strict0/128; actual frames reviewed show low capture without goal reorientation. Throw R2 completed normally, final5498 strict0/128 but now captures/lifts the cube, then holds without release. First Throw evaluation command used an invalid shortened task ID and exited during argument parsing; corrected to registered Mjlab-Throw-To-Bin-Franka before simulation. Actual reviewed video htcomf84; bounded unchanged R3 continuation preregistered.

Wave18 refreshed actual placement: one full training process perGPU1–7 (StrikeR2, ThrowR3, EdgeR2, LiftR8, PlaceR5, PegR4, CageR7), GPU0 remains0MiB. No numerical failure artifacts in these current runs. Holder20277 expiry2026-09-28T08:58:54 scheduler time, disk1.6TB free. training_wave18.json records manifests, source hashes, checkpoints and verified PIDs. New Cage/Stack/Lift/Reorient/Throw actual videos are included in the public summary refresh.

Wave18 public refresh verified: home newest-card navigation,24/24 actual video playback,14/10 status filters, search and390px mobile layout pass. All50 public HTML/JSON/video/poster files returnHTTP200 and match local SHA256s with TLS verification enabled. Gallery remains a concise point-in-time RL summary; full teacher goal14/24 remains active.

Wave18 scoreboard aoitgz9v finished and synced14/24. Remote API verified126 experiment-record files, including all new reviews, Lift/Cage preflights, settling audits, seven-run inventory and the50-file/24-video public verification. Source/records committed9315d68; full24-teacher goal remains active with seven PPO runs.

### Wave19

Previous goal turn made progress: reviewed strict results, launched finite-preflighted Cage/Lift plus bounded Throw continuation, verified current public videos and126 W&B records. Revalidated seven live PPO processes and GPU0 at0MiB. New Pivot FK audit covers128 recorded first episodes; median closest reward-target miss6.94mm, closing alignment0.948, approach alignment0.938, full wrist error22.42degrees; no sampled trajectory exceeds20degree board tilt. Endpoint IK at35mm/finger passes24/32; eight fail due to3.18–4.36mm finger/floor penetration. This weakens the hypothesis of a grossly reversed wrist and identifies target/floor clearance and actual tipping as next checks. No Pivot reward or physics change yet.

New evaluation traces capture initial randomized model arrays separately from post-reset trajectories. All24 registered configurations randomize only geom_friction without interval model events;3 tests verify immutable copies and the real strict loop under adversarial subset resets for both backend settings. Old traces cannot recover their original friction; new fields affect evidence only. PlaceR5 completed its full budget normally; final5399 strict evaluation is next onGPU5.

PlaceR5 final5399 strict35/128 (126timeouts,1ground,1bounds), improved0/128. Actual success/failure frames reviewed, videoq6mikssu. Trace now retains finite128×90×3 initial friction with two randomized pad geoms. R6 launched from5399 unchanged with inherited5e-5 LR onGPU5. Pivot40mm/finger plus1mm floor guard passes32/32 endpoint checks; pivot_v3 adds this reward target, full wrist/opening preference and proven opposed-contact held credit.66 focused checks and3 PPO updates passed finite online (map1pnz1); full retry waits for Strike final evaluation onGPU1. No certification added.

StrikeR2 final4998 strict20/128 (all128timeouts), improved9/128. Actual success/failure frames show strike/retraction and puck sliding; selected high-return failure is close but outside the native gate. Video utnqsso7. PivotR2 launched after the evaluator exited, using its original1500 and validated pivot_v3; no second full training process onGPU1. Public Place/Strike summaries and reviewed videos refreshed.

Wave19 actual placement verified: one full PPO process perGPU1–7, GPU0 remains0MiB. Seven manifests/PIDs retained in training_wave19.json; no current numerical failures. Holder expiry remains2026-09-28T08:58:54. Public refresh passes24-video playback, home newest-card navigation, filters/search/mobile layout and50/50 HTTP200 SHA256 matches. Full goal remains14/24; Peg is approaching its final budget and is the next strict evaluation.

Wave19 scoreboard ea54p9as finished,14/24; remote API verified133 experiment-record files, including new Pivot geometry/trace audits, preflight, Place/Strike reviews, inventory and public checks. Commit3613f6e retains the wave. PegR4 then completed its3000-update budget normally at3098 and its strict evaluation started onGPU6. Full24-teacher goal remains active.

### Wave20

Previous goal turn made progress: Place35/128 andStrike20/128, validated Pivot ramp retry, current public videos and133 W&B records. Revalidated six trainers plus running Peg evaluation; GPU0 remains0MiB. PegR4 final3098 strict0/128, upright no-contact hovering reviewed. Generalized closure audit CLI and materialized NPZ once;60/128 terminal geometries admit opposed contacts with13–19mm extra closure per finger. Preregistered own-policy gripper mean-0.25/std0.15 reset after finite PPO gate. Strike launch coverage improved45→97 substantial launches; endpoint calibration remains weak. Place5600 strict74/128; its full run continues.

Peg gripper-reset preflight completed finite and online (unyapx6f), saved inherited gyroON and LR1e-4. R5 launched from3098 onGPU6 with recorded mean-0.25/std0.15 initialization, preserving learned arm/normalizers. LiftR8 final8997 strict0/128;105 inside goal, none inside and settled. Exact-friction hold probe completed31 source states: strong closure dropsall31 onCPU/GPU; fixed20mm/finger retains23 GPU/22 CPU grasps with14 GPU endpoint passes. These are interventions, not RL success. Preregistered Lift mean0/std0.05 preflight and bounded R9.

Lift gentle preflight passed finite and online (lbs3dbh7); R9 launched from8997 with mean0/std0.05 onGPU4, preserving learned arm, lift_v7 and legacy backend. Peg width audit identifies118/128 center-ray misses; full projection rewards55mm opening versus26.3mm central section.52/60 counterfactual opposing-contact poses are ray misses; old fallback overestimates first-contact gap by27.1mm median. Added opt-in peg_v1 corrected fallback;74 focused tests pass. Preregistered small PPO preflight and exact-step replacement after R5 checkpoint retention/evaluation.

Wave20 final checks: Peg width preflight17npttwg passed; R5 exact step1898 stopped, retained3400 strict0/128 and actual failure frames reviewed. R6 launched originalR4/3098 with peg_v1 and recorded gripper reset onGPU6, W&B rec0k5q3. Initial command had an incorrect directory suffix and exited at argument validation before simulation; corrected to RL-020-R4-peg-gyro. CageR7 completed, strict97/128; EdgeR2 completed, strict0/128; actual clips reviewed. PlaceR6 failed after6329 in lane995; diagnostics retained, model6300 strict96/128. No new certification.

CageR8 launched from original11096 after evaluator exit, same2048×2000/cage_v4 onGPU7; verified PID2340222. Current full PPO processes onGPU1/2/4/6/7; GPU3/5 free between evaluations and further diagnosis. GPU0 verified0MiB. Place6300 actual frames reviewed; no confirmation at96/128. Five live run manifests/PIDs and two completed/stopped slots recorded in training_wave20.json.

Wave20 public refresh verified: newest homepage card opens /v4-rl/,24/24 actual video playback,14/10 filters, search and mobile layout pass. All50 public files match local SHA256 over verified TLS. Concise current-teacher summaries include Cage97/128 and Place96/128; full goal remains14/24.

Wave20 scoreboard yx39wuwn finished and synced14/24. Remote W&B API verified150 experiment-record files including wave20 inventory, actual video reviews and public verification. Website request completed; all24-teacher goal remains active. Next: diagnose captured Place6329/lane995 failure before restart, quantify Edge final failure and Strike calibration, and evaluate current Pivot/Throw/Lift/Peg/Cage runs at their limits. Disk1.6TB free, GPU0 unused, holder expiry unchanged.

### Wave21

Revalidated five full PPO processes onGPU1/2/4/6/7 andGPU0 at0MiB; holder expiry unchanged. Place6329/lane995 reproduction fails at first substep; cold-cache diagnostic passes all4. Added process-local full-Hessian replay ablation to isolate installed incremental Newton optimization without changing shared dependencies. Strike conservative LR refinement preregistered after launch/endpoint audit.

Strike refinement preflight49brwemr completed3 updates, all checkpoint tensors finite, Adam5e-5 confirmed; fullR3 launched original4998 onGPU3. Place full-Hessian ablation still fails. Eager solver audit identifies a small-T elliptic Hessian denominator defect; normalized outer-product form passes the original full-batch failure without clearing warmstart. Independent derivative/kernel tests and strict backend round-trip tests added.

Place passed3-update preflight and100 full-size PPO updates; R7 launched original6300 with cone compatibility. Throw/Pivot completed and strict0/128 with actual clips reviewed. Lift completed, strict0/128 and actual near-goal hold reviewed. Preregistered Lift cone preflight and Throw release-exploration preflight based on recorded failure/intervention evidence. No certification added.

Lift/Throw preflights passed and full retries launched. Peg3900 strict0/128 actual narrower aperture reviewed; R6 continues. Edge unchanged bounded R3 preregistered based on measured approach improvement; GPU1 evaluator has exited.

Wave21 placement verified: one full PPO process perGPU1–7, GPU0 at0MiB, no numerical failure artifacts in these seven current runs. training_wave21.json records source/manifests/PIDs and checkpoint labels. Cone compatibility enabled only for new PlaceR7/LiftR10 lineages; defaults and all14 certified configurations retain their saved backend. Holder expiry unchanged2026-09-28T08:58:54.

Wave21 public refresh verified:24 actual videos play, newest home card/navigation, filters/search/mobile pass,50 public files match local hashes over verified TLS. Reviewed Lift/Throw/Pivot/Peg updates published. Goal remains14/24; seven PPO runs are active.

CageR8 reached12600 with improving training metrics; launched one128-world strict validation beside its sole full trainer onGPU7 (about4.6GB trainer on80GB GPU). No second full trainer. Confirmation remains conditional on116+ successes.

CageR8/model12600 strict90/128, all timeouts, actual success/failure frames reviewed. Below prior97/128; no confirmation, full bounded run continues. LiftR9 exact terminal audit:107 inside5cm,5 settled overall,0 inside and settled; median speeds0.2848m/s and2.0058rad/s. Initial wave21 scoreboard kn3ebe2m finished, remote API verified170 records. Final refresh adds these two late audits/reviews.

Final wave21 gallery includes late Cage12600 clip/rate;24-playback, navigation/filter/search/mobile checks and50 HTTPS SHA256 matches pass after the change. Final W&B rollup includes Cage review and Lift settling audit.

Final wave21 scoreboard1o4fgggg finished; remote API verified172 experiment-record files including cone diagnostics/preflights, late Cage review, Lift settling audit and public checks. Seven Slurm training steps still live at handoff; next completed Cage/Peg budgets require strict evaluation. This goal turn made concrete progress through solver correction, recorded evaluations and finite-preflighted retries; full goal remains14/24, not complete.


### Wave22

**14/24 independent PPO RL teachers certified.** All24 have RL training evidence. Latest completed CageR8/final13095 strict99/128 and StrikeR3/final6997 strict21/128, both small improvements and below the116/128 validation gate. Place retained6300 remains96/128. New Cage/Strike actual success and failure frames reviewed; no confirmation or new certification. GPU0 remains unused.

Current full PPO runs: EdgeR3 GPU1, ThrowR4 GPU2, LiftR10 GPU4, PlaceR7 GPU5 and PegR6 GPU6. GPUs3/7 are free after completed Strike/Cage runs and evaluations. Five process placements verified; GPU0 remains0MiB. PlaceR7 and LiftR10 use the recorded opt-in dense elliptic Hessian correction; PegR6 retains free-body gyro compatibility. All14 certified teachers retain their saved original backend. Holder20277 expires2026-09-28T08:58:54 scheduler time.

Pivot R2 contact audit: median closest reward waypoint6.45mm, wrist error8.74deg, no sampled tilt above20deg. All363 sampled closest robot/board contacts use right pad with mean normal[0.333,-0.004,-0.943], mostly downward. A bounded32-pose IK shift audit finds no inward/upward-normal solution at0–40mm; a separately stated tipping-moment criterion passes25/32 at5mm,1/32 at10mm,0otherwise. This is collision geometry only, without integration or RL success. No new reward, physical model, success predicate or action change. Preserve this distinction before a dynamics probe/new preflight.


Wave22 refresh: reviewed final Cage13095 (99/128) and Strike6997 (21/128) clips published via untu_vps. Newest homepage card links to current RL teachers. Browser checks passed24 actual video playback,14/10 filters, search and mobile layout; all50 public files returnedHTTP200 and matched local SHA256 over verified TLS. No new certification.

Wave22 scoreboard24dce44e finished; remote API verified178 experiment-record files including both new reviews, Pivot contact/shift/tipping audits, current inventory and public verification. Both new video runs finished online. This turn made progress through two strict final evaluations, actual video reviews, geometric diagnosis and verified publication; full24-teacher goal remains active at14/24. Five bounded PPO runs continue. Next: finish Peg/Lift strict evaluations, test Pivot contact dynamics and diagnose Cage/Strike residual failures before further preregistered retries.


### Wave23

Previous turn made progress: final Cage99/128 and Strike21/128 strict results and reviewed videos, Pivot geometric evidence and verified public/W&B publication. Five full trainers revalidated live; GPU0/3/7 at0MiB.

Preregister PREFLIGHT-CAGE-REFINE-037: ownR8/model13095, unchanged cage_v4,64 environments×3 PPO updates, explicit Adam5e-5, preserve optimizer moments/normalizers/policy/gripper and legacy backend. Cage improved97→99/128 over2000 updates but remains below116; a bounded smaller-step refinement tests endpoint consistency. If finite with correct saved LR, launch RL-013-R9-cage-refine from original13095,2048×2000 onGPU7, same settings. Native gate/60D/8D/model unchanged. No confirmation unless strict validation passes116/128.


StrikeR3 endpoint audit:113/128 substantial launches versus97 before; median final error15.82cm,57 undershoots>8cm,41 overshoots>8cm and20 lateral misses>8cm. Peak-speed Coulomb prediction differs from measured endpoint by1.84mm median among substantial launches; peak may still contact hand, so this is empirical support for shaping, not a model proof. The bottleneck is now endpoint calibration. Preregister strike_v3: preserve strike_v2 approach/broad rewards/native20 bonus and add4×(exp(-predicted_error/.05)+exp(-actual_error/.05)/(1+XYspeed/.05)). This rewards precise flight and a settled near-goal endpoint; no gate/model/action changes. PREFLIGHT-STRIKE-PRECISION-038: ownR3/6997,64×3, inherited5e-5 Adam/normalizers/means/std/backend. After finite preflight, RL-023-R4-strike-precision from original6997,2048×2000 onGPU3 after Place evaluator exits.

Cage preflight13097 finite with5e-5 confirmed; fullR9 launched original13095 onGPU7. PlaceR7/7400 strict105/128, below116; full budget continues. PegR6 completed normally at5097; strict evaluation started onGPU6. Pivot constant-control CPU probes preserve actual source states/friction; four initial modes and20 small XY/Z variations all32cases show no tilt>20deg, no warnings or deep obstacle penetration. No Pivot reward change is justified by these negative dynamic probes.


PegR6/final5097 strict0/128, actual low capture reviewed. At last diagnostic123 active episodes, opposing-contact fraction50.4% versus0% at3900, aperture21.82mm, tip height4.71mm. Preregister RL-020-R7-peg-lift: original5097, unchanged peg_v1,2048×2000, preserve saved Adam/normalizers/gripper distributions/LR and gyroON/coneOFF, GPU6 after evaluator exit. The full R6 budget was finite and now acquired physical capture; no new reward or initialization reset and no new preflight needed for this unchanged continuation.

Strike precision tests65 pass (plus10 affected tests after preserving the disabled default branch), and3-update PPO preflight6999 is finite with inherited5e-5. FullR4 launched from original6997 onGPU3.


LiftR10/final11995 strict0/128 (123timeouts,4ground,1bounds), actual near-goal hold reviewed. Exact terminal audit115 inside5cm,0settled; median error11.70mm, linear0.3187m/s, angular2.0946rad/s. The opt-in cone correction alone did not solve policy settling. Existing lift_v7 has only5×continuous quiet credit and position/grasp shaping, no explicit native completion bonus. Preregister lift_v8: retain all existing terms, increase near-goal quiet weight5→15 and add25×the unchanged refreshed native Lift predicate. No success/model/action/backend change. PREFLIGHT-LIFT-COMPLETION-039 from original11995,64×3, preserve Adam1e-4/normalizers/gripper/arm/noise, coneON/gyroOFF. After finite PPO gate launch RL-002-R11-lift-completion, original11995,2048×2000 onGPU4 after evaluator exits.


Lift completion preflightwxlnwsna finished finite at11997 withAdam1e-4, coneON/gyroOFF;71 focused checks passed. FullR11 launched from original11995. Seven full training processes verified onGPUs1–7, GPU0 remains0MiB; current manifests/PIDs in training_wave23.json. No new certification.


Wave23 gallery refresh includes reviewed Place7400 (105/128), Peg5097 and Lift11995 clips. Newest homepage card/navigation,24 actual video playback,14/10 filters, search/mobile layout and50/50 HTTPS SHA256 matches passed. Full details remain in scoped W&B; fourteen certified teachers remain unchanged.

Wave23 scoreboardrjmyos4m finished; remote API verified189 experiment-record files, including all new reviews/audits/preflights, seven-run inventory and public checks. Full goal remains14/24 and active. This turn made progress through Place105/128, Peg capture acquisition, native Lift settling diagnosis, finite-preflighted Cage/Strike/Lift continuations and unchanged Peg continuation. Next bounded completions are Edge/Throw/Place; verify actual live steps before reuse.


### Wave24

Previous turn made progress: Place105/128, actual Peg capture acquisition and Lift settling diagnosis, finite-preflighted Cage/Strike/Lift retries and unchanged Peg continuation; public/W&B records verified. Seven full trainers revalidated live, GPU0 at0MiB.

ReorientR9 actual trajectory/contact audit:12643 sampled states,6723 opposing-contact states within drift bound;6562 (97.6%) have nonpositive axis cosine and zero old orientation credit.124/128 terminal axes remain beyond90deg, median92.54deg; no sampled episode reaches45deg. This is a reward plateau, not a success-rate replacement. Preregister reorient_v8: preserve contact/approach/lift terms, replace max(dot,0)^2 with(1+clamp(dot,-1,1))/2 for the current directed axis (absolute cosine for symmetric variants), still gated by held+valid drift; add25×refreshed unchanged native success. No model/60D/8D/predicate change. PREFLIGHT-REORIENT-AXIS-040: originalR9/8598,64×3, preserve arm/gripper/Adam1e-4/normalizers/noise and gyroON/coneOFF. Tiny preflight may co-locate onGPU1 with its sole2048-world Edge trainer (about4.6GB on80GB). After finite preflight and EdgeR3 final evaluation, launch RL-005-R10-reorient-axis from original8598,2048×2000 onGPU1. No second full trainer per GPU.


Reorient axis preflight completed at8600 with finite tensors, Adam1e-4 and gyroON/coneOFF;63 focused checks passed. FullR10 remains queued until EdgeR3 final budget and strict evaluation exitGPU1.

StackR6 trace audit:118/128 ever show opposing contact, only4 sampled episodes rise above the stack target;104 terminal opposing contacts, median height26.37mm,26.11mm below target and86.73mm XY error. There are zero terminal object/base or robot/base contacts; this does not support a persistent physical obstruction at the endpoint.10229 held samples remain more than20mm below target. R6 was a finite1000-update continuation; arm noise remains nonzero. Preregister RL-018-R7-stack-lift: originalR6/5198, unchanged completion_v6,2048×3000, inherited Adam1e-4/normalizers/gripper and coneOFF/gyroOFF. UseGPU5 after PlaceR7 completes and its strict evaluation/conditional confirmation finish; no second full trainer. This budget tests transport after acquired capture, without a new reward or initialization reset. Failed Place validation remains queued for subsequent refinement; certification criteria unchanged.


Additional Pivot native CPU press/climb grid:32 recorded poses×16 combinations of10–40mm inward and10–40mm upward constant joint-target shifts. All remain below20deg, without warnings/nonfinite states or robot/obstacle penetration beyond3mm. Configured/compiled timestep verified5ms; native actuator gains/biases present. This negative intervention evidence does not justify a new Pivot recipe yet.


EdgeR3/final6997 strict0/128, alltimeouts, actual pressing failure reviewed. Closest exposed pinch improves148.83→39.84mm, predominantly38.61mm vertical with9.76mm horizontal error; wrist error72.39deg. GPU1 evaluator exited and preflighted ReorientR10 launched from original8598.

PlaceR7 completed normally at8299; strict97/128 versus best7400=105/128,127timeouts1bounds, actual clips reviewed. No confirmation. CPU endpoint subcondition reconstruction is diagnostic: at7400 it disagrees with the authoritative GPU outcome on4/128 episodes, so no scores are reclassified. Failure components include containment/floor, angular settling and support contact; best105/128 checkpoint remains retained. GPU5 evaluator exited; proceed with the preregistered StackR7 continuation.

Place8299 CPU endpoint audit identifies angular settling failure in23/31 native failures, containment XY in8 and support contact in8 (conditions overlap). It reconstructs100 successes versus authoritative97, with3 disagreements; no reclassification. Generalized existing hold/release replay to Place and failed-only/stride selection. A30-world constant-control diagnostic co-locates onGPU5 with its sole full Stack trainer, restoring exact trace friction and saved cone backend. This is a CPU/GPU intervention comparison, not new RL evidence. ThrowR4 completed normally at9496; strict evaluation launched onGPU2.


ThrowR4/final9496 improves0→116/128 validation, but confirmation20260915 scores107/128; no certification. Actual validation success/failure clips reviewed; individual confirmation failures not inspected for tuning. Preregister PREFLIGHT-THROW-REFINE-041: original9496, unchanged throw_v4,64×3, Adam5e-5 override with moments/normalizers/learned arm+gripper/noise preserved, coneOFF/gyroOFF. After finite preflight, RL-024-R5-throw-refine original9496,2048×2000 onGPU2. Finish the full budget before validation; this improves the barely-passing validation margin. Retire failed confirmation seed20260915 for this next Throw candidate and freeze20260916 now. Do not retry9496 with a new seed. New candidate must pass both128 batches at116+; Throw-R5-evaluation-plan.json records the prospective schedule. evaluate_candidate accepts an explicit distinct confirmation seed, leaving other task defaults unchanged.
