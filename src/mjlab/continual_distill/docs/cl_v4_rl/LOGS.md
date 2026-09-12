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
