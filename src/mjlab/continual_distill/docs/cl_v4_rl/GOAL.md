# Goal — 24 RL teachers above 90%

Produce one retained RL checkpoint and observation normalizer for each of the 24 tasks in `active_tasks.json`. Each teacher must achieve **strictly greater than 90% success on its own task**. An average above 90% across tasks does not meet this goal. This phase trains separate RL teachers; continual distillation and a shared multitask student are later phases.

## Acceptance protocol

- Use the task's actual success predicate and an uninterrupted first episode per sampled initial condition. Do not count retries after automatic reset as additional chances in the same episode.
- Evaluate deterministic policy means with the same 60D observation and normalized 8D action contract. Match train/test task geometry, initialization distribution, physics and episode limits.
- Proposed acceptance gate: at least **116/128 successes (90.625%)** on held-out initial conditions, followed by another independent 128-episode confirmation batch also above 90%. Freeze and record evaluation seeds before evaluation; do not tune on the confirmation batch. This is a measured-rate criterion, not a statistical guarantee that the true rate exceeds 90%.
- Report counts, rate, evaluation seeds, checkpoint/hash, interface version, effective config, and termination breakdown. Keep success/failure videos and inspect high-return failures for reward exploitation.
- Retain one best verified checkpoint per task, with optimizer and normalizer for resuming, and a deterministic evaluation entry point. Previous scripted-teacher rates do not certify an RL teacher.

## Fixed scope

24 active tasks; Tool-Pull is deferred because the selected observation omits its independent stick pose. Keep all task code and registration. The four extra object variants are also outside this CL suite.

The user selected the working **60D field layout** and identical normalized **8D absolute joint-position actions**. Do not reintroduce the rejected 143D expansion or silently change the action meaning. Task-specific tuning may change training parameters or reward shaping, but record the reason, before/after behavior and validation. Changes to physics or success semantics create a new benchmark configuration and require renewed verification.
