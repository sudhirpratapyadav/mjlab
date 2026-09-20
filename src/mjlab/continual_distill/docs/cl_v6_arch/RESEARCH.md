# Research — modern training techniques for the shared residual MLP

Curated report from a parallel research pass (2026-09-20), scoped to our
exact setting: small residual MLP, JAX/Flax, supervised regression
(behavior-cloning KL to teacher), trained sequentially per task under SI.
Full citations at the bottom. Not exhaustive by design (user's instruction).

## Adopted this pass (P1, see LOG.md for implementation notes)

- **Gradient clipping** (global norm) — cheap; protects SI's per-parameter
  importance sum (`sum of -g_k . delta_theta_k`) from a single spiking batch
  permanently polluting it.
- **LR warmup + decay per task** — same schedule shape for every task, so
  SI's accumulated importance magnitudes stay comparable across tasks
  (per the report's explicit caveat).
- **Lower Adam beta2** (~0.95-0.98) for short runs — default beta2=0.999 has
  a ~1000-step averaging window; our per-task step counts are in that same
  ballpark, so the second-moment estimate barely warms up. Made it a flag,
  default lowered.
- **Near-zero init on the second Dense of each residual block** (SkipInit
  family) — residual branches start close to identity; cheap, well
  understood, becomes more relevant as depth grows. We're already pre-norm,
  so the marginal benefit is smaller than for post-norm nets per the report,
  but it's essentially free.
- **Weight decay stays at 0**, explicitly, not by accident — the report
  flags weight decay as directly fighting the SI anchor penalty (decay pulls
  to 0, SI pulls to the previous task's optimum). Documented so a future
  pass doesn't "fix" this by adding decay without re-centering it at the SI
  anchor.

## Queued for P2 (controlled A/B before adopting, not blind defaults)

- **EMA of weights for eval/deployment** — report calls this the easiest
  win: decouple entirely from SI (SI reads the raw training trajectory;
  EMA only used at eval/checkpoint-for-render time). Worth adding once P1's
  changes are validated.
- **Muon optimizer** for hidden Dense layers (AdamW/Adam kept for the task
  embedding table + output head) — genuinely relevant evidence beyond LLM
  hype (2025 MLP-benchmark paper, implicit-neural-representation work), but
  the report is explicit that Muon's whitened, cross-parameter update
  produces different weight-delta statistics than the diagonal-Adam updates
  essentially all published SI/EWC results assume. Needs an explicit A/B
  checking that anti-forgetting on early tasks still holds, not a default
  swap.
- **muP (maximal update parametrization)** — directly targets the width-16k
  training failure we already hit on the old architecture, but is a
  moderate implementation lift (per-layer LR/init multipliers keyed to
  fan-in). Deferred: this pass is about residual depth, not an aggressive
  width push; revisit if/when we deliberately scale width again.

## Explicitly skipped, with reason

- **Dropout / stochastic depth** — real CL evidence exists but in
  classification settings; directly conflicts with our hard requirement
  that SI's online importance integral sees a low-noise gradient/delta
  trajectory. Skip by default.
- **BatchNorm** (any form) — documented to actively hurt continual learning
  (running stats bias toward the current task). We're already on LayerNorm,
  which has no cross-batch running state — correct as-is, don't reconsider.
- **Sophia, Lion** — Sophia's benefit is scale-dependent (unproven at our
  size) and adds Hessian-diagonal noise near SI; Lion's sign-based updates
  decorrelate update magnitude from gradient magnitude, which the SI
  importance term (`grad . delta_theta`) depends on directly.
- **Schedule-free optimizers** — interesting (removes the need to fix a
  schedule length per task) but less battle-tested outside LLM/AlgoPerf
  contexts, and its internal iterate-averaging changes which "point" SI
  should read gradients from. Lower priority than Muon/muP.
- **Distributed-training infra, large-batch scaling rules, image
  augmentation** — not applicable at our scale/domain.
- **RMSNorm over LayerNorm** — marginal-at-best evidence either way; not
  worth the churn.

## Full report

<details>
<summary>Agent's complete curated report (click to expand)</summary>

### 1. Well-established, low-risk — adopt

**Gradient clipping (global-norm, e.g. 1.0).** Standard defense against gradient outliers (Pascanu et al. 2013). Low risk, cheap. For you it's extra important: SI's importance estimate is a running sum of `-g_k . Delta_theta_k` per parameter, and a single gradient spike (e.g., from a batch where the teacher's log-std is near a numerical extreme) permanently pollutes that sum for the rest of training. Clip on the *raw* loss gradient used for both the optimizer step and (if your SI implementation reads the applied gradient) the importance accumulation -- just be consistent about which gradient SI sees.

**LR warmup (short, few hundred steps) + decay (cosine or linear) per task.** Warmup is theoretically motivated as adaptive-LR variance reduction (Liu et al., *RAdam*, 2019) -- Adam's second-moment estimate is unreliable on the first few steps, especially right after a task switch when the loss landscape has jumped. Decay to a small floor over each task's fixed epoch budget is standard practice for short, non-open-ended runs like yours. **Caveat:** use the *same* schedule shape/hyperparameters for every task in the sequence -- varying it task-to-task makes SI's accumulated importance magnitudes non-comparable across tasks.

**AdamW (decoupled weight decay) instead of coupled Adam+L2.** Loshchilov & Hutter, ICLR 2019 -- well established that decoupling weight decay from the adaptive step is what makes weight decay behave like actual regularization under Adam. **Big caveat for you, see #4**: any nonzero weight decay actively fights the SI anchor penalty. Recommend switching the *implementation* to AdamW for hygiene, but keep `weight_decay~=0` unless you also re-center it.

**beta2 tuning for short runs.** Default `beta2=0.999` has an effective averaging window of ~1000 steps. With 100-500 epochs over a fixed offline dataset your per-task step count may be in that same ballpark, meaning the second-moment estimate barely warms up before the task ends. Lowering `beta2` to ~0.95-0.98 for short runs is a well-known practical fix (used e.g. in nanoGPT-style short-schedule setups); low risk, worth a quick sweep. `beta1=0.9`, `eps=1e-8` (or `1e-6` if you see LayerNorm-related instability) are fine as-is.

**EMA of weights for eval/deployment (Polyak averaging).** Well-established variance reduction (Polyak & Juditsky 1992; recent analysis in Morales-Brotons et al., "EMA of Weights in Deep Learning," 2024) -- smoother, better-calibrated eval weights essentially for free. **Good synergy with SI:** compute SI's importance integral over the raw ("fast") training trajectory as usual, and only apply EMA at eval/export time. This gets you the stability benefit without touching SI's bookkeeping at all -- arguably the single easiest win on this list.

**Residual-branch down-scaling at init (Fixup/SkipInit/DeepNorm family).** Fixup (Zhang, Dauphin, Ma, ICLR 2019) and SkipInit (De & Smith 2020) zero- or near-zero-initialize the last layer of each residual branch (or a learned scalar gate) so identity dominates early training; DeepNorm (Wang et al. 2022) generalizes this with depth-dependent scale factors and proved it out to 1000+ layers. You're pre-norm (LayerNorm before the branch), which already buys most of the gradient-scale benefit these methods target, so this is a smaller marginal win for you than for post-norm nets -- but it's cheap, well-understood, and becomes more relevant as you go to 8 blocks. Note this literature is almost entirely CNN/Transformer-validated, not MLP-specific, so treat the size of the benefit as somewhat uncertain for your architecture.

**LayerNorm over RMSNorm -- keep what you have.** RMSNorm (Zhang & Sennrich, 2019) drops the mean-centering step for a modest speed win (7-64% faster in various implementations) but the accuracy difference is small and can go either way; one MLP-Mixer comparison found LayerNorm slightly *better* than RMSNorm. At your scale, compute isn't the bottleneck, and centering may matter more for a low-dimensional (60-dim obs + 32-dim embedding) input than in huge hidden dims. **Skip switching -- not worth the churn.**

### 2. Promising newer techniques -- worth a controlled trial

**Muon optimizer** (Keller Jordan et al., late 2024, actively developed through 2025). Instead of Adam's per-coordinate scalar normalization, Muon orthogonalizes the update direction of each 2D weight matrix (via a Newton-Schulz iteration approximating spectral/steepest descent under the spectral norm), applied only to hidden 2D Dense weights -- biases, norm gains, embeddings, and the input/output layers are still handled by AdamW. This is squarely aimed at exactly your architecture: a stack of wide Dense layers. Evidence base beyond LLM pretraining hype now includes a dedicated MLP benchmark ("Benchmarking Optimizers for MLPs in Tabular Deep Learning," 2025 -- Muon beat AdamW consistently across MLP variants) and implicit-neural-representation work on plain ReLU MLPs -- genuinely relevant, not just transformer folklore. **Gotchas:** needs its own LR (typically a different scale than Adam's), still needs AdamW for the task-embedding table and output head, and -- important -- its update mixes information across an entire weight row/column rather than being per-parameter independent, which is an *untested* combination with SI's per-parameter importance bookkeeping. Treat as an A/B experiment, not a default swap.

**Maximal Update Parametrization (muP)** (Yang & Hu et al., "Tensor Programs V," 2021; now well-established practice at labs like OpenAI/Cerebras, with active 2024-2025 extensions). muP prescribes width-dependent init-variance and per-layer LR scaling so that the *same* base LR remains near-optimal as width grows. This directly targets the exact failure mode you already hit: width 16384 not training at the LR tuned for 4096-8192. Implementation is a moderate lift (per-layer multipliers keyed to fan-in) but is mechanical, well-documented (`microsoft/mup` reference repo), and specifically de-risks further width scaling rather than requiring you to rediscover LR/init pairings by trial and error each time you scale. A 2025 paper ("Weight Decay may matter more than muP for LR Transfer," arXiv 2510.19093) suggests the picture isn't fully settled -- flag this as "strong evidence, some open questions" rather than gospel.

**Schedule-free optimizers** (Defazio et al., "The Road Less Scheduled," Meta, 2024 -- won the MLCommons AlgoPerf 2024 self-tuning track). Uses a Polyak/Nesterov-style iterate averaging that removes the need to commit to a schedule length in advance. Interesting for your setting because you retune a schedule 15-24 times (once per task); schedule-free removes that per-task decision. Less consensus than the above: the paper itself notes sensitivity to momentum hyperparameters remains, and it's much less battle-tested outside LLM/AlgoPerf contexts. Also its internal parameter sequence (interpolated "y"/"z" points) differs from the raw trajectory -- if adopted, be deliberate about which point SI reads gradients/deltas from. Lower priority than Muon or muP.

### 3. Deprioritize / skip

- **Sophia** (diagonal-Hessian second-order optimizer): its claimed step-count savings grow with model/step scale; at your few-hundred-epoch, few-million-to-50M-param regime the benefit is unproven, and its Hessian-diagonal estimate adds exactly the kind of extra stochasticity you want to avoid near SI. Skip.
- **Lion** (sign-momentum optimizer): efficient in GPU-hours, but AdamW usually still wins on downstream quality, and Lion's sign-based updates decorrelate update magnitude from gradient magnitude -- bad for the `grad . delta_theta` product SI depends on. Skip.
- **Dropout / stochastic depth / other stochastic regularization**: real evidence it helps continual learning *in some settings* via implicit task-specific gating (Abbasi et al., "Dropout as an Implicit Gating Mechanism for CL," 2020) -- but that literature is classification/task-incremental, and it directly conflicts with your explicit hard requirement (SI's online importance integral assumes a comparatively clean, low-noise gradient/delta trajectory). Skip by default; only revisit as a deliberate, isolated ablation.
- **BatchNorm** (any form): well-documented to actively hurt continual learning -- its running statistics become biased toward whichever task is currently training, corrupting eval on prior tasks (Pham et al., "Continual Normalization," 2022; Lesort et al., "Diagnosing BN in Class-Incremental Learning," 2022). You're already on LayerNorm, which computes per-example statistics with no cross-batch/cross-task running state -- this is the right call; don't reconsider BN.
- **Distributed-training infra** (ZeRO/FSDP, LARS/LAMB large-batch scaling rules, linear-scaling batch/LR rules): irrelevant at your parameter count and single-GPU-scale batch sizes.
- **Image-domain data augmentation** (mixup/cutmix/RandAugment): not applicable to continuous state-action regression.

### 4. Continual-learning + SI interaction warnings

- **Weight decay vs. the SI anchor**: direct conflict. Weight decay pulls every parameter toward 0 each step; SI's quadratic penalty pulls each parameter toward its value at the end of the *previous* task. These fight whenever the previous-task optimum isn't near zero (essentially always). If you want weight decay's benefits, re-center it at the SI anchor point rather than the origin (this is the fix noted in the recent EWC-focused literature, e.g. "Elastic Weight Consolidation Done Right," 2026) -- otherwise keep it at 0, which is the current (correct-by-accident) setup.
- **BatchNorm running stats vs. task boundaries**: confirmed failure mode -- reaffirm LayerNorm as the safe choice; its lack of cross-batch running state is precisely why it doesn't have this problem.
- **Dropout/stochastic-depth vs. SI's online importance bookkeeping**: unstudied combination in the specific SI/EWC context, but mechanistically the added per-step noise directly corrupts the `grad . delta_theta` path integral SI accumulates online -- treat as unsafe until proven otherwise in a controlled test.
- **Gradient clipping vs. SI**: not a conflict if you're careful -- SI's original formulation (Zenke, Poole, Ganguli, ICML 2017) accumulates importance from the actual loss gradient at each step, independent of what clipping does to the applied optimizer update; just make sure the implementation is explicit about which gradient (pre- or post-clip) feeds the importance sum, and stays consistent across tasks.
- **Muon vs. SI**: mathematically SI's per-parameter importance sum doesn't require any particular optimizer, so it's *formally* compatible -- but Muon's whitened, cross-parameter update produces very different `Delta_theta` statistics than the diagonal Adam updates that essentially all published EWC/SI results are built on. Don't assume forgetting-control transfers; validate it explicitly (e.g., check that anti-forgetting metrics on early tasks hold up) if adopted.
- **EMA vs. SI**: no conflict if kept decoupled -- compute/accumulate SI importance on the raw training trajectory, use EMA weights only at eval/deployment time.

### Sources

- [Fixup Initialization (arXiv:1901.09321)](https://arxiv.org/abs/1901.09321)
- [SkipInit / De & Smith discussion (ICLR blog track)](https://iclr-blog-track.github.io/2022/03/25/unnormalized-resnets/)
- [DeepNet: Scaling Transformers to 1,000 Layers (arXiv:2203.00555)](https://arxiv.org/pdf/2203.00555)
- [Decoupled Weight Decay Regularization / AdamW (arXiv:1711.05101)](https://arxiv.org/pdf/1711.05101)
- [On the Variance of the Adaptive Learning Rate and Beyond / RAdam (arXiv:1908.03265)](https://arxiv.org/pdf/1908.03265)
- [Exponential Moving Average of Weights in Deep Learning (arXiv:2411.18704)](https://arxiv.org/pdf/2411.18704)
- [Root Mean Square Layer Normalization (arXiv:1910.07467)](https://arxiv.org/pdf/1910.07467)
- [Muon optimizer overview (Keller Jordan)](https://kellerjordan.github.io/posts/muon/)
- [Benchmarking Optimizers for MLPs in Tabular Deep Learning (arXiv:2604.15297)](https://arxiv.org/pdf/2604.15297)
- [The Road Less Scheduled / Schedule-Free (arXiv:2405.15682)](https://arxiv.org/abs/2405.15682)
- [Sophia optimizer (arXiv:2305.14342)](https://arxiv.org/pdf/2305.14342)
- [Continual Learning Through Synaptic Intelligence (arXiv:1703.04200)](https://arxiv.org/abs/1703.04200)
- [Dropout as an Implicit Gating Mechanism for CL (arXiv:2004.11545)](https://arxiv.org/pdf/2004.11545)
- [Continual Normalization: Rethinking BN for Online CL (arXiv:2203.16102)](https://arxiv.org/pdf/2203.16102)
- [Elastic Weight Consolidation Done Right for CL (arXiv:2603.18596)](https://arxiv.org/html/2603.18596v1)
- [Maximal Update Parametrization / muP (microsoft/mup)](https://github.com/microsoft/mup)
- [Weight Decay may matter more than muP for LR Transfer (arXiv:2510.19093)](https://arxiv.org/abs/2510.19093)

</details>
