# Survey: Continual / Lifelong Learning for Robotic Manipulation

Compiled 2026-07-11 via a deep-research pass (fan-out web search + adversarial
verification) plus targeted gap-filling searches. Scope: agents that learn
manipulation tasks **sequentially over time** — resisting catastrophic forgetting,
achieving forward/backward transfer, adapting fast — **including** work that does
this without "continual learning" in the title (skill libraries, sequential
multi-task, meta-RL fast adaptation, generalist/VLA policies with continual eval).

> Verified findings (3-vote adversarial check) are marked ✓VERIFIED. Gap-filled
> rows from targeted follow-up searches are marked ⋯sourced but not independently
> triple-verified. Where a "beats SOTA by N%" figure appears it is the authors'
> own single-benchmark headline unless noted.

---

## 1. Taxonomy of approaches

Continual-learning method families, as applied to manipulation. Most strong modern
methods are **hybrids** (e.g. a modular architecture that *also* uses replay).

**A. Regularization-based** — add a penalty that discourages moving weights
important to old tasks. EWC (Kirkpatrick 2017, Fisher-weighted), SI (Zenke 2017,
path-integral importance — the family used in this repo's own distillation work),
MAS (Aljundi 2018). On manipulation these are the standard *baselines* in
Continual-World and LIBERO; they prevent forgetting but tend to underperform on
forward transfer and cost plasticity.

**B. Replay / rehearsal** — store (or generate) old-task data and interleave it.
Experience Replay (ER) is the simplest and remains a **strong baseline**; CLEAR
(Rolnick 2019) is the canonical continual-RL replay method. For large pretrained
VLAs, replaying only ~2% of data suffices to nearly eliminate forgetting
(✓VERIFIED — "Pretrained VLAs are Surprisingly Resistant…"). Generative replay
avoids storing raw data.

**C. Architecture / modular** — dedicate parameters per task/skill so old knowledge
is structurally isolated. Progressive Neural Networks (Rusu 2016, incl. Jaco-arm
transfer), PackNet (Mallya 2018, iterative pruning+masking), Modulating Masks
(✓VERIFIED, fixed backbone + per-task masks, linear mask combination for transfer).
**This family dominates recent SOTA on LIBERO**:
- **LOTUS** (✓VERIFIED, ICRA 2024) — unsupervised continual skill discovery builds
  an ever-growing skill library; +11% over continual-IL SOTA on LIBERO; real robot.
- **TAIL** (✓VERIFIED, ICLR 2024) — per-task PEFT adapters (LoRA/bottleneck/prefix);
  TAIL-LoRA ~1% params, BWT≈0.
- **IsCiL** (✓VERIFIED, ICLR 2025) — prototype-based skill retrieval + isolated
  per-skill LoRA adapters (skill-level, shareable across tasks).
- **DMPEL** (✓VERIFIED, 2025) — growing LoRA expert library + router; "expert
  coefficient replay" (~6.7% overhead) reactivates frozen experts.

**D. Meta-learning / fast adaptation** — learn to adapt quickly to new tasks (MAML,
Reptile, PEARL, RL²). Not anti-forgetting per se, but central to the "adapt to new
tasks" half of the ask; Meta-World's ML10/ML45 are the standard meta-RL manipulation
suites.

**E. Foundation-model / VLA continual finetuning** — sequentially finetune large
pretrained policies (RT-X, OpenVLA, Octo, Pi0, GR00T). Key ✓VERIFIED finding: large
pretrained VLAs forget far less than from-scratch policies (Pi0 NBT −0.016, GR00T
0.027 vs BC-Transformer 0.245 with 2% ER). BUT on **real-world heterogeneous** task
sequences even pi0.5 degrades significantly (2026 real-robot study, ContinualVLA).

**F. Distillation-based** — transfer old-task behavior into the new model via a KL /
matching loss (this repo's own approach: per-task teachers → shared student with SI).
Often combined with replay.

**G. Dynamics / world-model CL** — continual learning of the world model (Dreamer-
style) rather than the policy. Underrepresented in verified manipulation evidence.

**H. Task-free / boundary-agnostic** — no task IDs at train or test time. **RWLA**
(✓VERIFIED) — ER + retrieval-based local adaptation before deployment, reusing the
replay memory; evaluated on LIBERO.

---

## 2. Benchmarks & simulators deep-dive

| Benchmark | Sim | #Tasks | What it tests | Key users |
|---|---|---|---|---|
| **LIBERO** (NeurIPS 2023 D&B) | robosuite/MuJoCo | **130** in 4 suites (Spatial, Object, Goal, LIBERO-100/-Long); +LIBERO-90 | Lifelong knowledge transfer: declarative vs procedural vs mixed. FWT/BWT/AUC. **Finding: plain sequential finetuning beats dedicated CL methods on forward transfer.** ✓VERIFIED | LOTUS, TAIL, IsCiL, DMPEL, RWLA, VLA-forgetting studies |
| **Continual World** (NeurIPS 2021) | MuJoCo (Meta-World) | CW10 / **CW20** (10/20 task sequence) | Continual RL: catastrophic forgetting, forward transfer, capacity-vs-retention trade-off. ✓VERIFIED | EWC/PackNet/replay baselines; continual-RL methods |
| **Meta-World** (CoRL 2019) | MuJoCo | MT10/MT50 (multitask), ML10 (10+5), ML45 (45+5) meta-RL | Multi-task & meta-RL fast adaptation; substrate for Continual World | Meta-RL & multitask methods |
| **RLBench** | CoppeliaSim/PyRep | 100 tasks | Multi-task manipulation; used as CL substrate in some works | (to fill) |
| **CausalWorld** | PyBullet | configurable | Systematic generalization / transfer | (to fill) |
| Real-robot setups | — | few (e.g. 4 sequential tasks, 500 traj each) | Real-world continual VLA; heterogeneity causes significant forgetting | ContinualVLA (2026), LOTUS (real) |

---

## 3. Consolidated comparison table

Columns: Paper · Category · Method · Benchmark · #Tasks · Sim/Real · Anti-forgetting
mechanism · Key result · Code.

| Paper (venue, yr) | Category | Method | Benchmark | #Tasks | Sim/Real | Anti-forget | Key result | Code |
|---|---|---|---|---|---|---|---|---|
| **LIBERO** (NeurIPS'23) | benchmark | — | (defines) | 130 | Sim | ER/EWC/PackNet baselines | seq-FT > CL methods on FWT ✓ | [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) ✓ |
| **Continual World** (NeurIPS'21) | benchmark | — | Meta-World | 10/20 | Sim | (evaluates families) | forgetting/FWT/capacity metrics ✓ | [awarelab/continual_world](https://github.com/awarelab/continual_world) ✓ |
| **LOTUS** (ICRA'24) | architecture/skill-lib (+replay) | unsup. continual skill discovery | LIBERO | (LIBERO suites) | Real+Sim | skill-library update + ER | +11% vs continual-IL SOTA ✓ | [UT-Austin-RPL/Lotus](https://github.com/UT-Austin-RPL/Lotus) ✓ |
| **TAIL** (ICLR'24) | architecture/adapter | per-task PEFT (LoRA/bottleneck/prefix) | LIBERO | (suites) | Sim | per-task adapter isolation | BWT≈0, ~1% params ✓ | (check) |
| **IsCiL** (ICLR'25) | architecture/adapter | prototype skill retrieval + per-skill LoRA | LIBERO-style | (skills) | Sim | skill-level adapter isolation | skill-shareable, low-shot ✓ | (check) |
| **DMPEL** (2025) | architecture/expert-lib (+coef replay) | LoRA expert library + router | LIBERO | (suites) | Sim | expert-coefficient replay (~6.7% oh) | beats SOTA, min params ✓ | [HarryLui98/DMPEL](https://github.com/HarryLui98/DMPEL) ✓ |
| **Modulating Masks** (TMLR'23) | architecture | fixed backbone + per-task masks | RL (PPO/IMPALA) | many | Sim | mask isolation; linear mask combine → FWT | solves sparse-reward unsolvable from scratch ✓ | (check) |
| **RWLA** (2024) | task-free replay | ER + retrieval local adaptation | LIBERO | (suites) | Sim | ER + retrieval "review" | task-free, no extra storage ✓ | (check) |
| **Pretrained-VLAs-Resist-Forgetting** (2026) | VLA replay | seq-FT + 2% ER | LIBERO | (suite seq) | Sim | small ER | Pi0 NBT −0.016 vs BC-Tf 0.245 ✓ | continual-vlas.github.io |
| **ContinualVLA / "VLA from real data?"** (2026) | VLA replay (real) | seq-FT + ER (impl. factors) | real robot | 4 | Real | ER | pi0.5 still degrades on heterogeneous real data ✓ | continual-vlas.github.io |

### Recent (2021–2026) — VLA / LIBERO-era ⋯sourced

| Paper (venue, yr) | Category | Method | Benchmark | #Tasks | Sim/Real | Anti-forget | Key result | Code |
|---|---|---|---|---|---|---|---|---|
| **PHASER** (2026) | replay (VLA) | phase-aware semantic ER + interference router | LIBERO CL (3 VLA backbones) | suites | Sim | phase-centric replay buffer | +31% ASR vs matched ER; 87.8% LIBERO-Goal | — |
| **CRL-VLA** (2026) | regularization (VLA) | dual-critic advantage regulation | LIBERO | suites | Sim | frozen critic anchors semantics | beats baselines fwt+forget | — |
| **VLA-Natural-CL-w/-RL** (2026) | VLA-CL via RL | RL fine-tuning | LIBERO (+ManiSkill2/RoboCasa) | — | Sim | RL FT naturally mitigates forgetting | > CL baselines retention+FWT | [UT-Austin-RobIn/continual-vla-rl](https://github.com/UT-Austin-RobIn/continual-vla-rl) |
| **REGEN** (2026) | generative replay / world-model | World-Action-Model pseudo-replay | sim + real (xArm7) | — | Both | recurrent generative replay | forgetting 96.3→60.5, FWT 50→80 (real) | — |
| **CLARE** (RA-L 2026) | architecture/adapter (VLA) | autonomous adapter routing+expansion | LIBERO + 5 real tasks | — | Both | modular adapters, no task-ID | no forgetting; beats exemplar methods | [tum-lsy.github.io/clare](https://tum-lsy.github.io/clare) |
| **M2Distill** (ICRA'25) | distillation | multi-modal latent + GMM-policy distill | LIBERO (Obj/Goal/Spatial) | 30 | Sim | latent-shift regulation | > prior SOTA all metrics | — |
| **SPECI** (2025) | skill-lib/prompt | expandable skill codebook + attn select | LIBERO-style | — | Sim | skill-codebook + mode-approx | strong bidir transfer | — |
| **PPL (Primitive Prompt Learning)** (CVPR'25) | architecture (prompt)+replay | frozen motion primitives + new prompts | large skill set | — | Both | prompt-freeze + ER | > SOTA | — |
| **Stellar-VLA** (2025) | architecture (MoE)+replay | DP nonparam + expert routing | LIBERO-goal/-long/-30 + real dual-arm | 10 (+3 real) | Both | expert routing + ~1% ER | ~50% success-rate gain | "soon" |
| **LiMoDE** (2026) | architecture (MoE)+skill-lib | dynamic-MoE, freeze prior experts | manipulation | — | Sim | frozen-expert + new lifelong experts | > SOTA, moderate overhead | — |
| **OMLA** (2025) | meta-learning+adapter | online meta-learned adapters | sim + real | — | Both | meta-objective across adapters | > adapter baselines | [ricky-zhu.github.io/OMLA](https://ricky-zhu.github.io/OMLA/) |
| **SkillsCrafter** (AAAI'26) | architecture/regularization | SVD skill-semantic subspace proj | sim + real | — | Both | subspace projection retention | > SOTA | — |
| **iManip** (2025) | replay+architecture | temporal replay + extendable PerceiverIO | RLBench | — | Sim | temporal replay + action prompt | mitigates severe forgetting | promised |
| **CRIL** (IROS'21) | generative replay | GAN traj gen + dynamics prediction | sim manip (pre-LIBERO) | — | Sim | pseudo-trajectory replay | foundational CIL baseline | — |
| **Continual Diffuser (CoD)** (2024) | replay+diffusion | rehearsal-buffer diffusion | 90-task offline suite | 90 | Sim | rehearsal | strong plasticity-stability | — |
| **Sparse Diffusion Policy (SDP)** (2024) | architecture (MoE)+diffusion | sparse per-task expert activation | multitask/continual | — | Sim | freeze prior experts, add new | no forget, sparse params | — |
| **SANE** (CoLLAs'22) | architecture (ensemble), task-free | self-activating neural ensembles | procedural visual RL | — | Sim | auto-activate/freeze modules | task-agnostic continual RL | — |
| **Continual-Dreamer** (2022) | world-model, task-free | Dreamer + reservoir replay | continual RL | — | Sim | world-model + reservoir replay | world models aid continual RL | — |
| **Life-Long World Model** (2023) | world-model | DreamerV2 + imagined replay | continual visual RL | — | Sim | frozen world-model imagined replay | continual visual RL | — |

### Classic era (2014–2020) — foundations & manipulation-adjacent ⋯sourced

| Paper (venue, yr) | Category | Method | Benchmark | #Tasks | Sim/Real | Anti-forget | Key result | Code |
|---|---|---|---|---|---|---|---|---|
| **PG-ELLA** (ICML'14) | skill-lib/shared-basis (RL) | online shared policy dictionary + sparse coeffs | control (cartpole etc.) | multi | Sim | shared basis + Hessian-stable coeffs | foundational lifelong-RL | — |
| **Cross-domain PG-ELLA** (IJCAI'15) | skill-lib (cross-domain RL) | shared repo + per-domain projections | cross-domain RL | multi | Sim | shared knowledge + projections | first cross-domain lifelong RL | — |
| **PG-ELLA search&rescue** (2015) | skill-lib (mobile robot) | PG-ELLA | robot terrain tasks | few | Sim(robot) | shared basis | accelerates task-seq learning | — |
| **Progressive Neural Nets** (2016) | architecture | frozen columns + lateral connections | Atari/maze/Jaco | seq | Sim | new column/task (zero forget) | immune to forgetting, +FWT | — |
| **Prog-Nets Sim2Real (Jaco)** (CoRL'17) | architecture/transfer | progressive columns sim→real | MuJoCo + real Jaco | few | **Real** | frozen sim col + new real col | classic real-robot CL/transfer | — |
| **EWC** (PNAS'17) | regularization | Fisher-weighted quadratic penalty | MNIST/Atari (→CW) | multi | Sim | anchor weights via Fisher | near-single-task retention | — |
| **SI** (ICML'17) | regularization | path-integral importance penalty | MNIST/CIFAR (→CW) | multi | Sim | online importance surrogate | EWC-level, cheaper (**this repo's family**) | 3rd-party |
| **MAS** (ECCV'18) | regularization | output-sensitivity importance (unsup) | vision (→CW) | multi | Sim | unsupervised importance | label-free importance | [rahafaljundi/MAS](https://github.com/rahafaljundi/MAS-Memory-Aware-Synapses) |
| **GEM** (NeurIPS'17) | replay (constraint) | episodic memory gradient projection | MNIST/CIFAR | 20 | Sim | project grad vs stored tasks | +backward transfer | [facebookresearch/GradientEpisodicMemory](https://github.com/facebookresearch/GradientEpisodicMemory) |
| **A-GEM** (ICLR'19) | replay (constraint) | avg-gradient single constraint | vision (→CW) | many | Sim | avg-memory gradient | GEM acc, cheaper | [facebookresearch/agem](https://github.com/facebookresearch/agem) |
| **CLEAR** (NeurIPS'19) | replay (continual RL) | off-policy replay + BC + on-policy | DMLab/Atari | seq | Sim | replay + behavioral cloning | forgetting ~eliminated, task-free | — |
| **PackNet** (CVPR'18) | architecture (param-isolation) | iterative prune + per-task masks | vision (→CW) | multi | Sim | disjoint masked subnets | **best in Continual-World (0.80)** | [arunmallya/packnet](https://github.com/arunmallya/packnet) |
| **RCL** (NeurIPS'18) | architecture (learned expand) | RL-controlled neuron addition | MNIST/CIFAR | multi | Sim | grow + freeze old | adaptive expansion | — |
| **Progress & Compress** (ICML'18) | distillation+regularization | active col → distill → online-EWC KB | Omniglot/Atari/maze | seq | Sim | distill + online-EWC | fixed-capacity, no replay | — |
| **H-DRLN** (AAAI'17) | architecture+distill (skills) | deep skill array + skill distillation | Minecraft | multi | Sim | frozen skill modules + distill | canonical skill-library lifelong RL | — |
| **MAML** (ICML'17) | meta-learning | learned init for fast adaptation | few-shot/RL (→Meta-World) | dist | Sim | — (fast adapt, not retention) | SOTA few-shot adaptation | [cbfinn/maml](https://github.com/cbfinn/maml) |
| **RL²** (2016) | meta-learning | RNN-encoded fast RL algorithm | bandits/MDPs/maze | dist | Sim | — (fast adapt) | RNN meta-learner rivals hand algos | — |
| **Reptile** (2018) | meta-learning | first-order MAML | few-shot | dist | Sim | — (fast adapt) | near-MAML, cheaper | [openai/supervised-reptile](https://github.com/openai/supervised-reptile) |
| **PEARL** (ICML'19) | meta-learning (context) | probabilistic task-latent + SAC | MuJoCo meta-RL | dist | Sim | — (fast adapt) | 20-100× more sample-efficient | [katerakelly/oyster](https://github.com/katerakelly/oyster) |
| **Continual World** (NeurIPS'21) | benchmark | evaluates EWC/MAS/A-GEM/PackNet/VCL on SAC | Meta-World (Sawyer) | CW10/CW20 | Sim | full spectrum | PackNet best 0.80; EWC/MAS FWT −0.17/−0.52 | [awarelab/continual_world](https://github.com/awarelab/continual_world) |

> "→CW" = introduced on vision/RL but became a **Continual-World manipulation
> baseline**. Meta-learning rows have no anti-forgetting mechanism — they target
> *fast adaptation*, the "adapt to new tasks" leg of the survey, not retention.

---

## 4. Cross-cutting takeaways (from verified findings)

1. **Forget-prevention ≠ transfer.** LIBERO's headline: dedicated CL methods prevent
   forgetting better, but plain **sequential finetuning wins on forward transfer**.
   Continual-World shows the same trade-off from the RL side: regularizers drive
   forgetting to ~0 but their forward transfer goes *negative* (EWC −0.17, MAS −0.52),
   while **PackNet (parameter isolation) is the best overall at 0.80**. Isolate-don't-
   penalize is the classic lesson; modern skill/adapter libraries are its descendants.
2. **Scale buys stability.** Large pretrained VLAs forget 2–4× less than from-scratch
   policies; small replay closes most of the remaining gap — *in sim*.
3. **Real-world breaks the optimism.** On heterogeneous real-robot task sequences,
   even strong VLAs (pi0.5) degrade significantly — the sim finding doesn't transfer
   cleanly.
4. **Architecture/modular dominates recent SOTA** (skill libraries, adapter/expert
   libraries), but nearly all are **hybrids** that also replay.

## 5. Open questions (from the research)

- How do classic methods (PG-ELLA, progressive nets, PackNet, EWC/SI/MAS) compare
  head-to-head with modern adapter/skill-library & VLA-replay methods on a common
  benchmark (LIBERO / Continual-World)?
- Does "pretrained VLAs resist forgetting w/ small replay" hold at real-world scale?
- For **forward transfer** specifically, which family actually delivers positive
  transfer rather than merely preventing forgetting?
- Generalization beyond LIBERO/Meta-World to **dexterous-hand** and **mobile
  manipulation** sequences (underrepresented).

---

*Coverage: ~50 papers across the taxonomy (10 adversarially-verified from the
deep-research pass + ~40 from targeted classic-era and recent-VLA follow-up searches).
Sources: arXiv, NeurIPS/ICML/ICLR/ICRA/CVPR/AAAI proceedings, GitHub. Verified rows
marked ✓; gap-fill rows are sourced but not independently triple-verified. A handful
of 2026 preprints (PHASER, CRL-VLA, Stellar-VLA, LiMoDE, REGEN) report exact
FWT/BWT/forgetting only in the paper body — pull precise numbers from their tables.
Name note: the continual-VLA "REGEN" = arXiv:2606.27374 (distinct from the unrelated
inverse-design "ReGen"). "Info-VLA" was not found as a distinct paper — CRL-VLA is the
closest information-theoretic continual-VLA work.*
