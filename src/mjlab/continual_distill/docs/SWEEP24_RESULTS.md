# 24-sequence RL-teacher sweep + mix experiments (2026-07-09/10)

wandb project: **continual_rl_mjlab_sweep**. lr 3e-5, 4096 student, 500 ep/task,
tasks {PushCuboid,PushButton,OpenDoor,OpenDrawer} RL teachers. 24 perms x 3 seeds.
Run group: `sweep24_p<NN>_s<seed>_1783605057` (an earlier attempt at lr 1e-4 was
scrapped — bistable regime — and archived under logs/old_lr1e4_sweep/).

## 24-permutation ranking (3-seed avg final accuracy)

BEST: **p06 PushCuboid -> OpenDrawer -> OpenDoor -> PushButton = 0.960**
| rank | perm | avg | sequence |
|---|---|---|---|
| 1 | p06 | 0.960 | PushCuboid, OpenDrawer, OpenDoor, PushButton |
| 2 | p03 | 0.951 | PushCuboid, OpenDoor, PushButton, OpenDrawer |
| 3 | p01 | 0.934 | PushCuboid, PushButton, OpenDoor, OpenDrawer |
| 4 | p05 | 0.928 | PushCuboid, OpenDrawer, PushButton, OpenDoor |
| 5 | p04 | 0.921 | PushCuboid, OpenDoor, OpenDrawer, PushButton |
| 6 | p02 | 0.895 | PushCuboid, PushButton, OpenDrawer, OpenDoor |
| ... | | | |
| 22 | p09 | 0.680 | PushButton, OpenDoor, PushCuboid, OpenDrawer |
| 23 | p12 | 0.673 | PushButton, OpenDrawer, OpenDoor, PushCuboid |
| 24 | p10 | 0.659 | PushButton, OpenDoor, OpenDrawer, PushCuboid |

**Pattern:** all top-6 sequences start with PushCuboid (learn the fragile/most-
forgettable task FIRST, then it's protected while easy planar tasks follow). All
bottom-3 start with PushButton and bury PushCuboid late (~0.66). Task ORDER matters
a lot: ~0.30 spread between best and worst.

## Mix-teacher experiments (on best seq p06)

Teachers: 2 good classical (PushButton ~100%, OpenDrawer 99.3%) + 2 good BC
(PushButton BC ~100%, OpenDrawer BC 99.5%). Cuboid & Door stay RL. 6 configs x 3
seeds, run group `mix_<cfg>_s<seed>`:
- 1cl: Drawer=classical | 2cl: Button+Drawer classical
- 1bc: Drawer=BC | 2bc: Button+Drawer BC
- crossA: Button=classical, Drawer=BC | crossB: Button=BC, Drawer=classical

### RESULTS (3-seed avg, p06 order Cuboid/Drawer/Door/Button)

| config | Button/Drawer teacher | Cuboid | Drawer | Door | Button | AVG |
|---|---|---|---|---|---|---|
| RL baseline | RL / RL | 0.86 | 0.98 | 1.00 | 0.98 | **0.960** |
| 1cl | RL / classical | 0.85 | 0.94 | 1.00 | 0.98 | 0.943 |
| crossB | BC / classical | 0.80 | 0.97 | 1.00 | 0.95 | 0.930 |
| 1bc | RL / BC | 0.78 | 0.96 | 1.00 | 0.93 | 0.917 |
| 2bc | BC / BC | 0.69 | 0.98 | 1.00 | 1.00 | 0.917 |
| crossA | classical / BC | 0.72 | 0.99 | 1.00 | 0.88 | 0.896 |
| 2cl | classical / classical | 0.75 | 0.97 | 1.00 | 0.83 | 0.887 |

run group: `mix_<cfg>_s<seed>_1783623195`.

**Findings:**
1. ALL teacher flavors are viable drop-ins: every mix is 0.87-0.94 vs RL's 0.96.
   Ranking RL >~ 1cl > crossB > 1bc ~ 2bc > crossA > 2cl.
2. The SWAPPED tasks distill well regardless of flavor (Drawer 0.94-0.99,
   Button 0.83-1.0) — the good classical(99%)/BC(99%) teachers transfer their skill.
3. The cost appears on PushCuboid (task 0), NOT the swapped tasks: Cuboid drops
   0.86 -> 0.69-0.85. Swapping LATER teachers mildly degrades the EARLIEST task's
   retention (collateral interaction, mild).
4. 1 swap ~= RL; 2 swaps cost a bit more (2cl 0.887 lowest, mainly Button 0.83).
5. BC >= classical for Button: 2bc keeps Button 1.0 vs 2cl 0.83 — cloned BC teacher
   distills more cleanly than raw classical for that task.

## Student-size sweep (on best seq p06, all RL teachers)

3 seeds each, lr 3e-5, 500 ep/task. Run group `size<dim>_s<seed>_1783664057`.

| student hidden dims | per-seed | mean ± std |
|---|---|---|
| 512-256-128 | 0.76, 0.68, 0.53 | 0.658 ± 0.095 |
| 1024-512-256 | 0.75, 0.77, 0.86 | 0.792 ± 0.046 |
| 2048-1024-512 | 0.85, 0.91, 0.86 | 0.872 ± 0.027 |
| 4096-2048-1024 | 0.94, 0.96, 0.98 | 0.960 ± 0.014 |
| 8192-4096-2048 | 0.76, 0.38, 0.83 | 0.656 ± 0.201 |

**Clean monotonic capacity curve: 0.66 -> 0.79 -> 0.87 -> 0.96.** Retention rises
with student size AND variance shrinks (±0.095 at 512 -> ±0.027 at 2048) — bigger
nets are better and more stable. Matches the earlier LiftCube-seq size sweep
(0.62/0.74/0.88/0.93) — trend is task-set-independent. 4096 best; 2048 is the
efficiency point (~0.87 at 1/4 the params). Verified each run built the correct
model (checkpoint layer dims: 512->[128,256,512], 2048->[512,1024,2048]).

**8192 REVERSES the trend (0.656, ±0.201):** capacity helps only up to 4096; going
bigger HURTS both mean AND stability. 8192-s1 collapses (PushButton 0.00, OpenDoor
0.45); even good seeds are patchy (Drawer 0.39-0.80). Signature: the larger net
changes loss-landscape sharpness so the lr-3e-5 fix (tuned at 4096) no longer fully
stabilizes it -> re-enters bistability/consolidation difficulty. **4096 is the sweet
spot; over-parameterization re-introduces instability.** 8192 fits memory fine
(~5GB/run on 80GB A100 -> RAM was never the limit; optimization is).

## Project summary

wandb `continual_rl_mjlab_sweep` now holds 99 runs:
- 24-seq RL sweep (72): best p06 Cuboid->Drawer->Door->Button = 0.960
- mix-teacher (18): all flavors viable, 0.89-0.94 vs RL 0.96
- student-size (9): monotonic 0.66/0.79/0.87 for 512/1024/2048 (vs 4096=0.96)

## Student-size sweep on WORST seq p10 (Button->Door->Drawer->Cuboid)

3 seeds, lr 3e-5, 500 ep. Run group `worstsize<dim>_s<seed>_1783670000`. 4096 from
sweep p10. [seed-2 wave still running; 7/12 as of writing — numbers are partial]

| size | WORST-seq mean | per-seed | BEST-seq (p06) for ref |
|---|---|---|---|
| 512 | 0.561 | 0.60, 0.52 | 0.658 |
| 1024 | 0.615 | 0.60, 0.63 | 0.792 |
| 2048 | 0.617 | 0.63, 0.61 | 0.872 |
| 4096 | 0.659 | 0.67, 0.65, 0.66 | 0.960 |
| 8192 | 0.598 | 0.60 (n=1) | 0.656 |

**KEY INTERACTION — ordering dominates capacity:**
- On the WORST ordering, size barely helps: nearly FLAT ~0.56 -> 0.66, plateaus by
  2048. Buys only ~+0.10 total.
- On the BEST ordering, size climbs steeply 0.66 -> 0.96 (+0.30).
- => **a bad task ordering CANNOT be fixed by adding capacity.** With PushCuboid
  buried last (its worst position), even 4096 reaches only 0.66; the SAME net gets
  0.96 with good ordering. Ordering matters far more than student size on a hard seq.
- 4096 is the peak on BOTH sequences; 8192 reverses on both (best 0.66, worst 0.60)
  -> over-parameterization instability is ordering-independent.
