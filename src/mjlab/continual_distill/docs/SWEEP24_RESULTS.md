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
[results pending]
