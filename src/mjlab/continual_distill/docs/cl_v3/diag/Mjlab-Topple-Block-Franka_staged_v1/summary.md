# diagnose — Mjlab-Topple-Block-Franka

2026-09-09T14:21:31 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 200 · device cuda:0  
**31/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToppleBlockClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = ADVANCE, 3 = PUNCH, 4 = RETREAT

## Histograms

- end phase, FAILING envs (1): 4 (RETREAT): 1
- end phase, successful envs (31): 4 (RETREAT): 31
- most-steps phase, failing envs: 4 (RETREAT): 1
- failure class: never_reached: 1
- termination among failing envs: none: 1
- success step (successful envs): median 64, max 115

## Reading

- **never_reached** — 1/1 failing envs (envs [23]); typical end phase 4 (RETREAT); final aperture median -0.003; closest |gripper_to_object| median 0.108; closest |object_to_goal| median 0.005; min goal_error median 0.869

## Failing envs

### env 23 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 24, 1 (DESCEND): 41, 2 (ADVANCE): 41, 3 (PUNCH): 37, 4 (RETREAT): 57
- aperture min -0.010 / max 0.093 / final -0.003
- closest |gripper_to_object| 0.108 at step 100 (final 0.143)
- closest |object_to_goal| 0.005 at step 14 (final 0.204); goal_error min 0.869 at step 144 / final 1.568
- object LIFTED at step 56 (max rise 0.035 m); LOST at step 57 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 27, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.081
- env 1: success at step 44, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.141
- env 2: success at step 64, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.115, lifted at 25
- env 3: success at step 67, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.102, lifted at 28
- env 4: success at step 77, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.090
- env 5: success at step 73, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.090
- env 6: success at step 84, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.129
- env 7: success at step 75, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.121
- env 8: success at step 115, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.101, lifted at 81
- env 9: success at step 19, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.130
- env 10: success at step 77, end phase 4 (RETREAT), aperture min -0.003, min |go| 0.089
- env 11: success at step 30, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.129
- env 12: success at step 73, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.120
- env 13: success at step 47, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.102
- env 14: success at step 88, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.115, lifted at 81
- env 15: success at step 63, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.130
- env 16: success at step 30, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.127
- env 17: success at step 74, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.109
- env 18: success at step 75, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.113
- env 19: success at step 64, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.124
- env 20: success at step 23, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.105
- env 21: success at step 97, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.098, lifted at 73
- env 22: success at step 21, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.140
- env 24: success at step 113, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.088, lifted at 80
- env 25: success at step 33, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.105
- env 26: success at step 83, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.099, lifted at 37
- env 27: success at step 46, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.128, lifted at 35
- env 28: success at step 74, end phase 4 (RETREAT), aperture min -0.003, min |go| 0.105
- env 29: success at step 62, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.122
- env 30: success at step 48, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.098
- env 31: success at step 29, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.108
