# diagnose — Mjlab-Edge-Grasp-Franka

2026-09-09T21:32:23 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**12/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `EdgeGraspClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = PUSH_HOVER, 1 = PUSH_DESCEND, 2 = PUSH_ADVANCE, 3 = RETREAT, 4 = SIDE_HOVER, 5 = SIDE_INSERT, 6 = CLOSE, 7 = LIFT, 8 = SIDE_DROP

## Histograms

- end phase, FAILING envs (20): 6 (CLOSE): 15, 7 (LIFT): 5
- end phase, successful envs (12): 6 (CLOSE): 5, 8 (SIDE_DROP): 2, 7 (LIFT): 2, 5 (SIDE_INSERT): 1, 3 (RETREAT): 1, 1 (PUSH_DESCEND): 1
- most-steps phase, failing envs: 5 (SIDE_INSERT): 15, 2 (PUSH_ADVANCE): 2, 8 (SIDE_DROP): 2, 7 (LIFT): 1
- failure class: terminated:ee_ground_collision: 15, never_lifted: 2, never_reached: 2, grasp_lost: 1
- termination among failing envs: ee_ground_collision: 15, none: 5
- failing envs that lifted the object: 4/20; lifted then lost: 3
- success step (successful envs): median 61, max 96

## Reading

- **terminated:ee_ground_collision** — 15/20 failing envs (envs [1, 3, 4, 5, 8, 11, 14, 15, 16, 20, 22, 25, 26, 30, 31]); typical end phase 6 (CLOSE); final aperture median 0.024; closest |gripper_to_object| median 0.044; closest |object_to_goal| median 0.073; min goal_error median 0.077
- **never_lifted** — 2/20 failing envs (envs [2, 29]); typical end phase 7 (LIFT); final aperture median 0.000; closest |gripper_to_object| median 0.058; closest |object_to_goal| median 0.108; min goal_error median 0.117
- **never_reached** — 2/20 failing envs (envs [13, 24]); typical end phase 7 (LIFT); final aperture median -0.002; closest |gripper_to_object| median 0.070; closest |object_to_goal| median 0.090; min goal_error median 0.095
- **grasp_lost** — 1/20 failing envs (envs [9]); typical end phase 7 (LIFT); final aperture median 0.002; closest |gripper_to_object| median 0.055; closest |object_to_goal| median 0.131; min goal_error median 0.142; lost at step median 29

## Failing envs

### env 1 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 169 steps; TERMINATED by `ee_ground_collision` at step 168
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 8, 8 (SIDE_DROP): 29
- aperture min -0.006 / max 0.080 / final 0.018
- closest |gripper_to_object| 0.086 at step 17 (final 0.090)
- closest |object_to_goal| 0.077 at step 23 (final 0.151); goal_error min 0.075 at step 23 / final 0.150
- object never lifted (max rise 0.011 m)

### env 2 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 7, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 6, 8 (SIDE_DROP): 29
- aperture min -0.002 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.057 at step 22 (final 0.119)
- closest |object_to_goal| 0.142 at step 1 (final 0.170); goal_error min 0.150 at step 1 / final 0.165
- object never lifted (max rise 0.000 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 172 steps; TERMINATED by `ee_ground_collision` at step 171
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 4, 2 (PUSH_ADVANCE): 17, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 14, 8 (SIDE_DROP): 32
- aperture min -0.006 / max 0.080 / final -0.001
- closest |gripper_to_object| 0.034 at step 164 (final 0.038)
- closest |object_to_goal| 0.073 at step 47 (final 0.160); goal_error min 0.078 at step 46 / final 0.149
- object never lifted (max rise 0.013 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 175 steps; TERMINATED by `ee_ground_collision` at step 174
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 36, 6 (CLOSE): 6, 8 (SIDE_DROP): 31
- aperture min -0.005 / max 0.095 / final 0.031
- closest |gripper_to_object| 0.091 at step 17 (final 0.166)
- closest |object_to_goal| 0.041 at step 21 (final 0.296); goal_error min 0.045 at step 21 / final 0.293
- object LIFTED at step 20 (max rise 0.023 m); LOST at step 24 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 143 steps; TERMINATED by `ee_ground_collision` at step 142
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 8, 8 (SIDE_DROP): 29
- aperture min -0.004 / max 0.084 / final 0.018
- closest |gripper_to_object| 0.033 at step 106 (final 0.085)
- closest |object_to_goal| 0.071 at step 27 (final 0.225); goal_error min 0.071 at step 26 / final 0.225
- object never lifted (max rise 0.005 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 174 steps; TERMINATED by `ee_ground_collision` at step 173
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 12, 1 (PUSH_DESCEND): 12, 2 (PUSH_ADVANCE): 39, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 21, 6 (CLOSE): 7, 8 (SIDE_DROP): 34
- aperture min -0.006 / max 0.089 / final 0.024
- closest |gripper_to_object| 0.030 at step 162 (final 0.065)
- closest |object_to_goal| 0.075 at step 71 (final 0.107); goal_error min 0.084 at step 72 / final 0.118
- object never lifted (max rise 0.003 m)

### env 9 — grasp_lost
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 3, 8 (SIDE_DROP): 29
- aperture min -0.012 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.055 at step 19 (final 0.111)
- closest |object_to_goal| 0.131 at step 197 (final 0.142); goal_error min 0.142 at step 293 / final 0.142
- object LIFTED at step 28 (max rise 0.021 m); LOST at step 29 (aperture collapsed while the object moved away)

### env 11 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 185 steps; TERMINATED by `ee_ground_collision` at step 184
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 8, 8 (SIDE_DROP): 32
- aperture min -0.004 / max 0.080 / final 0.018
- closest |gripper_to_object| 0.044 at step 108 (final 0.081)
- closest |object_to_goal| 0.069 at step 64 (final 0.196); goal_error min 0.075 at step 67 / final 0.184
- object never lifted (max rise 0.007 m)

### env 13 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 9, 8 (SIDE_DROP): 29
- aperture min 0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.068 at step 19 (final 0.264)
- closest |object_to_goal| 0.108 at step 23 (final 0.323); goal_error min 0.113 at step 23 / final 0.323
- object never lifted (max rise 0.013 m)

### env 14 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 227 steps; TERMINATED by `ee_ground_collision` at step 226
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 16, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 68, 6 (CLOSE): 35, 8 (SIDE_DROP): 5
- aperture min -0.006 / max 0.090 / final 0.027
- closest |gripper_to_object| 0.041 at step 139 (final 0.064)
- closest |object_to_goal| 0.077 at step 136 (final 0.197); goal_error min 0.080 at step 134 / final 0.206
- object LIFTED at step 133 (max rise 0.023 m); LOST at step 140 (aperture collapsed while the object moved away)

### env 15 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 148 steps; TERMINATED by `ee_ground_collision` at step 147
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 7, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 6, 8 (SIDE_DROP): 33
- aperture min -0.018 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.027 at step 120 (final 0.062)
- closest |object_to_goal| 0.070 at step 77 (final 0.115); goal_error min 0.072 at step 76 / final 0.116
- object never lifted (max rise 0.010 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 176 steps; TERMINATED by `ee_ground_collision` at step 175
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 7, 8 (SIDE_DROP): 29
- aperture min -0.005 / max 0.082 / final 0.024
- closest |gripper_to_object| 0.039 at step 118 (final 0.064)
- closest |object_to_goal| 0.081 at step 90 (final 0.182); goal_error min 0.089 at step 90 / final 0.175
- object never lifted (max rise 0.017 m)

### env 20 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 180 steps; TERMINATED by `ee_ground_collision` at step 179
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 12, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 33, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 36, 6 (CLOSE): 7, 8 (SIDE_DROP): 34
- aperture min -0.003 / max 0.093 / final 0.028
- closest |gripper_to_object| 0.045 at step 117 (final 0.059)
- closest |object_to_goal| 0.068 at step 91 (final 0.170); goal_error min 0.069 at step 92 / final 0.170
- object never lifted (max rise 0.010 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 257 steps; TERMINATED by `ee_ground_collision` at step 256
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 3, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 109, 6 (CLOSE): 42, 8 (SIDE_DROP): 30
- aperture min -0.002 / max 0.080 / final 0.004
- closest |gripper_to_object| 0.075 at step 256 (final 0.075)
- closest |object_to_goal| 0.091 at step 27 (final 0.209); goal_error min 0.101 at step 34 / final 0.216
- object never lifted (max rise 0.011 m)

### env 24 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 24, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 38, 8 (SIDE_DROP): 29
- aperture min -0.019 / max 0.080 / final -0.003
- closest |gripper_to_object| 0.073 at step 257 (final 0.090)
- closest |object_to_goal| 0.072 at step 50 (final 0.125); goal_error min 0.076 at step 49 / final 0.125
- object never lifted (max rise 0.001 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 151 steps; TERMINATED by `ee_ground_collision` at step 150
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 4, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 33, 5 (SIDE_INSERT): 33, 6 (CLOSE): 6, 8 (SIDE_DROP): 34
- aperture min -0.002 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.087 at step 76 (final 0.161)
- closest |object_to_goal| 0.081 at step 26 (final 0.313); goal_error min 0.079 at step 29 / final 0.308
- object never lifted (max rise 0.011 m)

### env 26 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 164 steps; TERMINATED by `ee_ground_collision` at step 163
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 7, 2 (PUSH_ADVANCE): 14, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 36, 6 (CLOSE): 7, 8 (SIDE_DROP): 30
- aperture min -0.008 / max 0.080 / final 0.024
- closest |gripper_to_object| 0.083 at step 97 (final 0.109)
- closest |object_to_goal| 0.078 at step 46 (final 0.205); goal_error min 0.083 at step 48 / final 0.196
- object never lifted (max rise 0.017 m)

### env 29 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 7 (LIFT); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 21, 2 (PUSH_ADVANCE): 7, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 17, 6 (CLOSE): 14, 7 (LIFT): 172, 8 (SIDE_DROP): 6
- aperture min -0.014 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.059 at step 117 (final 0.178)
- closest |object_to_goal| 0.074 at step 51 (final 0.128); goal_error min 0.085 at step 44 / final 0.136
- object never lifted (max rise 0.010 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 210 steps; TERMINATED by `ee_ground_collision` at step 209
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 7, 8 (SIDE_DROP): 35
- aperture min -0.012 / max 0.081 / final 0.025
- closest |gripper_to_object| 0.036 at step 196 (final 0.065)
- closest |object_to_goal| 0.070 at step 128 (final 0.099); goal_error min 0.074 at step 127 / final 0.105
- object LIFTED at step 91 (max rise 0.026 m); never lost

### env 31 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 254 steps; TERMINATED by `ee_ground_collision` at step 253
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 16, 2 (PUSH_ADVANCE): 35, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 71, 6 (CLOSE): 23, 8 (SIDE_DROP): 34
- aperture min -0.005 / max 0.089 / final 0.014
- closest |gripper_to_object| 0.081 at step 123 (final 0.121)
- closest |object_to_goal| 0.070 at step 86 (final 0.252); goal_error min 0.077 at step 85 / final 0.260
- object never lifted (max rise 0.001 m)

## Successful envs (one line each)

- env 0: success at step 95, end phase 6 (CLOSE), aperture min -0.001, min |go| 0.040, lifted at 94
- env 6: success at step 96, end phase 8 (SIDE_DROP), aperture min -0.014, min |go| 0.055, lifted at 96
- env 7: success at step 85, end phase 6 (CLOSE), aperture min -0.012, min |go| 0.042, lifted at 86
- env 10: success at step 94, end phase 5 (SIDE_INSERT), aperture min -0.013, min |go| 0.048, lifted at 95
- env 12: success at step 38, end phase 7 (LIFT), aperture min -0.005, min |go| 0.047, lifted at 38
- env 17: success at step 27, end phase 7 (LIFT), aperture min -0.010, min |go| 0.037, lifted at 27
- env 18: success at step 24, end phase 3 (RETREAT), aperture min -0.009, min |go| 0.089, lifted at 24
- env 19: success at step 26, end phase 1 (PUSH_DESCEND), aperture min -0.011, min |go| 0.090, lifted at 27
- env 21: success at step 84, end phase 8 (SIDE_DROP), aperture min -0.012, min |go| 0.058, lifted at 84
- env 23: success at step 35, end phase 6 (CLOSE), aperture min -0.010, min |go| 0.054, lifted at 36
- env 27: success at step 33, end phase 6 (CLOSE), aperture min -0.019, min |go| 0.069, lifted at 34
- env 28: success at step 89, end phase 6 (CLOSE), aperture min -0.012, min |go| 0.043, lifted at 89
