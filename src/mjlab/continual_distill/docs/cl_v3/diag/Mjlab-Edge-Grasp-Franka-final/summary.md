# diagnose — Mjlab-Edge-Grasp-Franka

2026-09-10T04:19:38 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**13/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `EdgeGraspClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = PUSH_HOVER, 1 = PUSH_DESCEND, 2 = PUSH_ADVANCE, 3 = RETREAT, 4 = SIDE_HOVER, 5 = SIDE_INSERT, 6 = CLOSE, 7 = LIFT, 8 = SIDE_DROP

## Histograms

- end phase, FAILING envs (19): 8 (SIDE_DROP): 8, 6 (CLOSE): 6, 7 (LIFT): 4, 5 (SIDE_INSERT): 1
- end phase, successful envs (13): 8 (SIDE_DROP): 7, 7 (LIFT): 4, 1 (PUSH_DESCEND): 1, 4 (SIDE_HOVER): 1
- most-steps phase, failing envs: 4 (SIDE_HOVER): 7, 5 (SIDE_INSERT): 6, 8 (SIDE_DROP): 5, 2 (PUSH_ADVANCE): 1
- failure class: terminated:ee_ground_collision: 14, never_lifted: 3, never_reached: 2
- termination among failing envs: ee_ground_collision: 14, none: 5
- failing envs that lifted the object: 3/19; lifted then lost: 3
- success step (successful envs): median 37, max 98

## Reading

- **terminated:ee_ground_collision** — 14/19 failing envs (envs [0, 3, 4, 6, 10, 12, 14, 15, 16, 18, 22, 24, 25, 30]); typical end phase 8 (SIDE_DROP); final aperture median 0.080; closest |gripper_to_object| median 0.076; closest |object_to_goal| median 0.074; min goal_error median 0.080
- **never_lifted** — 3/19 failing envs (envs [20, 26, 28]); typical end phase 7 (LIFT); final aperture median 0.001; closest |gripper_to_object| median 0.033; closest |object_to_goal| median 0.145; min goal_error median 0.152
- **never_reached** — 2/19 failing envs (envs [1, 7]); typical end phase 7 (LIFT); final aperture median 0.002; closest |gripper_to_object| median 0.075; closest |object_to_goal| median 0.122; min goal_error median 0.133

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 81 steps; TERMINATED by `ee_ground_collision` at step 80
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 4, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 4
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.087 at step 24 (final 0.132)
- closest |object_to_goal| 0.068 at step 30 (final 0.193); goal_error min 0.076 at step 32 / final 0.194
- object never lifted (max rise 0.003 m)

### env 1 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 14, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 3, 8 (SIDE_DROP): 36
- aperture min -0.014 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.084 at step 97 (final 0.118)
- closest |object_to_goal| 0.126 at step 28 (final 0.157); goal_error min 0.136 at step 1 / final 0.146
- object never lifted (max rise 0.000 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 189 steps; TERMINATED by `ee_ground_collision` at step 188
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 27, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 3, 8 (SIDE_DROP): 44
- aperture min -0.023 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.037 at step 183 (final 0.054)
- closest |object_to_goal| 0.083 at step 75 (final 0.105); goal_error min 0.087 at step 70 / final 0.111
- object never lifted (max rise 0.004 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 179 steps; TERMINATED by `ee_ground_collision` at step 178
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 13, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 38, 6 (CLOSE): 3, 8 (SIDE_DROP): 43
- aperture min 0.000 / max 0.081 / final 0.070
- closest |gripper_to_object| 0.043 at step 102 (final 0.068)
- closest |object_to_goal| 0.072 at step 53 (final 0.161); goal_error min 0.081 at step 52 / final 0.159
- object never lifted (max rise 0.004 m)

### env 6 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 113 steps; TERMINATED by `ee_ground_collision` at step 112
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 8
- aperture min -0.000 / max 0.080 / final 0.077
- closest |gripper_to_object| 0.072 at step 103 (final 0.081)
- closest |object_to_goal| 0.076 at step 67 (final 0.210); goal_error min 0.081 at step 64 / final 0.206
- object never lifted (max rise 0.003 m)

### env 7 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 12, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 2, 8 (SIDE_DROP): 36
- aperture min -0.014 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.066 at step 89 (final 0.117)
- closest |object_to_goal| 0.118 at step 214 (final 0.124); goal_error min 0.130 at step 33 / final 0.130
- object never lifted (max rise 0.000 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 83 steps; TERMINATED by `ee_ground_collision` at step 82
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 25, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 3, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 4
- aperture min -0.005 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.090 at step 26 (final 0.095)
- closest |object_to_goal| 0.077 at step 31 (final 0.234); goal_error min 0.079 at step 30 / final 0.239
- object never lifted (max rise 0.008 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 121 steps; TERMINATED by `ee_ground_collision` at step 120
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 23, 2 (PUSH_ADVANCE): 28, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 3
- aperture min -0.012 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.075 at step 46 (final 0.091)
- closest |object_to_goal| 0.066 at step 78 (final 0.194); goal_error min 0.076 at step 77 / final 0.192
- object LIFTED at step 53 (max rise 0.024 m); LOST at step 54 (aperture collapsed while the object moved away)

### env 14 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 208 steps; TERMINATED by `ee_ground_collision` at step 207
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 18, 2 (PUSH_ADVANCE): 34, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 4, 8 (SIDE_DROP): 42
- aperture min -0.014 / max 0.085 / final 0.053
- closest |gripper_to_object| 0.020 at step 200 (final 0.027)
- closest |object_to_goal| 0.073 at step 99 (final 0.126); goal_error min 0.080 at step 86 / final 0.132
- object never lifted (max rise 0.003 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 115 steps; TERMINATED by `ee_ground_collision` at step 114
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 14, 2 (PUSH_ADVANCE): 31, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 4
- aperture min -0.005 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.081 at step 113 (final 0.085)
- closest |object_to_goal| 0.068 at step 74 (final 0.197); goal_error min 0.077 at step 71 / final 0.198
- object never lifted (max rise 0.004 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 119 steps; TERMINATED by `ee_ground_collision` at step 118
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 14, 2 (PUSH_ADVANCE): 22, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 8
- aperture min -0.007 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.076 at step 28 (final 0.130)
- closest |object_to_goal| 0.074 at step 72 (final 0.176); goal_error min 0.075 at step 70 / final 0.174
- object never lifted (max rise 0.004 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 120 steps; TERMINATED by `ee_ground_collision` at step 119
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 6, 2 (PUSH_ADVANCE): 17, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 24
- aperture min -0.002 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.078 at step 27 (final 0.101)
- closest |object_to_goal| 0.084 at step 53 (final 0.182); goal_error min 0.081 at step 52 / final 0.179
- object LIFTED at step 41 (max rise 0.023 m); LOST at step 44 (aperture collapsed while the object moved away)

### env 20 — never_lifted
- ended in phase 5 (SIDE_INSERT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 31, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 97, 6 (CLOSE): 28, 8 (SIDE_DROP): 36
- aperture min -0.008 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.033 at step 58 (final 0.137)
- closest |object_to_goal| 0.162 at step 5 (final 0.198); goal_error min 0.168 at step 21 / final 0.195
- object never lifted (max rise 0.000 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 174 steps; TERMINATED by `ee_ground_collision` at step 173
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 19, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 3, 8 (SIDE_DROP): 38
- aperture min -0.009 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.079 at step 106 (final 0.109)
- closest |object_to_goal| 0.077 at step 58 (final 0.216); goal_error min 0.085 at step 58 / final 0.213
- object never lifted (max rise 0.003 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 250 steps; TERMINATED by `ee_ground_collision` at step 249
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 21, 2 (PUSH_ADVANCE): 38, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 66, 6 (CLOSE): 25, 8 (SIDE_DROP): 36
- aperture min -0.015 / max 0.080 / final -0.005
- closest |gripper_to_object| 0.070 at step 43 (final 0.090)
- closest |object_to_goal| 0.070 at step 80 (final 0.156); goal_error min 0.075 at step 80 / final 0.152
- object LIFTED at step 48 (max rise 0.027 m); LOST at step 49 (aperture collapsed while the object moved away)

### env 25 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 197 steps; TERMINATED by `ee_ground_collision` at step 196
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 35, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 14, 8 (SIDE_DROP): 36
- aperture min -0.008 / max 0.104 / final 0.080
- closest |gripper_to_object| 0.022 at step 157 (final 0.041)
- closest |object_to_goal| 0.078 at step 70 (final 0.155); goal_error min 0.088 at step 73 / final 0.151
- object never lifted (max rise 0.012 m)

### env 26 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 4, 8 (SIDE_DROP): 36
- aperture min -0.000 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.050 at step 24 (final 0.129)
- closest |object_to_goal| 0.145 at step 8 (final 0.154); goal_error min 0.152 at step 1 / final 0.158
- object never lifted (max rise 0.000 m)

### env 28 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 4, 8 (SIDE_DROP): 36
- aperture min -0.011 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.031 at step 67 (final 0.122)
- closest |object_to_goal| 0.139 at step 24 (final 0.172); goal_error min 0.145 at step 23 / final 0.168
- object never lifted (max rise 0.000 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 8 (SIDE_DROP) after 77 steps; TERMINATED by `ee_ground_collision` at step 76
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 7, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 8 (SIDE_DROP): 4
- aperture min -0.008 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.090 at step 17 (final 0.118)
- closest |object_to_goal| 0.069 at step 25 (final 0.184); goal_error min 0.076 at step 25 / final 0.190
- object never lifted (max rise 0.005 m)

## Successful envs (one line each)

- env 2: success at step 44, end phase 8 (SIDE_DROP), aperture min -0.014, min |go| 0.074, lifted at 45
- env 5: success at step 23, end phase 7 (LIFT), aperture min -0.013, min |go| 0.091, lifted at 23
- env 8: success at step 48, end phase 8 (SIDE_DROP), aperture min -0.005, min |go| 0.065, lifted at 26
- env 9: success at step 28, end phase 8 (SIDE_DROP), aperture min -0.008, min |go| 0.069, lifted at 29
- env 11: success at step 31, end phase 8 (SIDE_DROP), aperture min -0.019, min |go| 0.067, lifted at 32
- env 13: success at step 27, end phase 1 (PUSH_DESCEND), aperture min -0.005, min |go| 0.091, lifted at 28
- env 17: success at step 68, end phase 4 (SIDE_HOVER), aperture min -0.004, min |go| 0.058, lifted at 69
- env 19: success at step 37, end phase 8 (SIDE_DROP), aperture min -0.004, min |go| 0.014, lifted at 37
- env 21: success at step 70, end phase 7 (LIFT), aperture min -0.008, min |go| 0.062, lifted at 70
- env 23: success at step 59, end phase 8 (SIDE_DROP), aperture min -0.015, min |go| 0.071, lifted at 59
- env 27: success at step 24, end phase 8 (SIDE_DROP), aperture min 0.000, min |go| 0.050, lifted at 25
- env 29: success at step 24, end phase 7 (LIFT), aperture min -0.006, min |go| 0.096, lifted at 24
- env 31: success at step 98, end phase 7 (LIFT), aperture min -0.011, min |go| 0.046, lifted at 97
