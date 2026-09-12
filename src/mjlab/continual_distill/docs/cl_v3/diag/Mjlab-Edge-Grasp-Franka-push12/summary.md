# diagnose — Mjlab-Edge-Grasp-Franka

2026-09-10T03:36:42 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**9/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `EdgeGraspClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = PUSH_HOVER, 1 = PUSH_DESCEND, 2 = PUSH_ADVANCE, 3 = RETREAT, 4 = SIDE_HOVER, 5 = SIDE_INSERT, 6 = CLOSE, 7 = LIFT, 8 = SIDE_DROP

## Histograms

- end phase, FAILING envs (23): 6 (CLOSE): 19, 7 (LIFT): 4
- end phase, successful envs (9): 6 (CLOSE): 9
- most-steps phase, failing envs: 5 (SIDE_INSERT): 19, 8 (SIDE_DROP): 3, 4 (SIDE_HOVER): 1
- failure class: terminated:ee_ground_collision: 14, never_lifted: 5, never_reached: 4
- termination among failing envs: ee_ground_collision: 14, none: 9
- failing envs that lifted the object: 2/23; lifted then lost: 0
- success step (successful envs): median 81, max 94

## Reading

- **terminated:ee_ground_collision** — 14/23 failing envs (envs [2, 3, 4, 5, 7, 8, 10, 13, 14, 16, 17, 21, 24, 30]); typical end phase 6 (CLOSE); final aperture median 0.041; closest |gripper_to_object| median 0.076; closest |object_to_goal| median 0.075; min goal_error median 0.079
- **never_lifted** — 5/23 failing envs (envs [0, 11, 19, 26, 27]); typical end phase 6 (CLOSE); final aperture median 0.006; closest |gripper_to_object| median 0.044; closest |object_to_goal| median 0.142; min goal_error median 0.152
- **never_reached** — 4/23 failing envs (envs [15, 25, 28, 29]); typical end phase 7 (LIFT); final aperture median 0.003; closest |gripper_to_object| median 0.073; closest |object_to_goal| median 0.124; min goal_error median 0.135

## Failing envs

### env 0 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 7, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 60, 6 (CLOSE): 42, 7 (LIFT): 60, 8 (SIDE_DROP): 8
- aperture min -0.013 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.031 at step 70 (final 0.789)
- closest |object_to_goal| 0.150 at step 27 (final 0.191); goal_error min 0.157 at step 26 / final 0.179
- object never lifted (max rise 0.000 m)

### env 2 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 192 steps; TERMINATED by `ee_ground_collision` at step 191
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 25, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 35, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 36, 6 (CLOSE): 5, 8 (SIDE_DROP): 32
- aperture min -0.010 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.033 at step 27 (final 0.094)
- closest |object_to_goal| 0.099 at step 78 (final 0.214); goal_error min 0.104 at step 76 / final 0.209
- object never lifted (max rise 0.009 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 191 steps; TERMINATED by `ee_ground_collision` at step 190
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 13, 2 (PUSH_ADVANCE): 34, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 6, 8 (SIDE_DROP): 35
- aperture min -0.008 / max 0.081 / final 0.031
- closest |gripper_to_object| 0.093 at step 51 (final 0.117)
- closest |object_to_goal| 0.073 at step 76 (final 0.183); goal_error min 0.079 at step 77 / final 0.191
- object never lifted (max rise 0.000 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 186 steps; TERMINATED by `ee_ground_collision` at step 185
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 35, 5 (SIDE_INSERT): 36, 6 (CLOSE): 5, 8 (SIDE_DROP): 32
- aperture min -0.007 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.082 at step 24 (final 0.106)
- closest |object_to_goal| 0.079 at step 65 (final 0.201); goal_error min 0.078 at step 67 / final 0.202
- object never lifted (max rise 0.001 m)

### env 5 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 175 steps; TERMINATED by `ee_ground_collision` at step 174
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 5, 8 (SIDE_DROP): 31
- aperture min 0.000 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.093 at step 19 (final 0.238)
- closest |object_to_goal| 0.072 at step 24 (final 0.301); goal_error min 0.081 at step 24 / final 0.298
- object never lifted (max rise 0.012 m)

### env 7 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 172 steps; TERMINATED by `ee_ground_collision` at step 171
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 4, 8 (SIDE_DROP): 34
- aperture min 0.000 / max 0.080 / final 0.053
- closest |gripper_to_object| 0.074 at step 22 (final 0.169)
- closest |object_to_goal| 0.073 at step 63 (final 0.224); goal_error min 0.073 at step 62 / final 0.236
- object never lifted (max rise 0.005 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 151 steps; TERMINATED by `ee_ground_collision` at step 150
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 6, 8 (SIDE_DROP): 35
- aperture min -0.005 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.083 at step 21 (final 0.162)
- closest |object_to_goal| 0.073 at step 24 (final 0.240); goal_error min 0.075 at step 27 / final 0.245
- object never lifted (max rise 0.007 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 174 steps; TERMINATED by `ee_ground_collision` at step 173
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 12, 2 (PUSH_ADVANCE): 27, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 36, 6 (CLOSE): 5, 8 (SIDE_DROP): 29
- aperture min -0.006 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.031 at step 155 (final 0.032)
- closest |object_to_goal| 0.085 at step 64 (final 0.134); goal_error min 0.094 at step 64 / final 0.128
- object never lifted (max rise 0.003 m)

### env 11 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 26, 1 (PUSH_DESCEND): 14, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 32, 8 (SIDE_DROP): 29
- aperture min -0.015 / max 0.080 / final 0.053
- closest |gripper_to_object| 0.044 at step 25 (final 0.144)
- closest |object_to_goal| 0.142 at step 172 (final 0.144); goal_error min 0.153 at step 25 / final 0.154
- object never lifted (max rise 0.000 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 136 steps; TERMINATED by `ee_ground_collision` at step 135
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 7, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 34, 6 (CLOSE): 17, 8 (SIDE_DROP): 4
- aperture min -0.005 / max 0.089 / final 0.068
- closest |gripper_to_object| 0.043 at step 102 (final 0.108)
- closest |object_to_goal| 0.077 at step 75 (final 0.189); goal_error min 0.083 at step 96 / final 0.197
- object never lifted (max rise 0.019 m)

### env 14 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 169 steps; TERMINATED by `ee_ground_collision` at step 168
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 19, 8 (SIDE_DROP): 4
- aperture min -0.002 / max 0.082 / final 0.041
- closest |gripper_to_object| 0.036 at step 131 (final 0.099)
- closest |object_to_goal| 0.087 at step 122 (final 0.204); goal_error min 0.089 at step 123 / final 0.205
- object LIFTED at step 123 (max rise 0.028 m); never lost

### env 15 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 10, 8 (SIDE_DROP): 29
- aperture min -0.006 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.080 at step 33 (final 0.159)
- closest |object_to_goal| 0.098 at step 22 (final 0.125); goal_error min 0.113 at step 22 / final 0.125
- object never lifted (max rise 0.008 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 163 steps; TERMINATED by `ee_ground_collision` at step 162
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 6, 2 (PUSH_ADVANCE): 14, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 37, 6 (CLOSE): 6, 8 (SIDE_DROP): 29
- aperture min -0.011 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.066 at step 27 (final 0.080)
- closest |object_to_goal| 0.083 at step 49 (final 0.197); goal_error min 0.085 at step 50 / final 0.206
- object never lifted (max rise 0.006 m)

### env 17 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 140 steps; TERMINATED by `ee_ground_collision` at step 139
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 5, 8 (SIDE_DROP): 32
- aperture min 0.001 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.091 at step 18 (final 0.112)
- closest |object_to_goal| 0.060 at step 22 (final 0.194); goal_error min 0.059 at step 22 / final 0.191
- object LIFTED at step 21 (max rise 0.025 m); never lost

### env 19 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 39, 8 (SIDE_DROP): 29
- aperture min -0.012 / max 0.080 / final 0.008
- closest |gripper_to_object| 0.019 at step 82 (final 0.106)
- closest |object_to_goal| 0.134 at step 26 (final 0.201); goal_error min 0.143 at step 27 / final 0.199
- object never lifted (max rise 0.000 m)

### env 21 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 158 steps; TERMINATED by `ee_ground_collision` at step 157
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 21, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 38, 6 (CLOSE): 5, 8 (SIDE_DROP): 29
- aperture min -0.008 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.079 at step 104 (final 0.102)
- closest |object_to_goal| 0.067 at step 43 (final 0.214); goal_error min 0.075 at step 43 / final 0.213
- object never lifted (max rise 0.002 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 164 steps; TERMINATED by `ee_ground_collision` at step 163
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 17, 3 (RETREAT): 18, 4 (SIDE_HOVER): 35, 5 (SIDE_INSERT): 33, 6 (CLOSE): 5, 8 (SIDE_DROP): 33
- aperture min -0.000 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.083 at step 21 (final 0.104)
- closest |object_to_goal| 0.077 at step 44 (final 0.215); goal_error min 0.077 at step 46 / final 0.212
- object never lifted (max rise 0.005 m)

### env 25 — never_reached
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 15, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 37, 8 (SIDE_DROP): 29
- aperture min -0.014 / max 0.080 / final 0.014
- closest |gripper_to_object| 0.064 at step 107 (final 0.135)
- closest |object_to_goal| 0.125 at step 0 (final 0.155); goal_error min 0.133 at step 47 / final 0.149
- object never lifted (max rise 0.011 m)

### env 26 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 40, 8 (SIDE_DROP): 29
- aperture min -0.023 / max 0.082 / final 0.006
- closest |gripper_to_object| 0.058 at step 29 (final 0.146)
- closest |object_to_goal| 0.146 at step 1 (final 0.161); goal_error min 0.152 at step 60 / final 0.163
- object never lifted (max rise 0.002 m)

### env 27 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 6, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 6, 8 (SIDE_DROP): 29
- aperture min -0.002 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.045 at step 21 (final 0.151)
- closest |object_to_goal| 0.138 at step 1 (final 0.180); goal_error min 0.149 at step 1 / final 0.184
- object never lifted (max rise 0.000 m)

### env 28 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 6, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 42, 7 (LIFT): 4, 8 (SIDE_DROP): 29
- aperture min -0.004 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.065 at step 24 (final 0.134)
- closest |object_to_goal| 0.124 at step 299 (final 0.124); goal_error min 0.136 at step 299 / final 0.136
- object never lifted (max rise 0.004 m)

### env 29 — never_reached
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 13, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 41, 8 (SIDE_DROP): 29
- aperture min -0.013 / max 0.080 / final 0.005
- closest |gripper_to_object| 0.093 at step 109 (final 0.134)
- closest |object_to_goal| 0.132 at step 20 (final 0.146); goal_error min 0.142 at step 1 / final 0.151
- object never lifted (max rise 0.000 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 146 steps; TERMINATED by `ee_ground_collision` at step 145
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 4, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 10, 8 (SIDE_DROP): 32
- aperture min -0.004 / max 0.080 / final 0.011
- closest |gripper_to_object| 0.064 at step 83 (final 0.124)
- closest |object_to_goal| 0.065 at step 23 (final 0.221); goal_error min 0.075 at step 23 / final 0.228
- object never lifted (max rise 0.014 m)

## Successful envs (one line each)

- env 1: success at step 22, end phase 6 (CLOSE), aperture min -0.005, min |go| 0.064, lifted at 22
- env 6: success at step 90, end phase 6 (CLOSE), aperture min -0.001, min |go| 0.032, lifted at 90
- env 9: success at step 19, end phase 6 (CLOSE), aperture min -0.011, min |go| 0.078, lifted at 19
- env 12: success at step 81, end phase 6 (CLOSE), aperture min -0.014, min |go| 0.034, lifted at 81
- env 18: success at step 33, end phase 6 (CLOSE), aperture min -0.007, min |go| 0.046, lifted at 34
- env 20: success at step 93, end phase 6 (CLOSE), aperture min -0.006, min |go| 0.048, lifted at 93
- env 22: success at step 29, end phase 6 (CLOSE), aperture min -0.001, min |go| 0.018, lifted at 29
- env 23: success at step 87, end phase 6 (CLOSE), aperture min -0.013, min |go| 0.049, lifted at 88
- env 31: success at step 94, end phase 6 (CLOSE), aperture min -0.014, min |go| 0.053, lifted at 94
