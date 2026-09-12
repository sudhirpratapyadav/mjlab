# diagnose — Mjlab-Edge-Grasp-Franka

2026-09-10T03:16:26 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**12/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `EdgeGraspClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = PUSH_HOVER, 1 = PUSH_DESCEND, 2 = PUSH_ADVANCE, 3 = RETREAT, 4 = SIDE_HOVER, 5 = SIDE_INSERT, 6 = CLOSE, 7 = LIFT, 8 = SIDE_DROP

## Histograms

- end phase, FAILING envs (20): 6 (CLOSE): 15, 5 (SIDE_INSERT): 4, 7 (LIFT): 1
- end phase, successful envs (12): 6 (CLOSE): 6, 5 (SIDE_INSERT): 2, 1 (PUSH_DESCEND): 1, 8 (SIDE_DROP): 1, 7 (LIFT): 1, 3 (RETREAT): 1
- most-steps phase, failing envs: 5 (SIDE_INSERT): 10, 4 (SIDE_HOVER): 4, 8 (SIDE_DROP): 3, 2 (PUSH_ADVANCE): 2, 7 (LIFT): 1
- failure class: terminated:ee_ground_collision: 15, never_lifted: 3, never_reached: 2
- termination among failing envs: ee_ground_collision: 15, none: 5
- failing envs that lifted the object: 3/20; lifted then lost: 3
- success step (successful envs): median 30, max 99

## Reading

- **terminated:ee_ground_collision** — 15/20 failing envs (envs [2, 3, 4, 5, 6, 8, 11, 13, 15, 18, 19, 20, 25, 28, 29]); typical end phase 6 (CLOSE); final aperture median 0.041; closest |gripper_to_object| median 0.075; closest |object_to_goal| median 0.070; min goal_error median 0.075
- **never_lifted** — 3/20 failing envs (envs [9, 26, 31]); typical end phase 5 (SIDE_INSERT); final aperture median 0.080; closest |gripper_to_object| median 0.013; closest |object_to_goal| median 0.141; min goal_error median 0.146
- **never_reached** — 2/20 failing envs (envs [17, 30]); typical end phase 6 (CLOSE); final aperture median 0.042; closest |gripper_to_object| median 0.066; closest |object_to_goal| median 0.149; min goal_error median 0.156

## Failing envs

### env 2 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 148 steps; TERMINATED by `ee_ground_collision` at step 147
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 3, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 5, 8 (SIDE_DROP): 33
- aperture min -0.007 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.081 at step 125 (final 0.084)
- closest |object_to_goal| 0.064 at step 29 (final 0.197); goal_error min 0.066 at step 28 / final 0.193
- object never lifted (max rise 0.010 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 148 steps; TERMINATED by `ee_ground_collision` at step 147
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 3, 2 (PUSH_ADVANCE): 1, 3 (RETREAT): 18, 4 (SIDE_HOVER): 36, 5 (SIDE_INSERT): 33, 6 (CLOSE): 6, 8 (SIDE_DROP): 29
- aperture min -0.020 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.075 at step 22 (final 0.191)
- closest |object_to_goal| 0.064 at step 27 (final 0.286); goal_error min 0.062 at step 26 / final 0.278
- object LIFTED at step 25 (max rise 0.026 m); LOST at step 26 (aperture collapsed while the object moved away)

### env 4 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 162 steps; TERMINATED by `ee_ground_collision` at step 161
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 14, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 7, 8 (SIDE_DROP): 33
- aperture min -0.011 / max 0.080 / final 0.024
- closest |gripper_to_object| 0.075 at step 26 (final 0.087)
- closest |object_to_goal| 0.071 at step 45 (final 0.205); goal_error min 0.074 at step 45 / final 0.197
- object LIFTED at step 26 (max rise 0.020 m); LOST at step 27 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 146 steps; TERMINATED by `ee_ground_collision` at step 145
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 5, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 33, 6 (CLOSE): 5, 8 (SIDE_DROP): 29
- aperture min -0.012 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.076 at step 23 (final 0.103)
- closest |object_to_goal| 0.070 at step 39 (final 0.193); goal_error min 0.078 at step 37 / final 0.193
- object never lifted (max rise 0.003 m)

### env 6 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 151 steps; TERMINATED by `ee_ground_collision` at step 150
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 4, 2 (PUSH_ADVANCE): 3, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 6, 8 (SIDE_DROP): 34
- aperture min -0.008 / max 0.081 / final 0.038
- closest |gripper_to_object| 0.040 at step 90 (final 0.064)
- closest |object_to_goal| 0.067 at step 66 (final 0.160); goal_error min 0.072 at step 66 / final 0.157
- object never lifted (max rise 0.008 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 185 steps; TERMINATED by `ee_ground_collision` at step 184
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 35, 3 (RETREAT): 18, 4 (SIDE_HOVER): 34, 5 (SIDE_INSERT): 33, 6 (CLOSE): 3, 8 (SIDE_DROP): 32
- aperture min -0.004 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.089 at step 26 (final 0.153)
- closest |object_to_goal| 0.077 at step 77 (final 0.198); goal_error min 0.086 at step 74 / final 0.195
- object never lifted (max rise 0.003 m)

### env 9 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 7 (LIFT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 41, 6 (CLOSE): 42, 7 (LIFT): 66, 8 (SIDE_DROP): 7
- aperture min -0.014 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.013 at step 69 (final 0.811)
- closest |object_to_goal| 0.147 at step 22 (final 0.161); goal_error min 0.157 at step 21 / final 0.166
- object never lifted (max rise 0.000 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 144 steps; TERMINATED by `ee_ground_collision` at step 143
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 3, 3 (RETREAT): 18, 4 (SIDE_HOVER): 33, 5 (SIDE_INSERT): 35, 6 (CLOSE): 5, 8 (SIDE_DROP): 29
- aperture min -0.009 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.093 at step 20 (final 0.210)
- closest |object_to_goal| 0.068 at step 25 (final 0.305); goal_error min 0.073 at step 25 / final 0.305
- object never lifted (max rise 0.007 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 142 steps; TERMINATED by `ee_ground_collision` at step 141
- most steps in phase 8 (SIDE_DROP); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 4, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 19, 6 (CLOSE): 12, 8 (SIDE_DROP): 34
- aperture min -0.002 / max 0.080 / final 0.006
- closest |gripper_to_object| 0.048 at step 86 (final 0.216)
- closest |object_to_goal| 0.071 at step 67 (final 0.370); goal_error min 0.068 at step 68 / final 0.361
- object never lifted (max rise 0.010 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 153 steps; TERMINATED by `ee_ground_collision` at step 152
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 1, 2 (PUSH_ADVANCE): 4, 3 (RETREAT): 18, 4 (SIDE_HOVER): 37, 5 (SIDE_INSERT): 34, 6 (CLOSE): 6, 8 (SIDE_DROP): 30
- aperture min -0.006 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.082 at step 22 (final 0.105)
- closest |object_to_goal| 0.077 at step 32 (final 0.188); goal_error min 0.082 at step 31 / final 0.193
- object never lifted (max rise 0.014 m)

### env 17 — never_reached
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 41, 8 (SIDE_DROP): 29
- aperture min -0.012 / max 0.080 / final 0.005
- closest |gripper_to_object| 0.072 at step 99 (final 0.134)
- closest |object_to_goal| 0.143 at step 6 (final 0.155); goal_error min 0.152 at step 1 / final 0.166
- object never lifted (max rise 0.000 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 182 steps; TERMINATED by `ee_ground_collision` at step 181
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 7, 2 (PUSH_ADVANCE): 30, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 7, 8 (SIDE_DROP): 30
- aperture min -0.009 / max 0.080 / final 0.024
- closest |gripper_to_object| 0.044 at step 116 (final 0.144)
- closest |object_to_goal| 0.085 at step 66 (final 0.328); goal_error min 0.094 at step 66 / final 0.326
- object never lifted (max rise 0.016 m)

### env 19 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 183 steps; TERMINATED by `ee_ground_collision` at step 182
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 36, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 35, 6 (CLOSE): 5, 8 (SIDE_DROP): 31
- aperture min -0.005 / max 0.081 / final 0.041
- closest |gripper_to_object| 0.051 at step 118 (final 0.105)
- closest |object_to_goal| 0.069 at step 104 (final 0.186); goal_error min 0.075 at step 104 / final 0.176
- object never lifted (max rise 0.003 m)

### env 20 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 179 steps; TERMINATED by `ee_ground_collision` at step 178
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 8, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 38, 6 (CLOSE): 5, 8 (SIDE_DROP): 33
- aperture min -0.000 / max 0.081 / final 0.047
- closest |gripper_to_object| 0.069 at step 166 (final 0.093)
- closest |object_to_goal| 0.082 at step 67 (final 0.112); goal_error min 0.086 at step 63 / final 0.113
- object never lifted (max rise 0.008 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 129 steps; TERMINATED by `ee_ground_collision` at step 128
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 22, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 13, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 14, 8 (SIDE_DROP): 29
- aperture min -0.012 / max 0.080 / final 0.045
- closest |gripper_to_object| 0.059 at step 128 (final 0.059)
- closest |object_to_goal| 0.072 at step 44 (final 0.157); goal_error min 0.076 at step 44 / final 0.150
- object never lifted (max rise 0.016 m)

### env 26 — never_lifted
- ended in phase 5 (SIDE_INSERT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 25, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 92, 6 (CLOSE): 28, 8 (SIDE_DROP): 29
- aperture min -0.011 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.021 at step 83 (final 0.108)
- closest |object_to_goal| 0.139 at step 24 (final 0.165); goal_error min 0.144 at step 28 / final 0.171
- object never lifted (max rise 0.000 m)

### env 28 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 166 steps; TERMINATED by `ee_ground_collision` at step 165
- most steps in phase 4 (SIDE_HOVER); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 8, 2 (PUSH_ADVANCE): 14, 3 (RETREAT): 18, 4 (SIDE_HOVER): 35, 5 (SIDE_INSERT): 33, 6 (CLOSE): 5, 8 (SIDE_DROP): 29
- aperture min -0.002 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.081 at step 27 (final 0.112)
- closest |object_to_goal| 0.069 at step 67 (final 0.187); goal_error min 0.080 at step 64 / final 0.196
- object never lifted (max rise 0.012 m)

### env 29 — terminated:ee_ground_collision
- ended in phase 6 (CLOSE) after 150 steps; TERMINATED by `ee_ground_collision` at step 149
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 2, 2 (PUSH_ADVANCE): 2, 3 (RETREAT): 18, 4 (SIDE_HOVER): 34, 5 (SIDE_INSERT): 35, 6 (CLOSE): 5, 8 (SIDE_DROP): 30
- aperture min -0.011 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.053 at step 101 (final 0.078)
- closest |object_to_goal| 0.045 at step 30 (final 0.195); goal_error min 0.051 at step 29 / final 0.187
- object LIFTED at step 27 (max rise 0.024 m); LOST at step 28 (aperture collapsed while the object moved away)

### env 30 — never_reached
- ended in phase 5 (SIDE_INSERT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 99, 6 (CLOSE): 28, 8 (SIDE_DROP): 29
- aperture min -0.003 / max 0.092 / final 0.080
- closest |gripper_to_object| 0.060 at step 21 (final 0.143)
- closest |object_to_goal| 0.154 at step 3 (final 0.174); goal_error min 0.161 at step 22 / final 0.182
- object never lifted (max rise 0.000 m)

### env 31 — never_lifted
- ended in phase 5 (SIDE_INSERT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 24, 1 (PUSH_DESCEND): 26, 2 (PUSH_ADVANCE): 51, 3 (RETREAT): 18, 4 (SIDE_HOVER): 31, 5 (SIDE_INSERT): 93, 6 (CLOSE): 28, 8 (SIDE_DROP): 29
- aperture min -0.013 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.012 at step 96 (final 0.115)
- closest |object_to_goal| 0.141 at step 14 (final 0.220); goal_error min 0.146 at step 27 / final 0.218
- object never lifted (max rise 0.002 m)

## Successful envs (one line each)

- env 0: success at step 32, end phase 6 (CLOSE), aperture min -0.003, min |go| 0.066, lifted at 32
- env 1: success at step 23, end phase 1 (PUSH_DESCEND), aperture min -0.013, min |go| 0.084, lifted at 23
- env 7: success at step 28, end phase 6 (CLOSE), aperture min -0.008, min |go| 0.045, lifted at 29
- env 10: success at step 33, end phase 6 (CLOSE), aperture min -0.021, min |go| 0.029, lifted at 32
- env 12: success at step 20, end phase 6 (CLOSE), aperture min 0.000, min |go| 0.087, lifted at 19
- env 14: success at step 85, end phase 5 (SIDE_INSERT), aperture min -0.014, min |go| 0.037, lifted at 85
- env 16: success at step 41, end phase 6 (CLOSE), aperture min -0.003, min |go| 0.040, lifted at 41
- env 21: success at step 85, end phase 8 (SIDE_DROP), aperture min -0.014, min |go| 0.069, lifted at 86
- env 22: success at step 24, end phase 6 (CLOSE), aperture min 0.000, min |go| 0.059, lifted at 24
- env 23: success at step 99, end phase 7 (LIFT), aperture min -0.015, min |go| 0.032, lifted at 99
- env 24: success at step 29, end phase 5 (SIDE_INSERT), aperture min -0.008, min |go| 0.060, lifted at 29
- env 27: success at step 27, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.092, lifted at 27
