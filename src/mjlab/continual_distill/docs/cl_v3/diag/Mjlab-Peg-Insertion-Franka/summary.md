# diagnose — Mjlab-Peg-Insertion-Franka

2026-09-09T15:58:08 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**3/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `PegInsertionClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 5 = PLACE, 6 = RELEASE

## Histograms

- end phase, FAILING envs (29): 4: 16, 0 (HOVER): 4, 3: 3, 2: 2, 5 (PLACE): 2, 1: 1, 8: 1
- end phase, successful envs (3): 8: 2, 5 (PLACE): 1
- most-steps phase, failing envs: 4: 23, 0 (HOVER): 4, 8: 1, 1: 1
- failure class: terminated:ee_ground_collision: 23, never_lifted: 4, near_miss: 2
- termination among failing envs: ee_ground_collision: 23, none: 6
- failing envs that lifted the object: 5/29; lifted then lost: 4
- success step (successful envs): median 133, max 207

## Reading

- **terminated:ee_ground_collision** — 23/29 failing envs (envs [0, 2, 4, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 31]); typical end phase 4; final aperture median 0.000; closest |gripper_to_object| median 0.047; closest |object_to_goal| median 0.247; min goal_error median 0.254 (success < 0.015)
- **never_lifted** — 4/29 failing envs (envs [3, 5, 6, 27]); typical end phase 0 (HOVER); final aperture median 0.080; closest |gripper_to_object| median 0.019; closest |object_to_goal| median 0.285; min goal_error median 0.288 (success < 0.015)
- **near_miss** — 2/29 failing envs (envs [13, 29]); typical end phase 8; final aperture median 0.040; closest |gripper_to_object| median 0.041; closest |object_to_goal| median 0.050; min goal_error median 0.019 (success < 0.015)

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 2 after 310 steps; TERMINATED by `ee_ground_collision` at step 309
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 104, 1: 62, 2: 23, 3: 20, 4: 101
- aperture min -0.000 / max 0.082 / final 0.008
- closest |gripper_to_object| 0.035 at step 273 (final 0.066)
- closest |object_to_goal| 0.430 at step 30 (final 0.455); goal_error min 0.406 at step 256 / final 0.411 (success < 0.015)
- object never lifted (max rise 0.003 m)

### env 2 — terminated:ee_ground_collision
- ended in phase 4 after 249 steps; TERMINATED by `ee_ground_collision` at step 248
- most steps in phase 4; steps per phase 0 (HOVER): 56, 1: 61, 2: 12, 3: 20, 4: 100
- aperture min -0.000 / max 0.083 / final 0.000
- closest |gripper_to_object| 0.050 at step 124 (final 0.611)
- closest |object_to_goal| 0.392 at step 15 (final 0.411); goal_error min 0.368 at step 155 / final 0.369 (success < 0.015)
- object never lifted (max rise 0.003 m)

### env 3 — never_lifted
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0 (HOVER): 304, 1: 164, 2: 48, 3: 80, 4: 404
- aperture min -0.000 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.021 at step 717 (final 0.064)
- closest |object_to_goal| 0.275 at step 658 (final 0.287); goal_error min 0.237 at step 236 / final 0.238 (success < 0.015)
- object never lifted (max rise 0.004 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 4 after 183 steps; TERMINATED by `ee_ground_collision` at step 182
- most steps in phase 4; steps per phase 0 (HOVER): 54, 1: 5, 2: 12, 3: 20, 4: 92
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.028 at step 80 (final 0.503)
- closest |object_to_goal| 0.247 at step 48 (final 0.277); goal_error min 0.254 at step 3 / final 0.334 (success < 0.015)
- object never lifted (max rise 0.003 m)

### env 5 — never_lifted
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0 (HOVER): 280, 1: 188, 2: 48, 3: 80, 4: 404
- aperture min -0.009 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.017 at step 833 (final 0.413)
- closest |object_to_goal| 0.295 at step 68 (final 0.455); goal_error min 0.303 at step 1 / final 0.468 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 6 — never_lifted
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0 (HOVER): 322, 1: 146, 2: 48, 3: 80, 4: 404
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.013 at step 104 (final 0.212)
- closest |object_to_goal| 0.350 at step 75 (final 0.439); goal_error min 0.315 at step 76 / final 0.388 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 7 — terminated:ee_ground_collision
- ended in phase 4 after 237 steps; TERMINATED by `ee_ground_collision` at step 236
- most steps in phase 4; steps per phase 0 (HOVER): 51, 1: 61, 2: 12, 3: 20, 4: 93
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.025 at step 117 (final 0.499)
- closest |object_to_goal| 0.264 at step 82 (final 0.301); goal_error min 0.259 at step 74 / final 0.339 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 3 after 531 steps; TERMINATED by `ee_ground_collision` at step 530
- most steps in phase 4; steps per phase 0 (HOVER): 175, 1: 75, 2: 36, 3: 43, 4: 202
- aperture min -0.000 / max 0.080 / final 0.059
- closest |gripper_to_object| 0.026 at step 283 (final 0.034)
- closest |object_to_goal| 0.249 at step 96 (final 0.314); goal_error min 0.256 at step 1 / final 0.289 (success < 0.015)
- object never lifted (max rise 0.001 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 4 after 172 steps; TERMINATED by `ee_ground_collision` at step 171
- most steps in phase 4; steps per phase 0 (HOVER): 48, 1: 16, 2: 12, 3: 20, 4: 76
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.051 at step 70 (final 0.445)
- closest |object_to_goal| 0.195 at step 130 (final 0.212); goal_error min 0.175 at step 109 / final 0.175 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 4 after 163 steps; TERMINATED by `ee_ground_collision` at step 162
- most steps in phase 4; steps per phase 0 (HOVER): 47, 1: 2, 2: 12, 3: 20, 4: 82
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.065 at step 51 (final 0.551)
- closest |object_to_goal| 0.199 at step 66 (final 0.275); goal_error min 0.205 at step 3 / final 0.318 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 184 steps; TERMINATED by `ee_ground_collision` at step 183
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 52, 1: 38, 2: 12, 3: 20, 4: 42, 5 (PLACE): 20
- aperture min -0.001 / max 0.087 / final 0.002
- closest |gripper_to_object| 0.050 at step 101 (final 0.084)
- closest |object_to_goal| 0.074 at step 166 (final 0.094); goal_error min 0.090 at step 163 / final 0.098 (success < 0.015)
- object LIFTED at step 113 (max rise 0.093 m); LOST at step 114 (aperture collapsed while the object moved away)

### env 13 — near_miss
- ended in phase 8 after 1000 steps; ran to time-out
- most steps in phase 8; steps per phase 0 (HOVER): 31, 1: 5, 2: 12, 3: 20, 4: 42, 5 (PLACE): 71, 6 (RELEASE): 16, 7: 30, 8: 773
- aperture min 0.023 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.055 at step 217 (final 0.879)
- closest |object_to_goal| 0.047 at step 110 (final 0.090); goal_error min 0.019 at step 189 / final 0.036 (success < 0.015)
- object LIFTED at step 65 (max rise 0.121 m); never lost

### env 14 — terminated:ee_ground_collision
- ended in phase 3 after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 40, 1: 1, 2: 12, 3: 3
- aperture min 0.003 / max 0.080 / final 0.003
- closest |gripper_to_object| 0.046 at step 47 (final 0.055)
- closest |object_to_goal| 0.293 at step 44 (final 0.305); goal_error min 0.300 at step 42 / final 0.306 (success < 0.015)
- object never lifted (max rise 0.001 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 4 after 148 steps; TERMINATED by `ee_ground_collision` at step 147
- most steps in phase 4; steps per phase 0 (HOVER): 28, 1: 11, 2: 12, 3: 20, 4: 77
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.056 at step 42 (final 0.392)
- closest |object_to_goal| 0.123 at step 73 (final 0.129); goal_error min 0.092 at step 72 / final 0.115 (success < 0.015)
- object never lifted (max rise 0.001 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 4 after 167 steps; TERMINATED by `ee_ground_collision` at step 166
- most steps in phase 4; steps per phase 0 (HOVER): 46, 1: 15, 2: 12, 3: 20, 4: 74
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.047 at step 66 (final 0.486)
- closest |object_to_goal| 0.149 at step 1 (final 0.177); goal_error min 0.132 at step 81 / final 0.148 (success < 0.015)
- object never lifted (max rise 0.008 m)

### env 17 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 252 steps; TERMINATED by `ee_ground_collision` at step 251
- most steps in phase 4; steps per phase 0 (HOVER): 58, 1: 61, 2: 12, 3: 20, 4: 101
- aperture min -0.000 / max 0.082 / final 0.039
- closest |gripper_to_object| 0.020 at step 124 (final 0.501)
- closest |object_to_goal| 0.218 at step 138 (final 0.232); goal_error min 0.220 at step 88 / final 0.278 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 19 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 192 steps; TERMINATED by `ee_ground_collision` at step 191
- most steps in phase 1; steps per phase 0 (HOVER): 40, 1: 61, 2: 12, 3: 20, 4: 45, 5 (PLACE): 14
- aperture min 0.010 / max 0.088 / final 0.011
- closest |gripper_to_object| 0.053 at step 64 (final 0.086)
- closest |object_to_goal| 0.011 at step 191 (final 0.011); goal_error min 0.063 at step 190 / final 0.063 (success < 0.015)
- object LIFTED at step 126 (max rise 0.092 m); LOST at step 128 (aperture collapsed while the object moved away)

### env 20 — terminated:ee_ground_collision
- ended in phase 4 after 149 steps; TERMINATED by `ee_ground_collision` at step 148
- most steps in phase 4; steps per phase 0 (HOVER): 34, 1: 4, 2: 12, 3: 20, 4: 79
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.059 at step 41 (final 0.566)
- closest |object_to_goal| 0.256 at step 13 (final 0.288); goal_error min 0.262 at step 1 / final 0.295 (success < 0.015)
- object never lifted (max rise 0.005 m)

### env 21 — terminated:ee_ground_collision
- ended in phase 4 after 242 steps; TERMINATED by `ee_ground_collision` at step 241
- most steps in phase 4; steps per phase 0 (HOVER): 54, 1: 61, 2: 12, 3: 20, 4: 95
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.017 at step 67 (final 0.528)
- closest |object_to_goal| 0.280 at step 113 (final 0.288); goal_error min 0.321 at step 73 / final 0.324 (success < 0.015)
- object never lifted (max rise 0.001 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 181 steps; TERMINATED by `ee_ground_collision` at step 180
- most steps in phase 4; steps per phase 0 (HOVER): 43, 1: 5, 2: 12, 3: 20, 4: 101
- aperture min -0.000 / max 0.080 / final 0.049
- closest |gripper_to_object| 0.063 at step 45 (final 0.486)
- closest |object_to_goal| 0.310 at step 68 (final 0.329); goal_error min 0.318 at step 3 / final 0.348 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 23 — terminated:ee_ground_collision
- ended in phase 4 after 171 steps; TERMINATED by `ee_ground_collision` at step 170
- most steps in phase 4; steps per phase 0 (HOVER): 40, 1: 24, 2: 12, 3: 20, 4: 75
- aperture min -0.000 / max 0.083 / final -0.000
- closest |gripper_to_object| 0.053 at step 62 (final 0.374)
- closest |object_to_goal| 0.187 at step 84 (final 0.196); goal_error min 0.185 at step 57 / final 0.250 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 4 after 153 steps; TERMINATED by `ee_ground_collision` at step 152
- most steps in phase 4; steps per phase 0 (HOVER): 40, 1: 6, 2: 12, 3: 20, 4: 75
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.060 at step 49 (final 0.623)
- closest |object_to_goal| 0.407 at step 37 (final 0.458); goal_error min 0.404 at step 88 / final 0.405 (success < 0.015)
- object never lifted (max rise 0.011 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 4 after 173 steps; TERMINATED by `ee_ground_collision` at step 172
- most steps in phase 4; steps per phase 0 (HOVER): 44, 1: 15, 2: 12, 3: 20, 4: 82
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.050 at step 64 (final 0.558)
- closest |object_to_goal| 0.326 at step 37 (final 0.359); goal_error min 0.320 at step 64 / final 0.384 (success < 0.015)
- object never lifted (max rise 0.006 m)

### env 26 — terminated:ee_ground_collision
- ended in phase 3 after 314 steps; TERMINATED by `ee_ground_collision` at step 313
- most steps in phase 4; steps per phase 0 (HOVER): 82, 1: 86, 2: 24, 3: 21, 4: 101
- aperture min -0.000 / max 0.086 / final 0.005
- closest |gripper_to_object| 0.028 at step 274 (final 0.095)
- closest |object_to_goal| 0.095 at step 132 (final 0.169); goal_error min 0.091 at step 132 / final 0.180 (success < 0.015)
- object LIFTED at step 88 (max rise 0.125 m); LOST at step 139 (aperture collapsed while the object moved away)

### env 27 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0 (HOVER): 315, 1: 119, 2: 60, 3: 100, 4: 406
- aperture min -0.004 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.025 at step 117 (final 0.118)
- closest |object_to_goal| 0.264 at step 369 (final 0.282); goal_error min 0.273 at step 3 / final 0.298 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 28 — terminated:ee_ground_collision
- ended in phase 4 after 182 steps; TERMINATED by `ee_ground_collision` at step 181
- most steps in phase 4; steps per phase 0 (HOVER): 55, 1: 1, 2: 12, 3: 20, 4: 94
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.044 at step 76 (final 0.444)
- closest |object_to_goal| 0.126 at step 87 (final 0.139); goal_error min 0.131 at step 63 / final 0.147 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 29 — near_miss
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0 (HOVER): 240, 1: 121, 2: 60, 3: 100, 4: 479
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.026 at step 517 (final 0.585)
- closest |object_to_goal| 0.053 at step 616 (final 0.072); goal_error min 0.018 at step 616 / final 0.020 (success < 0.015)
- object LIFTED at step 519 (max rise 0.098 m); LOST at step 656 (aperture collapsed while the object moved away)

### env 30 — terminated:ee_ground_collision
- ended in phase 2 after 59 steps; TERMINATED by `ee_ground_collision` at step 58
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 40, 1: 11, 2: 8
- aperture min 0.018 / max 0.080 / final 0.018
- closest |gripper_to_object| 0.047 at step 53 (final 0.048)
- closest |object_to_goal| 0.391 at step 23 (final 0.418); goal_error min 0.399 at step 1 / final 0.468 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 31 — terminated:ee_ground_collision
- ended in phase 4 after 156 steps; TERMINATED by `ee_ground_collision` at step 155
- most steps in phase 4; steps per phase 0 (HOVER): 35, 1: 14, 2: 12, 3: 20, 4: 75
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.047 at step 57 (final 0.478)
- closest |object_to_goal| 0.236 at step 112 (final 0.241); goal_error min 0.197 at step 80 / final 0.201 (success < 0.015)
- object never lifted (max rise 0.004 m)

## Successful envs (one line each)

- env 1: success at step 132, end phase 8, aperture min 0.024, min |go| 0.055, lifted at 82
- env 10: success at step 207, end phase 5 (PLACE), aperture min 0.001, min |go| 0.047, lifted at 148
- env 18: success at step 133, end phase 8, aperture min 0.024, min |go| 0.055, lifted at 69
