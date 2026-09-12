# diagnose — Mjlab-Peg-Insertion-Franka

2026-09-10T05:00:21 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**7/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `PegInsertionClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 4 = CARRY, 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (25): 1: 10, 8 (DONE): 4, 5 (PLACE): 4, 3: 3, 0 (HOVER): 1, 6 (RELEASE): 1, 2: 1, 4 (CARRY): 1
- end phase, successful envs (7): 5 (PLACE): 3, 1: 2, 8 (DONE): 1, 2: 1
- most-steps phase, failing envs: 1: 8, 5 (PLACE): 7, 8 (DONE): 4, 4 (CARRY): 4, 0 (HOVER): 2
- failure class: terminated:ee_ground_collision: 14, grasp_lost: 6, lifted_not_at_goal: 3, never_lifted: 1, near_miss: 1
- termination among failing envs: ee_ground_collision: 14, none: 11
- failing envs that lifted the object: 21/25; lifted then lost: 17
- success step (successful envs): median 326, max 581

## Reading

- **terminated:ee_ground_collision** — 14/25 failing envs (envs [1, 5, 7, 8, 9, 10, 13, 15, 18, 24, 26, 28, 29, 30]); typical end phase 1; final aperture median 0.078; closest |gripper_to_object| median 0.024; closest |object_to_goal| median 0.029; min goal_error median 0.025 (success < 0.015)
- **grasp_lost** — 6/25 failing envs (envs [3, 19, 21, 22, 23, 27]); typical end phase 3; final aperture median 0.074; closest |gripper_to_object| median 0.012; closest |object_to_goal| median 0.101; min goal_error median 0.119 (success < 0.015); lost at step median 182
- **lifted_not_at_goal** — 3/25 failing envs (envs [6, 20, 25]); typical end phase 8 (DONE); final aperture median 0.080; closest |gripper_to_object| median 0.022; closest |object_to_goal| median 0.006; min goal_error median 0.035 (success < 0.015)
- **never_lifted** — 1/25 failing envs (envs [0]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.014; closest |object_to_goal| median 0.135; min goal_error median 0.154 (success < 0.015)
- **near_miss** — 1/25 failing envs (envs [17]); typical end phase 1; final aperture median 0.081; closest |gripper_to_object| median 0.003; closest |object_to_goal| median 0.017; min goal_error median 0.020 (success < 0.015)

## Failing envs

### env 0 — never_lifted
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 243, 1: 401, 2: 120, 3: 236
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.014 at step 864 (final 0.194)
- closest |object_to_goal| 0.135 at step 957 (final 0.149); goal_error min 0.154 at step 87 / final 0.196 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 1 — terminated:ee_ground_collision
- ended in phase 1 after 885 steps; TERMINATED by `ee_ground_collision` at step 884
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 173, 1: 178, 2: 30, 3: 41, 4 (CARRY): 221, 5 (PLACE): 242
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.003 at step 442 (final 0.062)
- closest |object_to_goal| 0.002 at step 230 (final 0.156); goal_error min 0.021 at step 300 / final 0.126 (success < 0.015)
- object LIFTED at step 96 (max rise 0.176 m); LOST at step 448 (aperture collapsed while the object moved away)

### env 3 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 255, 1: 392, 2: 90, 3: 162, 4 (CARRY): 101
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.011 at step 210 (final 0.192)
- closest |object_to_goal| 0.070 at step 304 (final 0.109); goal_error min 0.058 at step 183 / final 0.151 (success < 0.015)
- object LIFTED at step 83 (max rise 0.160 m); LOST at step 367 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 1 after 925 steps; TERMINATED by `ee_ground_collision` at step 924
- most steps in phase 1; steps per phase 0 (HOVER): 150, 1: 336, 2: 45, 3: 84, 4 (CARRY): 189, 5 (PLACE): 121
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.008 at step 924 (final 0.008)
- closest |object_to_goal| 0.067 at step 313 (final 0.218); goal_error min 0.040 at step 314 / final 0.175 (success < 0.015)
- object LIFTED at step 188 (max rise 0.176 m); LOST at step 326 (aperture collapsed while the object moved away)

### env 6 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 57, 1: 33, 2: 15, 3: 23, 4 (CARRY): 68, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 637
- aperture min 0.024 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.022 at step 340 (final 1.066)
- closest |object_to_goal| 0.023 at step 336 (final 0.131); goal_error min 0.037 at step 228 / final 0.136 (success < 0.015)
- object LIFTED at step 110 (max rise 0.174 m); never lost

### env 7 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 267 steps; TERMINATED by `ee_ground_collision` at step 266
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 43, 1: 19, 2: 15, 3: 27, 4 (CARRY): 85, 5 (PLACE): 78
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.028 at step 68 (final 0.282)
- closest |object_to_goal| 0.013 at step 218 (final 0.029); goal_error min 0.037 at step 188 / final 0.056 (success < 0.015)
- object LIFTED at step 83 (max rise 0.171 m); LOST at step 216 (aperture collapsed while the object moved away)

### env 8 — terminated:ee_ground_collision
- ended in phase 1 after 153 steps; TERMINATED by `ee_ground_collision` at step 152
- most steps in phase 1; steps per phase 0 (HOVER): 71, 1: 82
- aperture min 0.011 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.080 at step 124 (final 0.321)
- closest |object_to_goal| 0.357 at step 31 (final 0.389); goal_error min 0.358 at step 96 / final 0.441 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 379 steps; TERMINATED by `ee_ground_collision` at step 378
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 48, 1: 21, 2: 15, 3: 29, 4 (CARRY): 86, 5 (PLACE): 180
- aperture min 0.000 / max 0.087 / final 0.000
- closest |gripper_to_object| 0.017 at step 289 (final 0.304)
- closest |object_to_goal| 0.008 at step 263 (final 0.095); goal_error min 0.021 at step 329 / final 0.095 (success < 0.015)
- object LIFTED at step 89 (max rise 0.191 m); LOST at step 360 (aperture collapsed while the object moved away)

### env 10 — terminated:ee_ground_collision
- ended in phase 1 after 567 steps; TERMINATED by `ee_ground_collision` at step 566
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 118, 1: 95, 2: 15, 3: 26, 4 (CARRY): 192, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.011 at step 565 (final 0.017)
- closest |object_to_goal| 0.059 at step 215 (final 0.193); goal_error min 0.023 at step 217 / final 0.148 (success < 0.015)
- object LIFTED at step 89 (max rise 0.173 m); LOST at step 223 (aperture collapsed while the object moved away)

### env 13 — terminated:ee_ground_collision
- ended in phase 1 after 111 steps; TERMINATED by `ee_ground_collision` at step 110
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1: 40
- aperture min 0.076 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.025 at step 109 (final 0.035)
- closest |object_to_goal| 0.344 at step 21 (final 0.373); goal_error min 0.352 at step 1 / final 0.422 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 325 steps; TERMINATED by `ee_ground_collision` at step 324
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 47, 1: 22, 2: 15, 3: 24, 4 (CARRY): 96, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.030 at step 182 (final 0.427)
- closest |object_to_goal| 0.045 at step 216 (final 0.050); goal_error min 0.024 at step 224 / final 0.024 (success < 0.015)
- object LIFTED at step 89 (max rise 0.146 m); LOST at step 219 (aperture collapsed while the object moved away)

### env 17 — near_miss
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 215, 1: 223, 2: 60, 3: 109, 4 (CARRY): 272, 5 (PLACE): 121
- aperture min -0.012 / max 0.082 / final 0.081
- closest |gripper_to_object| 0.003 at step 622 (final 0.016)
- closest |object_to_goal| 0.017 at step 393 (final 0.118); goal_error min 0.020 at step 348 / final 0.130 (success < 0.015)
- object LIFTED at step 188 (max rise 0.182 m); LOST at step 656 (aperture collapsed while the object moved away)

### env 18 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 337 steps; TERMINATED by `ee_ground_collision` at step 336
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 50, 1: 25, 2: 15, 3: 25, 4 (CARRY): 88, 5 (PLACE): 121, 6 (RELEASE): 13
- aperture min 0.002 / max 0.080 / final 0.075
- closest |gripper_to_object| 0.035 at step 84 (final 0.233)
- closest |object_to_goal| 0.005 at step 330 (final 0.017); goal_error min 0.019 at step 248 / final 0.045 (success < 0.015)
- object LIFTED at step 96 (max rise 0.171 m); LOST at step 317 (aperture collapsed while the object moved away)

### env 19 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 50, 1: 19, 2: 15, 3: 30, 4 (CARRY): 29, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 690
- aperture min -0.003 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.004 at step 191 (final 1.092)
- closest |object_to_goal| 0.001 at step 181 (final 0.022); goal_error min 0.042 at step 146 / final 0.048 (success < 0.015)
- object LIFTED at step 99 (max rise 0.133 m); LOST at step 246 (aperture collapsed while the object moved away)

### env 20 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 43, 1: 121, 2: 15, 3: 27, 4 (CARRY): 98, 5 (PLACE): 242, 6 (RELEASE): 16, 7: 30, 8 (DONE): 408
- aperture min 0.024 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.022 at step 385 (final 1.064)
- closest |object_to_goal| 0.003 at step 527 (final 0.045); goal_error min 0.025 at step 441 / final 0.037 (success < 0.015)
- object LIFTED at step 184 (max rise 0.175 m); never lost

### env 21 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 252, 1: 305, 2: 135, 3: 207, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.015 at step 161 (final 0.162)
- closest |object_to_goal| 0.055 at step 168 (final 0.120); goal_error min 0.070 at step 165 / final 0.095 (success < 0.015)
- object LIFTED at step 88 (max rise 0.175 m); LOST at step 165 (aperture collapsed while the object moved away)

### env 22 — grasp_lost
- ended in phase 2 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 157, 1: 366, 2: 183, 3: 294
- aperture min -0.000 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.012 at step 159 (final 0.034)
- closest |object_to_goal| 0.211 at step 237 (final 0.215); goal_error min 0.265 at step 175 / final 0.265 (success < 0.015)
- object LIFTED at step 138 (max rise 0.129 m); LOST at step 199 (aperture collapsed while the object moved away)

### env 23 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 160, 1: 333, 2: 195, 3: 312
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.007 at step 117 (final 0.179)
- closest |object_to_goal| 0.204 at step 895 (final 0.218); goal_error min 0.254 at step 141 / final 0.254 (success < 0.015)
- object LIFTED at step 94 (max rise 0.139 m); LOST at step 156 (aperture collapsed while the object moved away)

### env 24 — terminated:ee_ground_collision
- ended in phase 4 (CARRY) after 209 steps; TERMINATED by `ee_ground_collision` at step 208
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 48, 1: 18, 2: 15, 3: 29, 4 (CARRY): 99
- aperture min 0.024 / max 0.080 / final 0.025
- closest |gripper_to_object| 0.023 at step 160 (final 0.034)
- closest |object_to_goal| 0.079 at step 134 (final 0.106); goal_error min 0.079 at step 181 / final 0.161 (success < 0.015)
- object LIFTED at step 86 (max rise 0.178 m); never lost

### env 25 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 45, 1: 21, 2: 15, 3: 29, 4 (CARRY): 71, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 652
- aperture min 0.024 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.012 at step 278 (final 1.070)
- closest |object_to_goal| 0.006 at step 267 (final 0.045); goal_error min 0.035 at step 289 / final 0.072 (success < 0.015)
- object LIFTED at step 87 (max rise 0.187 m); never lost

### env 26 — terminated:ee_ground_collision
- ended in phase 1 after 849 steps; TERMINATED by `ee_ground_collision` at step 848
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 98, 1: 196, 2: 30, 3: 58, 4 (CARRY): 225, 5 (PLACE): 242
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 658 (final 0.021)
- closest |object_to_goal| 0.005 at step 392 (final 0.295); goal_error min 0.027 at step 275 / final 0.307 (success < 0.015)
- object LIFTED at step 101 (max rise 0.174 m); LOST at step 762 (aperture collapsed while the object moved away)

### env 27 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 205, 1: 315, 2: 180, 3: 300
- aperture min -0.000 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.020 at step 455 (final 0.044)
- closest |object_to_goal| 0.133 at step 709 (final 0.137); goal_error min 0.168 at step 148 / final 0.172 (success < 0.015)
- object LIFTED at step 129 (max rise 0.032 m); LOST at step 137 (aperture collapsed while the object moved away)

### env 28 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 358 steps; TERMINATED by `ee_ground_collision` at step 357
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 38, 1: 19, 2: 15, 3: 28, 4 (CARRY): 40, 5 (PLACE): 218
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.022 at step 138 (final 0.409)
- closest |object_to_goal| 0.002 at step 233 (final 0.012); goal_error min 0.024 at step 186 / final 0.060 (success < 0.015)
- object LIFTED at step 77 (max rise 0.183 m); LOST at step 196 (aperture collapsed while the object moved away)

### env 29 — terminated:ee_ground_collision
- ended in phase 1 after 77 steps; TERMINATED by `ee_ground_collision` at step 76
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 46, 1: 31
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.026 at step 66 (final 0.036)
- closest |object_to_goal| 0.290 at step 54 (final 0.299); goal_error min 0.296 at step 76 / final 0.296 (success < 0.015)
- object never lifted (max rise 0.000 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 3 after 758 steps; TERMINATED by `ee_ground_collision` at step 757
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 110, 1: 144, 2: 30, 3: 41, 4 (CARRY): 191, 5 (PLACE): 242
- aperture min -0.000 / max 0.083 / final 0.044
- closest |gripper_to_object| 0.024 at step 249 (final 0.185)
- closest |object_to_goal| 0.009 at step 386 (final 0.129); goal_error min 0.022 at step 250 / final 0.104 (success < 0.015)
- object LIFTED at step 83 (max rise 0.185 m); LOST at step 390 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 2: success at step 326, end phase 1, aperture min -0.000, min |go| 0.018, lifted at 186
- env 4: success at step 217, end phase 8 (DONE), aperture min 0.024, min |go| 0.028, lifted at 89
- env 11: success at step 581, end phase 1, aperture min -0.000, min |go| 0.008, lifted at 187
- env 12: success at step 243, end phase 5 (PLACE), aperture min 0.024, min |go| 0.028, lifted at 96
- env 14: success at step 465, end phase 5 (PLACE), aperture min -0.000, min |go| 0.015, lifted at 190
- env 16: success at step 220, end phase 2, aperture min -0.000, min |go| 0.018, lifted at 76
- env 31: success at step 473, end phase 5 (PLACE), aperture min -0.000, min |go| 0.016, lifted at 90
