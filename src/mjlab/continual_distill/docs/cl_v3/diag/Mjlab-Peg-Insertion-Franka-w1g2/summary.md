# diagnose — Mjlab-Peg-Insertion-Franka

2026-09-10T03:01:21 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**4/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `PegInsertionClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 4 = CARRY, 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (28): 1: 9, 8 (DONE): 6, 5 (PLACE): 5, 3: 3, 2: 2, 0 (HOVER): 1, 4 (CARRY): 1, 6 (RELEASE): 1
- end phase, successful envs (4): 1: 3, 5 (PLACE): 1
- most-steps phase, failing envs: 1: 14, 8 (DONE): 6, 5 (PLACE): 6, 0 (HOVER): 1, 4 (CARRY): 1
- failure class: grasp_lost: 17, terminated:ee_ground_collision: 7, lifted_not_at_goal: 2, near_miss: 1, never_lifted: 1
- termination among failing envs: none: 21, ee_ground_collision: 7
- failing envs that lifted the object: 27/28; lifted then lost: 25
- success step (successful envs): median 201, max 228

## Reading

- **grasp_lost** — 17/28 failing envs (envs [1, 3, 7, 8, 9, 10, 11, 12, 14, 19, 20, 21, 23, 24, 27, 28, 29]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.020; closest |object_to_goal| median 0.027; min goal_error median 0.053 (success < 0.015); lost at step median 175
- **terminated:ee_ground_collision** — 7/28 failing envs (envs [4, 6, 16, 17, 26, 30, 31]); typical end phase 5 (PLACE); final aperture median 0.000; closest |gripper_to_object| median 0.053; closest |object_to_goal| median 0.004; min goal_error median 0.025 (success < 0.015)
- **lifted_not_at_goal** — 2/28 failing envs (envs [2, 5]); typical end phase 8 (DONE); final aperture median 0.080; closest |gripper_to_object| median 0.036; closest |object_to_goal| median 0.005; min goal_error median 0.032 (success < 0.015)
- **near_miss** — 1/28 failing envs (envs [18]); typical end phase 8 (DONE); final aperture median 0.080; closest |gripper_to_object| median 0.060; closest |object_to_goal| median 0.001; min goal_error median 0.021 (success < 0.015)
- **never_lifted** — 1/28 failing envs (envs [22]); typical end phase 1; final aperture median 0.079; closest |gripper_to_object| median 0.043; closest |object_to_goal| median 0.338; min goal_error median 0.349 (success < 0.015)

## Failing envs

### env 1 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 210, 1: 302, 2: 90, 3: 117, 4 (CARRY): 160, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.035
- closest |gripper_to_object| 0.012 at step 471 (final 0.030)
- closest |object_to_goal| 0.003 at step 257 (final 0.132); goal_error min 0.034 at step 219 / final 0.110 (success < 0.015)
- object LIFTED at step 84 (max rise 0.167 m); LOST at step 285 (aperture collapsed while the object moved away)

### env 2 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 28, 1: 23, 2: 15, 3: 18, 4 (CARRY): 28, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 721
- aperture min 0.024 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.023 at step 252 (final 0.737)
- closest |object_to_goal| 0.006 at step 196 (final 0.034); goal_error min 0.035 at step 149 / final 0.069 (success < 0.015)
- object LIFTED at step 72 (max rise 0.165 m); never lost

### env 3 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 198, 1: 339, 2: 90, 3: 135, 4 (CARRY): 117, 5 (PLACE): 121
- aperture min -0.000 / max 0.088 / final 0.000
- closest |gripper_to_object| 0.020 at step 757 (final 0.190)
- closest |object_to_goal| 0.018 at step 149 (final 0.091); goal_error min 0.049 at step 167 / final 0.113 (success < 0.015)
- object LIFTED at step 83 (max rise 0.163 m); LOST at step 175 (aperture collapsed while the object moved away)

### env 4 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 298 steps; TERMINATED by `ee_ground_collision` at step 297
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 40, 1: 23, 2: 15, 3: 19, 4 (CARRY): 66, 5 (PLACE): 135
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.058 at step 189 (final 0.550)
- closest |object_to_goal| 0.004 at step 248 (final 0.016); goal_error min 0.025 at step 199 / final 0.039 (success < 0.015)
- object LIFTED at step 85 (max rise 0.160 m); LOST at step 206 (aperture collapsed while the object moved away)

### env 5 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 47, 1: 26, 2: 15, 3: 19, 4 (CARRY): 16, 5 (PLACE): 83, 6 (RELEASE): 16, 7: 30, 8 (DONE): 748
- aperture min 0.020 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.050 at step 231 (final 0.772)
- closest |object_to_goal| 0.005 at step 208 (final 0.053); goal_error min 0.029 at step 154 / final 0.074 (success < 0.015)
- object LIFTED at step 95 (max rise 0.157 m); never lost

### env 6 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 200 steps; TERMINATED by `ee_ground_collision` at step 199
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 27, 1: 23, 2: 15, 3: 19, 4 (CARRY): 40, 5 (PLACE): 76
- aperture min 0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.053 at step 166 (final 0.315)
- closest |object_to_goal| 0.015 at step 167 (final 0.172); goal_error min 0.025 at step 165 / final 0.148 (success < 0.015)
- object LIFTED at step 71 (max rise 0.166 m); LOST at step 154 (aperture collapsed while the object moved away)

### env 7 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 145, 1: 569, 2: 75, 3: 110, 4 (CARRY): 101
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.017 at step 659 (final 0.047)
- closest |object_to_goal| 0.046 at step 168 (final 0.106); goal_error min 0.077 at step 141 / final 0.147 (success < 0.015)
- object LIFTED at step 84 (max rise 0.160 m); LOST at step 170 (aperture collapsed while the object moved away)

### env 8 — grasp_lost
- ended in phase 2 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 225, 1: 213, 2: 141, 3: 168, 4 (CARRY): 132, 5 (PLACE): 121
- aperture min -0.008 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.040 at step 113 (final 0.057)
- closest |object_to_goal| 0.041 at step 243 (final 0.169); goal_error min 0.034 at step 241 / final 0.121 (success < 0.015)
- object LIFTED at step 72 (max rise 0.165 m); LOST at step 142 (aperture collapsed while the object moved away)

### env 9 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 188, 1: 447, 2: 105, 3: 159, 4 (CARRY): 101
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.005 at step 651 (final 0.037)
- closest |object_to_goal| 0.028 at step 186 (final 0.200); goal_error min 0.069 at step 186 / final 0.251 (success < 0.015)
- object LIFTED at step 77 (max rise 0.165 m); LOST at step 263 (aperture collapsed while the object moved away)

### env 10 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 48, 1: 23, 2: 15, 3: 19, 4 (CARRY): 47, 5 (PLACE): 74, 6 (RELEASE): 16, 7: 30, 8 (DONE): 728
- aperture min 0.007 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.058 at step 163 (final 0.774)
- closest |object_to_goal| 0.007 at step 226 (final 0.061); goal_error min 0.032 at step 189 / final 0.035 (success < 0.015)
- object LIFTED at step 92 (max rise 0.163 m); LOST at step 207 (aperture collapsed while the object moved away)

### env 11 — grasp_lost
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 170, 1: 420, 2: 75, 3: 119, 4 (CARRY): 216
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.018 at step 940 (final 0.147)
- closest |object_to_goal| 0.011 at step 274 (final 0.188); goal_error min 0.044 at step 272 / final 0.142 (success < 0.015)
- object LIFTED at step 72 (max rise 0.158 m); LOST at step 310 (aperture collapsed while the object moved away)

### env 12 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 192, 1: 370, 2: 150, 3: 187, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.034 at step 633 (final 0.051)
- closest |object_to_goal| 0.044 at step 155 (final 0.138); goal_error min 0.061 at step 155 / final 0.192 (success < 0.015)
- object LIFTED at step 78 (max rise 0.161 m); LOST at step 164 (aperture collapsed while the object moved away)

### env 14 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 236, 1: 307, 2: 60, 3: 99, 4 (CARRY): 177, 5 (PLACE): 121
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.018 at step 649 (final 0.043)
- closest |object_to_goal| 0.022 at step 377 (final 0.350); goal_error min 0.034 at step 374 / final 0.296 (success < 0.015)
- object LIFTED at step 156 (max rise 0.163 m); LOST at step 385 (aperture collapsed while the object moved away)

### env 16 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 736 steps; TERMINATED by `ee_ground_collision` at step 735
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 48, 1: 25, 2: 15, 3: 19, 4 (CARRY): 173, 5 (PLACE): 456
- aperture min -0.005 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.011 at step 686 (final 0.232)
- closest |object_to_goal| 0.003 at step 379 (final 0.119); goal_error min 0.024 at step 488 / final 0.108 (success < 0.015)
- object LIFTED at step 95 (max rise 0.166 m); LOST at step 704 (aperture collapsed while the object moved away)

### env 17 — terminated:ee_ground_collision
- ended in phase 1 after 634 steps; TERMINATED by `ee_ground_collision` at step 633
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 135, 1: 112, 2: 30, 3: 43, 4 (CARRY): 193, 5 (PLACE): 121
- aperture min -0.001 / max 0.081 / final 0.056
- closest |gripper_to_object| 0.019 at step 504 (final 0.064)
- closest |object_to_goal| 0.004 at step 212 (final 0.087); goal_error min 0.024 at step 234 / final 0.043 (success < 0.015)
- object LIFTED at step 82 (max rise 0.170 m); LOST at step 281 (aperture collapsed while the object moved away)

### env 18 — near_miss
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 39, 1: 29, 2: 15, 3: 20, 4 (CARRY): 52, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 678
- aperture min -0.002 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.060 at step 168 (final 0.770)
- closest |object_to_goal| 0.001 at step 463 (final 0.022); goal_error min 0.021 at step 239 / final 0.052 (success < 0.015)
- object LIFTED at step 90 (max rise 0.162 m); LOST at step 205 (aperture collapsed while the object moved away)

### env 19 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 225, 1: 317, 2: 165, 3: 192, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.059 at step 478 (final 0.084)
- closest |object_to_goal| 0.019 at step 165 (final 0.221); goal_error min 0.055 at step 164 / final 0.206 (success < 0.015)
- object LIFTED at step 71 (max rise 0.159 m); LOST at step 171 (aperture collapsed while the object moved away)

### env 20 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 234, 1: 439, 2: 90, 3: 136, 4 (CARRY): 101
- aperture min -0.002 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.014 at step 924 (final 0.135)
- closest |object_to_goal| 0.058 at step 135 (final 0.116); goal_error min 0.082 at step 134 / final 0.138 (success < 0.015)
- object LIFTED at step 83 (max rise 0.154 m); LOST at step 133 (aperture collapsed while the object moved away)

### env 21 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 253, 1: 478, 2: 60, 3: 108, 4 (CARRY): 101
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.018 at step 525 (final 0.038)
- closest |object_to_goal| 0.027 at step 171 (final 0.267); goal_error min 0.079 at step 171 / final 0.236 (success < 0.015)
- object LIFTED at step 89 (max rise 0.168 m); LOST at step 182 (aperture collapsed while the object moved away)

### env 22 — never_lifted
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 229, 1: 294, 2: 210, 3: 267
- aperture min 0.000 / max 0.080 / final 0.079
- closest |gripper_to_object| 0.043 at step 615 (final 0.194)
- closest |object_to_goal| 0.338 at step 83 (final 0.356); goal_error min 0.349 at step 79 / final 0.404 (success < 0.015)
- object never lifted (max rise 0.006 m)

### env 23 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 312, 1: 451, 2: 60, 3: 76, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.062 at step 49 (final 0.239)
- closest |object_to_goal| 0.077 at step 114 (final 0.301); goal_error min 0.120 at step 114 / final 0.266 (success < 0.015)
- object LIFTED at step 67 (max rise 0.173 m); LOST at step 124 (aperture collapsed while the object moved away)

### env 24 — grasp_lost
- ended in phase 2 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 152, 1: 387, 2: 158, 3: 202, 4 (CARRY): 101
- aperture min 0.000 / max 0.083 / final 0.050
- closest |gripper_to_object| 0.020 at step 227 (final 0.045)
- closest |object_to_goal| 0.043 at step 175 (final 0.141); goal_error min 0.053 at step 194 / final 0.094 (success < 0.015)
- object LIFTED at step 74 (max rise 0.160 m); LOST at step 248 (aperture collapsed while the object moved away)

### env 26 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 381 steps; TERMINATED by `ee_ground_collision` at step 380
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 45, 1: 20, 2: 15, 3: 20, 4 (CARRY): 137, 5 (PLACE): 144
- aperture min -0.001 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.049 at step 244 (final 0.472)
- closest |object_to_goal| 0.005 at step 345 (final 0.017); goal_error min 0.032 at step 328 / final 0.037 (success < 0.015)
- object LIFTED at step 86 (max rise 0.166 m); LOST at step 229 (aperture collapsed while the object moved away)

### env 27 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 44, 1: 19, 2: 15, 3: 19, 4 (CARRY): 24, 5 (PLACE): 79, 6 (RELEASE): 16, 7: 30, 8 (DONE): 754
- aperture min -0.003 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.029 at step 169 (final 0.791)
- closest |object_to_goal| 0.002 at step 202 (final 0.195); goal_error min 0.030 at step 167 / final 0.150 (success < 0.015)
- object LIFTED at step 85 (max rise 0.159 m); LOST at step 159 (aperture collapsed while the object moved away)

### env 28 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 48, 1: 19, 2: 15, 3: 19, 4 (CARRY): 71, 5 (PLACE): 242, 6 (RELEASE): 16, 7: 30, 8 (DONE): 540
- aperture min 0.012 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.062 at step 85 (final 0.757)
- closest |object_to_goal| 0.012 at step 227 (final 0.067); goal_error min 0.029 at step 271 / final 0.047 (success < 0.015)
- object LIFTED at step 89 (max rise 0.159 m); LOST at step 338 (aperture collapsed while the object moved away)

### env 29 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 181, 1: 507, 2: 90, 3: 121, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.014 at step 990 (final 0.029)
- closest |object_to_goal| 0.096 at step 114 (final 0.210); goal_error min 0.088 at step 129 / final 0.162 (success < 0.015)
- object LIFTED at step 79 (max rise 0.160 m); LOST at step 126 (aperture collapsed while the object moved away)

### env 30 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 300 steps; TERMINATED by `ee_ground_collision` at step 299
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 32, 1: 28, 2: 15, 3: 18, 4 (CARRY): 29, 5 (PLACE): 178
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.063 at step 213 (final 0.227)
- closest |object_to_goal| 0.005 at step 298 (final 0.019); goal_error min 0.027 at step 182 / final 0.056 (success < 0.015)
- object LIFTED at step 82 (max rise 0.142 m); LOST at step 236 (aperture collapsed while the object moved away)

### env 31 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 250 steps; TERMINATED by `ee_ground_collision` at step 249
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 37, 1: 23, 2: 15, 3: 19, 4 (CARRY): 27, 5 (PLACE): 121, 6 (RELEASE): 8
- aperture min -0.000 / max 0.080 / final 0.062
- closest |gripper_to_object| 0.054 at step 165 (final 0.280)
- closest |object_to_goal| 0.002 at step 241 (final 0.013); goal_error min 0.017 at step 188 / final 0.048 (success < 0.015)
- object LIFTED at step 81 (max rise 0.164 m); LOST at step 186 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 196, end phase 5 (PLACE), aperture min 0.024, min |go| 0.065, lifted at 70
- env 13: success at step 206, end phase 1, aperture min -0.006, min |go| 0.006, lifted at 72
- env 15: success at step 228, end phase 1, aperture min -0.000, min |go| 0.014, lifted at 157
- env 25: success at step 167, end phase 1, aperture min -0.000, min |go| 0.032, lifted at 86
