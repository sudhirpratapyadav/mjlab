# diagnose — Mjlab-Peg-Insertion-Franka

2026-09-10T04:02:32 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**4/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `PegInsertionClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 4 = CARRY, 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (28): 1: 13, 5 (PLACE): 5, 0 (HOVER): 3, 3: 3, 8 (DONE): 2, 6 (RELEASE): 1, 2: 1
- end phase, successful envs (4): 1: 2, 3: 1, 6 (RELEASE): 1
- most-steps phase, failing envs: 1: 14, 0 (HOVER): 5, 5 (PLACE): 5, 8 (DONE): 2, 4 (CARRY): 2
- failure class: grasp_lost: 18, terminated:ee_ground_collision: 8, near_miss: 1, never_lifted: 1
- termination among failing envs: none: 20, ee_ground_collision: 8
- failing envs that lifted the object: 26/28; lifted then lost: 25
- success step (successful envs): median 348, max 385

## Reading

- **grasp_lost** — 18/28 failing envs (envs [0, 2, 3, 4, 7, 9, 14, 15, 17, 18, 20, 21, 25, 26, 27, 28, 30, 31]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.024; closest |object_to_goal| median 0.012; min goal_error median 0.031 (success < 0.015); lost at step median 201
- **terminated:ee_ground_collision** — 8/28 failing envs (envs [5, 6, 8, 10, 11, 19, 23, 29]); typical end phase 5 (PLACE); final aperture median 0.014; closest |gripper_to_object| median 0.058; closest |object_to_goal| median 0.021; min goal_error median 0.031 (success < 0.015)
- **near_miss** — 1/28 failing envs (envs [1]); typical end phase 0 (HOVER); final aperture median 0.080; closest |gripper_to_object| median 0.013; closest |object_to_goal| median 0.019; min goal_error median 0.020 (success < 0.015)
- **never_lifted** — 1/28 failing envs (envs [12]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.048; closest |object_to_goal| median 0.306; min goal_error median 0.311 (success < 0.015)

## Failing envs

### env 0 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 178, 1: 456, 2: 105, 3: 160, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.077
- closest |gripper_to_object| 0.024 at step 796 (final 0.173)
- closest |object_to_goal| 0.051 at step 152 (final 0.197); goal_error min 0.100 at step 152 / final 0.167 (success < 0.015)
- object LIFTED at step 84 (max rise 0.168 m); LOST at step 166 (aperture collapsed while the object moved away)

### env 1 — near_miss
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 226, 1: 310, 2: 75, 3: 116, 4 (CARRY): 152, 5 (PLACE): 121
- aperture min -0.000 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.013 at step 424 (final 0.177)
- closest |object_to_goal| 0.019 at step 168 (final 0.162); goal_error min 0.020 at step 276 / final 0.143 (success < 0.015)
- object LIFTED at step 88 (max rise 0.185 m); LOST at step 353 (aperture collapsed while the object moved away)

### env 2 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 230, 1: 222, 2: 135, 3: 164, 4 (CARRY): 128, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.040 at step 183 (final 0.074)
- closest |object_to_goal| 0.007 at step 182 (final 0.095); goal_error min 0.027 at step 174 / final 0.108 (success < 0.015)
- object LIFTED at step 74 (max rise 0.152 m); LOST at step 180 (aperture collapsed while the object moved away)

### env 3 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 186, 1: 302, 2: 105, 3: 158, 4 (CARRY): 128, 5 (PLACE): 121
- aperture min -0.000 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.004 at step 441 (final 0.050)
- closest |object_to_goal| 0.021 at step 126 (final 0.132); goal_error min 0.030 at step 145 / final 0.097 (success < 0.015)
- object LIFTED at step 70 (max rise 0.169 m); LOST at step 151 (aperture collapsed while the object moved away)

### env 4 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 222, 1: 215, 2: 120, 3: 155, 4 (CARRY): 167, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.022 at step 555 (final 0.044)
- closest |object_to_goal| 0.011 at step 209 (final 0.022); goal_error min 0.027 at step 246 / final 0.027 (success < 0.015)
- object LIFTED at step 77 (max rise 0.167 m); LOST at step 226 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 242 steps; TERMINATED by `ee_ground_collision` at step 241
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 31, 1: 21, 2: 15, 3: 19, 4 (CARRY): 56, 5 (PLACE): 100
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.038 at step 159 (final 0.474)
- closest |object_to_goal| 0.012 at step 170 (final 0.027); goal_error min 0.033 at step 172 / final 0.036 (success < 0.015)
- object LIFTED at step 73 (max rise 0.165 m); LOST at step 170 (aperture collapsed while the object moved away)

### env 6 — terminated:ee_ground_collision
- ended in phase 3 after 422 steps; TERMINATED by `ee_ground_collision` at step 421
- most steps in phase 1; steps per phase 0 (HOVER): 104, 1: 145, 2: 30, 3: 42, 4 (CARRY): 101
- aperture min 0.000 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.020 at step 305 (final 0.076)
- closest |object_to_goal| 0.091 at step 123 (final 0.210); goal_error min 0.074 at step 123 / final 0.167 (success < 0.015)
- object LIFTED at step 78 (max rise 0.167 m); LOST at step 120 (aperture collapsed while the object moved away)

### env 7 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 47, 1: 21, 2: 15, 3: 19, 4 (CARRY): 40, 5 (PLACE): 94, 6 (RELEASE): 16, 7: 30, 8 (DONE): 718
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.059 at step 188 (final 0.786)
- closest |object_to_goal| 0.002 at step 341 (final 0.009); goal_error min 0.028 at step 191 / final 0.049 (success < 0.015)
- object LIFTED at step 90 (max rise 0.158 m); LOST at step 207 (aperture collapsed while the object moved away)

### env 8 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 251 steps; TERMINATED by `ee_ground_collision` at step 250
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 31, 1: 20, 2: 15, 3: 19, 4 (CARRY): 35, 5 (PLACE): 121, 6 (RELEASE): 10
- aperture min -0.000 / max 0.080 / final 0.069
- closest |gripper_to_object| 0.059 at step 157 (final 0.399)
- closest |object_to_goal| 0.003 at step 204 (final 0.009); goal_error min 0.022 at step 169 / final 0.041 (success < 0.015)
- object LIFTED at step 73 (max rise 0.165 m); LOST at step 169 (aperture collapsed while the object moved away)

### env 9 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 210, 1: 206, 2: 120, 3: 152, 4 (CARRY): 191, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.019 at step 496 (final 0.202)
- closest |object_to_goal| 0.024 at step 221 (final 0.109); goal_error min 0.046 at step 217 / final 0.136 (success < 0.015)
- object LIFTED at step 95 (max rise 0.164 m); LOST at step 226 (aperture collapsed while the object moved away)

### env 10 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 250 steps; TERMINATED by `ee_ground_collision` at step 249
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 33, 1: 18, 2: 15, 3: 19, 4 (CARRY): 57, 5 (PLACE): 108
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.060 at step 65 (final 0.462)
- closest |object_to_goal| 0.015 at step 174 (final 0.029); goal_error min 0.027 at step 170 / final 0.047 (success < 0.015)
- object LIFTED at step 74 (max rise 0.159 m); LOST at step 162 (aperture collapsed while the object moved away)

### env 11 — terminated:ee_ground_collision
- ended in phase 1 after 646 steps; TERMINATED by `ee_ground_collision` at step 645
- most steps in phase 1; steps per phase 0 (HOVER): 161, 1: 292, 2: 75, 3: 118
- aperture min -0.000 / max 0.082 / final 0.075
- closest |gripper_to_object| 0.013 at step 361 (final 0.111)
- closest |object_to_goal| 0.306 at step 382 (final 0.564); goal_error min 0.339 at step 67 / final 0.509 (success < 0.015)
- object never lifted (max rise 0.007 m)

### env 12 — never_lifted
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 230, 1: 318, 2: 195, 3: 257
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.048 at step 269 (final 0.124)
- closest |object_to_goal| 0.306 at step 59 (final 0.353); goal_error min 0.311 at step 56 / final 0.392 (success < 0.015)
- object never lifted (max rise 0.006 m)

### env 14 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 265, 1: 272, 2: 165, 3: 197, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.053 at step 118 (final 0.093)
- closest |object_to_goal| 0.034 at step 125 (final 0.351); goal_error min 0.069 at step 125 / final 0.309 (success < 0.015)
- object LIFTED at step 78 (max rise 0.171 m); LOST at step 130 (aperture collapsed while the object moved away)

### env 15 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 202, 1: 328, 2: 45, 3: 71, 4 (CARRY): 233, 5 (PLACE): 121
- aperture min -0.005 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.009 at step 740 (final 0.595)
- closest |object_to_goal| 0.063 at step 133 (final 0.177); goal_error min 0.045 at step 149 / final 0.136 (success < 0.015)
- object LIFTED at step 84 (max rise 0.150 m); LOST at step 149 (aperture collapsed while the object moved away)

### env 17 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 272, 1: 228, 2: 75, 3: 120, 4 (CARRY): 184, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.023 at step 675 (final 0.129)
- closest |object_to_goal| 0.009 at step 202 (final 0.158); goal_error min 0.027 at step 201 / final 0.176 (success < 0.015)
- object LIFTED at step 74 (max rise 0.169 m); LOST at step 226 (aperture collapsed while the object moved away)

### env 18 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 166, 1: 433, 2: 60, 3: 97, 4 (CARRY): 123, 5 (PLACE): 121
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.014 at step 484 (final 0.037)
- closest |object_to_goal| 0.008 at step 143 (final 0.150); goal_error min 0.023 at step 168 / final 0.136 (success < 0.015)
- object LIFTED at step 91 (max rise 0.161 m); LOST at step 162 (aperture collapsed while the object moved away)

### env 19 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 192 steps; TERMINATED by `ee_ground_collision` at step 191
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 37, 1: 20, 2: 15, 3: 19, 4 (CARRY): 58, 5 (PLACE): 43
- aperture min 0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.062 at step 62 (final 0.136)
- closest |object_to_goal| 0.027 at step 182 (final 0.032); goal_error min 0.025 at step 185 / final 0.028 (success < 0.015)
- object LIFTED at step 79 (max rise 0.159 m); LOST at step 174 (aperture collapsed while the object moved away)

### env 20 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 29, 1: 21, 2: 15, 3: 19, 4 (CARRY): 15, 5 (PLACE): 121, 6 (RELEASE): 16, 7: 30, 8 (DONE): 734
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.042 at step 234 (final 0.788)
- closest |object_to_goal| 0.005 at step 157 (final 0.025); goal_error min 0.024 at step 206 / final 0.053 (success < 0.015)
- object LIFTED at step 72 (max rise 0.167 m); LOST at step 194 (aperture collapsed while the object moved away)

### env 21 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 136, 1: 286, 2: 45, 3: 61, 4 (CARRY): 230, 5 (PLACE): 242
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.037 at step 883 (final 0.048)
- closest |object_to_goal| 0.001 at step 250 (final 0.191); goal_error min 0.032 at step 181 / final 0.240 (success < 0.015)
- object LIFTED at step 85 (max rise 0.164 m); LOST at step 360 (aperture collapsed while the object moved away)

### env 23 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 267 steps; TERMINATED by `ee_ground_collision` at step 266
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 34, 1: 22, 2: 15, 3: 19, 4 (CARRY): 62, 5 (PLACE): 115
- aperture min 0.024 / max 0.080 / final 0.028
- closest |gripper_to_object| 0.057 at step 149 (final 0.101)
- closest |object_to_goal| 0.006 at step 233 (final 0.038); goal_error min 0.028 at step 173 / final 0.058 (success < 0.015)
- object LIFTED at step 78 (max rise 0.157 m); never lost

### env 25 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 221, 1: 335, 2: 150, 3: 193, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.051 at step 195 (final 0.066)
- closest |object_to_goal| 0.043 at step 199 (final 0.120); goal_error min 0.063 at step 199 / final 0.116 (success < 0.015)
- object LIFTED at step 77 (max rise 0.160 m); LOST at step 275 (aperture collapsed while the object moved away)

### env 26 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 213, 1: 197, 2: 120, 3: 151, 4 (CARRY): 198, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.069
- closest |gripper_to_object| 0.019 at step 746 (final 0.168)
- closest |object_to_goal| 0.020 at step 193 (final 0.091); goal_error min 0.050 at step 217 / final 0.053 (success < 0.015)
- object LIFTED at step 84 (max rise 0.170 m); LOST at step 216 (aperture collapsed while the object moved away)

### env 27 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 319, 1: 425, 2: 60, 3: 95, 4 (CARRY): 101
- aperture min -0.000 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.012 at step 456 (final 0.043)
- closest |object_to_goal| 0.079 at step 110 (final 0.136); goal_error min 0.122 at step 110 / final 0.185 (success < 0.015)
- object LIFTED at step 80 (max rise 0.165 m); LOST at step 117 (aperture collapsed while the object moved away)

### env 28 — grasp_lost
- ended in phase 2 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 165, 1: 333, 2: 108, 3: 139, 4 (CARRY): 134, 5 (PLACE): 121
- aperture min 0.000 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.060 at step 216 (final 0.068)
- closest |object_to_goal| 0.006 at step 215 (final 0.392); goal_error min 0.029 at step 158 / final 0.356 (success < 0.015)
- object LIFTED at step 79 (max rise 0.176 m); LOST at step 324 (aperture collapsed while the object moved away)

### env 29 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 302 steps; TERMINATED by `ee_ground_collision` at step 301
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 32, 1: 21, 2: 15, 3: 18, 4 (CARRY): 97, 5 (PLACE): 119
- aperture min -0.008 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.061 at step 114 (final 0.409)
- closest |object_to_goal| 0.043 at step 161 (final 0.258); goal_error min 0.087 at step 183 / final 0.215 (success < 0.015)
- object LIFTED at step 74 (max rise 0.161 m); LOST at step 188 (aperture collapsed while the object moved away)

### env 30 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 226, 1: 177, 2: 90, 3: 123, 4 (CARRY): 263, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.008 at step 468 (final 0.112)
- closest |object_to_goal| 0.011 at step 175 (final 0.097); goal_error min 0.036 at step 176 / final 0.131 (success < 0.015)
- object LIFTED at step 78 (max rise 0.164 m); LOST at step 196 (aperture collapsed while the object moved away)

### env 31 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 199, 1: 387, 2: 60, 3: 87, 4 (CARRY): 146, 5 (PLACE): 121
- aperture min -0.001 / max 0.080 / final 0.076
- closest |gripper_to_object| 0.045 at step 239 (final 0.197)
- closest |object_to_goal| 0.013 at step 210 (final 0.109); goal_error min 0.027 at step 182 / final 0.107 (success < 0.015)
- object LIFTED at step 80 (max rise 0.172 m); LOST at step 234 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 13: success at step 385, end phase 3, aperture min -0.000, min |go| 0.019, lifted at 73
- env 16: success at step 361, end phase 1, aperture min -0.000, min |go| 0.006, lifted at 76
- env 22: success at step 163, end phase 6 (RELEASE), aperture min 0.024, min |go| 0.068, lifted at 85
- env 24: success at step 336, end phase 1, aperture min -0.001, min |go| 0.005, lifted at 85
