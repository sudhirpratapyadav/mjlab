# diagnose — Mjlab-Peg-Insertion-Franka

2026-09-09T22:14:49 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**4/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `PegInsertionClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 4 = CARRY, 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (28): 1: 14, 5 (PLACE): 5, 3: 4, 0 (HOVER): 3, 2: 1, 8 (DONE): 1
- end phase, successful envs (4): 5 (PLACE): 1, 1: 1, 3: 1, 2: 1
- most-steps phase, failing envs: 1: 20, 5 (PLACE): 5, 0 (HOVER): 2, 8 (DONE): 1
- failure class: grasp_lost: 18, terminated:ee_ground_collision: 8, lifted_not_at_goal: 1, never_lifted: 1
- termination among failing envs: none: 20, ee_ground_collision: 8
- failing envs that lifted the object: 26/28; lifted then lost: 25
- success step (successful envs): median 205, max 323

## Reading

- **grasp_lost** — 18/28 failing envs (envs [0, 2, 4, 6, 8, 9, 10, 12, 13, 19, 20, 21, 23, 26, 27, 28, 29, 30]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.027; closest |object_to_goal| median 0.041; min goal_error median 0.078 (success < 0.015); lost at step median 169
- **terminated:ee_ground_collision** — 8/28 failing envs (envs [1, 5, 7, 15, 16, 18, 25, 31]); typical end phase 5 (PLACE); final aperture median 0.000; closest |gripper_to_object| median 0.039; closest |object_to_goal| median 0.034; min goal_error median 0.035 (success < 0.015)
- **lifted_not_at_goal** — 1/28 failing envs (envs [14]); typical end phase 8 (DONE); final aperture median 0.080; closest |gripper_to_object| median 0.053; closest |object_to_goal| median 0.004; min goal_error median 0.028 (success < 0.015)
- **never_lifted** — 1/28 failing envs (envs [17]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.015; closest |object_to_goal| median 0.225; min goal_error median 0.432 (success < 0.015)

## Failing envs

### env 0 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 138, 1: 536, 2: 90, 3: 135, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 287 (final 0.168)
- closest |object_to_goal| 0.043 at step 142 (final 0.270); goal_error min 0.090 at step 142 / final 0.267 (success < 0.015)
- object LIFTED at step 73 (max rise 0.160 m); LOST at step 168 (aperture collapsed while the object moved away)

### env 1 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 286 steps; TERMINATED by `ee_ground_collision` at step 285
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 57, 1: 41, 2: 30, 3: 41, 4 (CARRY): 29, 5 (PLACE): 88
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.056 at step 136 (final 0.192)
- closest |object_to_goal| 0.036 at step 243 (final 0.051); goal_error min 0.023 at step 237 / final 0.030 (success < 0.015)
- object LIFTED at step 156 (max rise 0.162 m); LOST at step 237 (aperture collapsed while the object moved away)

### env 2 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 244, 1: 286, 2: 165, 3: 204, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.078
- closest |gripper_to_object| 0.060 at step 409 (final 0.204)
- closest |object_to_goal| 0.037 at step 180 (final 0.268); goal_error min 0.082 at step 180 / final 0.315 (success < 0.015)
- object LIFTED at step 83 (max rise 0.156 m); LOST at step 185 (aperture collapsed while the object moved away)

### env 4 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 291, 1: 282, 2: 135, 3: 191, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.038 at step 927 (final 0.092)
- closest |object_to_goal| 0.046 at step 151 (final 0.181); goal_error min 0.084 at step 151 / final 0.190 (success < 0.015)
- object LIFTED at step 78 (max rise 0.167 m); LOST at step 154 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 1 after 692 steps; TERMINATED by `ee_ground_collision` at step 691
- most steps in phase 1; steps per phase 0 (HOVER): 152, 1: 157, 2: 45, 3: 67, 4 (CARRY): 150, 5 (PLACE): 121
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.016 at step 685 (final 0.036)
- closest |object_to_goal| 0.020 at step 156 (final 0.136); goal_error min 0.034 at step 168 / final 0.138 (success < 0.015)
- object LIFTED at step 74 (max rise 0.163 m); LOST at step 190 (aperture collapsed while the object moved away)

### env 6 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 156, 1: 397, 2: 60, 3: 93, 4 (CARRY): 173, 5 (PLACE): 121
- aperture min -0.009 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.017 at step 692 (final 0.099)
- closest |object_to_goal| 0.036 at step 162 (final 0.344); goal_error min 0.061 at step 161 / final 0.397 (success < 0.015)
- object LIFTED at step 79 (max rise 0.155 m); LOST at step 169 (aperture collapsed while the object moved away)

### env 7 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 467 steps; TERMINATED by `ee_ground_collision` at step 466
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 30, 1: 22, 2: 15, 3: 18, 4 (CARRY): 97, 5 (PLACE): 285
- aperture min 0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.024 at step 260 (final 0.298)
- closest |object_to_goal| 0.003 at step 192 (final 0.068); goal_error min 0.025 at step 432 / final 0.040 (success < 0.015)
- object LIFTED at step 73 (max rise 0.160 m); LOST at step 438 (aperture collapsed while the object moved away)

### env 8 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 111, 1: 577, 2: 90, 3: 121, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.039 at step 971 (final 0.044)
- closest |object_to_goal| 0.043 at step 180 (final 0.159); goal_error min 0.073 at step 180 / final 0.198 (success < 0.015)
- object LIFTED at step 74 (max rise 0.165 m); LOST at step 336 (aperture collapsed while the object moved away)

### env 9 — grasp_lost
- ended in phase 2 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 141, 1: 533, 2: 100, 3: 125, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.011
- closest |gripper_to_object| 0.032 at step 261 (final 0.039)
- closest |object_to_goal| 0.074 at step 208 (final 0.168); goal_error min 0.078 at step 207 / final 0.169 (success < 0.015)
- object LIFTED at step 88 (max rise 0.173 m); LOST at step 199 (aperture collapsed while the object moved away)

### env 10 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 185, 1: 411, 2: 120, 3: 183, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.022 at step 948 (final 0.164)
- closest |object_to_goal| 0.040 at step 161 (final 0.260); goal_error min 0.077 at step 161 / final 0.206 (success < 0.015)
- object LIFTED at step 88 (max rise 0.173 m); LOST at step 166 (aperture collapsed while the object moved away)

### env 12 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 153, 1: 528, 2: 90, 3: 128, 4 (CARRY): 101
- aperture min 0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.029 at step 270 (final 0.107)
- closest |object_to_goal| 0.042 at step 194 (final 0.363); goal_error min 0.080 at step 194 / final 0.317 (success < 0.015)
- object LIFTED at step 84 (max rise 0.160 m); LOST at step 378 (aperture collapsed while the object moved away)

### env 13 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 253, 1: 263, 2: 165, 3: 218, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.048 at step 745 (final 0.139)
- closest |object_to_goal| 0.083 at step 126 (final 0.154); goal_error min 0.116 at step 142 / final 0.209 (success < 0.015)
- object LIFTED at step 89 (max rise 0.168 m); LOST at step 149 (aperture collapsed while the object moved away)

### env 14 — lifted_not_at_goal
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 28, 1: 22, 2: 15, 3: 20, 4 (CARRY): 24, 5 (PLACE): 41, 6 (RELEASE): 16, 7: 30, 8 (DONE): 804
- aperture min 0.024 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.053 at step 161 (final 0.803)
- closest |object_to_goal| 0.004 at step 341 (final 0.022); goal_error min 0.028 at step 145 / final 0.052 (success < 0.015)
- object LIFTED at step 72 (max rise 0.166 m); never lost

### env 15 — terminated:ee_ground_collision
- ended in phase 1 after 818 steps; TERMINATED by `ee_ground_collision` at step 817
- most steps in phase 1; steps per phase 0 (HOVER): 171, 1: 471, 2: 60, 3: 116
- aperture min 0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.008 at step 393 (final 0.028)
- closest |object_to_goal| 0.339 at step 44 (final 0.365); goal_error min 0.344 at step 68 / final 0.407 (success < 0.015)
- object never lifted (max rise 0.002 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 263 steps; TERMINATED by `ee_ground_collision` at step 262
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 37, 1: 20, 2: 15, 3: 19, 4 (CARRY): 75, 5 (PLACE): 97
- aperture min -0.001 / max 0.080 / final -0.001
- closest |gripper_to_object| 0.055 at step 125 (final 0.335)
- closest |object_to_goal| 0.032 at step 185 (final 0.053); goal_error min 0.027 at step 180 / final 0.040 (success < 0.015)
- object LIFTED at step 80 (max rise 0.152 m); LOST at step 169 (aperture collapsed while the object moved away)

### env 17 — never_lifted
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 314, 1: 440, 2: 90, 3: 156
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.015 at step 830 (final 0.112)
- closest |object_to_goal| 0.225 at step 0 (final 0.558); goal_error min 0.432 at step 75 / final 0.555 (success < 0.015)
- object never lifted (max rise 0.008 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 207 steps; TERMINATED by `ee_ground_collision` at step 206
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 29, 1: 19, 2: 15, 3: 19, 4 (CARRY): 22, 5 (PLACE): 103
- aperture min -0.001 / max 0.080 / final -0.001
- closest |gripper_to_object| 0.062 at step 69 (final 0.089)
- closest |object_to_goal| 0.004 at step 183 (final 0.095); goal_error min 0.035 at step 134 / final 0.089 (success < 0.015)
- object LIFTED at step 69 (max rise 0.157 m); LOST at step 151 (aperture collapsed while the object moved away)

### env 19 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 175, 1: 401, 2: 75, 3: 110, 4 (CARRY): 118, 5 (PLACE): 121
- aperture min -0.000 / max 0.085 / final 0.081
- closest |gripper_to_object| 0.014 at step 974 (final 0.019)
- closest |object_to_goal| 0.023 at step 121 (final 0.178); goal_error min 0.031 at step 136 / final 0.134 (success < 0.015)
- object LIFTED at step 73 (max rise 0.164 m); LOST at step 128 (aperture collapsed while the object moved away)

### env 20 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 218, 1: 370, 2: 135, 3: 176, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.037 at step 974 (final 0.155)
- closest |object_to_goal| 0.069 at step 143 (final 0.333); goal_error min 0.105 at step 143 / final 0.292 (success < 0.015)
- object LIFTED at step 95 (max rise 0.170 m); LOST at step 150 (aperture collapsed while the object moved away)

### env 21 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 196, 1: 507, 2: 75, 3: 121, 4 (CARRY): 101
- aperture min -0.000 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.004 at step 917 (final 0.143)
- closest |object_to_goal| 0.003 at step 406 (final 0.130); goal_error min 0.026 at step 200 / final 0.171 (success < 0.015)
- object LIFTED at step 77 (max rise 0.170 m); LOST at step 371 (aperture collapsed while the object moved away)

### env 23 — grasp_lost
- ended in phase 3 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 212, 1: 237, 2: 135, 3: 158, 4 (CARRY): 137, 5 (PLACE): 121
- aperture min -0.000 / max 0.083 / final 0.001
- closest |gripper_to_object| 0.017 at step 542 (final 0.061)
- closest |object_to_goal| 0.020 at step 135 (final 0.140); goal_error min 0.036 at step 150 / final 0.136 (success < 0.015)
- object LIFTED at step 72 (max rise 0.157 m); LOST at step 155 (aperture collapsed while the object moved away)

### env 25 — terminated:ee_ground_collision
- ended in phase 5 (PLACE) after 262 steps; TERMINATED by `ee_ground_collision` at step 261
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 43, 1: 25, 2: 15, 3: 19, 4 (CARRY): 42, 5 (PLACE): 118
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.058 at step 129 (final 0.707)
- closest |object_to_goal| 0.036 at step 164 (final 0.132); goal_error min 0.055 at step 164 / final 0.089 (success < 0.015)
- object LIFTED at step 90 (max rise 0.164 m); LOST at step 171 (aperture collapsed while the object moved away)

### env 26 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 131, 1: 364, 2: 45, 3: 65, 4 (CARRY): 153, 5 (PLACE): 242
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.023 at step 980 (final 0.035)
- closest |object_to_goal| 0.003 at step 182 (final 0.217); goal_error min 0.033 at step 152 / final 0.163 (success < 0.015)
- object LIFTED at step 85 (max rise 0.159 m); LOST at step 442 (aperture collapsed while the object moved away)

### env 27 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 217, 1: 238, 2: 135, 3: 171, 4 (CARRY): 118, 5 (PLACE): 121
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.026 at step 946 (final 0.045)
- closest |object_to_goal| 0.004 at step 201 (final 0.010); goal_error min 0.024 at step 141 / final 0.060 (success < 0.015)
- object LIFTED at step 65 (max rise 0.151 m); LOST at step 204 (aperture collapsed while the object moved away)

### env 28 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 237, 1: 205, 2: 135, 3: 176, 4 (CARRY): 126, 5 (PLACE): 121
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.017 at step 653 (final 0.153)
- closest |object_to_goal| 0.001 at step 788 (final 0.020); goal_error min 0.035 at step 145 / final 0.057 (success < 0.015)
- object LIFTED at step 83 (max rise 0.147 m); LOST at step 172 (aperture collapsed while the object moved away)

### env 29 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 148, 1: 562, 2: 75, 3: 114, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.028 at step 676 (final 0.044)
- closest |object_to_goal| 0.057 at step 133 (final 0.170); goal_error min 0.086 at step 128 / final 0.159 (success < 0.015)
- object LIFTED at step 81 (max rise 0.166 m); LOST at step 143 (aperture collapsed while the object moved away)

### env 30 — grasp_lost
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0 (HOVER): 161, 1: 444, 2: 120, 3: 174, 4 (CARRY): 101
- aperture min -0.000 / max 0.080 / final 0.012
- closest |gripper_to_object| 0.030 at step 556 (final 0.215)
- closest |object_to_goal| 0.043 at step 155 (final 0.217); goal_error min 0.082 at step 154 / final 0.218 (success < 0.015)
- object LIFTED at step 81 (max rise 0.173 m); LOST at step 169 (aperture collapsed while the object moved away)

### env 31 — terminated:ee_ground_collision
- ended in phase 1 after 660 steps; TERMINATED by `ee_ground_collision` at step 659
- most steps in phase 1; steps per phase 0 (HOVER): 122, 1: 208, 2: 30, 3: 40, 4 (CARRY): 139, 5 (PLACE): 121
- aperture min -0.000 / max 0.082 / final 0.079
- closest |gripper_to_object| 0.017 at step 654 (final 0.034)
- closest |object_to_goal| 0.054 at step 132 (final 0.119); goal_error min 0.087 at step 151 / final 0.152 (success < 0.015)
- object LIFTED at step 82 (max rise 0.172 m); LOST at step 149 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 3: success at step 323, end phase 5 (PLACE), aperture min -0.006, min |go| 0.060, lifted at 77
- env 11: success at step 161, end phase 1, aperture min -0.000, min |go| 0.001, lifted at 86
- env 22: success at step 250, end phase 3, aperture min -0.002, min |go| 0.006, lifted at 80
- env 24: success at step 134, end phase 2, aperture min -0.000, min |go| 0.017, lifted at 90
