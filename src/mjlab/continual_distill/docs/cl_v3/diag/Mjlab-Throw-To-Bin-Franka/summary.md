# diagnose — Mjlab-Throw-To-Bin-Franka

2026-09-09T13:04:07 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 250 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Teacher `ThrowToBinClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = CLIMB, 4 = REACH, 5 = DAMP, 6 = RELEASE

## Histograms

- end phase, FAILING envs (32): 4 (REACH): 15, 5 (DAMP): 11, 2 (CLOSE): 5, 1 (DESCEND): 1
- end phase, successful envs (0): —
- most-steps phase, failing envs: 4 (REACH): 26, 0 (HOVER): 6
- failure class: lifted_not_at_goal: 18, terminated:ee_ground_collision: 13, never_lifted: 1
- termination among failing envs: none: 19, ee_ground_collision: 13
- failing envs that lifted the object: 25/32; lifted then lost: 0

## Reading

- **lifted_not_at_goal** — 18/32 failing envs (envs [0, 1, 2, 3, 4, 6, 7, 8, 10, 12, 14, 15, 18, 22, 24, 26, 28, 29]); typical end phase 4 (REACH); final aperture median 0.053; closest |gripper_to_object| median 0.004; closest |object_to_goal| median 0.276; min goal_error median 0.281
- **terminated:ee_ground_collision** — 13/32 failing envs (envs [5, 9, 11, 13, 16, 17, 19, 20, 21, 23, 25, 27, 30]); typical end phase 4 (REACH); final aperture median 0.071; closest |gripper_to_object| median 0.005; closest |object_to_goal| median 0.313; min goal_error median 0.322
- **never_lifted** — 1/32 failing envs (envs [31]); typical end phase 5 (DAMP); final aperture median 0.000; closest |gripper_to_object| median 0.013; closest |object_to_goal| median 0.471; min goal_error median 0.482

## Failing envs

### env 0 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 17, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 117
- aperture min 0.046 / max 0.086 / final 0.049
- closest |gripper_to_object| 0.003 at step 191 (final 0.007)
- closest |object_to_goal| 0.266 at step 248 (final 0.269); goal_error min 0.268 at step 249 / final 0.268
- object LIFTED at step 138 (max rise 0.076 m); never lost

### env 1 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 8, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 5
- aperture min 0.042 / max 0.083 / final 0.054
- closest |gripper_to_object| 0.012 at step 101 (final 0.024)
- closest |object_to_goal| 0.287 at step 246 (final 0.289); goal_error min 0.293 at step 247 / final 0.294
- object LIFTED at step 129 (max rise 0.080 m); never lost

### env 2 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 15, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 119
- aperture min 0.063 / max 0.080 / final 0.063
- closest |gripper_to_object| 0.004 at step 88 (final 0.014)
- closest |object_to_goal| 0.296 at step 247 (final 0.298); goal_error min 0.297 at step 249 / final 0.297
- object LIFTED at step 128 (max rise 0.077 m); never lost

### env 3 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 13, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121
- aperture min 0.063 / max 0.083 / final 0.063
- closest |gripper_to_object| 0.003 at step 203 (final 0.017)
- closest |object_to_goal| 0.242 at step 249 (final 0.242); goal_error min 0.243 at step 249 / final 0.243
- object LIFTED at step 103 (max rise 0.081 m); never lost

### env 4 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 17, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 117
- aperture min 0.050 / max 0.084 / final 0.050
- closest |gripper_to_object| 0.002 at step 153 (final 0.020)
- closest |object_to_goal| 0.296 at step 248 (final 0.308); goal_error min 0.304 at step 249 / final 0.304
- object LIFTED at step 139 (max rise 0.077 m); never lost

### env 5 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 92 steps; TERMINATED by `ee_ground_collision` at step 91
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.002 at step 91 (final 0.002)
- closest |object_to_goal| 0.355 at step 27 (final 0.371); goal_error min 0.366 at step 3 / final 0.366
- object never lifted (max rise 0.000 m)

### env 6 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 3, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 10
- aperture min 0.047 / max 0.084 / final 0.053
- closest |gripper_to_object| 0.005 at step 200 (final 0.008)
- closest |object_to_goal| 0.251 at step 248 (final 0.265); goal_error min 0.257 at step 249 / final 0.257
- object LIFTED at step 119 (max rise 0.086 m); never lost

### env 7 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 7, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 6
- aperture min 0.048 / max 0.086 / final 0.048
- closest |gripper_to_object| 0.009 at step 121 (final 0.015)
- closest |object_to_goal| 0.255 at step 247 (final 0.262); goal_error min 0.259 at step 249 / final 0.259
- object LIFTED at step 128 (max rise 0.084 m); never lost

### env 8 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 9, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 4
- aperture min 0.049 / max 0.082 / final 0.049
- closest |gripper_to_object| 0.011 at step 141 (final 0.024)
- closest |object_to_goal| 0.298 at step 248 (final 0.305); goal_error min 0.299 at step 249 / final 0.299
- object LIFTED at step 135 (max rise 0.084 m); never lost

### env 9 — terminated:ee_ground_collision
- ended in phase 5 (DAMP) after 248 steps; TERMINATED by `ee_ground_collision` at step 247
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 6, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 5
- aperture min 0.050 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.008 at step 234 (final 0.019)
- closest |object_to_goal| 0.313 at step 247 (final 0.313); goal_error min 0.322 at step 247 / final 0.322
- object LIFTED at step 129 (max rise 0.077 m); never lost

### env 10 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 7, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 6
- aperture min 0.060 / max 0.086 / final 0.060
- closest |gripper_to_object| 0.004 at step 248 (final 0.015)
- closest |object_to_goal| 0.253 at step 246 (final 0.258); goal_error min 0.249 at step 249 / final 0.249
- object LIFTED at step 127 (max rise 0.087 m); never lost

### env 11 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 244 steps; TERMINATED by `ee_ground_collision` at step 243
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 17, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 111
- aperture min 0.045 / max 0.081 / final 0.046
- closest |gripper_to_object| 0.002 at step 212 (final 0.018)
- closest |object_to_goal| 0.263 at step 242 (final 0.270); goal_error min 0.264 at step 243 / final 0.264
- object LIFTED at step 136 (max rise 0.078 m); never lost

### env 12 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 8, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 5
- aperture min 0.044 / max 0.084 / final 0.045
- closest |gripper_to_object| 0.009 at step 89 (final 0.019)
- closest |object_to_goal| 0.306 at step 248 (final 0.314); goal_error min 0.309 at step 249 / final 0.309
- object LIFTED at step 125 (max rise 0.084 m); never lost

### env 13 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 244 steps; TERMINATED by `ee_ground_collision` at step 243
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 17, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 111
- aperture min 0.045 / max 0.081 / final 0.046
- closest |gripper_to_object| 0.002 at step 228 (final 0.007)
- closest |object_to_goal| 0.297 at step 242 (final 0.307); goal_error min 0.301 at step 243 / final 0.301
- object LIFTED at step 140 (max rise 0.076 m); never lost

### env 14 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 17, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 117
- aperture min 0.045 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.002 at step 98 (final 0.030)
- closest |object_to_goal| 0.295 at step 240 (final 0.303); goal_error min 0.303 at step 249 / final 0.303
- object LIFTED at step 137 (max rise 0.067 m); never lost

### env 15 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 16, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 118
- aperture min 0.045 / max 0.082 / final 0.068
- closest |gripper_to_object| 0.003 at step 95 (final 0.022)
- closest |object_to_goal| 0.250 at step 247 (final 0.252); goal_error min 0.258 at step 249 / final 0.258
- object LIFTED at step 138 (max rise 0.075 m); never lost

### env 16 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 250 steps; TERMINATED by `ee_ground_collision` at step 249
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 16, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 118
- aperture min 0.045 / max 0.080 / final 0.047
- closest |gripper_to_object| 0.002 at step 122 (final 0.010)
- closest |object_to_goal| 0.247 at step 246 (final 0.256); goal_error min 0.248 at step 249 / final 0.248
- object LIFTED at step 136 (max rise 0.079 m); never lost

### env 17 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 93 steps; TERMINATED by `ee_ground_collision` at step 92
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 20, 2 (CLOSE): 2
- aperture min 0.076 / max 0.082 / final 0.081
- closest |gripper_to_object| 0.007 at step 90 (final 0.016)
- closest |object_to_goal| 0.407 at step 78 (final 0.428); goal_error min 0.418 at step 92 / final 0.418
- object never lifted (max rise 0.000 m)

### env 18 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 15, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 119
- aperture min 0.050 / max 0.081 / final 0.050
- closest |gripper_to_object| 0.002 at step 103 (final 0.015)
- closest |object_to_goal| 0.241 at step 249 (final 0.241); goal_error min 0.245 at step 249 / final 0.245
- object LIFTED at step 127 (max rise 0.081 m); never lost

### env 19 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 93 steps; TERMINATED by `ee_ground_collision` at step 92
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19, 2 (CLOSE): 3
- aperture min 0.071 / max 0.082 / final 0.071
- closest |gripper_to_object| 0.011 at step 91 (final 0.016)
- closest |object_to_goal| 0.321 at step 68 (final 0.333); goal_error min 0.330 at step 88 / final 0.330
- object never lifted (max rise 0.000 m)

### env 20 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 239 steps; TERMINATED by `ee_ground_collision` at step 238
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 104
- aperture min 0.051 / max 0.081 / final 0.051
- closest |gripper_to_object| 0.006 at step 108 (final 0.025)
- closest |object_to_goal| 0.311 at step 236 (final 0.319); goal_error min 0.315 at step 238 / final 0.315
- object LIFTED at step 146 (max rise 0.058 m); never lost

### env 21 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 92 steps; TERMINATED by `ee_ground_collision` at step 91
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 21
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.011 at step 90 (final 0.013)
- closest |object_to_goal| 0.355 at step 42 (final 0.358); goal_error min 0.368 at step 3 / final 0.368
- object never lifted (max rise 0.000 m)

### env 22 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 8, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 5
- aperture min 0.063 / max 0.080 / final 0.063
- closest |gripper_to_object| 0.008 at step 179 (final 0.015)
- closest |object_to_goal| 0.304 at step 249 (final 0.304); goal_error min 0.306 at step 249 / final 0.306
- object LIFTED at step 130 (max rise 0.084 m); never lost

### env 23 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 247 steps; TERMINATED by `ee_ground_collision` at step 246
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 112
- aperture min 0.043 / max 0.080 / final 0.052
- closest |gripper_to_object| 0.002 at step 207 (final 0.011)
- closest |object_to_goal| 0.255 at step 244 (final 0.262); goal_error min 0.260 at step 246 / final 0.260
- object LIFTED at step 139 (max rise 0.067 m); never lost

### env 24 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 16, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 118
- aperture min 0.046 / max 0.084 / final 0.084
- closest |gripper_to_object| 0.002 at step 93 (final 0.035)
- closest |object_to_goal| 0.317 at step 248 (final 0.317); goal_error min 0.326 at step 249 / final 0.326
- object LIFTED at step 137 (max rise 0.064 m); never lost

### env 25 — terminated:ee_ground_collision
- ended in phase 4 (REACH) after 245 steps; TERMINATED by `ee_ground_collision` at step 244
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 20, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 109
- aperture min 0.048 / max 0.083 / final 0.048
- closest |gripper_to_object| 0.001 at step 131 (final 0.008)
- closest |object_to_goal| 0.197 at step 241 (final 0.205); goal_error min 0.200 at step 244 / final 0.200
- object LIFTED at step 144 (max rise 0.068 m); never lost

### env 26 — lifted_not_at_goal
- ended in phase 4 (REACH) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 16, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 118
- aperture min 0.046 / max 0.080 / final 0.049
- closest |gripper_to_object| 0.001 at step 142 (final 0.004)
- closest |object_to_goal| 0.215 at step 246 (final 0.221); goal_error min 0.222 at step 249 / final 0.222
- object LIFTED at step 138 (max rise 0.075 m); never lost

### env 27 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 88 steps; TERMINATED by `ee_ground_collision` at step 87
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 65, 1 (DESCEND): 21, 2 (CLOSE): 2
- aperture min 0.076 / max 0.082 / final 0.082
- closest |gripper_to_object| 0.005 at step 85 (final 0.014)
- closest |object_to_goal| 0.472 at step 3 (final 0.491); goal_error min 0.483 at step 3 / final 0.483
- object never lifted (max rise 0.000 m)

### env 28 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 7, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 6
- aperture min 0.063 / max 0.086 / final 0.065
- closest |gripper_to_object| 0.007 at step 182 (final 0.011)
- closest |object_to_goal| 0.289 at step 249 (final 0.289); goal_error min 0.295 at step 249 / final 0.295
- object LIFTED at step 123 (max rise 0.086 m); never lost

### env 29 — lifted_not_at_goal
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 10, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 3
- aperture min 0.049 / max 0.080 / final 0.053
- closest |gripper_to_object| 0.005 at step 89 (final 0.021)
- closest |object_to_goal| 0.220 at step 249 (final 0.220); goal_error min 0.228 at step 248 / final 0.229
- object LIFTED at step 114 (max rise 0.076 m); never lost

### env 30 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 91 steps; TERMINATED by `ee_ground_collision` at step 90
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 18, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 90 (final 0.006)
- closest |object_to_goal| 0.332 at step 81 (final 0.339); goal_error min 0.342 at step 3 / final 0.342
- object never lifted (max rise 0.000 m)

### env 31 — never_lifted
- ended in phase 5 (DAMP) after 250 steps; ran to time-out
- most steps in phase 4 (REACH); steps per phase 0 (HOVER): 71, 1 (DESCEND): 3, 2 (CLOSE): 15, 3 (CLIMB): 30, 4 (REACH): 121, 5 (DAMP): 10
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.013 at step 96 (final 0.341)
- closest |object_to_goal| 0.471 at step 15 (final 0.582); goal_error min 0.482 at step 3 / final 0.583
- object never lifted (max rise 0.004 m)

## Successful envs (one line each)

