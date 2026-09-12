# diagnose — Mjlab-Tool-Pull-Franka

2026-09-10T04:26:51 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 600 · device cuda:0  
**2/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToolPullClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = CHECK, 4 = CLEAR, 5 = ADVANCE, 6 = SWEEP, 7 = PULL, 8 = RELEASE, 9 = PUSH_HOVER, 10 = PUSH_DESCEND, 11 = PUSH, 12 = DONE

## Histograms

- end phase, FAILING envs (30): 11 (PUSH): 28, 10 (PUSH_DESCEND): 2
- end phase, successful envs (2): 11 (PUSH): 2
- most-steps phase, failing envs: 9 (PUSH_HOVER): 26, 10 (PUSH_DESCEND): 4
- failure class: terminated:ee_ground_collision: 28, never_lifted: 1, terminated:object_out_of_bounds: 1
- termination among failing envs: ee_ground_collision: 28, none: 1, object_out_of_bounds: 1
- failing envs that lifted the object: 1/30; lifted then lost: 1
- success step (successful envs): median 94, max 104

## Reading

- **terminated:ee_ground_collision** — 28/30 failing envs (envs [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]); typical end phase 11 (PUSH); final aperture median -0.001; closest |gripper_to_object| median 0.033; closest |object_to_goal| median 0.231; min goal_error median 0.242 (success < 0.070)
- **never_lifted** — 1/30 failing envs (envs [2]); typical end phase 11 (PUSH); final aperture median 0.026; closest |gripper_to_object| median 0.017; closest |object_to_goal| median 0.226; min goal_error median 0.231 (success < 0.070)
- **terminated:object_out_of_bounds** — 1/30 failing envs (envs [16]); typical end phase 10 (PUSH_DESCEND); final aperture median -0.000; closest |gripper_to_object| median 0.019; closest |object_to_goal| median 0.103; min goal_error median 0.100 (success < 0.070)

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 434 steps; TERMINATED by `ee_ground_collision` at step 433
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 170, 10 (PUSH_DESCEND): 150, 11 (PUSH): 114
- aperture min -0.007 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.016 at step 35 (final 0.048)
- closest |object_to_goal| 0.232 at step 411 (final 0.249); goal_error min 0.243 at step 433 / final 0.243 (success < 0.070)
- object never lifted (max rise 0.017 m)

### env 2 — never_lifted
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 257, 10 (PUSH_DESCEND): 187, 11 (PUSH): 156
- aperture min -0.008 / max 0.076 / final 0.026
- closest |gripper_to_object| 0.017 at step 569 (final 0.068)
- closest |object_to_goal| 0.226 at step 589 (final 0.226); goal_error min 0.231 at step 583 / final 0.232 (success < 0.070)
- object never lifted (max rise 0.015 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 483 steps; TERMINATED by `ee_ground_collision` at step 482
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 197, 10 (PUSH_DESCEND): 163, 11 (PUSH): 123
- aperture min -0.006 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.016 at step 457 (final 0.068)
- closest |object_to_goal| 0.226 at step 393 (final 0.239); goal_error min 0.230 at step 482 / final 0.230 (success < 0.070)
- object never lifted (max rise 0.015 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 11, 10 (PUSH_DESCEND): 9, 11 (PUSH): 2
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.061 at step 19 (final 0.064)
- closest |object_to_goal| 0.201 at step 19 (final 0.212); goal_error min 0.211 at step 4 / final 0.211 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 5 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 24 steps; TERMINATED by `ee_ground_collision` at step 23
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 12, 10 (PUSH_DESCEND): 11, 11 (PUSH): 1
- aperture min -0.003 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.054 at step 22 (final 0.056)
- closest |object_to_goal| 0.273 at step 5 (final 0.287); goal_error min 0.283 at step 3 / final 0.285 (success < 0.070)
- object never lifted (max rise 0.002 m)

### env 6 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 18 steps; TERMINATED by `ee_ground_collision` at step 17
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 8, 10 (PUSH_DESCEND): 8, 11 (PUSH): 2
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.057 at step 16 (final 0.060)
- closest |object_to_goal| 0.230 at step 6 (final 0.232); goal_error min 0.241 at step 4 / final 0.241 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 7 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 189 steps; TERMINATED by `ee_ground_collision` at step 188
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 70, 10 (PUSH_DESCEND): 65, 11 (PUSH): 54
- aperture min -0.008 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.021 at step 76 (final 0.048)
- closest |object_to_goal| 0.187 at step 182 (final 0.193); goal_error min 0.198 at step 160 / final 0.198 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 18 steps; TERMINATED by `ee_ground_collision` at step 17
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 11, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.059 at step 16 (final 0.072)
- closest |object_to_goal| 0.242 at step 4 (final 0.248); goal_error min 0.253 at step 1 / final 0.253 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 42 steps; TERMINATED by `ee_ground_collision` at step 41
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 28, 10 (PUSH_DESCEND): 12, 11 (PUSH): 2
- aperture min -0.003 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.041 at step 41 (final 0.041)
- closest |object_to_goal| 0.223 at step 25 (final 0.233); goal_error min 0.232 at step 41 / final 0.232 (success < 0.070)
- object never lifted (max rise 0.001 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 86 steps; TERMINATED by `ee_ground_collision` at step 85
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 38, 10 (PUSH_DESCEND): 27, 11 (PUSH): 21
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.025 at step 59 (final 0.061)
- closest |object_to_goal| 0.207 at step 84 (final 0.208); goal_error min 0.218 at step 58 / final 0.218 (success < 0.070)
- object never lifted (max rise 0.001 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 82 steps; TERMINATED by `ee_ground_collision` at step 81
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 44, 10 (PUSH_DESCEND): 24, 11 (PUSH): 14
- aperture min -0.007 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.023 at step 55 (final 0.052)
- closest |object_to_goal| 0.241 at step 62 (final 0.256); goal_error min 0.253 at step 81 / final 0.253 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 15, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.050 at step 21 (final 0.050)
- closest |object_to_goal| 0.259 at step 13 (final 0.265); goal_error min 0.268 at step 3 / final 0.268 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 19 steps; TERMINATED by `ee_ground_collision` at step 18
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 12, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.041 at step 18 (final 0.041)
- closest |object_to_goal| 0.296 at step 16 (final 0.313); goal_error min 0.308 at step 3 / final 0.308 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 20 steps; TERMINATED by `ee_ground_collision` at step 19
- most steps in phase 10 (PUSH_DESCEND); steps per phase 9 (PUSH_HOVER): 7, 10 (PUSH_DESCEND): 11, 11 (PUSH): 2
- aperture min -0.005 / max 0.076 / final -0.005
- closest |gripper_to_object| 0.049 at step 18 (final 0.049)
- closest |object_to_goal| 0.226 at step 7 (final 0.239); goal_error min 0.236 at step 4 / final 0.236 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 16 — terminated:object_out_of_bounds
- ended in phase 10 (PUSH_DESCEND) after 217 steps; TERMINATED by `object_out_of_bounds` at step 216
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 103, 10 (PUSH_DESCEND): 58, 11 (PUSH): 56
- aperture min -0.007 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.019 at step 98 (final 0.195)
- closest |object_to_goal| 0.103 at step 178 (final 0.428); goal_error min 0.100 at step 180 / final 0.433 (success < 0.070)
- object LIFTED at step 182 (max rise 0.028 m); LOST at step 185 (aperture collapsed while the object moved away)

### env 17 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 87 steps; TERMINATED by `ee_ground_collision` at step 86
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 43, 10 (PUSH_DESCEND): 27, 11 (PUSH): 17
- aperture min -0.007 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.019 at step 58 (final 0.065)
- closest |object_to_goal| 0.208 at step 74 (final 0.227); goal_error min 0.219 at step 86 / final 0.219 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 24 steps; TERMINATED by `ee_ground_collision` at step 23
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 16, 10 (PUSH_DESCEND): 6, 11 (PUSH): 2
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.074 at step 17 (final 0.076)
- closest |object_to_goal| 0.260 at step 19 (final 0.271); goal_error min 0.270 at step 3 / final 0.270 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 19 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 65 steps; TERMINATED by `ee_ground_collision` at step 64
- most steps in phase 10 (PUSH_DESCEND); steps per phase 9 (PUSH_HOVER): 21, 10 (PUSH_DESCEND): 30, 11 (PUSH): 14
- aperture min -0.008 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.018 at step 37 (final 0.059)
- closest |object_to_goal| 0.194 at step 54 (final 0.199); goal_error min 0.204 at step 64 / final 0.204 (success < 0.070)
- object never lifted (max rise 0.001 m)

### env 20 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 82 steps; TERMINATED by `ee_ground_collision` at step 81
- most steps in phase 10 (PUSH_DESCEND); steps per phase 9 (PUSH_HOVER): 20, 10 (PUSH_DESCEND): 32, 11 (PUSH): 30
- aperture min -0.009 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.023 at step 52 (final 0.059)
- closest |object_to_goal| 0.134 at step 69 (final 0.144); goal_error min 0.140 at step 55 / final 0.140 (success < 0.070)
- object never lifted (max rise 0.003 m)

### env 21 — terminated:ee_ground_collision
- ended in phase 10 (PUSH_DESCEND) after 24 steps; TERMINATED by `ee_ground_collision` at step 23
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 16, 10 (PUSH_DESCEND): 8
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.060 at step 23 (final 0.060)
- closest |object_to_goal| 0.276 at step 21 (final 0.277); goal_error min 0.285 at step 1 / final 0.285 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 19 steps; TERMINATED by `ee_ground_collision` at step 18
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 11, 10 (PUSH_DESCEND): 7, 11 (PUSH): 1
- aperture min -0.002 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.055 at step 18 (final 0.055)
- closest |object_to_goal| 0.266 at step 6 (final 0.269); goal_error min 0.274 at step 18 / final 0.274 (success < 0.070)
- object never lifted (max rise 0.002 m)

### env 23 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 106 steps; TERMINATED by `ee_ground_collision` at step 105
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 41, 10 (PUSH_DESCEND): 28, 11 (PUSH): 37
- aperture min -0.005 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.023 at step 80 (final 0.063)
- closest |object_to_goal| 0.141 at step 104 (final 0.150); goal_error min 0.148 at step 85 / final 0.148 (success < 0.070)
- object never lifted (max rise 0.004 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 18 steps; TERMINATED by `ee_ground_collision` at step 17
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 11, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.061 at step 16 (final 0.067)
- closest |object_to_goal| 0.246 at step 5 (final 0.262); goal_error min 0.257 at step 5 / final 0.257 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 493 steps; TERMINATED by `ee_ground_collision` at step 492
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 216, 10 (PUSH_DESCEND): 157, 11 (PUSH): 120
- aperture min -0.007 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.017 at step 259 (final 0.059)
- closest |object_to_goal| 0.257 at step 458 (final 0.261); goal_error min 0.263 at step 492 / final 0.263 (success < 0.070)
- object never lifted (max rise 0.001 m)

### env 26 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 114 steps; TERMINATED by `ee_ground_collision` at step 113
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 43, 10 (PUSH_DESCEND): 42, 11 (PUSH): 29
- aperture min -0.004 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.022 at step 83 (final 0.062)
- closest |object_to_goal| 0.199 at step 107 (final 0.211); goal_error min 0.210 at step 81 / final 0.210 (success < 0.070)
- object never lifted (max rise 0.016 m)

### env 27 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 13, 10 (PUSH_DESCEND): 8, 11 (PUSH): 1
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.061 at step 21 (final 0.061)
- closest |object_to_goal| 0.266 at step 12 (final 0.275); goal_error min 0.273 at step 4 / final 0.273 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 28 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 255 steps; TERMINATED by `ee_ground_collision` at step 254
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 120, 10 (PUSH_DESCEND): 75, 11 (PUSH): 60
- aperture min -0.005 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.017 at step 55 (final 0.057)
- closest |object_to_goal| 0.239 at step 242 (final 0.240); goal_error min 0.245 at step 254 / final 0.245 (success < 0.070)
- object never lifted (max rise 0.013 m)

### env 29 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 124 steps; TERMINATED by `ee_ground_collision` at step 123
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 59, 10 (PUSH_DESCEND): 39, 11 (PUSH): 26
- aperture min -0.007 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.022 at step 94 (final 0.065)
- closest |object_to_goal| 0.242 at step 119 (final 0.247); goal_error min 0.250 at step 123 / final 0.250 (success < 0.070)
- object never lifted (max rise 0.001 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 25 steps; TERMINATED by `ee_ground_collision` at step 24
- most steps in phase 10 (PUSH_DESCEND); steps per phase 9 (PUSH_HOVER): 7, 10 (PUSH_DESCEND): 16, 11 (PUSH): 2
- aperture min -0.001 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.053 at step 23 (final 0.057)
- closest |object_to_goal| 0.203 at step 20 (final 0.206); goal_error min 0.211 at step 24 / final 0.211 (success < 0.070)
- object never lifted (max rise 0.000 m)

### env 31 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 412 steps; TERMINATED by `ee_ground_collision` at step 411
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 179, 10 (PUSH_DESCEND): 130, 11 (PUSH): 103
- aperture min -0.007 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.016 at step 55 (final 0.065)
- closest |object_to_goal| 0.228 at step 397 (final 0.243); goal_error min 0.239 at step 411 / final 0.239 (success < 0.070)
- object never lifted (max rise 0.000 m)

## Successful envs (one line each)

- env 1: success at step 84, end phase 11 (PUSH), aperture min -0.010, min |go| 0.025, lifted at 138
- env 14: success at step 104, end phase 11 (PUSH), aperture min -0.006, min |go| 0.026
