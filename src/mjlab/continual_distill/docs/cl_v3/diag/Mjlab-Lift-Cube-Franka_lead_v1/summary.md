# diagnose — Mjlab-Lift-Cube-Franka

2026-09-09T20:35:59 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 1000 · device cuda:0  
**103/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `LiftCubeClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = LIFT

## Histograms

- end phase, FAILING envs (25): 2 (CLOSE): 15, 1 (DESCEND): 9, 3 (LIFT): 1
- end phase, successful envs (103): 3 (LIFT): 94, 2 (CLOSE): 5, 1 (DESCEND): 4
- most-steps phase, failing envs: 0 (HOVER): 18, 1 (DESCEND): 7
- failure class: terminated:ee_ground_collision: 21, never_lifted: 4
- termination among failing envs: ee_ground_collision: 21, none: 4
- failing envs that lifted the object: 1/25; lifted then lost: 1
- success step (successful envs): median 123, max 824

## Reading

- **terminated:ee_ground_collision** — 21/25 failing envs (envs [0, 12, 13, 24, 30, 33, 44, 47, 50, 60, 65, 75, 95, 98, 106, 107, 109, 111, 115, 117, 123]); typical end phase 2 (CLOSE); final aperture median 0.080; closest |gripper_to_object| median 0.009; closest |object_to_goal| median 0.270; min goal_error median 0.276 (success < 0.050)
- **never_lifted** — 4/25 failing envs (envs [2, 31, 116, 118]); typical end phase 2 (CLOSE); final aperture median 0.081; closest |gripper_to_object| median 0.016; closest |object_to_goal| median 0.312; min goal_error median 0.319 (success < 0.050)

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 50 steps; TERMINATED by `ee_ground_collision` at step 49
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 24, 1 (DESCEND): 24, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 48 (final 0.015)
- closest |object_to_goal| 0.437 at step 30 (final 0.447); goal_error min 0.447 at step 1 / final 0.449 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 2 — never_lifted
- ended in phase 2 (CLOSE) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 202, 1 (DESCEND): 379, 2 (CLOSE): 149, 3 (LIFT): 270
- aperture min 0.000 / max 0.086 / final 0.083
- closest |gripper_to_object| 0.007 at step 184 (final 0.029)
- closest |object_to_goal| 0.310 at step 1 (final 0.359); goal_error min 0.318 at step 1 / final 0.360 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 55 steps; TERMINATED by `ee_ground_collision` at step 54
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 28, 1 (DESCEND): 25, 2 (CLOSE): 2
- aperture min 0.076 / max 0.085 / final 0.080
- closest |gripper_to_object| 0.012 at step 51 (final 0.026)
- closest |object_to_goal| 0.221 at step 0 (final 0.240); goal_error min 0.234 at step 1 / final 0.239 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 27, 1 (DESCEND): 22, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.008 at step 48 (final 0.016)
- closest |object_to_goal| 0.268 at step 1 (final 0.289); goal_error min 0.275 at step 1 / final 0.280 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 54 steps; TERMINATED by `ee_ground_collision` at step 53
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 33, 1 (DESCEND): 20, 2 (CLOSE): 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 53 (final 0.007)
- closest |object_to_goal| 0.422 at step 2 (final 0.447); goal_error min 0.423 at step 1 / final 0.438 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 27, 1 (DESCEND): 26
- aperture min 0.076 / max 0.084 / final 0.081
- closest |gripper_to_object| 0.009 at step 52 (final 0.009)
- closest |object_to_goal| 0.268 at step 9 (final 0.287); goal_error min 0.272 at step 1 / final 0.279 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 31 — never_lifted
- ended in phase 3 (LIFT) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 196, 1 (DESCEND): 364, 2 (CLOSE): 150, 3 (LIFT): 290
- aperture min 0.000 / max 0.084 / final 0.001
- closest |gripper_to_object| 0.019 at step 346 (final 0.114)
- closest |object_to_goal| 0.314 at step 90 (final 0.345); goal_error min 0.319 at step 89 / final 0.349 (success < 0.050)
- object never lifted (max rise 0.004 m)

### env 33 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 67 steps; TERMINATED by `ee_ground_collision` at step 66
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 41, 1 (DESCEND): 26
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 65 (final 0.010)
- closest |object_to_goal| 0.305 at step 54 (final 0.320); goal_error min 0.315 at step 1 / final 0.318 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 44 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 48 steps; TERMINATED by `ee_ground_collision` at step 47
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 24, 1 (DESCEND): 24
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.014 at step 46 (final 0.023)
- closest |object_to_goal| 0.244 at step 0 (final 0.264); goal_error min 0.254 at step 1 / final 0.256 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 47 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 47 steps; TERMINATED by `ee_ground_collision` at step 46
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 23, 1 (DESCEND): 24
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.012 at step 46 (final 0.012)
- closest |object_to_goal| 0.317 at step 29 (final 0.321); goal_error min 0.320 at step 1 / final 0.326 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 50 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 30, 1 (DESCEND): 26, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.010 at step 56 (final 0.017)
- closest |object_to_goal| 0.264 at step 0 (final 0.277); goal_error min 0.269 at step 1 / final 0.277 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 60 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 54 steps; TERMINATED by `ee_ground_collision` at step 53
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 29, 1 (DESCEND): 25
- aperture min 0.076 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.013 at step 52 (final 0.016)
- closest |object_to_goal| 0.143 at step 53 (final 0.143); goal_error min 0.150 at step 1 / final 0.155 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 65 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 33, 1 (DESCEND): 25, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 58 (final 0.017)
- closest |object_to_goal| 0.474 at step 1 (final 0.490); goal_error min 0.481 at step 1 / final 0.492 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 75 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 40, 1 (DESCEND): 20, 2 (CLOSE): 3
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.008 at step 60 (final 0.022)
- closest |object_to_goal| 0.384 at step 61 (final 0.397); goal_error min 0.394 at step 62 / final 0.394 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 95 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 31, 1 (DESCEND): 24, 2 (CLOSE): 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 55 (final 0.007)
- closest |object_to_goal| 0.321 at step 44 (final 0.327); goal_error min 0.327 at step 1 / final 0.334 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 98 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 31, 1 (DESCEND): 24, 2 (CLOSE): 2
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.008 at step 54 (final 0.018)
- closest |object_to_goal| 0.470 at step 0 (final 0.496); goal_error min 0.480 at step 1 / final 0.487 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 106 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 52 steps; TERMINATED by `ee_ground_collision` at step 51
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 30, 1 (DESCEND): 21, 2 (CLOSE): 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 51 (final 0.009)
- closest |object_to_goal| 0.171 at step 0 (final 0.183); goal_error min 0.182 at step 15 / final 0.182 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 107 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 35, 1 (DESCEND): 23
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 56 (final 0.009)
- closest |object_to_goal| 0.291 at step 17 (final 0.304); goal_error min 0.295 at step 1 / final 0.306 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 109 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 54 steps; TERMINATED by `ee_ground_collision` at step 53
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 27, 1 (DESCEND): 25, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.011 at step 52 (final 0.012)
- closest |object_to_goal| 0.209 at step 0 (final 0.234); goal_error min 0.216 at step 1 / final 0.228 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 111 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 658 steps; TERMINATED by `ee_ground_collision` at step 657
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 164, 1 (DESCEND): 206, 2 (CLOSE): 90, 3 (LIFT): 198
- aperture min 0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.009 at step 657 (final 0.009)
- closest |object_to_goal| 0.104 at step 603 (final 0.270); goal_error min 0.108 at step 603 / final 0.259 (success < 0.050)
- object LIFTED at step 561 (max rise 0.314 m); LOST at step 603 (aperture collapsed while the object moved away)

### env 115 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 31, 1 (DESCEND): 24, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 55 (final 0.018)
- closest |object_to_goal| 0.246 at step 1 (final 0.251); goal_error min 0.252 at step 1 / final 0.259 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 116 — never_lifted
- ended in phase 1 (DESCEND) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 221, 1 (DESCEND): 329, 2 (CLOSE): 150, 3 (LIFT): 300
- aperture min 0.000 / max 0.085 / final 0.080
- closest |gripper_to_object| 0.017 at step 266 (final 0.163)
- closest |object_to_goal| 0.402 at step 191 (final 0.440); goal_error min 0.411 at step 93 / final 0.430 (success < 0.050)
- object never lifted (max rise 0.006 m)

### env 117 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 47 steps; TERMINATED by `ee_ground_collision` at step 46
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 21, 1 (DESCEND): 26
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 44 (final 0.010)
- closest |object_to_goal| 0.270 at step 0 (final 0.286); goal_error min 0.276 at step 1 / final 0.289 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 118 — never_lifted
- ended in phase 2 (CLOSE) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 181, 1 (DESCEND): 399, 2 (CLOSE): 150, 3 (LIFT): 270
- aperture min 0.000 / max 0.087 / final 0.082
- closest |gripper_to_object| 0.015 at step 80 (final 0.041)
- closest |object_to_goal| 0.285 at step 0 (final 0.340); goal_error min 0.290 at step 1 / final 0.347 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 123 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 33, 1 (DESCEND): 21, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.008 at step 55 (final 0.008)
- closest |object_to_goal| 0.299 at step 0 (final 0.337); goal_error min 0.305 at step 1 / final 0.329 (success < 0.050)
- object never lifted (max rise 0.000 m)

## Successful envs (one line each)

- env 1: success at step 122, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 75
- env 3: success at step 314, end phase 3 (LIFT), aperture min 0.000, min |go| 0.011, lifted at 269
- env 4: success at step 80, end phase 3 (LIFT), aperture min 0.042, min |go| 0.001, lifted at 70
- env 5: success at step 122, end phase 3 (LIFT), aperture min 0.006, min |go| 0.003, lifted at 82
- env 6: success at step 117, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 77
- env 7: success at step 122, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 80
- env 8: success at step 121, end phase 1 (DESCEND), aperture min 0.009, min |go| 0.004, lifted at 75
- env 9: success at step 207, end phase 3 (LIFT), aperture min 0.000, min |go| 0.001, lifted at 170
- env 10: success at step 230, end phase 3 (LIFT), aperture min 0.000, min |go| 0.003, lifted at 184
- env 11: success at step 225, end phase 3 (LIFT), aperture min 0.000, min |go| 0.005, lifted at 181
- env 14: success at step 112, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 75
- env 15: success at step 120, end phase 2 (CLOSE), aperture min 0.020, min |go| 0.005, lifted at 88
- env 16: success at step 206, end phase 3 (LIFT), aperture min 0.000, min |go| 0.000, lifted at 167
- env 17: success at step 119, end phase 3 (LIFT), aperture min 0.013, min |go| 0.001, lifted at 83
- env 18: success at step 130, end phase 1 (DESCEND), aperture min 0.017, min |go| 0.005, lifted at 86
- env 19: success at step 121, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 74
- env 20: success at step 120, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 76
- env 21: success at step 118, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 68
- env 22: success at step 118, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 85
- env 23: success at step 130, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 78
- env 25: success at step 131, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 79
- env 26: success at step 121, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 71
- env 27: success at step 237, end phase 3 (LIFT), aperture min 0.000, min |go| 0.005, lifted at 191
- env 28: success at step 129, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 84
- env 29: success at step 130, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 77
- env 32: success at step 120, end phase 3 (LIFT), aperture min 0.013, min |go| 0.001, lifted at 86
- env 34: success at step 118, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 76
- env 35: success at step 526, end phase 2 (CLOSE), aperture min 0.000, min |go| 0.004, lifted at 489
- env 36: success at step 212, end phase 3 (LIFT), aperture min 0.000, min |go| 0.001, lifted at 175
- env 37: success at step 120, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 71
- env 38: success at step 131, end phase 1 (DESCEND), aperture min 0.007, min |go| 0.002, lifted at 84
- env 39: success at step 110, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 81
- env 40: success at step 115, end phase 3 (LIFT), aperture min 0.046, min |go| 0.001, lifted at 69
- env 41: success at step 421, end phase 3 (LIFT), aperture min 0.000, min |go| 0.006, lifted at 384
- env 42: success at step 804, end phase 3 (LIFT), aperture min 0.000, min |go| 0.007, lifted at 771
- env 43: success at step 127, end phase 1 (DESCEND), aperture min 0.013, min |go| 0.003, lifted at 76
- env 45: success at step 123, end phase 3 (LIFT), aperture min 0.015, min |go| 0.001, lifted at 76
- env 46: success at step 102, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 72
- env 48: success at step 313, end phase 3 (LIFT), aperture min 0.000, min |go| 0.008, lifted at 277
- env 49: success at step 216, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 171
- env 51: success at step 125, end phase 2 (CLOSE), aperture min 0.004, min |go| 0.012, lifted at 91
- env 52: success at step 126, end phase 3 (LIFT), aperture min 0.018, min |go| 0.001, lifted at 83
- env 53: success at step 399, end phase 3 (LIFT), aperture min 0.000, min |go| 0.007, lifted at 365
- env 54: success at step 116, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 81
- env 55: success at step 218, end phase 3 (LIFT), aperture min 0.000, min |go| 0.001, lifted at 174
- env 56: success at step 127, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 75
- env 57: success at step 113, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 81
- env 58: success at step 123, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 79
- env 59: success at step 117, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 84
- env 61: success at step 123, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 88
- env 62: success at step 112, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 72
- env 63: success at step 115, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 75
- env 64: success at step 526, end phase 3 (LIFT), aperture min 0.000, min |go| 0.005, lifted at 474
- env 66: success at step 131, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 78
- env 67: success at step 125, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 80
- env 68: success at step 125, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 83
- env 69: success at step 126, end phase 3 (LIFT), aperture min 0.011, min |go| 0.002, lifted at 87
- env 70: success at step 115, end phase 2 (CLOSE), aperture min 0.014, min |go| 0.005, lifted at 73
- env 71: success at step 131, end phase 3 (LIFT), aperture min 0.011, min |go| 0.004, lifted at 90
- env 72: success at step 824, end phase 3 (LIFT), aperture min 0.000, min |go| 0.003, lifted at 781
- env 73: success at step 123, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 77
- env 74: success at step 135, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 83
- env 76: success at step 122, end phase 3 (LIFT), aperture min 0.006, min |go| 0.001, lifted at 92
- env 77: success at step 123, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 86
- env 78: success at step 112, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 74
- env 79: success at step 118, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 76
- env 80: success at step 127, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 83
- env 81: success at step 138, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 76
- env 82: success at step 117, end phase 3 (LIFT), aperture min 0.042, min |go| 0.001, lifted at 75
- env 83: success at step 123, end phase 3 (LIFT), aperture min 0.043, min |go| 0.005, lifted at 80
- env 84: success at step 115, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 69
- env 85: success at step 121, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 78
- env 86: success at step 122, end phase 3 (LIFT), aperture min 0.008, min |go| 0.000, lifted at 84
- env 87: success at step 120, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 72
- env 88: success at step 208, end phase 3 (LIFT), aperture min 0.000, min |go| 0.001, lifted at 174
- env 89: success at step 223, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 191
- env 90: success at step 221, end phase 3 (LIFT), aperture min 0.000, min |go| 0.001, lifted at 177
- env 91: success at step 124, end phase 3 (LIFT), aperture min 0.005, min |go| 0.001, lifted at 88
- env 92: success at step 508, end phase 3 (LIFT), aperture min 0.000, min |go| 0.008, lifted at 478
- env 93: success at step 119, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 84
- env 94: success at step 119, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 79
- env 96: success at step 120, end phase 3 (LIFT), aperture min 0.014, min |go| 0.001, lifted at 91
- env 97: success at step 124, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 71
- env 99: success at step 424, end phase 3 (LIFT), aperture min 0.000, min |go| 0.009, lifted at 394
- env 100: success at step 105, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 73
- env 101: success at step 121, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 82
- env 102: success at step 117, end phase 3 (LIFT), aperture min 0.000, min |go| 0.006, lifted at 82
- env 103: success at step 92, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 71
- env 104: success at step 127, end phase 2 (CLOSE), aperture min 0.010, min |go| 0.004, lifted at 75
- env 105: success at step 109, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 71
- env 108: success at step 105, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 64
- env 110: success at step 116, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 76
- env 112: success at step 121, end phase 3 (LIFT), aperture min 0.015, min |go| 0.001, lifted at 85
- env 113: success at step 706, end phase 3 (LIFT), aperture min 0.000, min |go| 0.008, lifted at 670
- env 114: success at step 123, end phase 3 (LIFT), aperture min 0.044, min |go| 0.003, lifted at 79
- env 119: success at step 337, end phase 3 (LIFT), aperture min 0.000, min |go| 0.008, lifted at 288
- env 120: success at step 141, end phase 3 (LIFT), aperture min 0.042, min |go| 0.006, lifted at 98
- env 121: success at step 119, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 80
- env 122: success at step 610, end phase 3 (LIFT), aperture min 0.000, min |go| 0.004, lifted at 574
- env 124: success at step 218, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 177
- env 125: success at step 521, end phase 3 (LIFT), aperture min 0.000, min |go| 0.007, lifted at 277
- env 126: success at step 312, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 268
- env 127: success at step 520, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 478
