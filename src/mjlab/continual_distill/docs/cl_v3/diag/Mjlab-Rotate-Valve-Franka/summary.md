# diagnose — Mjlab-Rotate-Valve-Franka

2026-09-09T13:29:01 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 400 · device cuda:0  
**15/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `RotateValveClassicalPolicy`; primary object `asset`; mechanism joint `valve_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_valve_collision'].  
Phases: 0 = ALIGN, 1 = SEAT, 2 = ARC_FOLLOW, 3 = PINCH

## Histograms

- end phase, FAILING envs (17): 0 (ALIGN): 6, 2 (ARC_FOLLOW): 5, 1 (SEAT): 4, 3 (PINCH): 2
- end phase, successful envs (15): 1 (SEAT): 6, 0 (ALIGN): 5, 2 (ARC_FOLLOW): 4
- most-steps phase, failing envs: 2 (ARC_FOLLOW): 10, 1 (SEAT): 7
- failure class: mechanism_short: 13, near_miss: 2, stalled_phase_2: 1, stalled_phase_1: 1
- termination among failing envs: none: 17
- success step (successful envs): median 233, max 391

## Reading

- **mechanism_short** — 13/17 failing envs (envs [3, 4, 5, 6, 9, 11, 13, 15, 16, 20, 23, 26, 29]); typical end phase 0 (ALIGN); final aperture median 0.041; closest |gripper_to_object| median 0.023; closest |object_to_goal| median 0.005; min goal_error median 2.238 (success < 0.200); mechanism reached median 2.475 of target 4.712
- **near_miss** — 2/17 failing envs (envs [19, 24]); typical end phase 0 (ALIGN); final aperture median 0.043; closest |gripper_to_object| median 0.019; closest |object_to_goal| median 0.004; min goal_error median 0.276 (success < 0.200); mechanism reached median 4.437 of target 4.712
- **stalled_phase_2** — 1/17 failing envs (envs [17]); typical end phase 2 (ARC_FOLLOW); final aperture median 0.001; closest |gripper_to_object| median 0.013; closest |object_to_goal| median 0.005; min goal_error median 0.453 (success < 0.200); mechanism reached median 4.260 of target 4.712
- **stalled_phase_1** — 1/17 failing envs (envs [22]); typical end phase 1 (SEAT); final aperture median 0.077; closest |gripper_to_object| median 0.012; closest |object_to_goal| median 0.004; min goal_error median 0.344 (success < 0.200); mechanism reached median 4.368 of target 4.712

## Failing envs

### env 3 — mechanism_short
- ended in phase 1 (SEAT) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 102, 1 (SEAT): 150, 2 (ARC_FOLLOW): 128, 3 (PINCH): 20
- aperture min -0.017 / max 0.097 / final 0.081
- closest |gripper_to_object| 0.020 at step 377 (final 0.224)
- closest |object_to_goal| 0.029 at step 156 (final 0.131); goal_error min 3.505 at step 155 / final 4.711 (success < 0.200)
- mechanism joint: reached 1.207 (final 0.001) of target 4.712

### env 4 — mechanism_short
- ended in phase 2 (ARC_FOLLOW) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 132, 1 (SEAT): 143, 2 (ARC_FOLLOW): 109, 3 (PINCH): 16
- aperture min 0.030 / max 0.102 / final 0.042
- closest |gripper_to_object| 0.005 at step 71 (final 0.105)
- closest |object_to_goal| 0.009 at step 71 (final 0.051); goal_error min 1.431 at step 103 / final 2.536 (success < 0.200)
- mechanism joint: reached 3.282 (final 2.176) of target 4.712

### env 5 — mechanism_short
- ended in phase 3 (PINCH) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 88, 1 (SEAT): 147, 2 (ARC_FOLLOW): 151, 3 (PINCH): 14
- aperture min -0.016 / max 0.093 / final 0.064
- closest |gripper_to_object| 0.029 at step 54 (final 0.213)
- closest |object_to_goal| 0.002 at step 280 (final 0.089); goal_error min 2.050 at step 399 / final 2.050 (success < 0.200)
- mechanism joint: reached 2.663 (final 2.663) of target 4.712

### env 6 — mechanism_short
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 91, 1 (SEAT): 142, 2 (ARC_FOLLOW): 151, 3 (PINCH): 16
- aperture min -0.017 / max 0.080 / final 0.041
- closest |gripper_to_object| 0.027 at step 395 (final 0.058)
- closest |object_to_goal| 0.003 at step 293 (final 0.045); goal_error min 2.742 at step 139 / final 3.548 (success < 0.200)
- mechanism joint: reached 1.970 (final 1.165) of target 4.712

### env 9 — mechanism_short
- ended in phase 2 (ARC_FOLLOW) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 60, 1 (SEAT): 103, 2 (ARC_FOLLOW): 221, 3 (PINCH): 16
- aperture min -0.018 / max 0.098 / final -0.001
- closest |gripper_to_object| 0.029 at step 150 (final 0.185)
- closest |object_to_goal| 0.003 at step 344 (final 0.010); goal_error min 2.842 at step 235 / final 3.123 (success < 0.200)
- mechanism joint: reached 1.870 (final 1.589) of target 4.712

### env 11 — mechanism_short
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 96, 1 (SEAT): 127, 2 (ARC_FOLLOW): 161, 3 (PINCH): 16
- aperture min -0.006 / max 0.100 / final 0.018
- closest |gripper_to_object| 0.018 at step 378 (final 0.029)
- closest |object_to_goal| 0.009 at step 387 (final 0.058); goal_error min 2.394 at step 399 / final 2.394 (success < 0.200)
- mechanism joint: reached 2.318 (final 2.318) of target 4.712

### env 13 — mechanism_short
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 137, 1 (SEAT): 163, 2 (ARC_FOLLOW): 76, 3 (PINCH): 24
- aperture min -0.011 / max 0.101 / final 0.001
- closest |gripper_to_object| 0.013 at step 395 (final 0.034)
- closest |object_to_goal| 0.005 at step 374 (final 0.101); goal_error min 1.713 at step 320 / final 1.830 (success < 0.200)
- mechanism joint: reached 2.999 (final 2.883) of target 4.712

### env 15 — mechanism_short
- ended in phase 2 (ARC_FOLLOW) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 108, 1 (SEAT): 190, 2 (ARC_FOLLOW): 78, 3 (PINCH): 24
- aperture min -0.017 / max 0.109 / final 0.078
- closest |gripper_to_object| 0.023 at step 391 (final 0.032)
- closest |object_to_goal| 0.005 at step 127 (final 0.137); goal_error min 2.218 at step 182 / final 4.807 (success < 0.200)
- mechanism joint: reached 2.495 (final -0.094) of target 4.712

### env 16 — mechanism_short
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 66, 1 (SEAT): 129, 2 (ARC_FOLLOW): 189, 3 (PINCH): 16
- aperture min -0.015 / max 0.094 / final -0.006
- closest |gripper_to_object| 0.024 at step 337 (final 0.076)
- closest |object_to_goal| 0.009 at step 309 (final 0.152); goal_error min 0.935 at step 379 / final 1.245 (success < 0.200)
- mechanism joint: reached 3.778 (final 3.467) of target 4.712

### env 17 — stalled_phase_2
- ended in phase 2 (ARC_FOLLOW) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 57, 1 (SEAT): 128, 2 (ARC_FOLLOW): 199, 3 (PINCH): 16
- aperture min -0.016 / max 0.094 / final 0.001
- closest |gripper_to_object| 0.013 at step 48 (final 0.094)
- closest |object_to_goal| 0.005 at step 104 (final 0.133); goal_error min 0.453 at step 326 / final 1.508 (success < 0.200)
- mechanism joint: reached 4.260 (final 3.204) of target 4.712

### env 19 — near_miss
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 109, 1 (SEAT): 177, 2 (ARC_FOLLOW): 90, 3 (PINCH): 24
- aperture min -0.013 / max 0.107 / final 0.064
- closest |gripper_to_object| 0.012 at step 196 (final 0.096)
- closest |object_to_goal| 0.005 at step 99 (final 0.131); goal_error min 0.279 at step 208 / final 4.712 (success < 0.200)
- mechanism joint: reached 4.434 (final 0.001) of target 4.712

### env 20 — mechanism_short
- ended in phase 3 (PINCH) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 50, 1 (SEAT): 86, 2 (ARC_FOLLOW): 252, 3 (PINCH): 12
- aperture min -0.016 / max 0.092 / final 0.047
- closest |gripper_to_object| 0.031 at step 57 (final 0.226)
- closest |object_to_goal| 0.003 at step 385 (final 0.087); goal_error min 2.238 at step 398 / final 2.238 (success < 0.200)
- mechanism joint: reached 2.475 (final 2.475) of target 4.712

### env 22 — stalled_phase_1
- ended in phase 1 (SEAT) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 89, 1 (SEAT): 183, 2 (ARC_FOLLOW): 100, 3 (PINCH): 28
- aperture min -0.015 / max 0.087 / final 0.077
- closest |gripper_to_object| 0.012 at step 218 (final 0.079)
- closest |object_to_goal| 0.004 at step 129 (final 0.014); goal_error min 0.344 at step 227 / final 3.387 (success < 0.200)
- mechanism joint: reached 4.368 (final 1.326) of target 4.712

### env 23 — mechanism_short
- ended in phase 1 (SEAT) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 60, 1 (SEAT): 122, 2 (ARC_FOLLOW): 202, 3 (PINCH): 16
- aperture min -0.016 / max 0.090 / final 0.000
- closest |gripper_to_object| 0.012 at step 65 (final 0.183)
- closest |object_to_goal| 0.004 at step 321 (final 0.079); goal_error min 2.252 at step 398 / final 2.267 (success < 0.200)
- mechanism joint: reached 2.460 (final 2.446) of target 4.712

### env 24 — near_miss
- ended in phase 1 (SEAT) after 400 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 86, 1 (SEAT): 171, 2 (ARC_FOLLOW): 127, 3 (PINCH): 16
- aperture min -0.017 / max 0.095 / final 0.023
- closest |gripper_to_object| 0.026 at step 53 (final 0.050)
- closest |object_to_goal| 0.002 at step 222 (final 0.011); goal_error min 0.272 at step 317 / final 3.096 (success < 0.200)
- mechanism joint: reached 4.440 (final 1.616) of target 4.712

### env 26 — mechanism_short
- ended in phase 0 (ALIGN) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 55, 1 (SEAT): 82, 2 (ARC_FOLLOW): 251, 3 (PINCH): 12
- aperture min 0.004 / max 0.091 / final 0.065
- closest |gripper_to_object| 0.011 at step 288 (final 0.043)
- closest |object_to_goal| 0.013 at step 250 (final 0.105); goal_error min 1.897 at step 388 / final 1.930 (success < 0.200)
- mechanism joint: reached 2.815 (final 2.782) of target 4.712

### env 29 — mechanism_short
- ended in phase 2 (ARC_FOLLOW) after 400 steps; ran to time-out
- most steps in phase 2 (ARC_FOLLOW); steps per phase 0 (ALIGN): 42, 1 (SEAT): 58, 2 (ARC_FOLLOW): 292, 3 (PINCH): 8
- aperture min -0.004 / max 0.098 / final 0.000
- closest |gripper_to_object| 0.033 at step 51 (final 0.190)
- closest |object_to_goal| 0.052 at step 257 (final 0.062); goal_error min 3.864 at step 189 / final 3.864 (success < 0.200)
- mechanism joint: reached 0.848 (final 0.848) of target 4.712

## Successful envs (one line each)

- env 0: success at step 312, end phase 1 (SEAT), aperture min -0.014, min |go| 0.019
- env 1: success at step 176, end phase 1 (SEAT), aperture min -0.016, min |go| 0.010
- env 2: success at step 174, end phase 0 (ALIGN), aperture min -0.020, min |go| 0.017
- env 7: success at step 261, end phase 2 (ARC_FOLLOW), aperture min -0.017, min |go| 0.017
- env 8: success at step 194, end phase 2 (ARC_FOLLOW), aperture min -0.015, min |go| 0.029
- env 10: success at step 230, end phase 2 (ARC_FOLLOW), aperture min -0.015, min |go| 0.015
- env 12: success at step 370, end phase 0 (ALIGN), aperture min -0.004, min |go| 0.006
- env 14: success at step 336, end phase 2 (ARC_FOLLOW), aperture min 0.000, min |go| 0.014
- env 18: success at step 173, end phase 1 (SEAT), aperture min -0.012, min |go| 0.013
- env 21: success at step 391, end phase 0 (ALIGN), aperture min -0.015, min |go| 0.024
- env 25: success at step 353, end phase 0 (ALIGN), aperture min -0.015, min |go| 0.017
- env 27: success at step 175, end phase 1 (SEAT), aperture min -0.012, min |go| 0.019
- env 28: success at step 307, end phase 1 (SEAT), aperture min -0.013, min |go| 0.025
- env 30: success at step 233, end phase 1 (SEAT), aperture min -0.017, min |go| 0.018
- env 31: success at step 145, end phase 0 (ALIGN), aperture min -0.015, min |go| 0.017
