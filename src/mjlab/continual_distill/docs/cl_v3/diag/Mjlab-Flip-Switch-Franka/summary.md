# diagnose — Mjlab-Flip-Switch-Franka

2026-09-09T13:25:00 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 150 · device cuda:0  
**19/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `FlipSwitchClassicalPolicy`; primary object `asset`; mechanism joint `switch_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_switch_collision'].  
Phases: 0 = ALIGN, 1 = SEAT, 2 = STROKE

## Histograms

- end phase, FAILING envs (13): 2 (STROKE): 9, 1 (SEAT): 3, 0 (ALIGN): 1
- end phase, successful envs (19): 1 (SEAT): 12, 2 (STROKE): 6, 0 (ALIGN): 1
- most-steps phase, failing envs: 1 (SEAT): 13
- failure class: terminated:ee_ground_collision: 9, stalled_phase_1: 2, mechanism_short: 1, stalled_phase_2: 1
- termination among failing envs: ee_ground_collision: 9, none: 4
- success step (successful envs): median 70, max 124

## Reading

- **terminated:ee_ground_collision** — 9/13 failing envs (envs [3, 4, 9, 18, 24, 26, 27, 30, 31]); typical end phase 2 (STROKE); final aperture median 0.000; closest |gripper_to_object| median 0.030; closest |object_to_goal| median 0.061; min goal_error median 1.311 (success < 0.150); mechanism reached median -0.785 of target 0.524
- **stalled_phase_1** — 2/13 failing envs (envs [1, 14]); typical end phase 1 (SEAT); final aperture median 0.000; closest |gripper_to_object| median 0.015; closest |object_to_goal| median 0.060; min goal_error median 1.311 (success < 0.150); mechanism reached median -0.785 of target 0.524
- **mechanism_short** — 1/13 failing envs (envs [2]); typical end phase 1 (SEAT); final aperture median -0.012; closest |gripper_to_object| median 0.017; closest |object_to_goal| median 0.036; min goal_error median 0.316 (success < 0.150); mechanism reached median 0.208 of target 0.524
- **stalled_phase_2** — 1/13 failing envs (envs [5]); typical end phase 2 (STROKE); final aperture median -0.002; closest |gripper_to_object| median 0.014; closest |object_to_goal| median 0.064; min goal_error median 1.311 (success < 0.150); mechanism reached median -0.785 of target 0.524

## Failing envs

### env 1 — stalled_phase_1
- ended in phase 1 (SEAT) after 150 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 7, 1 (SEAT): 112, 2 (STROKE): 31
- aperture min -0.002 / max 0.077 / final 0.000
- closest |gripper_to_object| 0.007 at step 94 (final 0.065)
- closest |object_to_goal| 0.061 at step 13 (final 0.084); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 2 — mechanism_short
- ended in phase 1 (SEAT) after 150 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 17, 1 (SEAT): 71, 2 (STROKE): 62
- aperture min -0.012 / max 0.076 / final -0.012
- closest |gripper_to_object| 0.017 at step 64 (final 0.042)
- closest |object_to_goal| 0.036 at step 124 (final 0.082); goal_error min 0.316 at step 67 / final 1.380 (success < 0.150)
- mechanism joint: reached 0.208 (final -0.856) of target 0.524

### env 3 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 105 steps; TERMINATED by `ee_ground_collision` at step 104
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 1, 1 (SEAT): 77, 2 (STROKE): 27
- aperture min -0.011 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.019 at step 55 (final 0.295)
- closest |object_to_goal| 0.062 at step 44 (final 0.072); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 4 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 102 steps; TERMINATED by `ee_ground_collision` at step 101
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 9, 1 (SEAT): 71, 2 (STROKE): 22
- aperture min -0.014 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.029 at step 59 (final 0.236)
- closest |object_to_goal| 0.061 at step 79 (final 0.072); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 5 — stalled_phase_2
- ended in phase 2 (STROKE) after 150 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 3, 1 (SEAT): 133, 2 (STROKE): 14
- aperture min -0.008 / max 0.083 / final -0.002
- closest |gripper_to_object| 0.014 at step 113 (final 0.078)
- closest |object_to_goal| 0.064 at step 0 (final 0.070); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 9 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 94 steps; TERMINATED by `ee_ground_collision` at step 93
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 1, 1 (SEAT): 67, 2 (STROKE): 26
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.026 at step 49 (final 0.256)
- closest |object_to_goal| 0.062 at step 0 (final 0.079); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 14 — stalled_phase_1
- ended in phase 1 (SEAT) after 150 steps; ran to time-out
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 12, 1 (SEAT): 107, 2 (STROKE): 31
- aperture min -0.018 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.023 at step 129 (final 0.086)
- closest |object_to_goal| 0.059 at step 108 (final 0.084); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 18 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 80 steps; TERMINATED by `ee_ground_collision` at step 79
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 14, 1 (SEAT): 44, 2 (STROKE): 22
- aperture min -0.002 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.039 at step 64 (final 0.156)
- closest |object_to_goal| 0.037 at step 66 (final 0.074); goal_error min 0.567 at step 66 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.043 (final -0.788) of target 0.524

### env 24 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 101 steps; TERMINATED by `ee_ground_collision` at step 100
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 1, 1 (SEAT): 77, 2 (STROKE): 23
- aperture min -0.011 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.030 at step 59 (final 0.250)
- closest |object_to_goal| 0.062 at step 47 (final 0.078); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 26 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 73 steps; TERMINATED by `ee_ground_collision` at step 72
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 10, 1 (SEAT): 46, 2 (STROKE): 17
- aperture min -0.002 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.040 at step 51 (final 0.142)
- closest |object_to_goal| 0.061 at step 43 (final 0.069); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 27 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 67 steps; TERMINATED by `ee_ground_collision` at step 66
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 7, 1 (SEAT): 47, 2 (STROKE): 13
- aperture min -0.001 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.036 at step 50 (final 0.160)
- closest |object_to_goal| 0.062 at step 33 (final 0.078); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 30 — terminated:ee_ground_collision
- ended in phase 0 (ALIGN) after 132 steps; TERMINATED by `ee_ground_collision` at step 131
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 19, 1 (SEAT): 82, 2 (STROKE): 31
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.047 at step 78 (final 0.283)
- closest |object_to_goal| 0.061 at step 22 (final 0.081); goal_error min 1.311 at step 16 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.785 (final -0.788) of target 0.524

### env 31 — terminated:ee_ground_collision
- ended in phase 2 (STROKE) after 101 steps; TERMINATED by `ee_ground_collision` at step 100
- most steps in phase 1 (SEAT); steps per phase 0 (ALIGN): 1, 1 (SEAT): 75, 2 (STROKE): 25
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.023 at step 54 (final 0.272)
- closest |object_to_goal| 0.060 at step 53 (final 0.073); goal_error min 1.038 at step 53 / final 1.311 (success < 0.150)
- mechanism joint: reached -0.515 (final -0.788) of target 0.524

## Successful envs (one line each)

- env 0: success at step 84, end phase 2 (STROKE), aperture min -0.004, min |go| 0.026
- env 6: success at step 75, end phase 1 (SEAT), aperture min -0.003, min |go| 0.012
- env 7: success at step 71, end phase 1 (SEAT), aperture min -0.003, min |go| 0.008
- env 8: success at step 66, end phase 1 (SEAT), aperture min -0.008, min |go| 0.017
- env 10: success at step 64, end phase 1 (SEAT), aperture min -0.000, min |go| 0.014
- env 11: success at step 70, end phase 1 (SEAT), aperture min -0.007, min |go| 0.018
- env 12: success at step 73, end phase 1 (SEAT), aperture min -0.001, min |go| 0.012
- env 13: success at step 77, end phase 1 (SEAT), aperture min -0.005, min |go| 0.014
- env 15: success at step 61, end phase 1 (SEAT), aperture min -0.010, min |go| 0.019
- env 16: success at step 65, end phase 2 (STROKE), aperture min -0.007, min |go| 0.026
- env 17: success at step 69, end phase 2 (STROKE), aperture min -0.004, min |go| 0.019
- env 19: success at step 52, end phase 2 (STROKE), aperture min -0.002, min |go| 0.021
- env 20: success at step 124, end phase 1 (SEAT), aperture min -0.001, min |go| 0.015
- env 21: success at step 95, end phase 2 (STROKE), aperture min -0.003, min |go| 0.022
- env 22: success at step 65, end phase 1 (SEAT), aperture min -0.003, min |go| 0.011
- env 23: success at step 71, end phase 1 (SEAT), aperture min -0.000, min |go| 0.014
- env 25: success at step 65, end phase 1 (SEAT), aperture min -0.004, min |go| 0.012
- env 28: success at step 74, end phase 0 (ALIGN), aperture min -0.005, min |go| 0.010
- env 29: success at step 67, end phase 2 (STROKE), aperture min -0.006, min |go| 0.027
