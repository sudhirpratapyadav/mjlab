# diagnose — Mjlab-Turn-Lever-Franka

2026-09-09T13:37:13 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 150 · device cuda:0  
**116/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `TurnLeverClassicalPolicy`; primary object `asset`; mechanism joint `lever_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_lever_collision'].  
Phases: unnamed — numbers are the teacher's `_phase` values.

## Histograms

- end phase, FAILING envs (12): 2: 6, 1: 6
- end phase, successful envs (116): 2: 93, 1: 23
- most-steps phase, failing envs: 2: 6, 1: 6
- failure class: mechanism_short: 7, mechanism_not_moved: 3, near_miss: 2
- termination among failing envs: none: 12
- success step (successful envs): median 67, max 147

## Reading

- **mechanism_short** — 7/12 failing envs (envs [39, 74, 76, 85, 108, 114, 122]); typical end phase 1; final aperture median -0.001; closest |gripper_to_object| median 0.021; closest |object_to_goal| median 0.064; min goal_error median 0.594 (success < 0.150); mechanism reached median -0.977 of target -1.571
- **mechanism_not_moved** — 3/12 failing envs (envs [55, 94, 115]); typical end phase 2; final aperture median -0.002; closest |gripper_to_object| median 0.024; closest |object_to_goal| median 0.157; min goal_error median 1.571 (success < 0.150); mechanism reached median -0.000 of target -1.571
- **near_miss** — 2/12 failing envs (envs [58, 61]); typical end phase 1; final aperture median 0.007; closest |gripper_to_object| median 0.017; closest |object_to_goal| median 0.013; min goal_error median 0.184 (success < 0.150); mechanism reached median -1.387 of target -1.571

## Failing envs

### env 39 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 37, 1: 32, 2: 81
- aperture min -0.003 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.022 at step 77 (final 0.066)
- closest |object_to_goal| 0.145 at step 120 (final 0.156); goal_error min 1.404 at step 93 / final 1.404 (success < 0.150)
- mechanism joint: reached -0.167 (final -0.167) of target -1.571

### env 55 — mechanism_not_moved
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 27, 1: 31, 2: 92
- aperture min -0.002 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.028 at step 63 (final 0.058)
- closest |object_to_goal| 0.157 at step 20 (final 0.167); goal_error min 1.571 at step 79 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.000 (final -0.000) of target -1.571

### env 58 — near_miss
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 36, 1: 94, 2: 20
- aperture min -0.006 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.015 at step 70 (final 0.074)
- closest |object_to_goal| 0.011 at step 105 (final 0.100); goal_error min 0.161 at step 106 / final 0.807 (success < 0.150)
- mechanism joint: reached -1.410 (final -0.764) of target -1.571

### env 61 — near_miss
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 33, 1: 28, 2: 89
- aperture min -0.000 / max 0.088 / final 0.014
- closest |gripper_to_object| 0.020 at step 135 (final 0.029)
- closest |object_to_goal| 0.016 at step 68 (final 0.077); goal_error min 0.207 at step 68 / final 0.636 (success < 0.150)
- mechanism joint: reached -1.364 (final -0.935) of target -1.571

### env 74 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 90, 2: 29
- aperture min -0.017 / max 0.076 / final 0.002
- closest |gripper_to_object| 0.018 at step 58 (final 0.073)
- closest |object_to_goal| 0.026 at step 75 (final 0.052); goal_error min 0.257 at step 77 / final 0.503 (success < 0.150)
- mechanism joint: reached -1.314 (final -1.067) of target -1.571

### env 76 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 89, 2: 29
- aperture min -0.015 / max 0.076 / final -0.010
- closest |gripper_to_object| 0.022 at step 70 (final 0.139)
- closest |object_to_goal| 0.099 at step 111 (final 0.118); goal_error min 0.963 at step 101 / final 0.963 (success < 0.150)
- mechanism joint: reached -0.608 (final -0.608) of target -1.571

### env 85 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 35, 1: 49, 2: 66
- aperture min -0.017 / max 0.076 / final 0.014
- closest |gripper_to_object| 0.016 at step 69 (final 0.047)
- closest |object_to_goal| 0.024 at step 97 (final 0.058); goal_error min 0.239 at step 101 / final 0.545 (success < 0.150)
- mechanism joint: reached -1.332 (final -1.026) of target -1.571

### env 94 — mechanism_not_moved
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 26, 1: 32, 2: 92
- aperture min -0.003 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.021 at step 64 (final 0.068)
- closest |object_to_goal| 0.157 at step 131 (final 0.176); goal_error min 1.571 at step 81 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.000 (final -0.000) of target -1.571

### env 108 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 27, 1: 75, 2: 48
- aperture min -0.008 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.020 at step 50 (final 0.118)
- closest |object_to_goal| 0.064 at step 140 (final 0.068); goal_error min 0.594 at step 135 / final 0.594 (success < 0.150)
- mechanism joint: reached -0.977 (final -0.977) of target -1.571

### env 114 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 91, 2: 28
- aperture min -0.014 / max 0.076 / final -0.006
- closest |gripper_to_object| 0.022 at step 69 (final 0.134)
- closest |object_to_goal| 0.108 at step 114 (final 0.125); goal_error min 1.032 at step 99 / final 1.032 (success < 0.150)
- mechanism joint: reached -0.539 (final -0.539) of target -1.571

### env 115 — mechanism_not_moved
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 2; steps per phase 0: 32, 1: 33, 2: 85
- aperture min -0.003 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.024 at step 71 (final 0.055)
- closest |object_to_goal| 0.155 at step 7 (final 0.169); goal_error min 1.571 at step 87 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.000 (final -0.000) of target -1.571

### env 122 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 83, 2: 34
- aperture min -0.013 / max 0.076 / final 0.014
- closest |gripper_to_object| 0.021 at step 84 (final 0.064)
- closest |object_to_goal| 0.024 at step 132 (final 0.113); goal_error min 0.283 at step 132 / final 1.074 (success < 0.150)
- mechanism joint: reached -1.288 (final -0.497) of target -1.571

## Successful envs (one line each)

- env 0: success at step 68, end phase 2, aperture min -0.016, min |go| 0.015
- env 1: success at step 85, end phase 1, aperture min -0.006, min |go| 0.019
- env 2: success at step 83, end phase 2, aperture min -0.009, min |go| 0.016
- env 3: success at step 66, end phase 2, aperture min -0.013, min |go| 0.016
- env 4: success at step 62, end phase 2, aperture min -0.012, min |go| 0.016
- env 5: success at step 60, end phase 2, aperture min -0.015, min |go| 0.012
- env 6: success at step 63, end phase 2, aperture min -0.016, min |go| 0.016
- env 7: success at step 128, end phase 2, aperture min -0.015, min |go| 0.019
- env 8: success at step 60, end phase 2, aperture min -0.010, min |go| 0.018
- env 9: success at step 65, end phase 2, aperture min -0.014, min |go| 0.018
- env 10: success at step 59, end phase 1, aperture min -0.007, min |go| 0.024
- env 11: success at step 50, end phase 2, aperture min -0.014, min |go| 0.020
- env 12: success at step 106, end phase 2, aperture min -0.009, min |go| 0.012
- env 13: success at step 68, end phase 2, aperture min -0.018, min |go| 0.021
- env 14: success at step 57, end phase 2, aperture min -0.015, min |go| 0.013
- env 15: success at step 140, end phase 2, aperture min -0.009, min |go| 0.020
- env 16: success at step 62, end phase 2, aperture min -0.017, min |go| 0.013
- env 17: success at step 53, end phase 2, aperture min -0.016, min |go| 0.013
- env 18: success at step 86, end phase 2, aperture min -0.007, min |go| 0.017
- env 19: success at step 67, end phase 2, aperture min -0.014, min |go| 0.028
- env 20: success at step 60, end phase 2, aperture min -0.012, min |go| 0.015
- env 21: success at step 61, end phase 2, aperture min -0.014, min |go| 0.010
- env 22: success at step 54, end phase 2, aperture min -0.014, min |go| 0.012
- env 23: success at step 85, end phase 2, aperture min -0.006, min |go| 0.019
- env 24: success at step 89, end phase 1, aperture min -0.008, min |go| 0.033
- env 25: success at step 58, end phase 2, aperture min -0.014, min |go| 0.020
- env 26: success at step 94, end phase 2, aperture min -0.009, min |go| 0.016
- env 27: success at step 79, end phase 1, aperture min -0.012, min |go| 0.016
- env 28: success at step 59, end phase 2, aperture min -0.013, min |go| 0.015
- env 29: success at step 90, end phase 2, aperture min -0.014, min |go| 0.025
- env 30: success at step 64, end phase 2, aperture min -0.012, min |go| 0.017
- env 31: success at step 68, end phase 2, aperture min -0.013, min |go| 0.019
- env 32: success at step 78, end phase 2, aperture min -0.016, min |go| 0.018
- env 33: success at step 64, end phase 2, aperture min -0.013, min |go| 0.014
- env 34: success at step 79, end phase 2, aperture min -0.015, min |go| 0.018
- env 35: success at step 62, end phase 2, aperture min -0.013, min |go| 0.016
- env 36: success at step 87, end phase 2, aperture min -0.006, min |go| 0.015
- env 37: success at step 60, end phase 2, aperture min -0.017, min |go| 0.015
- env 38: success at step 59, end phase 2, aperture min -0.016, min |go| 0.020
- env 40: success at step 146, end phase 1, aperture min -0.006, min |go| 0.031
- env 41: success at step 74, end phase 1, aperture min -0.008, min |go| 0.018
- env 42: success at step 147, end phase 1, aperture min -0.002, min |go| 0.023
- env 43: success at step 69, end phase 2, aperture min -0.016, min |go| 0.019
- env 44: success at step 93, end phase 2, aperture min -0.007, min |go| 0.017
- env 45: success at step 66, end phase 2, aperture min -0.015, min |go| 0.016
- env 46: success at step 91, end phase 1, aperture min -0.010, min |go| 0.014
- env 47: success at step 79, end phase 2, aperture min -0.016, min |go| 0.013
- env 48: success at step 62, end phase 1, aperture min -0.014, min |go| 0.011
- env 49: success at step 64, end phase 2, aperture min -0.013, min |go| 0.021
- env 50: success at step 94, end phase 2, aperture min -0.004, min |go| 0.023
- env 51: success at step 60, end phase 2, aperture min -0.018, min |go| 0.020
- env 52: success at step 65, end phase 2, aperture min -0.014, min |go| 0.016
- env 53: success at step 63, end phase 2, aperture min -0.014, min |go| 0.014
- env 54: success at step 67, end phase 1, aperture min -0.012, min |go| 0.021
- env 56: success at step 67, end phase 2, aperture min -0.015, min |go| 0.016
- env 57: success at step 64, end phase 2, aperture min -0.013, min |go| 0.013
- env 59: success at step 106, end phase 2, aperture min -0.006, min |go| 0.019
- env 60: success at step 64, end phase 2, aperture min -0.015, min |go| 0.018
- env 62: success at step 117, end phase 2, aperture min -0.015, min |go| 0.025
- env 63: success at step 61, end phase 2, aperture min -0.013, min |go| 0.010
- env 64: success at step 70, end phase 1, aperture min -0.006, min |go| 0.016
- env 65: success at step 63, end phase 2, aperture min -0.016, min |go| 0.016
- env 66: success at step 141, end phase 2, aperture min -0.011, min |go| 0.028
- env 67: success at step 61, end phase 2, aperture min -0.009, min |go| 0.023
- env 68: success at step 61, end phase 2, aperture min -0.014, min |go| 0.012
- env 69: success at step 68, end phase 2, aperture min -0.018, min |go| 0.013
- env 70: success at step 90, end phase 2, aperture min -0.008, min |go| 0.020
- env 71: success at step 85, end phase 2, aperture min -0.008, min |go| 0.016
- env 72: success at step 129, end phase 2, aperture min -0.015, min |go| 0.020
- env 73: success at step 120, end phase 2, aperture min -0.021, min |go| 0.016
- env 75: success at step 60, end phase 2, aperture min -0.015, min |go| 0.010
- env 77: success at step 60, end phase 2, aperture min -0.015, min |go| 0.012
- env 78: success at step 79, end phase 1, aperture min -0.008, min |go| 0.016
- env 79: success at step 53, end phase 2, aperture min -0.015, min |go| 0.014
- env 80: success at step 66, end phase 2, aperture min -0.019, min |go| 0.021
- env 81: success at step 58, end phase 2, aperture min -0.016, min |go| 0.017
- env 82: success at step 110, end phase 2, aperture min -0.014, min |go| 0.023
- env 83: success at step 59, end phase 2, aperture min -0.016, min |go| 0.018
- env 84: success at step 87, end phase 1, aperture min -0.008, min |go| 0.022
- env 86: success at step 66, end phase 2, aperture min -0.013, min |go| 0.014
- env 87: success at step 64, end phase 1, aperture min -0.015, min |go| 0.019
- env 88: success at step 59, end phase 2, aperture min -0.015, min |go| 0.020
- env 89: success at step 62, end phase 2, aperture min -0.013, min |go| 0.015
- env 90: success at step 58, end phase 2, aperture min -0.017, min |go| 0.019
- env 91: success at step 56, end phase 2, aperture min -0.013, min |go| 0.011
- env 92: success at step 80, end phase 2, aperture min -0.005, min |go| 0.013
- env 93: success at step 67, end phase 2, aperture min -0.013, min |go| 0.011
- env 95: success at step 53, end phase 2, aperture min -0.017, min |go| 0.015
- env 96: success at step 87, end phase 2, aperture min -0.010, min |go| 0.022
- env 97: success at step 68, end phase 2, aperture min -0.011, min |go| 0.020
- env 98: success at step 67, end phase 2, aperture min -0.013, min |go| 0.017
- env 99: success at step 91, end phase 1, aperture min -0.005, min |go| 0.014
- env 100: success at step 94, end phase 2, aperture min -0.006, min |go| 0.022
- env 101: success at step 68, end phase 1, aperture min -0.016, min |go| 0.015
- env 102: success at step 87, end phase 1, aperture min -0.010, min |go| 0.023
- env 103: success at step 61, end phase 2, aperture min -0.014, min |go| 0.008
- env 104: success at step 57, end phase 2, aperture min -0.012, min |go| 0.012
- env 105: success at step 62, end phase 2, aperture min -0.015, min |go| 0.017
- env 106: success at step 123, end phase 2, aperture min -0.006, min |go| 0.014
- env 107: success at step 101, end phase 1, aperture min -0.019, min |go| 0.022
- env 109: success at step 87, end phase 2, aperture min -0.014, min |go| 0.016
- env 110: success at step 62, end phase 1, aperture min -0.013, min |go| 0.013
- env 111: success at step 80, end phase 2, aperture min -0.014, min |go| 0.018
- env 112: success at step 90, end phase 2, aperture min -0.015, min |go| 0.019
- env 113: success at step 142, end phase 1, aperture min -0.009, min |go| 0.025
- env 116: success at step 120, end phase 2, aperture min -0.013, min |go| 0.018
- env 117: success at step 69, end phase 1, aperture min -0.013, min |go| 0.019
- env 118: success at step 67, end phase 2, aperture min -0.018, min |go| 0.015
- env 119: success at step 62, end phase 2, aperture min -0.012, min |go| 0.017
- env 120: success at step 59, end phase 1, aperture min -0.015, min |go| 0.016
- env 121: success at step 61, end phase 2, aperture min -0.017, min |go| 0.021
- env 123: success at step 79, end phase 2, aperture min -0.014, min |go| 0.020
- env 124: success at step 111, end phase 1, aperture min -0.015, min |go| 0.020
- env 125: success at step 59, end phase 2, aperture min -0.016, min |go| 0.020
- env 126: success at step 60, end phase 2, aperture min -0.009, min |go| 0.025
- env 127: success at step 66, end phase 2, aperture min -0.018, min |go| 0.017
