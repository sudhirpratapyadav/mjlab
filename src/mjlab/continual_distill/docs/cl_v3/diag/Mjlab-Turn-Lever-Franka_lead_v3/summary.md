# diagnose — Mjlab-Turn-Lever-Franka

2026-09-09T18:39:42 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 150 · device cuda:0  
**109/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `TurnLeverClassicalPolicy`; primary object `asset`; mechanism joint `lever_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_lever_collision'].  
Phases: unnamed — numbers are the teacher's `_phase` values.

## Histograms

- end phase, FAILING envs (19): 1: 15, 2: 4
- end phase, successful envs (109): 2: 93, 1: 16
- most-steps phase, failing envs: 1: 19
- failure class: mechanism_short: 18, near_miss: 1
- termination among failing envs: none: 19
- success step (successful envs): median 75, max 146

## Reading

- **mechanism_short** — 18/19 failing envs (envs [1, 10, 11, 15, 22, 24, 39, 62, 75, 92, 100, 102, 103, 105, 112, 115, 117, 125]); typical end phase 1; final aperture median -0.000; closest |gripper_to_object| median 0.022; closest |object_to_goal| median 0.093; min goal_error median 0.890 (success < 0.150); mechanism reached median -0.681 of target -1.571
- **near_miss** — 1/19 failing envs (envs [32]); typical end phase 1; final aperture median -0.000; closest |gripper_to_object| median 0.018; closest |object_to_goal| median 0.012; min goal_error median 0.180 (success < 0.150); mechanism reached median -1.391 of target -1.571

## Failing envs

### env 1 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 98, 2: 18
- aperture min -0.009 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.022 at step 76 (final 0.072)
- closest |object_to_goal| 0.066 at step 81 (final 0.126); goal_error min 0.652 at step 85 / final 1.074 (success < 0.150)
- mechanism joint: reached -0.918 (final -0.497) of target -1.571

### env 10 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 105, 2: 11
- aperture min -0.015 / max 0.076 / final -0.013
- closest |gripper_to_object| 0.026 at step 64 (final 0.134)
- closest |object_to_goal| 0.093 at step 107 (final 0.103); goal_error min 0.920 at step 81 / final 0.920 (success < 0.150)
- mechanism joint: reached -0.651 (final -0.651) of target -1.571

### env 11 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 30, 1: 85, 2: 35
- aperture min -0.013 / max 0.087 / final 0.009
- closest |gripper_to_object| 0.023 at step 69 (final 0.062)
- closest |object_to_goal| 0.054 at step 128 (final 0.103); goal_error min 0.547 at step 129 / final 0.911 (success < 0.150)
- mechanism joint: reached -1.024 (final -0.660) of target -1.571

### env 15 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 36, 1: 88, 2: 26
- aperture min -0.015 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.030 at step 74 (final 0.148)
- closest |object_to_goal| 0.107 at step 137 (final 0.116); goal_error min 1.004 at step 121 / final 1.004 (success < 0.150)
- mechanism joint: reached -0.567 (final -0.567) of target -1.571

### env 22 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 35, 1: 88, 2: 27
- aperture min -0.014 / max 0.076 / final -0.014
- closest |gripper_to_object| 0.030 at step 109 (final 0.128)
- closest |object_to_goal| 0.121 at step 125 (final 0.133); goal_error min 1.160 at step 120 / final 1.160 (success < 0.150)
- mechanism joint: reached -0.411 (final -0.411) of target -1.571

### env 24 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 26, 1: 83, 2: 41
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.018 at step 63 (final 0.049)
- closest |object_to_goal| 0.120 at step 145 (final 0.127); goal_error min 1.129 at step 147 / final 1.129 (success < 0.150)
- mechanism joint: reached -0.442 (final -0.442) of target -1.571

### env 32 — near_miss
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 95, 2: 23
- aperture min -0.015 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.018 at step 78 (final 0.075)
- closest |object_to_goal| 0.012 at step 99 (final 0.032); goal_error min 0.180 at step 87 / final 0.180 (success < 0.150)
- mechanism joint: reached -1.391 (final -1.391) of target -1.571

### env 39 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 25, 1: 113, 2: 12
- aperture min -0.010 / max 0.076 / final -0.005
- closest |gripper_to_object| 0.023 at step 67 (final 0.069)
- closest |object_to_goal| 0.105 at step 80 (final 0.179); goal_error min 1.003 at step 76 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.568 (final -0.000) of target -1.571

### env 62 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 35, 1: 89, 2: 26
- aperture min -0.014 / max 0.076 / final -0.013
- closest |gripper_to_object| 0.029 at step 66 (final 0.130)
- closest |object_to_goal| 0.097 at step 138 (final 0.103); goal_error min 0.950 at step 115 / final 0.950 (success < 0.150)
- mechanism joint: reached -0.621 (final -0.621) of target -1.571

### env 75 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 68, 2: 49
- aperture min -0.013 / max 0.076 / final 0.009
- closest |gripper_to_object| 0.016 at step 125 (final 0.027)
- closest |object_to_goal| 0.043 at step 149 (final 0.043); goal_error min 0.385 at step 149 / final 0.385 (success < 0.150)
- mechanism joint: reached -1.186 (final -1.186) of target -1.571

### env 92 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 93, 2: 26
- aperture min -0.015 / max 0.076 / final -0.014
- closest |gripper_to_object| 0.024 at step 65 (final 0.124)
- closest |object_to_goal| 0.105 at step 135 (final 0.113); goal_error min 0.995 at step 114 / final 0.995 (success < 0.150)
- mechanism joint: reached -0.576 (final -0.576) of target -1.571

### env 100 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 24, 1: 97, 2: 29
- aperture min -0.015 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.018 at step 57 (final 0.096)
- closest |object_to_goal| 0.028 at step 134 (final 0.037); goal_error min 0.322 at step 132 / final 0.322 (success < 0.150)
- mechanism joint: reached -1.249 (final -1.249) of target -1.571

### env 102 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 95, 2: 21
- aperture min -0.010 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.020 at step 78 (final 0.101)
- closest |object_to_goal| 0.018 at step 147 (final 0.029); goal_error min 0.231 at step 90 / final 0.231 (success < 0.150)
- mechanism joint: reached -1.340 (final -1.340) of target -1.571

### env 103 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 75, 2: 42
- aperture min -0.010 / max 0.076 / final -0.005
- closest |gripper_to_object| 0.023 at step 101 (final 0.026)
- closest |object_to_goal| 0.093 at step 148 (final 0.105); goal_error min 0.860 at step 149 / final 0.860 (success < 0.150)
- mechanism joint: reached -0.711 (final -0.711) of target -1.571

### env 105 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 81, 2: 36
- aperture min -0.008 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.019 at step 106 (final 0.023)
- closest |object_to_goal| 0.073 at step 146 (final 0.080); goal_error min 0.681 at step 149 / final 0.681 (success < 0.150)
- mechanism joint: reached -0.890 (final -0.890) of target -1.571

### env 112 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 36, 1: 96, 2: 18
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.020 at step 140 (final 0.047)
- closest |object_to_goal| 0.082 at step 122 (final 0.175); goal_error min 0.807 at step 85 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.764 (final 0.000) of target -1.571

### env 115 — mechanism_short
- ended in phase 2 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 95, 2: 22
- aperture min -0.007 / max 0.097 / final -0.000
- closest |gripper_to_object| 0.019 at step 140 (final 0.034)
- closest |object_to_goal| 0.129 at step 98 (final 0.150); goal_error min 1.245 at step 103 / final 1.307 (success < 0.150)
- mechanism joint: reached -0.326 (final -0.263) of target -1.571

### env 117 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 103, 2: 15
- aperture min -0.016 / max 0.076 / final -0.006
- closest |gripper_to_object| 0.030 at step 61 (final 0.084)
- closest |object_to_goal| 0.142 at step 99 (final 0.171); goal_error min 1.397 at step 73 / final 1.571 (success < 0.150)
- mechanism joint: reached -0.174 (final -0.000) of target -1.571

### env 125 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 96, 2: 20
- aperture min -0.007 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 66 (final 0.068)
- closest |object_to_goal| 0.056 at step 149 (final 0.056); goal_error min 0.553 at step 132 / final 0.553 (success < 0.150)
- mechanism joint: reached -1.018 (final -1.018) of target -1.571

## Successful envs (one line each)

- env 0: success at step 73, end phase 1, aperture min -0.020, min |go| 0.017
- env 2: success at step 142, end phase 2, aperture min -0.005, min |go| 0.026
- env 3: success at step 74, end phase 2, aperture min -0.012, min |go| 0.014
- env 4: success at step 74, end phase 2, aperture min -0.018, min |go| 0.026
- env 5: success at step 75, end phase 2, aperture min -0.014, min |go| 0.029
- env 6: success at step 71, end phase 2, aperture min -0.014, min |go| 0.014
- env 7: success at step 71, end phase 2, aperture min -0.015, min |go| 0.012
- env 8: success at step 75, end phase 1, aperture min -0.012, min |go| 0.022
- env 9: success at step 128, end phase 2, aperture min -0.014, min |go| 0.016
- env 12: success at step 74, end phase 2, aperture min -0.014, min |go| 0.013
- env 13: success at step 72, end phase 2, aperture min -0.013, min |go| 0.012
- env 14: success at step 69, end phase 2, aperture min -0.019, min |go| 0.017
- env 16: success at step 77, end phase 2, aperture min -0.013, min |go| 0.012
- env 17: success at step 77, end phase 2, aperture min -0.014, min |go| 0.023
- env 18: success at step 81, end phase 2, aperture min -0.014, min |go| 0.017
- env 19: success at step 76, end phase 2, aperture min -0.013, min |go| 0.011
- env 20: success at step 74, end phase 1, aperture min -0.013, min |go| 0.024
- env 21: success at step 71, end phase 2, aperture min -0.014, min |go| 0.014
- env 23: success at step 76, end phase 2, aperture min -0.016, min |go| 0.015
- env 25: success at step 77, end phase 2, aperture min -0.014, min |go| 0.013
- env 26: success at step 77, end phase 2, aperture min -0.015, min |go| 0.014
- env 27: success at step 71, end phase 2, aperture min -0.013, min |go| 0.010
- env 28: success at step 129, end phase 2, aperture min -0.010, min |go| 0.010
- env 29: success at step 83, end phase 2, aperture min -0.014, min |go| 0.025
- env 30: success at step 80, end phase 2, aperture min -0.014, min |go| 0.016
- env 31: success at step 79, end phase 2, aperture min -0.014, min |go| 0.017
- env 33: success at step 73, end phase 2, aperture min -0.014, min |go| 0.015
- env 34: success at step 82, end phase 2, aperture min -0.015, min |go| 0.013
- env 35: success at step 76, end phase 2, aperture min -0.014, min |go| 0.013
- env 36: success at step 66, end phase 2, aperture min -0.015, min |go| 0.016
- env 37: success at step 99, end phase 2, aperture min -0.018, min |go| 0.016
- env 38: success at step 72, end phase 2, aperture min -0.014, min |go| 0.023
- env 40: success at step 74, end phase 2, aperture min -0.013, min |go| 0.012
- env 41: success at step 74, end phase 1, aperture min -0.015, min |go| 0.027
- env 42: success at step 75, end phase 1, aperture min -0.006, min |go| 0.020
- env 43: success at step 68, end phase 2, aperture min -0.013, min |go| 0.011
- env 44: success at step 68, end phase 2, aperture min -0.013, min |go| 0.008
- env 45: success at step 77, end phase 2, aperture min -0.015, min |go| 0.016
- env 46: success at step 74, end phase 2, aperture min -0.014, min |go| 0.010
- env 47: success at step 73, end phase 1, aperture min -0.009, min |go| 0.024
- env 48: success at step 80, end phase 2, aperture min -0.010, min |go| 0.020
- env 49: success at step 68, end phase 2, aperture min -0.015, min |go| 0.014
- env 50: success at step 74, end phase 2, aperture min -0.014, min |go| 0.012
- env 51: success at step 79, end phase 2, aperture min -0.015, min |go| 0.020
- env 52: success at step 71, end phase 2, aperture min -0.013, min |go| 0.012
- env 53: success at step 77, end phase 2, aperture min -0.013, min |go| 0.013
- env 54: success at step 73, end phase 1, aperture min -0.014, min |go| 0.026
- env 55: success at step 73, end phase 2, aperture min -0.019, min |go| 0.030
- env 56: success at step 66, end phase 1, aperture min -0.014, min |go| 0.019
- env 57: success at step 75, end phase 2, aperture min -0.015, min |go| 0.013
- env 58: success at step 77, end phase 2, aperture min -0.015, min |go| 0.015
- env 59: success at step 66, end phase 2, aperture min -0.011, min |go| 0.019
- env 60: success at step 71, end phase 2, aperture min -0.017, min |go| 0.016
- env 61: success at step 76, end phase 2, aperture min -0.014, min |go| 0.014
- env 63: success at step 69, end phase 2, aperture min -0.013, min |go| 0.011
- env 64: success at step 69, end phase 2, aperture min -0.015, min |go| 0.012
- env 65: success at step 72, end phase 2, aperture min -0.014, min |go| 0.012
- env 66: success at step 74, end phase 2, aperture min -0.018, min |go| 0.019
- env 67: success at step 75, end phase 2, aperture min -0.014, min |go| 0.010
- env 68: success at step 68, end phase 2, aperture min -0.013, min |go| 0.015
- env 69: success at step 75, end phase 2, aperture min -0.014, min |go| 0.023
- env 70: success at step 73, end phase 2, aperture min -0.013, min |go| 0.011
- env 71: success at step 76, end phase 2, aperture min -0.016, min |go| 0.019
- env 72: success at step 77, end phase 2, aperture min -0.015, min |go| 0.014
- env 73: success at step 84, end phase 2, aperture min -0.013, min |go| 0.014
- env 74: success at step 78, end phase 2, aperture min -0.009, min |go| 0.017
- env 76: success at step 74, end phase 1, aperture min -0.014, min |go| 0.029
- env 77: success at step 74, end phase 2, aperture min -0.014, min |go| 0.027
- env 78: success at step 75, end phase 1, aperture min -0.010, min |go| 0.013
- env 79: success at step 65, end phase 2, aperture min -0.018, min |go| 0.013
- env 80: success at step 70, end phase 2, aperture min -0.015, min |go| 0.015
- env 81: success at step 77, end phase 2, aperture min -0.015, min |go| 0.014
- env 82: success at step 71, end phase 2, aperture min -0.013, min |go| 0.012
- env 83: success at step 64, end phase 2, aperture min -0.001, min |go| 0.020
- env 84: success at step 78, end phase 2, aperture min -0.015, min |go| 0.014
- env 85: success at step 78, end phase 2, aperture min -0.010, min |go| 0.020
- env 86: success at step 74, end phase 2, aperture min -0.008, min |go| 0.016
- env 87: success at step 74, end phase 2, aperture min -0.015, min |go| 0.021
- env 88: success at step 73, end phase 1, aperture min -0.014, min |go| 0.023
- env 89: success at step 76, end phase 2, aperture min -0.014, min |go| 0.024
- env 90: success at step 70, end phase 2, aperture min -0.015, min |go| 0.012
- env 91: success at step 77, end phase 2, aperture min -0.015, min |go| 0.015
- env 93: success at step 73, end phase 1, aperture min -0.005, min |go| 0.018
- env 94: success at step 74, end phase 2, aperture min -0.015, min |go| 0.017
- env 95: success at step 72, end phase 2, aperture min -0.003, min |go| 0.015
- env 96: success at step 84, end phase 2, aperture min -0.014, min |go| 0.012
- env 97: success at step 77, end phase 2, aperture min -0.014, min |go| 0.012
- env 98: success at step 75, end phase 2, aperture min -0.012, min |go| 0.015
- env 99: success at step 146, end phase 2, aperture min -0.004, min |go| 0.024
- env 101: success at step 75, end phase 2, aperture min -0.012, min |go| 0.013
- env 104: success at step 78, end phase 2, aperture min -0.018, min |go| 0.019
- env 106: success at step 75, end phase 2, aperture min -0.014, min |go| 0.016
- env 107: success at step 75, end phase 2, aperture min -0.014, min |go| 0.011
- env 108: success at step 72, end phase 2, aperture min -0.014, min |go| 0.012
- env 109: success at step 68, end phase 1, aperture min -0.015, min |go| 0.023
- env 110: success at step 78, end phase 2, aperture min -0.015, min |go| 0.013
- env 111: success at step 76, end phase 2, aperture min -0.015, min |go| 0.017
- env 113: success at step 81, end phase 2, aperture min -0.013, min |go| 0.012
- env 114: success at step 74, end phase 2, aperture min -0.015, min |go| 0.014
- env 116: success at step 97, end phase 2, aperture min -0.009, min |go| 0.019
- env 118: success at step 75, end phase 2, aperture min -0.015, min |go| 0.014
- env 119: success at step 72, end phase 2, aperture min -0.013, min |go| 0.010
- env 120: success at step 78, end phase 1, aperture min -0.014, min |go| 0.016
- env 121: success at step 75, end phase 2, aperture min -0.015, min |go| 0.017
- env 122: success at step 75, end phase 2, aperture min -0.014, min |go| 0.010
- env 123: success at step 72, end phase 2, aperture min -0.015, min |go| 0.018
- env 124: success at step 74, end phase 1, aperture min -0.015, min |go| 0.019
- env 126: success at step 74, end phase 1, aperture min -0.015, min |go| 0.031
- env 127: success at step 76, end phase 2, aperture min -0.014, min |go| 0.012
