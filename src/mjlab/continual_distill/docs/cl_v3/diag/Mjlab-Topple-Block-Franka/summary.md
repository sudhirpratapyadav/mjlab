# diagnose — Mjlab-Topple-Block-Franka

2026-09-09T13:56:57 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 200 · device cuda:0  
**112/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToppleBlockClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = SEAT, 2 = PUNCH, 3 = RETREAT

## Histograms

- end phase, FAILING envs (16): 2 (PUNCH): 13, 3 (RETREAT): 3
- end phase, successful envs (112): 2 (PUNCH): 68, 3 (RETREAT): 44
- most-steps phase, failing envs: 1 (SEAT): 11, 2 (PUNCH): 4, 3 (RETREAT): 1
- failure class: terminated:ee_ground_collision: 15, never_reached: 1
- termination among failing envs: ee_ground_collision: 15, none: 1
- success step (successful envs): median 47, max 121

## Reading

- **terminated:ee_ground_collision** — 15/16 failing envs (envs [4, 5, 10, 11, 23, 32, 44, 65, 70, 73, 74, 84, 118, 121, 123]); typical end phase 2 (PUNCH); final aperture median 0.000; closest |gripper_to_object| median 0.118; closest |object_to_goal| median 0.004; min goal_error median 0.840
- **never_reached** — 1/16 failing envs (envs [102]); typical end phase 3 (RETREAT); final aperture median -0.006; closest |gripper_to_object| median 0.101; closest |object_to_goal| median 0.002; min goal_error median 1.418

## Failing envs

### env 4 — terminated:ee_ground_collision
- ended in phase 3 (RETREAT) after 161 steps; TERMINATED by `ee_ground_collision` at step 160
- most steps in phase 2 (PUNCH); steps per phase 0 (HOVER): 27, 1 (SEAT): 51, 2 (PUNCH): 70, 3 (RETREAT): 13
- aperture min -0.011 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.108 at step 83 (final 0.193)
- closest |object_to_goal| 0.003 at step 12 (final 0.334); goal_error min 1.159 at step 30 / final 1.570
- object LIFTED at step 65 (max rise 0.027 m); LOST at step 66 (aperture collapsed while the object moved away)

### env 5 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 86 steps; TERMINATED by `ee_ground_collision` at step 85
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 5, 1 (SEAT): 51, 2 (PUNCH): 30
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.149 at step 69 (final 0.289)
- closest |object_to_goal| 0.004 at step 7 (final 0.031); goal_error min 1.366 at step 57 / final 1.571
- object LIFTED at step 31 (max rise 0.025 m); LOST at step 33 (aperture collapsed while the object moved away)

### env 10 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 130 steps; TERMINATED by `ee_ground_collision` at step 129
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 35, 1 (SEAT): 51, 2 (PUNCH): 44
- aperture min -0.010 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.123 at step 23 (final 0.213)
- closest |object_to_goal| 0.005 at step 17 (final 0.251); goal_error min 1.032 at step 54 / final 1.568
- object LIFTED at step 104 (max rise 0.027 m); LOST at step 105 (aperture collapsed while the object moved away)

### env 11 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 131 steps; TERMINATED by `ee_ground_collision` at step 130
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 33, 1 (SEAT): 51, 2 (PUNCH): 47
- aperture min -0.013 / max 0.087 / final 0.000
- closest |gripper_to_object| 0.110 at step 50 (final 0.243)
- closest |object_to_goal| 0.002 at step 12 (final 0.237); goal_error min 0.914 at step 37 / final 1.342
- object LIFTED at step 105 (max rise 0.033 m); LOST at step 110 (aperture collapsed while the object moved away)

### env 23 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 148 steps; TERMINATED by `ee_ground_collision` at step 147
- most steps in phase 2 (PUNCH); steps per phase 0 (HOVER): 35, 1 (SEAT): 51, 2 (PUNCH): 62
- aperture min -0.014 / max 0.093 / final 0.000
- closest |gripper_to_object| 0.115 at step 85 (final 0.187)
- closest |object_to_goal| 0.004 at step 37 (final 0.366); goal_error min 0.000 at step 146 / final 0.000
- object LIFTED at step 79 (max rise 0.031 m); LOST at step 80 (aperture collapsed while the object moved away)

### env 32 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 154 steps; TERMINATED by `ee_ground_collision` at step 153
- most steps in phase 2 (PUNCH); steps per phase 0 (HOVER): 37, 1 (SEAT): 51, 2 (PUNCH): 66
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.123 at step 108 (final 0.153)
- closest |object_to_goal| 0.003 at step 37 (final 0.330); goal_error min 0.020 at step 153 / final 0.020
- object LIFTED at step 53 (max rise 0.035 m); LOST at step 60 (aperture collapsed while the object moved away)

### env 44 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 134 steps; TERMINATED by `ee_ground_collision` at step 133
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 35, 1 (SEAT): 51, 2 (PUNCH): 48
- aperture min -0.013 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.114 at step 45 (final 0.220)
- closest |object_to_goal| 0.006 at step 17 (final 0.271); goal_error min 1.007 at step 55 / final 1.571
- object LIFTED at step 99 (max rise 0.027 m); LOST at step 101 (aperture collapsed while the object moved away)

### env 65 — terminated:ee_ground_collision
- ended in phase 3 (RETREAT) after 168 steps; TERMINATED by `ee_ground_collision` at step 167
- most steps in phase 2 (PUNCH); steps per phase 0 (HOVER): 35, 1 (SEAT): 51, 2 (PUNCH): 70, 3 (RETREAT): 12
- aperture min -0.013 / max 0.076 / final 0.002
- closest |gripper_to_object| 0.118 at step 84 (final 0.131)
- closest |object_to_goal| 0.003 at step 9 (final 0.244); goal_error min 0.840 at step 57 / final 1.270
- object LIFTED at step 85 (max rise 0.028 m); LOST at step 87 (aperture collapsed while the object moved away)

### env 70 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 128 steps; TERMINATED by `ee_ground_collision` at step 127
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 28, 1 (SEAT): 51, 2 (PUNCH): 49
- aperture min -0.011 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.122 at step 36 (final 0.225)
- closest |object_to_goal| 0.007 at step 1 (final 0.249); goal_error min 0.995 at step 47 / final 1.571
- object LIFTED at step 97 (max rise 0.027 m); LOST at step 100 (aperture collapsed while the object moved away)

### env 73 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 117 steps; TERMINATED by `ee_ground_collision` at step 116
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 28, 1 (SEAT): 51, 2 (PUNCH): 38
- aperture min -0.010 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.126 at step 100 (final 0.325)
- closest |object_to_goal| 0.004 at step 33 (final 0.336); goal_error min 0.000 at step 115 / final 0.000
- object LIFTED at step 43 (max rise 0.023 m); LOST at step 45 (aperture collapsed while the object moved away)

### env 74 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 118 steps; TERMINATED by `ee_ground_collision` at step 117
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 29, 1 (SEAT): 51, 2 (PUNCH): 38
- aperture min -0.009 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.129 at step 48 (final 0.239)
- closest |object_to_goal| 0.004 at step 22 (final 0.208); goal_error min 1.450 at step 54 / final 1.571
- object LIFTED at step 94 (max rise 0.027 m); LOST at step 96 (aperture collapsed while the object moved away)

### env 84 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 129 steps; TERMINATED by `ee_ground_collision` at step 128
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 41, 1 (SEAT): 51, 2 (PUNCH): 37
- aperture min -0.009 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.120 at step 95 (final 0.206)
- closest |object_to_goal| 0.005 at step 0 (final 0.297); goal_error min 0.515 at step 128 / final 0.515
- object LIFTED at step 105 (max rise 0.031 m); LOST at step 106 (aperture collapsed while the object moved away)

### env 102 — never_reached
- ended in phase 3 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 3 (RETREAT); steps per phase 0 (HOVER): 42, 1 (SEAT): 16, 2 (PUNCH): 70, 3 (RETREAT): 72
- aperture min -0.014 / max 0.076 / final -0.006
- closest |gripper_to_object| 0.101 at step 90 (final 0.129)
- closest |object_to_goal| 0.002 at step 18 (final 0.109); goal_error min 1.418 at step 40 / final 1.568
- object LIFTED at step 32 (max rise 0.032 m); LOST at step 34 (aperture collapsed while the object moved away)

### env 118 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 124 steps; TERMINATED by `ee_ground_collision` at step 123
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 30, 1 (SEAT): 51, 2 (PUNCH): 43
- aperture min -0.007 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.112 at step 59 (final 0.245)
- closest |object_to_goal| 0.004 at step 4 (final 0.171); goal_error min 0.824 at step 123 / final 0.824
- object LIFTED at step 48 (max rise 0.035 m); LOST at step 49 (aperture collapsed while the object moved away)

### env 121 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 121 steps; TERMINATED by `ee_ground_collision` at step 120
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 26, 1 (SEAT): 51, 2 (PUNCH): 44
- aperture min -0.003 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.088 at step 90 (final 0.505)
- closest |object_to_goal| 0.001 at step 37 (final 0.212); goal_error min 0.682 at step 92 / final 1.543
- non-grasp task: object moved 0.009 m toward the goal (|object_to_goal| 0.212 at the end); max rise 0.017 m

### env 123 — terminated:ee_ground_collision
- ended in phase 2 (PUNCH) after 130 steps; TERMINATED by `ee_ground_collision` at step 129
- most steps in phase 1 (SEAT); steps per phase 0 (HOVER): 34, 1 (SEAT): 51, 2 (PUNCH): 45
- aperture min -0.007 / max 0.092 / final 0.000
- closest |gripper_to_object| 0.105 at step 43 (final 0.257)
- closest |object_to_goal| 0.005 at step 11 (final 0.087); goal_error min 0.692 at step 69 / final 0.751
- object LIFTED at step 25 (max rise 0.027 m); LOST at step 26 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 99, end phase 2 (PUNCH), aperture min -0.010, min |go| 0.120
- env 1: success at step 47, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.062
- env 2: success at step 33, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.063
- env 3: success at step 96, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.117, lifted at 21
- env 6: success at step 45, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.069
- env 7: success at step 52, end phase 3 (RETREAT), aperture min -0.010, min |go| 0.072
- env 8: success at step 37, end phase 2 (PUNCH), aperture min -0.005, min |go| 0.071
- env 9: success at step 75, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.097
- env 12: success at step 28, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.090
- env 13: success at step 32, end phase 2 (PUNCH), aperture min -0.011, min |go| 0.051, lifted at 67
- env 14: success at step 45, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.063
- env 15: success at step 38, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.068
- env 16: success at step 28, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.062
- env 17: success at step 88, end phase 3 (RETREAT), aperture min -0.009, min |go| 0.089
- env 18: success at step 45, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.065
- env 19: success at step 99, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.121
- env 20: success at step 67, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.068
- env 21: success at step 113, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.130, lifted at 39
- env 22: success at step 53, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.055
- env 24: success at step 93, end phase 2 (PUNCH), aperture min -0.010, min |go| 0.122
- env 25: success at step 27, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.089
- env 26: success at step 44, end phase 2 (PUNCH), aperture min -0.014, min |go| 0.066
- env 27: success at step 45, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.070
- env 28: success at step 38, end phase 2 (PUNCH), aperture min -0.003, min |go| 0.077
- env 29: success at step 35, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.056
- env 30: success at step 43, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.058
- env 31: success at step 52, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.066
- env 33: success at step 41, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.084, lifted at 25
- env 34: success at step 28, end phase 2 (PUNCH), aperture min -0.010, min |go| 0.057, lifted at 20
- env 35: success at step 58, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.080
- env 36: success at step 60, end phase 3 (RETREAT), aperture min -0.010, min |go| 0.089, lifted at 48
- env 37: success at step 53, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.069
- env 38: success at step 47, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.061
- env 39: success at step 38, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.095, lifted at 20
- env 40: success at step 47, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.068
- env 41: success at step 48, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.077
- env 42: success at step 25, end phase 2 (PUNCH), aperture min -0.002, min |go| 0.083
- env 43: success at step 106, end phase 2 (PUNCH), aperture min -0.017, min |go| 0.110
- env 45: success at step 19, end phase 2 (PUNCH), aperture min -0.001, min |go| 0.083, lifted at 68
- env 46: success at step 46, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.068
- env 47: success at step 68, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.078
- env 48: success at step 56, end phase 2 (PUNCH), aperture min -0.011, min |go| 0.090
- env 49: success at step 21, end phase 2 (PUNCH), aperture min -0.014, min |go| 0.081
- env 50: success at step 64, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.081
- env 51: success at step 119, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.130, lifted at 48
- env 52: success at step 118, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.113, lifted at 52
- env 53: success at step 46, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.069
- env 54: success at step 46, end phase 2 (PUNCH), aperture min -0.011, min |go| 0.073
- env 55: success at step 42, end phase 2 (PUNCH), aperture min -0.005, min |go| 0.066
- env 56: success at step 35, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.068, lifted at 20
- env 57: success at step 97, end phase 2 (PUNCH), aperture min -0.001, min |go| 0.090, lifted at 24
- env 58: success at step 45, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.059
- env 59: success at step 53, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.063
- env 60: success at step 64, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.065, lifted at 48
- env 61: success at step 111, end phase 3 (RETREAT), aperture min -0.006, min |go| 0.116, lifted at 55
- env 62: success at step 38, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.074
- env 63: success at step 62, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.089
- env 64: success at step 121, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.114, lifted at 65
- env 66: success at step 38, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.066
- env 67: success at step 27, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.065
- env 68: success at step 105, end phase 2 (PUNCH), aperture min -0.011, min |go| 0.102
- env 69: success at step 43, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.066
- env 71: success at step 46, end phase 2 (PUNCH), aperture min -0.001, min |go| 0.078
- env 72: success at step 106, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.113
- env 75: success at step 60, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.111, lifted at 34
- env 76: success at step 48, end phase 3 (RETREAT), aperture min -0.014, min |go| 0.071
- env 77: success at step 39, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.055
- env 78: success at step 106, end phase 2 (PUNCH), aperture min -0.010, min |go| 0.126, lifted at 23
- env 79: success at step 47, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.067
- env 80: success at step 96, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.125
- env 81: success at step 41, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.065, lifted at 102
- env 82: success at step 94, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.131, lifted at 21
- env 83: success at step 52, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.064
- env 85: success at step 42, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.063
- env 86: success at step 50, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.060
- env 87: success at step 30, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.056
- env 88: success at step 107, end phase 2 (PUNCH), aperture min -0.009, min |go| 0.119
- env 89: success at step 29, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.085
- env 90: success at step 47, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.060
- env 91: success at step 95, end phase 3 (RETREAT), aperture min -0.008, min |go| 0.126, lifted at 54
- env 92: success at step 48, end phase 3 (RETREAT), aperture min -0.012, min |go| 0.063
- env 93: success at step 96, end phase 2 (PUNCH), aperture min -0.012, min |go| 0.122, lifted at 51
- env 94: success at step 30, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.056
- env 95: success at step 47, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.119, lifted at 31
- env 96: success at step 32, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.065, lifted at 22
- env 97: success at step 36, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.054
- env 98: success at step 103, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.117
- env 99: success at step 58, end phase 3 (RETREAT), aperture min -0.014, min |go| 0.087, lifted at 27
- env 100: success at step 38, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.062
- env 101: success at step 19, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.063
- env 103: success at step 98, end phase 2 (PUNCH), aperture min -0.015, min |go| 0.115
- env 104: success at step 105, end phase 2 (PUNCH), aperture min -0.007, min |go| 0.128, lifted at 53
- env 105: success at step 93, end phase 2 (PUNCH), aperture min -0.004, min |go| 0.079
- env 106: success at step 36, end phase 2 (PUNCH), aperture min -0.011, min |go| 0.070, lifted at 91
- env 107: success at step 39, end phase 3 (RETREAT), aperture min -0.009, min |go| 0.064
- env 108: success at step 55, end phase 3 (RETREAT), aperture min -0.003, min |go| 0.107
- env 109: success at step 23, end phase 2 (PUNCH), aperture min -0.010, min |go| 0.073
- env 110: success at step 49, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.068
- env 111: success at step 47, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.072
- env 112: success at step 49, end phase 3 (RETREAT), aperture min -0.013, min |go| 0.060
- env 113: success at step 46, end phase 3 (RETREAT), aperture min -0.010, min |go| 0.056, lifted at 25
- env 114: success at step 53, end phase 2 (PUNCH), aperture min -0.012, min |go| 0.075
- env 115: success at step 35, end phase 2 (PUNCH), aperture min -0.006, min |go| 0.059, lifted at 63
- env 116: success at step 55, end phase 2 (PUNCH), aperture min -0.000, min |go| 0.084
- env 117: success at step 115, end phase 3 (RETREAT), aperture min -0.008, min |go| 0.116, lifted at 49
- env 119: success at step 48, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.064
- env 120: success at step 34, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.057
- env 122: success at step 30, end phase 2 (PUNCH), aperture min -0.005, min |go| 0.076
- env 124: success at step 58, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.059
- env 125: success at step 41, end phase 3 (RETREAT), aperture min -0.009, min |go| 0.058
- env 126: success at step 54, end phase 3 (RETREAT), aperture min -0.011, min |go| 0.054
- env 127: success at step 37, end phase 2 (PUNCH), aperture min -0.008, min |go| 0.065, lifted at 24
