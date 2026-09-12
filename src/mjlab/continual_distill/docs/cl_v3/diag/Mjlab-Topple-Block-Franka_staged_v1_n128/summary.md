# diagnose — Mjlab-Topple-Block-Franka

2026-09-09T18:43:10 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 200 · device cuda:0  
**114/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToppleBlockClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = ADVANCE, 3 = PUNCH, 4 = RETREAT

## Histograms

- end phase, FAILING envs (14): 4 (RETREAT): 14
- end phase, successful envs (114): 4 (RETREAT): 114
- most-steps phase, failing envs: 4 (RETREAT): 10, 3 (PUNCH): 4
- failure class: never_reached: 12, terminated:ee_ground_collision: 2
- termination among failing envs: none: 12, ee_ground_collision: 2
- success step (successful envs): median 55, max 183

## Reading

- **never_reached** — 12/14 failing envs (envs [11, 25, 36, 42, 59, 90, 93, 99, 106, 110, 120, 125]); typical end phase 4 (RETREAT); final aperture median 0.000; closest |gripper_to_object| median 0.096; closest |object_to_goal| median 0.003; min goal_error median 0.825
- **terminated:ee_ground_collision** — 2/14 failing envs (envs [13, 53]); typical end phase 4 (RETREAT); final aperture median 0.000; closest |gripper_to_object| median 0.116; closest |object_to_goal| median 0.003; min goal_error median 1.175

## Failing envs

### env 11 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 28, 1 (DESCEND): 28, 2 (ADVANCE): 41, 3 (PUNCH): 35, 4 (RETREAT): 68
- aperture min -0.012 / max 0.080 / final 0.004
- closest |gripper_to_object| 0.117 at step 142 (final 0.132)
- closest |object_to_goal| 0.001 at step 21 (final 0.294); goal_error min 0.607 at step 136 / final 1.082
- object LIFTED at step 78 (max rise 0.034 m); LOST at step 79 (aperture collapsed while the object moved away)

### env 13 — terminated:ee_ground_collision
- ended in phase 4 (RETREAT) after 176 steps; TERMINATED by `ee_ground_collision` at step 175
- most steps in phase 3 (PUNCH); steps per phase 0 (HOVER): 27, 1 (DESCEND): 28, 2 (ADVANCE): 41, 3 (PUNCH): 70, 4 (RETREAT): 10
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.113 at step 40 (final 0.679)
- closest |object_to_goal| 0.004 at step 1 (final 0.282); goal_error min 1.233 at step 79 / final 1.571
- object LIFTED at step 75 (max rise 0.034 m); LOST at step 76 (aperture collapsed while the object moved away)

### env 25 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 38, 1 (DESCEND): 25, 2 (ADVANCE): 12, 3 (PUNCH): 5, 4 (RETREAT): 120
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.095 at step 78 (final 0.218)
- closest |object_to_goal| 0.002 at step 0 (final 0.049); goal_error min 0.812 at step 81 / final 1.571
- non-grasp task: object moved 0.000 m toward the goal (|object_to_goal| 0.049 at the end); max rise 0.018 m

### env 36 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 29, 1 (DESCEND): 28, 2 (ADVANCE): 29, 4 (RETREAT): 114
- aperture min -0.005 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.120 at step 73 (final 0.207)
- closest |object_to_goal| 0.003 at step 18 (final 0.179); goal_error min 0.835 at step 87 / final 1.571
- object LIFTED at step 77 (max rise 0.031 m); LOST at step 78 (aperture collapsed while the object moved away)

### env 42 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 25, 1 (DESCEND): 27, 2 (ADVANCE): 41, 3 (PUNCH): 33, 4 (RETREAT): 74
- aperture min -0.007 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.096 at step 99 (final 0.246)
- closest |object_to_goal| 0.003 at step 51 (final 0.335); goal_error min 0.695 at step 134 / final 1.563
- object LIFTED at step 72 (max rise 0.036 m); LOST at step 75 (aperture collapsed while the object moved away)

### env 53 — terminated:ee_ground_collision
- ended in phase 4 (RETREAT) after 179 steps; TERMINATED by `ee_ground_collision` at step 178
- most steps in phase 3 (PUNCH); steps per phase 0 (HOVER): 31, 1 (DESCEND): 27, 2 (ADVANCE): 41, 3 (PUNCH): 70, 4 (RETREAT): 10
- aperture min -0.007 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.119 at step 46 (final 0.417)
- closest |object_to_goal| 0.002 at step 45 (final 0.271); goal_error min 1.118 at step 85 / final 1.571
- object LIFTED at step 77 (max rise 0.034 m); LOST at step 79 (aperture collapsed while the object moved away)

### env 59 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 32, 1 (DESCEND): 28, 2 (ADVANCE): 41, 3 (PUNCH): 21, 4 (RETREAT): 78
- aperture min -0.009 / max 0.087 / final 0.000
- closest |gripper_to_object| 0.114 at step 107 (final 0.375)
- closest |object_to_goal| 0.001 at step 20 (final 0.129); goal_error min 0.822 at step 123 / final 1.571
- object LIFTED at step 82 (max rise 0.033 m); LOST at step 87 (aperture collapsed while the object moved away)

### env 90 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 31, 1 (DESCEND): 31, 2 (ADVANCE): 15, 3 (PUNCH): 9, 4 (RETREAT): 114
- aperture min -0.008 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.086 at step 85 (final 0.140)
- closest |object_to_goal| 0.003 at step 8 (final 0.092); goal_error min 0.840 at step 87 / final 1.570
- object LIFTED at step 166 (max rise 0.026 m); LOST at step 170 (aperture collapsed while the object moved away)

### env 93 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 3 (PUNCH); steps per phase 0 (HOVER): 40, 1 (DESCEND): 41, 2 (ADVANCE): 11, 3 (PUNCH): 70, 4 (RETREAT): 38
- aperture min -0.006 / max 0.083 / final -0.000
- closest |gripper_to_object| 0.085 at step 87 (final 0.142)
- closest |object_to_goal| 0.001 at step 6 (final 0.191); goal_error min 1.201 at step 98 / final 1.568
- object LIFTED at step 65 (max rise 0.030 m); LOST at step 102 (aperture collapsed while the object moved away)

### env 99 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 3 (PUNCH); steps per phase 0 (HOVER): 35, 1 (DESCEND): 41, 2 (ADVANCE): 20, 3 (PUNCH): 70, 4 (RETREAT): 34
- aperture min -0.011 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.084 at step 130 (final 0.137)
- closest |object_to_goal| 0.003 at step 7 (final 0.160); goal_error min 1.173 at step 69 / final 1.547
- object LIFTED at step 47 (max rise 0.032 m); LOST at step 49 (aperture collapsed while the object moved away)

### env 106 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 11, 1 (DESCEND): 41, 2 (ADVANCE): 41, 3 (PUNCH): 31, 4 (RETREAT): 76
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.113 at step 99 (final 0.170)
- closest |object_to_goal| 0.001 at step 160 (final 0.047); goal_error min 0.858 at step 125 / final 1.236
- non-grasp task: object moved 0.010 m toward the goal (|object_to_goal| 0.047 at the end); max rise 0.011 m

### env 110 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 8, 1 (DESCEND): 41, 2 (ADVANCE): 41, 3 (PUNCH): 38, 4 (RETREAT): 72
- aperture min -0.002 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.090 at step 149 (final 0.110)
- closest |object_to_goal| 0.005 at step 16 (final 0.168); goal_error min 0.829 at step 129 / final 1.343
- object LIFTED at step 42 (max rise 0.033 m); LOST at step 43 (aperture collapsed while the object moved away)

### env 120 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 11, 1 (DESCEND): 41, 2 (ADVANCE): 41, 3 (PUNCH): 23, 4 (RETREAT): 84
- aperture min -0.004 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.088 at step 63 (final 0.171)
- closest |object_to_goal| 0.003 at step 14 (final 0.164); goal_error min 0.815 at step 199 / final 0.815
- object LIFTED at step 34 (max rise 0.025 m); LOST at step 35 (aperture collapsed while the object moved away)

### env 125 — never_reached
- ended in phase 4 (RETREAT) after 200 steps; ran to time-out
- most steps in phase 4 (RETREAT); steps per phase 0 (HOVER): 11, 1 (DESCEND): 41, 2 (ADVANCE): 41, 3 (PUNCH): 34, 4 (RETREAT): 73
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.134 at step 99 (final 0.156)
- closest |object_to_goal| 0.005 at step 92 (final 0.093); goal_error min 0.589 at step 198 / final 0.589
- object LIFTED at step 52 (max rise 0.024 m); LOST at step 56 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 77, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.098
- env 1: success at step 33, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.114
- env 2: success at step 59, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.113
- env 3: success at step 51, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.133
- env 4: success at step 76, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.090
- env 5: success at step 28, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.081
- env 6: success at step 75, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.098
- env 7: success at step 46, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.105
- env 8: success at step 31, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.102
- env 9: success at step 65, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.125, lifted at 51
- env 10: success at step 98, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.089, lifted at 67
- env 12: success at step 85, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.124, lifted at 72
- env 14: success at step 27, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.117
- env 15: success at step 82, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.090
- env 16: success at step 60, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.115
- env 17: success at step 52, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.097
- env 18: success at step 79, end phase 4 (RETREAT), aperture min -0.001, min |go| 0.087
- env 19: success at step 103, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.089
- env 20: success at step 53, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.112
- env 21: success at step 54, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.106
- env 22: success at step 57, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.111
- env 23: success at step 79, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.091
- env 24: success at step 31, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.109
- env 26: success at step 29, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.125
- env 27: success at step 54, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.135
- env 28: success at step 57, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.109
- env 29: success at step 129, end phase 4 (RETREAT), aperture min -0.009, min |go| 0.112, lifted at 58
- env 30: success at step 70, end phase 4 (RETREAT), aperture min -0.010, min |go| 0.079, lifted at 24
- env 31: success at step 30, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.112
- env 32: success at step 63, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.122
- env 33: success at step 23, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.107
- env 34: success at step 65, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.125
- env 35: success at step 55, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.113
- env 37: success at step 62, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.120
- env 38: success at step 88, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.091, lifted at 65
- env 39: success at step 124, end phase 4 (RETREAT), aperture min -0.011, min |go| 0.127, lifted at 33
- env 40: success at step 21, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.133
- env 41: success at step 25, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.135
- env 43: success at step 52, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.119
- env 44: success at step 47, end phase 4 (RETREAT), aperture min -0.011, min |go| 0.143, lifted at 27
- env 45: success at step 46, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.099
- env 46: success at step 32, end phase 4 (RETREAT), aperture min -0.003, min |go| 0.114
- env 47: success at step 78, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.101
- env 48: success at step 85, end phase 4 (RETREAT), aperture min -0.001, min |go| 0.091
- env 49: success at step 68, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.127, lifted at 56
- env 50: success at step 47, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.125
- env 51: success at step 28, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.115
- env 52: success at step 34, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.106, lifted at 28
- env 54: success at step 118, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.118, lifted at 82
- env 55: success at step 39, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.114
- env 56: success at step 23, end phase 4 (RETREAT), aperture min -0.003, min |go| 0.100
- env 57: success at step 24, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.127
- env 58: success at step 87, end phase 4 (RETREAT), aperture min -0.009, min |go| 0.064, lifted at 44
- env 60: success at step 103, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.090, lifted at 76
- env 61: success at step 74, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.084
- env 62: success at step 30, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.114
- env 63: success at step 55, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.117
- env 64: success at step 37, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.105
- env 65: success at step 92, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.080, lifted at 57
- env 66: success at step 33, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.105
- env 67: success at step 73, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.100
- env 68: success at step 38, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.136
- env 69: success at step 58, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.099
- env 70: success at step 84, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.097
- env 71: success at step 29, end phase 4 (RETREAT), aperture min -0.012, min |go| 0.137
- env 72: success at step 71, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.088
- env 73: success at step 97, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.118
- env 74: success at step 65, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.128
- env 75: success at step 183, end phase 4 (RETREAT), aperture min -0.010, min |go| 0.110, lifted at 50
- env 76: success at step 35, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.108
- env 77: success at step 79, end phase 4 (RETREAT), aperture min -0.001, min |go| 0.100
- env 78: success at step 31, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.131
- env 79: success at step 63, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.126, lifted at 50
- env 80: success at step 27, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.137
- env 81: success at step 75, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.090
- env 82: success at step 32, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.086
- env 83: success at step 34, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.083
- env 84: success at step 34, end phase 4 (RETREAT), aperture min -0.010, min |go| 0.110
- env 85: success at step 66, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.116
- env 86: success at step 32, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.108
- env 87: success at step 63, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.101, lifted at 23
- env 88: success at step 39, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.093
- env 89: success at step 83, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.124, lifted at 35
- env 91: success at step 54, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.117
- env 92: success at step 89, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.117
- env 94: success at step 37, end phase 4 (RETREAT), aperture min -0.013, min |go| 0.115, lifted at 26
- env 95: success at step 58, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.118
- env 96: success at step 51, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.114
- env 97: success at step 104, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.082, lifted at 32
- env 98: success at step 32, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.124
- env 100: success at step 79, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.072, lifted at 64
- env 101: success at step 46, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.101
- env 102: success at step 34, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.109
- env 103: success at step 71, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.134
- env 104: success at step 46, end phase 4 (RETREAT), aperture min -0.005, min |go| 0.114
- env 105: success at step 32, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.106
- env 107: success at step 81, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.094
- env 108: success at step 59, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.115
- env 109: success at step 75, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.129, lifted at 51
- env 111: success at step 78, end phase 4 (RETREAT), aperture min -0.001, min |go| 0.097
- env 112: success at step 56, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.115
- env 113: success at step 55, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.102
- env 114: success at step 79, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.086
- env 115: success at step 60, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.086
- env 116: success at step 42, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.135
- env 117: success at step 70, end phase 4 (RETREAT), aperture min -0.002, min |go| 0.091
- env 118: success at step 48, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.110
- env 119: success at step 48, end phase 4 (RETREAT), aperture min -0.007, min |go| 0.067
- env 121: success at step 37, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.139, lifted at 28
- env 122: success at step 44, end phase 4 (RETREAT), aperture min -0.000, min |go| 0.091, lifted at 32
- env 123: success at step 31, end phase 4 (RETREAT), aperture min -0.008, min |go| 0.103
- env 124: success at step 173, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.109
- env 126: success at step 78, end phase 4 (RETREAT), aperture min -0.006, min |go| 0.116
- env 127: success at step 38, end phase 4 (RETREAT), aperture min -0.004, min |go| 0.083
