# diagnose — Mjlab-Edge-Grasp-Franka

2026-09-09T15:59:44 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `EdgeGraspClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = PUSH_HOVER, 1 = PUSH_DESCEND, 2 = PUSH_ADVANCE, 3 = RETREAT, 4 = SIDE_HOVER, 5 = SIDE_INSERT, 6 = CLOSE, 7 = LIFT

## Histograms

- end phase, FAILING envs (32): 5 (SIDE_INSERT): 21, 7 (LIFT): 10, 6 (CLOSE): 1
- end phase, successful envs (0): —
- most-steps phase, failing envs: 5 (SIDE_INSERT): 16, 2 (PUSH_ADVANCE): 15, 3 (RETREAT): 1
- failure class: terminated:ee_ground_collision: 20, never_reached: 12
- termination among failing envs: ee_ground_collision: 20, none: 12
- failing envs that lifted the object: 0/32; lifted then lost: 0

## Reading

- **terminated:ee_ground_collision** — 20/32 failing envs (envs [2, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 21, 22, 24, 25, 27, 28, 30]); typical end phase 5 (SIDE_INSERT); final aperture median 0.080; closest |gripper_to_object| median 0.083; closest |object_to_goal| median 0.139; min goal_error median 0.149
- **never_reached** — 12/32 failing envs (envs [0, 1, 3, 4, 5, 14, 15, 20, 23, 26, 29, 31]); typical end phase 7 (LIFT); final aperture median 0.000; closest |gripper_to_object| median 0.084; closest |object_to_goal| median 0.070; min goal_error median 0.075

## Failing envs

### env 0 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 17, 3 (RETREAT): 24, 4 (SIDE_HOVER): 6, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 65
- aperture min -0.001 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.087 at step 28 (final 0.414)
- closest |object_to_goal| 0.066 at step 174 (final 0.246); goal_error min 0.072 at step 171 / final 0.246
- object never lifted (max rise 0.000 m)

### env 1 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 20, 3 (RETREAT): 24, 4 (SIDE_HOVER): 8, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 67
- aperture min -0.008 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.081 at step 34 (final 0.387)
- closest |object_to_goal| 0.068 at step 157 (final 0.265); goal_error min 0.071 at step 157 / final 0.260
- object never lifted (max rise 0.001 m)

### env 2 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 116 steps; TERMINATED by `ee_ground_collision` at step 115
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 12, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 24, 4 (SIDE_HOVER): 10, 5 (SIDE_INSERT): 26
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.087 at step 46 (final 0.301)
- closest |object_to_goal| 0.075 at step 57 (final 0.083); goal_error min 0.087 at step 58 / final 0.088
- object never lifted (max rise 0.001 m)

### env 3 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 24, 4 (SIDE_HOVER): 7, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 64
- aperture min -0.008 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.083 at step 32 (final 0.381)
- closest |object_to_goal| 0.084 at step 177 (final 0.153); goal_error min 0.088 at step 178 / final 0.153
- object never lifted (max rise 0.000 m)

### env 4 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 11, 3 (RETREAT): 24, 4 (SIDE_HOVER): 42, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 41
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.085 at step 189 (final 0.261)
- closest |object_to_goal| 0.070 at step 152 (final 0.186); goal_error min 0.081 at step 152 / final 0.182
- object never lifted (max rise 0.002 m)

### env 5 — never_reached
- ended in phase 5 (SIDE_INSERT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 55, 3 (RETREAT): 24, 4 (SIDE_HOVER): 50, 5 (SIDE_INSERT): 121, 6 (CLOSE): 16
- aperture min -0.004 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.075 at step 234 (final 0.226)
- closest |object_to_goal| 0.067 at step 198 (final 0.136); goal_error min 0.073 at step 197 / final 0.145
- object never lifted (max rise 0.000 m)

### env 6 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 162 steps; TERMINATED by `ee_ground_collision` at step 161
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 16
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.082 at step 22 (final 0.254)
- closest |object_to_goal| 0.135 at step 63 (final 0.155); goal_error min 0.145 at step 1 / final 0.148
- object never lifted (max rise 0.000 m)

### env 7 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 156 steps; TERMINATED by `ee_ground_collision` at step 155
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 8, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 15
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.085 at step 112 (final 0.240)
- closest |object_to_goal| 0.151 at step 109 (final 0.166); goal_error min 0.163 at step 153 / final 0.163
- object never lifted (max rise 0.000 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 157 steps; TERMINATED by `ee_ground_collision` at step 156
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 16
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.086 at step 42 (final 0.252)
- closest |object_to_goal| 0.147 at step 42 (final 0.159); goal_error min 0.159 at step 156 / final 0.159
- object never lifted (max rise 0.000 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 157 steps; TERMINATED by `ee_ground_collision` at step 156
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 15
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.080 at step 116 (final 0.245)
- closest |object_to_goal| 0.150 at step 5 (final 0.171); goal_error min 0.160 at step 28 / final 0.169
- object never lifted (max rise 0.000 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 153 steps; TERMINATED by `ee_ground_collision` at step 152
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 14
- aperture min -0.014 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.081 at step 97 (final 0.243)
- closest |object_to_goal| 0.154 at step 32 (final 0.173); goal_error min 0.165 at step 29 / final 0.170
- object never lifted (max rise 0.000 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 109 steps; TERMINATED by `ee_ground_collision` at step 108
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 24, 4 (SIDE_HOVER): 14, 5 (SIDE_INSERT): 20
- aperture min -0.007 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.076 at step 20 (final 0.411)
- closest |object_to_goal| 0.072 at step 64 (final 0.182); goal_error min 0.081 at step 84 / final 0.174
- object never lifted (max rise 0.000 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 160 steps; TERMINATED by `ee_ground_collision` at step 159
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 20, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 15
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.083 at step 111 (final 0.245)
- closest |object_to_goal| 0.150 at step 86 (final 0.169); goal_error min 0.162 at step 48 / final 0.164
- object never lifted (max rise 0.000 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 155 steps; TERMINATED by `ee_ground_collision` at step 154
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 15
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.063 at step 88 (final 0.132)
- closest |object_to_goal| 0.159 at step 13 (final 0.282); goal_error min 0.162 at step 27 / final 0.282
- object never lifted (max rise 0.012 m)

### env 14 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 25, 3 (RETREAT): 24, 4 (SIDE_HOVER): 48, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 20
- aperture min -0.006 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.076 at step 200 (final 0.192)
- closest |object_to_goal| 0.069 at step 160 (final 0.146); goal_error min 0.074 at step 164 / final 0.153
- object never lifted (max rise 0.000 m)

### env 15 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 13, 2 (PUSH_ADVANCE): 31, 3 (RETREAT): 24, 4 (SIDE_HOVER): 43, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 21
- aperture min -0.006 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.082 at step 35 (final 0.171)
- closest |object_to_goal| 0.072 at step 171 (final 0.252); goal_error min 0.075 at step 173 / final 0.255
- object never lifted (max rise 0.000 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 108 steps; TERMINATED by `ee_ground_collision` at step 107
- most steps in phase 3 (RETREAT); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 21, 3 (RETREAT): 24, 4 (SIDE_HOVER): 13, 5 (SIDE_INSERT): 16
- aperture min -0.002 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.088 at step 45 (final 0.287)
- closest |object_to_goal| 0.102 at step 76 (final 0.125); goal_error min 0.112 at step 63 / final 0.113
- object never lifted (max rise 0.000 m)

### env 17 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 163 steps; TERMINATED by `ee_ground_collision` at step 162
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 17, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 20
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.084 at step 81 (final 0.250)
- closest |object_to_goal| 0.150 at step 109 (final 0.151); goal_error min 0.161 at step 160 / final 0.161
- object never lifted (max rise 0.000 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 104 steps; TERMINATED by `ee_ground_collision` at step 103
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 13, 2 (PUSH_ADVANCE): 20, 3 (RETREAT): 24, 4 (SIDE_HOVER): 6, 5 (SIDE_INSERT): 25
- aperture min -0.003 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.083 at step 22 (final 0.298)
- closest |object_to_goal| 0.074 at step 102 (final 0.090); goal_error min 0.085 at step 52 / final 0.085
- object never lifted (max rise 0.003 m)

### env 19 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 112 steps; TERMINATED by `ee_ground_collision` at step 111
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 14, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 41, 3 (RETREAT): 24, 4 (SIDE_HOVER): 21, 5 (SIDE_INSERT): 3
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.088 at step 55 (final 0.425)
- closest |object_to_goal| 0.082 at step 69 (final 0.193); goal_error min 0.088 at step 76 / final 0.199
- object never lifted (max rise 0.000 m)

### env 20 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 27, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 19, 3 (RETREAT): 24, 4 (SIDE_HOVER): 8, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 57
- aperture min -0.002 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.091 at step 42 (final 0.318)
- closest |object_to_goal| 0.074 at step 167 (final 0.175); goal_error min 0.080 at step 173 / final 0.180
- object never lifted (max rise 0.000 m)

### env 21 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 109 steps; TERMINATED by `ee_ground_collision` at step 108
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 19, 1 (PUSH_DESCEND): 12, 2 (PUSH_ADVANCE): 21, 3 (RETREAT): 24, 4 (SIDE_HOVER): 8, 5 (SIDE_INSERT): 25
- aperture min -0.008 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.082 at step 38 (final 0.323)
- closest |object_to_goal| 0.070 at step 101 (final 0.086); goal_error min 0.081 at step 54 / final 0.081
- object never lifted (max rise 0.000 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 158 steps; TERMINATED by `ee_ground_collision` at step 157
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 15
- aperture min -0.014 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.072 at step 79 (final 0.239)
- closest |object_to_goal| 0.151 at step 30 (final 0.182); goal_error min 0.161 at step 30 / final 0.174
- object never lifted (max rise 0.000 m)

### env 23 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 49, 3 (RETREAT): 24, 4 (SIDE_HOVER): 26, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 19
- aperture min -0.014 / max 0.083 / final 0.000
- closest |gripper_to_object| 0.091 at step 38 (final 0.133)
- closest |object_to_goal| 0.065 at step 183 (final 0.256); goal_error min 0.072 at step 186 / final 0.258
- object never lifted (max rise 0.003 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 170 steps; TERMINATED by `ee_ground_collision` at step 169
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 25, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 18
- aperture min -0.014 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.092 at step 125 (final 0.270)
- closest |object_to_goal| 0.144 at step 26 (final 0.161); goal_error min 0.153 at step 1 / final 0.156
- object never lifted (max rise 0.000 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 107 steps; TERMINATED by `ee_ground_collision` at step 106
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 21, 1 (PUSH_DESCEND): 10, 2 (PUSH_ADVANCE): 18, 3 (RETREAT): 24, 4 (SIDE_HOVER): 8, 5 (SIDE_INSERT): 26
- aperture min -0.003 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.081 at step 24 (final 0.297)
- closest |object_to_goal| 0.080 at step 104 (final 0.089); goal_error min 0.092 at step 57 / final 0.093
- object never lifted (max rise 0.001 m)

### env 26 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 8, 2 (PUSH_ADVANCE): 11, 3 (RETREAT): 24, 4 (SIDE_HOVER): 5, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 80
- aperture min -0.002 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.087 at step 31 (final 0.370)
- closest |object_to_goal| 0.072 at step 145 (final 0.246); goal_error min 0.077 at step 147 / final 0.242
- object never lifted (max rise 0.000 m)

### env 27 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 117 steps; TERMINATED by `ee_ground_collision` at step 116
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 23, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 32, 3 (RETREAT): 24, 4 (SIDE_HOVER): 10, 5 (SIDE_INSERT): 19
- aperture min -0.009 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.085 at step 41 (final 0.302)
- closest |object_to_goal| 0.088 at step 103 (final 0.109); goal_error min 0.099 at step 71 / final 0.100
- object never lifted (max rise 0.001 m)

### env 28 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 158 steps; TERMINATED by `ee_ground_collision` at step 157
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 15, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 91, 3 (RETREAT): 24, 4 (SIDE_HOVER): 1, 5 (SIDE_INSERT): 16
- aperture min -0.015 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.086 at step 18 (final 0.263)
- closest |object_to_goal| 0.143 at step 13 (final 0.150); goal_error min 0.154 at step 1 / final 0.156
- object never lifted (max rise 0.000 m)

### env 29 — never_reached
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 11, 2 (PUSH_ADVANCE): 54, 3 (RETREAT): 24, 4 (SIDE_HOVER): 44, 5 (SIDE_INSERT): 122, 6 (CLOSE): 27
- aperture min -0.015 / max 0.080 / final 0.008
- closest |gripper_to_object| 0.076 at step 232 (final 0.225)
- closest |object_to_goal| 0.072 at step 206 (final 0.157); goal_error min 0.079 at step 202 / final 0.152
- object never lifted (max rise 0.001 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 5 (SIDE_INSERT) after 103 steps; TERMINATED by `ee_ground_collision` at step 102
- most steps in phase 2 (PUSH_ADVANCE); steps per phase 0 (PUSH_HOVER): 18, 1 (PUSH_DESCEND): 8, 2 (PUSH_ADVANCE): 26, 3 (RETREAT): 24, 4 (SIDE_HOVER): 8, 5 (SIDE_INSERT): 19
- aperture min -0.011 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.079 at step 32 (final 0.291)
- closest |object_to_goal| 0.089 at step 96 (final 0.102); goal_error min 0.099 at step 59 / final 0.100
- object never lifted (max rise 0.000 m)

### env 31 — never_reached
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (SIDE_INSERT); steps per phase 0 (PUSH_HOVER): 16, 1 (PUSH_DESCEND): 9, 2 (PUSH_ADVANCE): 53, 3 (RETREAT): 24, 4 (SIDE_HOVER): 7, 5 (SIDE_INSERT): 122, 6 (CLOSE): 32, 7 (LIFT): 37
- aperture min -0.015 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.085 at step 48 (final 0.221)
- closest |object_to_goal| 0.066 at step 192 (final 0.187); goal_error min 0.074 at step 192 / final 0.180
- object never lifted (max rise 0.000 m)

## Successful envs (one line each)

