# diagnose — Mjlab-Push-Cuboid-Franka

2026-09-09T14:11:53 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 150 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `PushCuboidClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = APPROACH, 1 = DESCEND, 2 = ENTER, 3 = PUSH, 4 = HOLD, 5 = RESEAT_UP, 6 = RESEAT_SIDE

## Histograms

- end phase, FAILING envs (32): 3 (PUSH): 24, 1 (DESCEND): 8
- end phase, successful envs (0): —
- most-steps phase, failing envs: 0 (APPROACH): 30, 3 (PUSH): 2
- failure class: terminated:ee_ground_collision: 32
- termination among failing envs: ee_ground_collision: 32

## Reading

- **terminated:ee_ground_collision** — 32/32 failing envs (envs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]); typical end phase 3 (PUSH); final aperture median -0.000; closest |gripper_to_object| median 0.050; closest |object_to_goal| median 0.148; min goal_error median 0.155 (success < 0.020)

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 35, 1 (DESCEND): 25, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.050 at step 61 (final 0.057)
- closest |object_to_goal| 0.150 at step 21 (final 0.153); goal_error min 0.159 at step 62 / final 0.159 (success < 0.020)
- non-grasp task: object moved 0.013 m toward the goal (|object_to_goal| 0.153 at the end); max rise 0.000 m

### env 1 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 48 steps; TERMINATED by `ee_ground_collision` at step 47
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 34, 1 (DESCEND): 14
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.056 at step 44 (final 0.060)
- closest |object_to_goal| 0.099 at step 37 (final 0.117); goal_error min 0.111 at step 1 / final 0.111 (success < 0.020)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.117 at the end); max rise 0.000 m

### env 2 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 66 steps; TERMINATED by `ee_ground_collision` at step 65
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 44, 1 (DESCEND): 22
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.056 at step 63 (final 0.067)
- closest |object_to_goal| 0.104 at step 4 (final 0.123); goal_error min 0.117 at step 6 / final 0.117 (success < 0.020)
- non-grasp task: object moved 0.022 m toward the goal (|object_to_goal| 0.123 at the end); max rise 0.000 m

### env 3 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 74 steps; TERMINATED by `ee_ground_collision` at step 73
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 45, 1 (DESCEND): 25, 3 (PUSH): 4
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.055 at step 64 (final 0.066)
- closest |object_to_goal| 0.185 at step 7 (final 0.211); goal_error min 0.198 at step 4 / final 0.198 (success < 0.020)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.211 at the end); max rise 0.000 m

### env 4 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 30, 1 (DESCEND): 18, 3 (PUSH): 9
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.045 at step 52 (final 0.051)
- closest |object_to_goal| 0.200 at step 56 (final 0.200); goal_error min 0.198 at step 56 / final 0.198 (success < 0.020)
- non-grasp task: object moved 0.035 m toward the goal (|object_to_goal| 0.200 at the end); max rise 0.004 m

### env 5 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 59 steps; TERMINATED by `ee_ground_collision` at step 58
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 39, 1 (DESCEND): 14, 3 (PUSH): 6
- aperture min -0.001 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.040 at step 58 (final 0.040)
- closest |object_to_goal| 0.207 at step 9 (final 0.209); goal_error min 0.212 at step 58 / final 0.212 (success < 0.020)
- non-grasp task: object moved 0.009 m toward the goal (|object_to_goal| 0.209 at the end); max rise 0.000 m

### env 6 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 61 steps; TERMINATED by `ee_ground_collision` at step 60
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 38, 1 (DESCEND): 14, 3 (PUSH): 9
- aperture min -0.000 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.052 at step 57 (final 0.052)
- closest |object_to_goal| 0.165 at step 60 (final 0.165); goal_error min 0.165 at step 60 / final 0.165 (success < 0.020)
- non-grasp task: object moved 0.037 m toward the goal (|object_to_goal| 0.165 at the end); max rise 0.008 m

### env 7 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 31, 1 (DESCEND): 21, 3 (PUSH): 11
- aperture min -0.002 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.044 at step 62 (final 0.044)
- closest |object_to_goal| 0.116 at step 62 (final 0.116); goal_error min 0.116 at step 62 / final 0.116 (success < 0.020)
- non-grasp task: object moved 0.033 m toward the goal (|object_to_goal| 0.116 at the end); max rise 0.000 m

### env 8 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 28, 1 (DESCEND): 13, 3 (PUSH): 12
- aperture min -0.001 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.044 at step 42 (final 0.063)
- closest |object_to_goal| 0.148 at step 52 (final 0.148); goal_error min 0.147 at step 52 / final 0.147 (success < 0.020)
- non-grasp task: object moved 0.086 m toward the goal (|object_to_goal| 0.148 at the end); max rise 0.007 m

### env 9 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 31, 1 (DESCEND): 25, 3 (PUSH): 4
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.053 at step 59 (final 0.053)
- closest |object_to_goal| 0.105 at step 2 (final 0.112); goal_error min 0.116 at step 6 / final 0.116 (success < 0.020)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.112 at the end); max rise 0.000 m

### env 10 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 50 steps; TERMINATED by `ee_ground_collision` at step 49
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 32, 1 (DESCEND): 18
- aperture min -0.001 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.040 at step 46 (final 0.056)
- closest |object_to_goal| 0.129 at step 43 (final 0.138); goal_error min 0.140 at step 49 / final 0.140 (success < 0.020)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.138 at the end); max rise 0.000 m

### env 11 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 45 steps; TERMINATED by `ee_ground_collision` at step 44
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 30, 1 (DESCEND): 15
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.063 at step 40 (final 0.067)
- closest |object_to_goal| 0.313 at step 15 (final 0.328); goal_error min 0.324 at step 3 / final 0.324 (success < 0.020)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.328 at the end); max rise 0.000 m

### env 12 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 27, 1 (DESCEND): 12, 3 (PUSH): 12
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.040 at step 41 (final 0.041)
- closest |object_to_goal| 0.116 at step 50 (final 0.116); goal_error min 0.113 at step 50 / final 0.113 (success < 0.020)
- non-grasp task: object moved 0.060 m toward the goal (|object_to_goal| 0.116 at the end); max rise 0.002 m

### env 13 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 48 steps; TERMINATED by `ee_ground_collision` at step 47
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 31, 1 (DESCEND): 13, 3 (PUSH): 4
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.050 at step 45 (final 0.052)
- closest |object_to_goal| 0.242 at step 2 (final 0.247); goal_error min 0.253 at step 5 / final 0.253 (success < 0.020)
- non-grasp task: object moved 0.003 m toward the goal (|object_to_goal| 0.247 at the end); max rise 0.000 m

### env 14 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 52 steps; TERMINATED by `ee_ground_collision` at step 51
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 30, 1 (DESCEND): 16, 3 (PUSH): 6
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.046 at step 49 (final 0.050)
- closest |object_to_goal| 0.246 at step 13 (final 0.254); goal_error min 0.252 at step 51 / final 0.252 (success < 0.020)
- non-grasp task: object moved 0.003 m toward the goal (|object_to_goal| 0.254 at the end); max rise 0.000 m

### env 15 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 87 steps; TERMINATED by `ee_ground_collision` at step 86
- most steps in phase 3 (PUSH); steps per phase 0 (APPROACH): 35, 1 (DESCEND): 13, 3 (PUSH): 39
- aperture min -0.003 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.046 at step 63 (final 0.060)
- closest |object_to_goal| 0.092 at step 85 (final 0.103); goal_error min 0.098 at step 86 / final 0.098 (success < 0.020)
- non-grasp task: object moved 0.167 m toward the goal (|object_to_goal| 0.103 at the end); max rise 0.002 m

### env 16 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 30, 1 (DESCEND): 25, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.060 at step 37 (final 0.073)
- closest |object_to_goal| 0.133 at step 7 (final 0.152); goal_error min 0.145 at step 7 / final 0.145 (success < 0.020)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.152 at the end); max rise 0.000 m

### env 17 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 72 steps; TERMINATED by `ee_ground_collision` at step 71
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 44, 1 (DESCEND): 25, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.052 at step 71 (final 0.052)
- closest |object_to_goal| 0.121 at step 38 (final 0.131); goal_error min 0.132 at step 1 / final 0.132 (success < 0.020)
- non-grasp task: object moved 0.012 m toward the goal (|object_to_goal| 0.131 at the end); max rise 0.000 m

### env 18 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 34, 1 (DESCEND): 17
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.050 at step 43 (final 0.065)
- closest |object_to_goal| 0.458 at step 24 (final 0.467); goal_error min 0.467 at step 3 / final 0.467 (success < 0.020)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.467 at the end); max rise 0.000 m

### env 19 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 49 steps; TERMINATED by `ee_ground_collision` at step 48
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 32, 1 (DESCEND): 17
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.047 at step 48 (final 0.047)
- closest |object_to_goal| 0.418 at step 23 (final 0.431); goal_error min 0.429 at step 3 / final 0.429 (success < 0.020)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.431 at the end); max rise 0.000 m

### env 20 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 73 steps; TERMINATED by `ee_ground_collision` at step 72
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 45, 1 (DESCEND): 25, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.059 at step 71 (final 0.066)
- closest |object_to_goal| 0.190 at step 62 (final 0.203); goal_error min 0.203 at step 4 / final 0.203 (success < 0.020)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.203 at the end); max rise 0.000 m

### env 21 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 74 steps; TERMINATED by `ee_ground_collision` at step 73
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 39, 1 (DESCEND): 25, 3 (PUSH): 10
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.036 at step 68 (final 0.045)
- closest |object_to_goal| 0.159 at step 73 (final 0.159); goal_error min 0.160 at step 73 / final 0.160 (success < 0.020)
- non-grasp task: object moved 0.046 m toward the goal (|object_to_goal| 0.159 at the end); max rise 0.007 m

### env 22 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 45 steps; TERMINATED by `ee_ground_collision` at step 44
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 27, 1 (DESCEND): 16, 3 (PUSH): 2
- aperture min -0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.048 at step 42 (final 0.062)
- closest |object_to_goal| 0.326 at step 44 (final 0.326); goal_error min 0.337 at step 4 / final 0.337 (success < 0.020)
- non-grasp task: object moved 0.005 m toward the goal (|object_to_goal| 0.326 at the end); max rise 0.000 m

### env 23 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 71 steps; TERMINATED by `ee_ground_collision` at step 70
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 41, 1 (DESCEND): 15, 3 (PUSH): 15
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.040 at step 59 (final 0.049)
- closest |object_to_goal| 0.157 at step 70 (final 0.157); goal_error min 0.153 at step 70 / final 0.153 (success < 0.020)
- non-grasp task: object moved 0.103 m toward the goal (|object_to_goal| 0.157 at the end); max rise 0.012 m

### env 24 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 35, 1 (DESCEND): 14, 3 (PUSH): 4
- aperture min -0.000 / max 0.076 / final 0.002
- closest |gripper_to_object| 0.055 at step 51 (final 0.059)
- closest |object_to_goal| 0.144 at step 15 (final 0.154); goal_error min 0.156 at step 48 / final 0.156 (success < 0.020)
- non-grasp task: object moved 0.023 m toward the goal (|object_to_goal| 0.154 at the end); max rise 0.000 m

### env 25 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 64 steps; TERMINATED by `ee_ground_collision` at step 63
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 39, 1 (DESCEND): 22, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.061 at step 63 (final 0.061)
- closest |object_to_goal| 0.111 at step 62 (final 0.120); goal_error min 0.123 at step 7 / final 0.123 (success < 0.020)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.120 at the end); max rise 0.000 m

### env 26 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 29, 1 (DESCEND): 25, 3 (PUSH): 3
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.050 at step 38 (final 0.060)
- closest |object_to_goal| 0.148 at step 15 (final 0.149); goal_error min 0.158 at step 6 / final 0.158 (success < 0.020)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.149 at the end); max rise 0.000 m

### env 27 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 47 steps; TERMINATED by `ee_ground_collision` at step 46
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 25, 1 (DESCEND): 9, 3 (PUSH): 13
- aperture min -0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.037 at step 38 (final 0.056)
- closest |object_to_goal| 0.083 at step 46 (final 0.083); goal_error min 0.086 at step 46 / final 0.086 (success < 0.020)
- non-grasp task: object moved 0.065 m toward the goal (|object_to_goal| 0.083 at the end); max rise 0.002 m

### env 28 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 142 steps; TERMINATED by `ee_ground_collision` at step 141
- most steps in phase 3 (PUSH); steps per phase 0 (APPROACH): 26, 1 (DESCEND): 27, 3 (PUSH): 69, 5 (RESEAT_UP): 20
- aperture min -0.001 / max 0.076 / final 0.003
- closest |gripper_to_object| 0.028 at step 98 (final 0.065)
- closest |object_to_goal| 0.149 at step 124 (final 0.149); goal_error min 0.154 at step 123 / final 0.157 (success < 0.020)
- non-grasp task: object moved 0.089 m toward the goal (|object_to_goal| 0.149 at the end); max rise 0.005 m

### env 29 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 36, 1 (DESCEND): 15, 3 (PUSH): 12
- aperture min -0.001 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.054 at step 54 (final 0.068)
- closest |object_to_goal| 0.099 at step 61 (final 0.111); goal_error min 0.101 at step 62 / final 0.101 (success < 0.020)
- non-grasp task: object moved 0.058 m toward the goal (|object_to_goal| 0.111 at the end); max rise 0.005 m

### env 30 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 41 steps; TERMINATED by `ee_ground_collision` at step 40
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 27, 1 (DESCEND): 14
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.054 at step 35 (final 0.072)
- closest |object_to_goal| 0.119 at step 4 (final 0.135); goal_error min 0.128 at step 5 / final 0.128 (success < 0.020)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.135 at the end); max rise 0.000 m

### env 31 — terminated:ee_ground_collision
- ended in phase 3 (PUSH) after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0 (APPROACH); steps per phase 0 (APPROACH): 31, 1 (DESCEND): 13, 3 (PUSH): 9
- aperture min -0.000 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.045 at step 50 (final 0.061)
- closest |object_to_goal| 0.262 at step 52 (final 0.262); goal_error min 0.263 at step 52 / final 0.263 (success < 0.020)
- non-grasp task: object moved 0.054 m toward the goal (|object_to_goal| 0.262 at the end); max rise 0.009 m

## Successful envs (one line each)

