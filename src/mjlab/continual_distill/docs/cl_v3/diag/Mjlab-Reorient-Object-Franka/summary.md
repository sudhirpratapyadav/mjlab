# diagnose — Mjlab-Reorient-Object-Franka

2026-09-09T19:13:32 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**1/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ReorientObjectClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 3 = LIFT, 4 = CARRY, 8 = DONE

## Histograms

- end phase, FAILING envs (31): 1: 16, 2: 7, 8 (DONE): 5, 0: 3
- end phase, successful envs (1): 8 (DONE): 1
- most-steps phase, failing envs: 0: 26, 4 (CARRY): 5
- failure class: terminated:ee_ground_collision: 30, never_reached: 1
- termination among failing envs: ee_ground_collision: 30, none: 1
- failing envs that lifted the object: 4/31; lifted then lost: 0
- success step (successful envs): median 80, max 80

## Reading

- **terminated:ee_ground_collision** — 30/31 failing envs (envs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 30, 31]); typical end phase 1; final aperture median 0.080; closest |gripper_to_object| median 0.017; closest |object_to_goal| median 0.003; min goal_error median 1.555
- **never_reached** — 1/31 failing envs (envs [17]); typical end phase 0; final aperture median 0.080; closest |gripper_to_object| median 0.074; closest |object_to_goal| median 0.001; min goal_error median 1.555

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 0 after 26 steps; TERMINATED by `ee_ground_collision` at step 25
- most steps in phase 0; steps per phase 0: 26
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.117 at step 25 (final 0.117)
- closest |object_to_goal| 0.003 at step 21 (final 0.005); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 1 — terminated:ee_ground_collision
- ended in phase 1 after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0; steps per phase 0: 36, 1: 15
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.016 at step 46 (final 0.035)
- closest |object_to_goal| 0.004 at step 48 (final 0.011); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 2 — terminated:ee_ground_collision
- ended in phase 1 after 47 steps; TERMINATED by `ee_ground_collision` at step 46
- most steps in phase 0; steps per phase 0: 36, 1: 11
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 46 (final 0.009)
- closest |object_to_goal| 0.001 at step 28 (final 0.004); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 3 — terminated:ee_ground_collision
- ended in phase 1 after 144 steps; TERMINATED by `ee_ground_collision` at step 143
- most steps in phase 0; steps per phase 0: 129, 1: 15
- aperture min 0.076 / max 0.082 / final 0.082
- closest |gripper_to_object| 0.032 at step 140 (final 0.034)
- closest |object_to_goal| 0.004 at step 93 (final 0.015); goal_error min 1.555 at step 2 / final 1.612
- object never lifted (max rise 0.000 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 1 after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0; steps per phase 0: 44, 1: 13
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.030 at step 54 (final 0.042)
- closest |object_to_goal| 0.001 at step 47 (final 0.008); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 5 — terminated:ee_ground_collision
- ended in phase 1 after 71 steps; TERMINATED by `ee_ground_collision` at step 70
- most steps in phase 0; steps per phase 0: 54, 1: 17
- aperture min 0.076 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.014 at step 66 (final 0.027)
- closest |object_to_goal| 0.002 at step 21 (final 0.070); goal_error min 1.526 at step 70 / final 1.526
- object never lifted (max rise 0.000 m)

### env 6 — terminated:ee_ground_collision
- ended in phase 8 (DONE) after 145 steps; TERMINATED by `ee_ground_collision` at step 144
- most steps in phase 4 (CARRY); steps per phase 0: 36, 1: 13, 2: 14, 3 (LIFT): 20, 4 (CARRY): 55, 8 (DONE): 7
- aperture min 0.030 / max 0.080 / final 0.031
- closest |gripper_to_object| 0.004 at step 73 (final 0.022)
- closest |object_to_goal| 0.003 at step 56 (final 0.365); goal_error min 1.555 at step 2 / final 2.598
- object LIFTED at step 77 (max rise 0.075 m); never lost

### env 7 — terminated:ee_ground_collision
- ended in phase 1 after 383 steps; TERMINATED by `ee_ground_collision` at step 382
- most steps in phase 0; steps per phase 0: 362, 1: 21
- aperture min 0.076 / max 0.083 / final 0.083
- closest |gripper_to_object| 0.028 at step 382 (final 0.028)
- closest |object_to_goal| 0.001 at step 355 (final 0.019); goal_error min 1.555 at step 2 / final 1.587
- object never lifted (max rise 0.000 m)

### env 8 — terminated:ee_ground_collision
- ended in phase 1 after 84 steps; TERMINATED by `ee_ground_collision` at step 83
- most steps in phase 0; steps per phase 0: 68, 1: 16
- aperture min 0.076 / max 0.082 / final 0.082
- closest |gripper_to_object| 0.027 at step 82 (final 0.036)
- closest |object_to_goal| 0.003 at step 83 (final 0.003); goal_error min 1.555 at step 2 / final 1.570
- object never lifted (max rise 0.000 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 2 after 49 steps; TERMINATED by `ee_ground_collision` at step 48
- most steps in phase 0; steps per phase 0: 34, 1: 12, 2: 3
- aperture min 0.068 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.012 at step 46 (final 0.025)
- closest |object_to_goal| 0.002 at step 1 (final 0.012); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 2 after 46 steps; TERMINATED by `ee_ground_collision` at step 45
- most steps in phase 0; steps per phase 0: 34, 1: 11, 2: 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.009 at step 45 (final 0.009)
- closest |object_to_goal| 0.004 at step 13 (final 0.016); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 1 after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0; steps per phase 0: 46, 1: 14
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.035 at step 58 (final 0.054)
- closest |object_to_goal| 0.001 at step 48 (final 0.005); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 12 — terminated:ee_ground_collision
- ended in phase 1 after 42 steps; TERMINATED by `ee_ground_collision` at step 41
- most steps in phase 0; steps per phase 0: 32, 1: 10
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.013 at step 41 (final 0.013)
- closest |object_to_goal| 0.001 at step 24 (final 0.010); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 13 — terminated:ee_ground_collision
- ended in phase 1 after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0; steps per phase 0: 43, 1: 13
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.025 at step 55 (final 0.025)
- closest |object_to_goal| 0.003 at step 39 (final 0.017); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 14 — terminated:ee_ground_collision
- ended in phase 8 (DONE) after 140 steps; TERMINATED by `ee_ground_collision` at step 139
- most steps in phase 4 (CARRY); steps per phase 0: 34, 1: 10, 2: 14, 3 (LIFT): 20, 4 (CARRY): 55, 8 (DONE): 7
- aperture min 0.026 / max 0.080 / final 0.027
- closest |gripper_to_object| 0.003 at step 88 (final 0.010)
- closest |object_to_goal| 0.003 at step 29 (final 0.411); goal_error min 1.153 at step 77 / final 1.932
- object LIFTED at step 68 (max rise 0.095 m); never lost

### env 15 — terminated:ee_ground_collision
- ended in phase 2 after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0; steps per phase 0: 43, 1: 13, 2: 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 56 (final 0.024)
- closest |object_to_goal| 0.003 at step 13 (final 0.014); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 16 — terminated:ee_ground_collision
- ended in phase 2 after 52 steps; TERMINATED by `ee_ground_collision` at step 51
- most steps in phase 0; steps per phase 0: 36, 1: 14, 2: 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.017 at step 48 (final 0.022)
- closest |object_to_goal| 0.003 at step 14 (final 0.018); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 17 — never_reached
- ended in phase 0 after 1000 steps; ran to time-out
- most steps in phase 0; steps per phase 0: 1000
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.074 at step 56 (final 0.095)
- closest |object_to_goal| 0.001 at step 377 (final 0.009); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 18 — terminated:ee_ground_collision
- ended in phase 8 (DONE) after 143 steps; TERMINATED by `ee_ground_collision` at step 142
- most steps in phase 4 (CARRY); steps per phase 0: 37, 1: 9, 2: 14, 3 (LIFT): 20, 4 (CARRY): 55, 8 (DONE): 8
- aperture min 0.021 / max 0.080 / final 0.022
- closest |gripper_to_object| 0.007 at step 46 (final 0.020)
- closest |object_to_goal| 0.002 at step 25 (final 0.362); goal_error min 0.109 at step 128 / final 1.311
- object LIFTED at step 74 (max rise 0.059 m); never lost

### env 19 — terminated:ee_ground_collision
- ended in phase 1 after 43 steps; TERMINATED by `ee_ground_collision` at step 42
- most steps in phase 0; steps per phase 0: 33, 1: 10
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.021 at step 42 (final 0.021)
- closest |object_to_goal| 0.002 at step 13 (final 0.012); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 20 — terminated:ee_ground_collision
- ended in phase 1 after 68 steps; TERMINATED by `ee_ground_collision` at step 67
- most steps in phase 0; steps per phase 0: 53, 1: 15
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.018 at step 67 (final 0.018)
- closest |object_to_goal| 0.003 at step 24 (final 0.006); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 21 — terminated:ee_ground_collision
- ended in phase 0 after 27 steps; TERMINATED by `ee_ground_collision` at step 26
- most steps in phase 0; steps per phase 0: 27
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.149 at step 26 (final 0.149)
- closest |object_to_goal| 0.004 at step 8 (final 0.013); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 22 — terminated:ee_ground_collision
- ended in phase 2 after 45 steps; TERMINATED by `ee_ground_collision` at step 44
- most steps in phase 0; steps per phase 0: 40, 1: 3, 2: 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.008 at step 43 (final 0.017)
- closest |object_to_goal| 0.004 at step 26 (final 0.010); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 23 — terminated:ee_ground_collision
- ended in phase 1 after 202 steps; TERMINATED by `ee_ground_collision` at step 201
- most steps in phase 0; steps per phase 0: 179, 1: 23
- aperture min 0.076 / max 0.083 / final 0.083
- closest |gripper_to_object| 0.028 at step 199 (final 0.038)
- closest |object_to_goal| 0.002 at step 173 (final 0.005); goal_error min 1.555 at step 2 / final 1.605
- object never lifted (max rise 0.000 m)

### env 24 — terminated:ee_ground_collision
- ended in phase 2 after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0; steps per phase 0: 37, 1: 15, 2: 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.013 at step 52 (final 0.013)
- closest |object_to_goal| 0.002 at step 17 (final 0.014); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 1 after 62 steps; TERMINATED by `ee_ground_collision` at step 61
- most steps in phase 0; steps per phase 0: 48, 1: 14
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.035 at step 59 (final 0.051)
- closest |object_to_goal| 0.003 at step 2 (final 0.007); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 26 — terminated:ee_ground_collision
- ended in phase 8 (DONE) after 157 steps; TERMINATED by `ee_ground_collision` at step 156
- most steps in phase 4 (CARRY); steps per phase 0: 47, 1: 13, 2: 14, 3 (LIFT): 20, 4 (CARRY): 55, 8 (DONE): 8
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.017 at step 60 (final 0.356)
- closest |object_to_goal| 0.002 at step 44 (final 0.178); goal_error min 1.550 at step 74 / final 1.584
- object never lifted (max rise 0.000 m)

### env 27 — terminated:ee_ground_collision
- ended in phase 2 after 67 steps; TERMINATED by `ee_ground_collision` at step 66
- most steps in phase 0; steps per phase 0: 51, 1: 14, 2: 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.003 at step 65 (final 0.015)
- closest |object_to_goal| 0.003 at step 3 (final 0.017); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 29 — terminated:ee_ground_collision
- ended in phase 1 after 68 steps; TERMINATED by `ee_ground_collision` at step 67
- most steps in phase 0; steps per phase 0: 54, 1: 14
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.034 at step 65 (final 0.038)
- closest |object_to_goal| 0.002 at step 37 (final 0.011); goal_error min 1.555 at step 2 / final 1.608
- object never lifted (max rise 0.000 m)

### env 30 — terminated:ee_ground_collision
- ended in phase 1 after 44 steps; TERMINATED by `ee_ground_collision` at step 43
- most steps in phase 0; steps per phase 0: 35, 1: 9
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.011 at step 41 (final 0.018)
- closest |object_to_goal| 0.003 at step 1 (final 0.011); goal_error min 1.555 at step 2 / final 1.615
- object never lifted (max rise 0.000 m)

### env 31 — terminated:ee_ground_collision
- ended in phase 8 (DONE) after 160 steps; TERMINATED by `ee_ground_collision` at step 159
- most steps in phase 4 (CARRY); steps per phase 0: 45, 1: 18, 2: 14, 3 (LIFT): 20, 4 (CARRY): 55, 8 (DONE): 8
- aperture min 0.029 / max 0.082 / final 0.031
- closest |gripper_to_object| 0.008 at step 63 (final 0.022)
- closest |object_to_goal| 0.001 at step 6 (final 0.439); goal_error min 0.071 at step 103 / final 0.143
- object LIFTED at step 92 (max rise 0.084 m); never lost

## Successful envs (one line each)

- env 28: success at step 80, end phase 8 (DONE), aperture min 0.025, min |go| 0.011, lifted at 83
