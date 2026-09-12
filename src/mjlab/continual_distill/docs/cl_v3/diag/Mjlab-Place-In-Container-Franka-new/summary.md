# diagnose — Mjlab-Place-In-Container-Franka

2026-09-09T20:28:55 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**25/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `PlaceInContainerClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (7): 1: 5, 6 (RELEASE): 2
- end phase, successful envs (25): 1: 11, 8 (DONE): 11, 2: 2, 0: 1
- most-steps phase, failing envs: 1: 6, 5 (PLACE): 1
- failure class: grasp_lost: 5, terminated:ee_ground_collision: 2
- termination among failing envs: none: 5, ee_ground_collision: 2
- failing envs that lifted the object: 7/7; lifted then lost: 7
- success step (successful envs): median 229, max 932

## Reading

- **grasp_lost** — 5/7 failing envs (envs [8, 11, 14, 22, 25]); typical end phase 1; final aperture median 0.081; closest |gripper_to_object| median 0.006; closest |object_to_goal| median 0.107; min goal_error median 0.116; lost at step median 399
- **terminated:ee_ground_collision** — 2/7 failing envs (envs [9, 28]); typical end phase 6 (RELEASE); final aperture median 0.080; closest |gripper_to_object| median 0.010; closest |object_to_goal| median 0.132; min goal_error median 0.134

## Failing envs

### env 8 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 78, 1: 587, 2: 75, 3: 105, 4: 22, 5 (PLACE): 42, 6 (RELEASE): 60, 8 (DONE): 31
- aperture min 0.000 / max 0.097 / final 0.068
- closest |gripper_to_object| 0.004 at step 139 (final 0.059)
- closest |object_to_goal| 0.115 at step 479 (final 0.136); goal_error min 0.127 at step 234 / final 0.129
- object LIFTED at step 73 (max rise 0.194 m); LOST at step 399 (aperture collapsed while the object moved away)

### env 9 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 297 steps; TERMINATED by `ee_ground_collision` at step 296
- most steps in phase 5 (PLACE); steps per phase 0: 41, 1: 58, 2: 30, 3: 51, 4: 20, 5 (PLACE): 65, 6 (RELEASE): 32
- aperture min -0.000 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.011 at step 152 (final 0.919)
- closest |object_to_goal| 0.123 at step 201 (final 0.319); goal_error min 0.129 at step 201 / final 0.316
- object LIFTED at step 162 (max rise 0.195 m); LOST at step 209 (aperture collapsed while the object moved away)

### env 11 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 79, 1: 549, 2: 90, 3: 138, 4: 22, 5 (PLACE): 31, 6 (RELEASE): 60, 8 (DONE): 31
- aperture min 0.000 / max 0.093 / final 0.080
- closest |gripper_to_object| 0.007 at step 369 (final 0.070)
- closest |object_to_goal| 0.050 at step 855 (final 0.068); goal_error min 0.048 at step 383 / final 0.063
- object LIFTED at step 234 (max rise 0.179 m); LOST at step 569 (aperture collapsed while the object moved away)

### env 14 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 115, 1: 476, 2: 75, 3: 105, 4: 103, 5 (PLACE): 35, 6 (RELEASE): 60, 8 (DONE): 31
- aperture min -0.000 / max 0.096 / final 0.093
- closest |gripper_to_object| 0.007 at step 316 (final 0.102)
- closest |object_to_goal| 0.107 at step 656 (final 0.127); goal_error min 0.116 at step 445 / final 0.118
- object LIFTED at step 84 (max rise 0.189 m); LOST at step 112 (aperture collapsed while the object moved away)

### env 22 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 73, 1: 604, 2: 75, 3: 100, 4: 21, 5 (PLACE): 36, 6 (RELEASE): 60, 8 (DONE): 31
- aperture min 0.001 / max 0.096 / final 0.081
- closest |gripper_to_object| 0.006 at step 151 (final 0.070)
- closest |object_to_goal| 0.098 at step 752 (final 0.107); goal_error min 0.111 at step 241 / final 0.111
- object LIFTED at step 76 (max rise 0.191 m); LOST at step 410 (aperture collapsed while the object moved away)

### env 25 — grasp_lost
- ended in phase 1 after 1000 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 89, 1: 580, 2: 75, 3: 104, 4: 22, 5 (PLACE): 39, 6 (RELEASE): 60, 8 (DONE): 31
- aperture min -0.004 / max 0.094 / final 0.086
- closest |gripper_to_object| 0.003 at step 60 (final 0.098)
- closest |object_to_goal| 0.117 at step 706 (final 0.131); goal_error min 0.127 at step 260 / final 0.128
- object LIFTED at step 79 (max rise 0.184 m); LOST at step 397 (aperture collapsed while the object moved away)

### env 28 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 390 steps; TERMINATED by `ee_ground_collision` at step 389
- most steps in phase 1; steps per phase 0: 54, 1: 81, 2: 45, 3: 74, 4: 19, 5 (PLACE): 65, 6 (RELEASE): 52
- aperture min -0.000 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.008 at step 48 (final 0.840)
- closest |object_to_goal| 0.141 at step 273 (final 0.248); goal_error min 0.140 at step 273 / final 0.244
- object LIFTED at step 68 (max rise 0.189 m); LOST at step 159 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 171, end phase 1, aperture min 0.009, min |go| 0.005, lifted at 83
- env 1: success at step 222, end phase 1, aperture min -0.000, min |go| 0.001, lifted at 75
- env 2: success at step 227, end phase 8 (DONE), aperture min 0.042, min |go| 0.003, lifted at 76
- env 3: success at step 153, end phase 1, aperture min 0.009, min |go| 0.003, lifted at 71
- env 4: success at step 174, end phase 8 (DONE), aperture min 0.010, min |go| 0.006, lifted at 76
- env 5: success at step 239, end phase 8 (DONE), aperture min 0.043, min |go| 0.004, lifted at 71
- env 6: success at step 378, end phase 1, aperture min 0.000, min |go| 0.006, lifted at 240
- env 7: success at step 163, end phase 8 (DONE), aperture min 0.050, min |go| 0.003, lifted at 72
- env 10: success at step 163, end phase 1, aperture min -0.006, min |go| 0.007, lifted at 81
- env 12: success at step 241, end phase 1, aperture min -0.011, min |go| 0.005, lifted at 73
- env 13: success at step 221, end phase 8 (DONE), aperture min 0.041, min |go| 0.002, lifted at 73
- env 15: success at step 128, end phase 2, aperture min -0.000, min |go| 0.006, lifted at 83
- env 16: success at step 551, end phase 8 (DONE), aperture min 0.000, min |go| 0.004, lifted at 398
- env 17: success at step 932, end phase 8 (DONE), aperture min 0.000, min |go| 0.002, lifted at 68
- env 18: success at step 150, end phase 8 (DONE), aperture min -0.004, min |go| 0.005, lifted at 64
- env 19: success at step 450, end phase 2, aperture min -0.001, min |go| 0.003, lifted at 80
- env 20: success at step 374, end phase 8 (DONE), aperture min -0.000, min |go| 0.003, lifted at 84
- env 21: success at step 264, end phase 1, aperture min 0.000, min |go| 0.003, lifted at 80
- env 23: success at step 229, end phase 1, aperture min 0.007, min |go| 0.003, lifted at 66
- env 24: success at step 240, end phase 1, aperture min 0.000, min |go| 0.007, lifted at 81
- env 26: success at step 218, end phase 0, aperture min -0.000, min |go| 0.012, lifted at 164
- env 27: success at step 432, end phase 8 (DONE), aperture min -0.007, min |go| 0.017, lifted at 342
- env 29: success at step 152, end phase 8 (DONE), aperture min 0.005, min |go| 0.009, lifted at 77
- env 30: success at step 377, end phase 1, aperture min 0.000, min |go| 0.005, lifted at 240
- env 31: success at step 231, end phase 1, aperture min 0.000, min |go| 0.005, lifted at 75
