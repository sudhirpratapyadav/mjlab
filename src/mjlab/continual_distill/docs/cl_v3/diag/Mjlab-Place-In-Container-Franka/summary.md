# diagnose — Mjlab-Place-In-Container-Franka

2026-09-09T15:19:58 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**8/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `PlaceInContainerClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 5 = PLACE, 6 = RELEASE, 8 = DONE

## Histograms

- end phase, FAILING envs (24): 4: 21, 1: 2, 2: 1
- end phase, successful envs (8): 8 (DONE): 8
- most-steps phase, failing envs: 4: 21, 0: 3
- failure class: never_lifted: 10, grasp_lost: 7, terminated:ee_ground_collision: 6, lifted_not_at_goal: 1
- termination among failing envs: none: 18, ee_ground_collision: 6
- failing envs that lifted the object: 11/24; lifted then lost: 10
- success step (successful envs): median 162, max 180

## Reading

- **never_lifted** — 10/24 failing envs (envs [3, 6, 8, 9, 10, 12, 13, 14, 20, 29]); typical end phase 4; final aperture median 0.000; closest |gripper_to_object| median 0.010; closest |object_to_goal| median 0.293; min goal_error median 0.303
- **grasp_lost** — 7/24 failing envs (envs [2, 11, 15, 18, 22, 24, 27]); typical end phase 4; final aperture median 0.000; closest |gripper_to_object| median 0.005; closest |object_to_goal| median 0.145; min goal_error median 0.154; lost at step median 126
- **terminated:ee_ground_collision** — 6/24 failing envs (envs [0, 1, 16, 17, 25, 31]); typical end phase 4; final aperture median 0.040; closest |gripper_to_object| median 0.011; closest |object_to_goal| median 0.195; min goal_error median 0.204
- **lifted_not_at_goal** — 1/24 failing envs (envs [4]); typical end phase 4; final aperture median 0.044; closest |gripper_to_object| median 0.003; closest |object_to_goal| median 0.138; min goal_error median 0.151

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 2 after 49 steps; TERMINATED by `ee_ground_collision` at step 48
- most steps in phase 0; steps per phase 0: 36, 1: 11, 2: 2
- aperture min 0.076 / max 0.085 / final 0.084
- closest |gripper_to_object| 0.015 at step 46 (final 0.017)
- closest |object_to_goal| 0.367 at step 48 (final 0.367); goal_error min 0.368 at step 48 / final 0.368
- object never lifted (max rise 0.000 m)

### env 1 — terminated:ee_ground_collision
- ended in phase 1 after 42 steps; TERMINATED by `ee_ground_collision` at step 41
- most steps in phase 0; steps per phase 0: 28, 1: 14
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.015 at step 41 (final 0.015)
- closest |object_to_goal| 0.349 at step 20 (final 0.358); goal_error min 0.359 at step 3 / final 0.359
- object never lifted (max rise 0.000 m)

### env 2 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 28, 1: 18, 2: 12, 3: 26, 4: 916
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.006 at step 46 (final 1.092)
- closest |object_to_goal| 0.145 at step 114 (final 0.170); goal_error min 0.150 at step 113 / final 0.171
- object LIFTED at step 74 (max rise 0.102 m); LOST at step 110 (aperture collapsed while the object moved away)

### env 3 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 37, 1: 12, 2: 12, 3: 26, 4: 913
- aperture min -0.000 / max 0.087 / final -0.000
- closest |gripper_to_object| 0.013 at step 48 (final 1.201)
- closest |object_to_goal| 0.292 at step 132 (final 0.295); goal_error min 0.301 at step 76 / final 0.302
- object never lifted (max rise 0.003 m)

### env 4 — lifted_not_at_goal
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 35, 1: 19, 2: 12, 3: 26, 4: 908
- aperture min 0.031 / max 0.090 / final 0.044
- closest |gripper_to_object| 0.003 at step 821 (final 0.007)
- closest |object_to_goal| 0.138 at step 820 (final 0.141); goal_error min 0.151 at step 977 / final 0.151
- object LIFTED at step 74 (max rise 0.110 m); never lost

### env 6 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 37, 1: 38, 2: 12, 3: 26, 4: 887
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.016 at step 58 (final 0.707)
- closest |object_to_goal| 0.234 at step 454 (final 0.243); goal_error min 0.242 at step 64 / final 0.246
- object never lifted (max rise 0.004 m)

### env 8 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 31, 1: 24, 2: 12, 3: 26, 4: 907
- aperture min -0.000 / max 0.083 / final 0.000
- closest |gripper_to_object| 0.012 at step 46 (final 0.753)
- closest |object_to_goal| 0.302 at step 55 (final 0.325); goal_error min 0.309 at step 56 / final 0.331
- object never lifted (max rise 0.008 m)

### env 9 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 36, 1: 13, 2: 12, 3: 26, 4: 913
- aperture min -0.000 / max 0.083 / final 0.000
- closest |gripper_to_object| 0.015 at step 48 (final 0.831)
- closest |object_to_goal| 0.411 at step 49 (final 0.433); goal_error min 0.418 at step 50 / final 0.433
- object never lifted (max rise 0.008 m)

### env 10 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 30, 1: 61, 2: 12, 3: 26, 4: 871
- aperture min -0.000 / max 0.086 / final 0.000
- closest |gripper_to_object| 0.008 at step 47 (final 0.949)
- closest |object_to_goal| 0.391 at step 50 (final 0.568); goal_error min 0.399 at step 50 / final 0.580
- object never lifted (max rise 0.015 m)

### env 11 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 42, 1: 20, 2: 12, 3: 26, 4: 900
- aperture min -0.000 / max 0.085 / final -0.000
- closest |gripper_to_object| 0.003 at step 67 (final 1.133)
- closest |object_to_goal| 0.155 at step 131 (final 0.193); goal_error min 0.159 at step 133 / final 0.187
- object LIFTED at step 81 (max rise 0.107 m); LOST at step 145 (aperture collapsed while the object moved away)

### env 12 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 30, 1: 25, 2: 12, 3: 26, 4: 907
- aperture min -0.000 / max 0.088 / final 0.000
- closest |gripper_to_object| 0.006 at step 54 (final 0.749)
- closest |object_to_goal| 0.251 at step 47 (final 0.298); goal_error min 0.260 at step 47 / final 0.303
- object never lifted (max rise 0.009 m)

### env 13 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 43, 1: 24, 2: 12, 3: 26, 4: 895
- aperture min -0.000 / max 0.091 / final 0.000
- closest |gripper_to_object| 0.010 at step 74 (final 0.683)
- closest |object_to_goal| 0.218 at step 80 (final 0.280); goal_error min 0.229 at step 80 / final 0.281
- object never lifted (max rise 0.012 m)

### env 14 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 33, 1: 61, 2: 12, 3: 26, 4: 868
- aperture min -0.000 / max 0.082 / final 0.000
- closest |gripper_to_object| 0.011 at step 50 (final 1.217)
- closest |object_to_goal| 0.294 at step 84 (final 0.353); goal_error min 0.304 at step 54 / final 0.346
- object never lifted (max rise 0.002 m)

### env 15 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 32, 1: 19, 2: 12, 3: 26, 4: 911
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.005 at step 238 (final 1.190)
- closest |object_to_goal| 0.150 at step 112 (final 0.187); goal_error min 0.159 at step 109 / final 0.185
- object LIFTED at step 76 (max rise 0.112 m); LOST at step 455 (aperture collapsed while the object moved away)

### env 16 — terminated:ee_ground_collision
- ended in phase 1 after 44 steps; TERMINATED by `ee_ground_collision` at step 43
- most steps in phase 0; steps per phase 0: 27, 1: 17
- aperture min 0.076 / max 0.082 / final 0.080
- closest |gripper_to_object| 0.014 at step 40 (final 0.028)
- closest |object_to_goal| 0.235 at step 35 (final 0.241); goal_error min 0.246 at step 38 / final 0.247
- object never lifted (max rise 0.000 m)

### env 17 — terminated:ee_ground_collision
- ended in phase 4 after 182 steps; TERMINATED by `ee_ground_collision` at step 181
- most steps in phase 4; steps per phase 0: 38, 1: 16, 2: 12, 3: 26, 4: 90
- aperture min -0.000 / max 0.086 / final 0.000
- closest |gripper_to_object| 0.002 at step 89 (final 0.449)
- closest |object_to_goal| 0.156 at step 122 (final 0.192); goal_error min 0.158 at step 120 / final 0.196
- object LIFTED at step 77 (max rise 0.107 m); LOST at step 125 (aperture collapsed while the object moved away)

### env 18 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 31, 1: 17, 2: 12, 3: 26, 4: 914
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.007 at step 64 (final 1.087)
- closest |object_to_goal| 0.144 at step 259 (final 0.155); goal_error min 0.154 at step 132 / final 0.154
- object LIFTED at step 73 (max rise 0.109 m); LOST at step 122 (aperture collapsed while the object moved away)

### env 20 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 31, 1: 25, 2: 12, 3: 26, 4: 906
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.010 at step 57 (final 1.130)
- closest |object_to_goal| 0.328 at step 60 (final 0.344); goal_error min 0.332 at step 60 / final 0.350
- object never lifted (max rise 0.004 m)

### env 22 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 41, 1: 15, 2: 12, 3: 26, 4: 906
- aperture min -0.012 / max 0.080 / final -0.009
- closest |gripper_to_object| 0.003 at step 103 (final 0.243)
- closest |object_to_goal| 0.150 at step 123 (final 0.164); goal_error min 0.154 at step 128 / final 0.164
- object LIFTED at step 79 (max rise 0.109 m); LOST at step 126 (aperture collapsed while the object moved away)

### env 24 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 35, 1: 21, 2: 12, 3: 26, 4: 906
- aperture min -0.000 / max 0.089 / final 0.000
- closest |gripper_to_object| 0.012 at step 73 (final 0.658)
- closest |object_to_goal| 0.141 at step 121 (final 0.171); goal_error min 0.143 at step 119 / final 0.163
- object LIFTED at step 79 (max rise 0.109 m); LOST at step 115 (aperture collapsed while the object moved away)

### env 25 — terminated:ee_ground_collision
- ended in phase 4 after 184 steps; TERMINATED by `ee_ground_collision` at step 183
- most steps in phase 4; steps per phase 0: 34, 1: 17, 2: 12, 3: 26, 4: 95
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.009 at step 77 (final 0.464)
- closest |object_to_goal| 0.154 at step 114 (final 0.182); goal_error min 0.162 at step 120 / final 0.185
- object LIFTED at step 75 (max rise 0.115 m); LOST at step 120 (aperture collapsed while the object moved away)

### env 27 — grasp_lost
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 29, 1: 16, 2: 12, 3: 26, 4: 917
- aperture min 0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.002 at step 87 (final 0.304)
- closest |object_to_goal| 0.123 at step 504 (final 0.144); goal_error min 0.132 at step 108 / final 0.134
- object LIFTED at step 66 (max rise 0.109 m); LOST at step 982 (aperture collapsed while the object moved away)

### env 29 — never_lifted
- ended in phase 4 after 1000 steps; ran to time-out
- most steps in phase 4; steps per phase 0: 20, 1: 57, 2: 12, 3: 26, 4: 885
- aperture min -0.000 / max 0.090 / final 0.000
- closest |gripper_to_object| 0.003 at step 61 (final 1.221)
- closest |object_to_goal| 0.280 at step 68 (final 0.315); goal_error min 0.287 at step 68 / final 0.323
- object never lifted (max rise 0.007 m)

### env 31 — terminated:ee_ground_collision
- ended in phase 4 after 780 steps; TERMINATED by `ee_ground_collision` at step 779
- most steps in phase 4; steps per phase 0: 39, 1: 16, 2: 12, 3: 26, 4: 687
- aperture min -0.009 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.003 at step 94 (final 0.341)
- closest |object_to_goal| 0.131 at step 193 (final 0.148); goal_error min 0.133 at step 192 / final 0.144
- object LIFTED at step 74 (max rise 0.110 m); LOST at step 208 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 5: success at step 147, end phase 8 (DONE), aperture min 0.046, min |go| 0.004, lifted at 68
- env 7: success at step 156, end phase 8 (DONE), aperture min 0.042, min |go| 0.004, lifted at 70
- env 19: success at step 170, end phase 8 (DONE), aperture min 0.000, min |go| 0.008, lifted at 95
- env 21: success at step 180, end phase 8 (DONE), aperture min 0.046, min |go| 0.004, lifted at 87
- env 23: success at step 162, end phase 8 (DONE), aperture min 0.013, min |go| 0.007, lifted at 76
- env 26: success at step 163, end phase 8 (DONE), aperture min 0.046, min |go| 0.003, lifted at 75
- env 28: success at step 152, end phase 8 (DONE), aperture min 0.046, min |go| 0.003, lifted at 72
- env 30: success at step 172, end phase 8 (DONE), aperture min 0.048, min |go| 0.005, lifted at 84
