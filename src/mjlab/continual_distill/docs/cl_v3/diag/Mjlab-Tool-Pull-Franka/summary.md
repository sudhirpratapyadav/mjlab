# diagnose — Mjlab-Tool-Pull-Franka

2026-09-09T16:01:59 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 600 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToolPullClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = ALIGN, 1 = DESCEND, 2 = PULL, 3 = DONE

## Histograms

- end phase, FAILING envs (32): 2 (PULL): 31, 1 (DESCEND): 1
- end phase, successful envs (0): —
- most-steps phase, failing envs: 2 (PULL): 24, 0 (ALIGN): 8
- failure class: object_not_moved: 15, object_moved_short: 9, terminated:ee_ground_collision: 8
- termination among failing envs: none: 24, ee_ground_collision: 8

## Reading

- **object_not_moved** — 15/32 failing envs (envs [4, 5, 6, 9, 11, 14, 15, 16, 18, 20, 21, 22, 24, 26, 27]); typical end phase 2 (PULL); final aperture median 0.016; closest |gripper_to_object| median 0.014; closest |object_to_goal| median 0.244; min goal_error median 0.255 (success < 0.070)
- **object_moved_short** — 9/32 failing envs (envs [0, 2, 3, 10, 12, 23, 25, 28, 31]); typical end phase 2 (PULL); final aperture median 0.007; closest |gripper_to_object| median 0.014; closest |object_to_goal| median 0.220; min goal_error median 0.232 (success < 0.070)
- **terminated:ee_ground_collision** — 8/32 failing envs (envs [1, 7, 8, 13, 17, 19, 29, 30]); typical end phase 2 (PULL); final aperture median 0.000; closest |gripper_to_object| median 0.054; closest |object_to_goal| median 0.263; min goal_error median 0.272 (success < 0.070)

## Failing envs

### env 0 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 26, 1 (DESCEND): 13, 2 (PULL): 561
- aperture min -0.003 / max 0.076 / final 0.007
- closest |gripper_to_object| 0.013 at step 219 (final 0.027)
- closest |object_to_goal| 0.215 at step 63 (final 0.233); goal_error min 0.227 at step 126 / final 0.235 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.233 at the end); max rise 0.000 m

### env 1 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 19 steps; TERMINATED by `ee_ground_collision` at step 18
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 10, 1 (DESCEND): 8, 2 (PULL): 1
- aperture min 0.001 / max 0.076 / final 0.003
- closest |gripper_to_object| 0.056 at step 18 (final 0.056)
- closest |object_to_goal| 0.245 at step 4 (final 0.249); goal_error min 0.253 at step 5 / final 0.255 (success < 0.070)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.249 at the end); max rise 0.000 m

### env 2 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 7, 1 (DESCEND): 17, 2 (PULL): 576
- aperture min -0.002 / max 0.076 / final 0.006
- closest |gripper_to_object| 0.013 at step 410 (final 0.032)
- closest |object_to_goal| 0.239 at step 290 (final 0.245); goal_error min 0.250 at step 381 / final 0.251 (success < 0.070)
- non-grasp task: object moved 0.033 m toward the goal (|object_to_goal| 0.245 at the end); max rise 0.000 m

### env 3 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 12, 1 (DESCEND): 13, 2 (PULL): 575
- aperture min -0.002 / max 0.076 / final 0.010
- closest |gripper_to_object| 0.014 at step 168 (final 0.031)
- closest |object_to_goal| 0.220 at step 579 (final 0.228); goal_error min 0.232 at step 309 / final 0.233 (success < 0.070)
- non-grasp task: object moved 0.049 m toward the goal (|object_to_goal| 0.228 at the end); max rise 0.000 m

### env 4 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 26, 1 (DESCEND): 10, 2 (PULL): 564
- aperture min -0.004 / max 0.076 / final 0.002
- closest |gripper_to_object| 0.013 at step 76 (final 0.018)
- closest |object_to_goal| 0.272 at step 145 (final 0.282); goal_error min 0.284 at step 70 / final 0.288 (success < 0.070)
- non-grasp task: object moved 0.012 m toward the goal (|object_to_goal| 0.282 at the end); max rise 0.000 m

### env 5 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 29, 1 (DESCEND): 11, 2 (PULL): 560
- aperture min -0.005 / max 0.076 / final 0.008
- closest |gripper_to_object| 0.014 at step 223 (final 0.020)
- closest |object_to_goal| 0.268 at step 89 (final 0.279); goal_error min 0.279 at step 60 / final 0.285 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.279 at the end); max rise 0.000 m

### env 6 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 26, 1 (DESCEND): 13, 2 (PULL): 561
- aperture min -0.001 / max 0.076 / final 0.021
- closest |gripper_to_object| 0.013 at step 425 (final 0.033)
- closest |object_to_goal| 0.257 at step 57 (final 0.269); goal_error min 0.269 at step 51 / final 0.276 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.269 at the end); max rise 0.000 m

### env 7 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 20 steps; TERMINATED by `ee_ground_collision` at step 19
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 11, 1 (DESCEND): 7, 2 (PULL): 2
- aperture min 0.001 / max 0.076 / final 0.021
- closest |gripper_to_object| 0.054 at step 18 (final 0.066)
- closest |object_to_goal| 0.241 at step 17 (final 0.270); goal_error min 0.251 at step 4 / final 0.264 (success < 0.070)
- non-grasp task: object moved 0.010 m toward the goal (|object_to_goal| 0.270 at the end); max rise 0.001 m

### env 8 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 24 steps; TERMINATED by `ee_ground_collision` at step 23
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 18, 1 (DESCEND): 4, 2 (PULL): 2
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.060 at step 19 (final 0.083)
- closest |object_to_goal| 0.257 at step 0 (final 0.269); goal_error min 0.264 at step 4 / final 0.264 (success < 0.070)
- non-grasp task: object moved 0.000 m toward the goal (|object_to_goal| 0.269 at the end); max rise 0.000 m

### env 9 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 25, 1 (DESCEND): 13, 2 (PULL): 562
- aperture min -0.005 / max 0.076 / final 0.006
- closest |gripper_to_object| 0.014 at step 499 (final 0.022)
- closest |object_to_goal| 0.244 at step 175 (final 0.256); goal_error min 0.255 at step 87 / final 0.260 (success < 0.070)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.256 at the end); max rise 0.000 m

### env 10 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 7, 1 (DESCEND): 11, 2 (PULL): 582
- aperture min -0.005 / max 0.076 / final 0.006
- closest |gripper_to_object| 0.014 at step 276 (final 0.032)
- closest |object_to_goal| 0.263 at step 96 (final 0.271); goal_error min 0.274 at step 33 / final 0.282 (success < 0.070)
- non-grasp task: object moved 0.023 m toward the goal (|object_to_goal| 0.271 at the end); max rise 0.000 m

### env 11 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 10, 1 (DESCEND): 13, 2 (PULL): 577
- aperture min -0.007 / max 0.076 / final 0.015
- closest |gripper_to_object| 0.014 at step 416 (final 0.025)
- closest |object_to_goal| 0.229 at step 34 (final 0.245); goal_error min 0.240 at step 53 / final 0.248 (success < 0.070)
- non-grasp task: object moved 0.006 m toward the goal (|object_to_goal| 0.245 at the end); max rise 0.000 m

### env 12 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 8, 1 (DESCEND): 20, 2 (PULL): 572
- aperture min -0.001 / max 0.076 / final 0.020
- closest |gripper_to_object| 0.014 at step 583 (final 0.020)
- closest |object_to_goal| 0.200 at step 291 (final 0.208); goal_error min 0.211 at step 318 / final 0.216 (success < 0.070)
- non-grasp task: object moved 0.070 m toward the goal (|object_to_goal| 0.208 at the end); max rise 0.000 m

### env 13 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 16, 1 (DESCEND): 4, 2 (PULL): 2
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.051 at step 20 (final 0.061)
- closest |object_to_goal| 0.250 at step 1 (final 0.263); goal_error min 0.258 at step 5 / final 0.258 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.263 at the end); max rise 0.000 m

### env 14 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 9, 1 (DESCEND): 12, 2 (PULL): 579
- aperture min -0.003 / max 0.076 / final 0.021
- closest |gripper_to_object| 0.014 at step 411 (final 0.019)
- closest |object_to_goal| 0.239 at step 43 (final 0.265); goal_error min 0.249 at step 35 / final 0.256 (success < 0.070)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.265 at the end); max rise 0.000 m

### env 15 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 10, 1 (DESCEND): 11, 2 (PULL): 579
- aperture min -0.004 / max 0.076 / final 0.017
- closest |gripper_to_object| 0.013 at step 342 (final 0.035)
- closest |object_to_goal| 0.227 at step 78 (final 0.255); goal_error min 0.239 at step 34 / final 0.246 (success < 0.070)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.255 at the end); max rise 0.000 m

### env 16 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 9, 1 (DESCEND): 19, 2 (PULL): 572
- aperture min -0.003 / max 0.076 / final 0.024
- closest |gripper_to_object| 0.013 at step 548 (final 0.030)
- closest |object_to_goal| 0.244 at step 26 (final 0.253); goal_error min 0.255 at step 41 / final 0.263 (success < 0.070)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.253 at the end); max rise 0.000 m

### env 17 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 14, 1 (DESCEND): 6, 2 (PULL): 2
- aperture min -0.001 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.055 at step 20 (final 0.058)
- closest |object_to_goal| 0.272 at step 12 (final 0.293); goal_error min 0.284 at step 1 / final 0.284 (success < 0.070)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.293 at the end); max rise 0.000 m

### env 18 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 11, 1 (DESCEND): 19, 2 (PULL): 570
- aperture min -0.003 / max 0.076 / final 0.017
- closest |gripper_to_object| 0.014 at step 336 (final 0.016)
- closest |object_to_goal| 0.250 at step 12 (final 0.263); goal_error min 0.260 at step 49 / final 0.268 (success < 0.070)
- non-grasp task: object moved 0.002 m toward the goal (|object_to_goal| 0.263 at the end); max rise 0.000 m

### env 19 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 23 steps; TERMINATED by `ee_ground_collision` at step 22
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 12, 1 (DESCEND): 10, 2 (PULL): 1
- aperture min -0.003 / max 0.076 / final -0.003
- closest |gripper_to_object| 0.052 at step 21 (final 0.067)
- closest |object_to_goal| 0.281 at step 11 (final 0.294); goal_error min 0.286 at step 22 / final 0.286 (success < 0.070)
- non-grasp task: object moved 0.005 m toward the goal (|object_to_goal| 0.294 at the end); max rise 0.003 m

### env 20 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 7, 1 (DESCEND): 13, 2 (PULL): 580
- aperture min -0.003 / max 0.076 / final 0.012
- closest |gripper_to_object| 0.013 at step 131 (final 0.034)
- closest |object_to_goal| 0.233 at step 19 (final 0.253); goal_error min 0.244 at step 48 / final 0.251 (success < 0.070)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.253 at the end); max rise 0.000 m

### env 21 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 9, 1 (DESCEND): 18, 2 (PULL): 573
- aperture min -0.003 / max 0.076 / final 0.016
- closest |gripper_to_object| 0.013 at step 404 (final 0.026)
- closest |object_to_goal| 0.244 at step 122 (final 0.258); goal_error min 0.255 at step 36 / final 0.263 (success < 0.070)
- non-grasp task: object moved 0.018 m toward the goal (|object_to_goal| 0.258 at the end); max rise 0.000 m

### env 22 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 7, 1 (DESCEND): 11, 2 (PULL): 582
- aperture min -0.003 / max 0.076 / final 0.016
- closest |gripper_to_object| 0.013 at step 400 (final 0.022)
- closest |object_to_goal| 0.236 at step 79 (final 0.245); goal_error min 0.246 at step 33 / final 0.253 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.245 at the end); max rise 0.000 m

### env 23 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 27, 1 (DESCEND): 11, 2 (PULL): 562
- aperture min -0.004 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.013 at step 599 (final 0.013)
- closest |object_to_goal| 0.245 at step 78 (final 0.251); goal_error min 0.258 at step 82 / final 0.262 (success < 0.070)
- non-grasp task: object moved 0.027 m toward the goal (|object_to_goal| 0.251 at the end); max rise 0.000 m

### env 24 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 9, 1 (DESCEND): 11, 2 (PULL): 580
- aperture min -0.001 / max 0.076 / final 0.022
- closest |gripper_to_object| 0.014 at step 364 (final 0.027)
- closest |object_to_goal| 0.249 at step 277 (final 0.269); goal_error min 0.261 at step 111 / final 0.265 (success < 0.070)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.269 at the end); max rise 0.000 m

### env 25 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 11, 1 (DESCEND): 11, 2 (PULL): 578
- aperture min -0.003 / max 0.076 / final 0.004
- closest |gripper_to_object| 0.014 at step 497 (final 0.028)
- closest |object_to_goal| 0.189 at step 492 (final 0.212); goal_error min 0.199 at step 598 / final 0.199 (success < 0.070)
- non-grasp task: object moved 0.065 m toward the goal (|object_to_goal| 0.212 at the end); max rise 0.001 m

### env 26 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 12, 1 (DESCEND): 9, 2 (PULL): 579
- aperture min -0.004 / max 0.076 / final 0.012
- closest |gripper_to_object| 0.014 at step 376 (final 0.021)
- closest |object_to_goal| 0.290 at step 50 (final 0.310); goal_error min 0.303 at step 41 / final 0.308 (success < 0.070)
- non-grasp task: object moved 0.013 m toward the goal (|object_to_goal| 0.310 at the end); max rise 0.000 m

### env 27 — object_not_moved
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 7, 1 (DESCEND): 9, 2 (PULL): 584
- aperture min -0.004 / max 0.076 / final 0.014
- closest |gripper_to_object| 0.014 at step 54 (final 0.019)
- closest |object_to_goal| 0.216 at step 101 (final 0.228); goal_error min 0.226 at step 60 / final 0.234 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.228 at the end); max rise 0.000 m

### env 28 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 30, 1 (DESCEND): 13, 2 (PULL): 557
- aperture min -0.003 / max 0.076 / final 0.009
- closest |gripper_to_object| 0.014 at step 469 (final 0.017)
- closest |object_to_goal| 0.228 at step 61 (final 0.239); goal_error min 0.240 at step 66 / final 0.247 (success < 0.070)
- non-grasp task: object moved 0.031 m toward the goal (|object_to_goal| 0.239 at the end); max rise 0.000 m

### env 29 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 23 steps; TERMINATED by `ee_ground_collision` at step 22
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 16, 1 (DESCEND): 7
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.054 at step 22 (final 0.054)
- closest |object_to_goal| 0.270 at step 9 (final 0.273); goal_error min 0.280 at step 4 / final 0.280 (success < 0.070)
- non-grasp task: object moved 0.010 m toward the goal (|object_to_goal| 0.273 at the end); max rise 0.000 m

### env 30 — terminated:ee_ground_collision
- ended in phase 2 (PULL) after 19 steps; TERMINATED by `ee_ground_collision` at step 18
- most steps in phase 0 (ALIGN); steps per phase 0 (ALIGN): 11, 1 (DESCEND): 7, 2 (PULL): 1
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.052 at step 18 (final 0.052)
- closest |object_to_goal| 0.279 at step 18 (final 0.279); goal_error min 0.286 at step 3 / final 0.286 (success < 0.070)
- non-grasp task: object moved 0.009 m toward the goal (|object_to_goal| 0.279 at the end); max rise 0.000 m

### env 31 — object_moved_short
- ended in phase 2 (PULL) after 600 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (ALIGN): 26, 1 (DESCEND): 11, 2 (PULL): 563
- aperture min -0.002 / max 0.076 / final 0.016
- closest |gripper_to_object| 0.014 at step 591 (final 0.031)
- closest |object_to_goal| 0.194 at step 117 (final 0.203); goal_error min 0.204 at step 327 / final 0.209 (success < 0.070)
- non-grasp task: object moved 0.050 m toward the goal (|object_to_goal| 0.203 at the end); max rise 0.000 m

## Successful envs (one line each)

