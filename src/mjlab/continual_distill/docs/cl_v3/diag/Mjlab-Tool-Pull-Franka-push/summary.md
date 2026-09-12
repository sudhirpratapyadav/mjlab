# diagnose — Mjlab-Tool-Pull-Franka

2026-09-10T02:30:45 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 600 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToolPullClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = CHECK, 4 = CLEAR, 5 = ADVANCE, 6 = SWEEP, 7 = PULL, 8 = RELEASE, 9 = PUSH_HOVER, 10 = PUSH_DESCEND, 11 = PUSH, 12 = DONE

## Histograms

- end phase, FAILING envs (32): 11 (PUSH): 25, 9 (PUSH_HOVER): 5, 10 (PUSH_DESCEND): 2
- end phase, successful envs (0): —
- most-steps phase, failing envs: 11 (PUSH): 22, 9 (PUSH_HOVER): 9, 10 (PUSH_DESCEND): 1
- failure class: object_moved_short: 21, object_not_moved: 7, terminated:ee_ground_collision: 4
- termination among failing envs: none: 28, ee_ground_collision: 4

## Reading

- **object_moved_short** — 21/32 failing envs (envs [0, 1, 2, 4, 6, 8, 10, 11, 12, 14, 16, 19, 20, 21, 22, 23, 24, 25, 28, 29, 30]); typical end phase 11 (PUSH); final aperture median 0.000; closest |gripper_to_object| median 0.017; closest |object_to_goal| median 0.214; min goal_error median 0.224 (success < 0.070)
- **object_not_moved** — 7/32 failing envs (envs [3, 7, 9, 13, 15, 26, 27]); typical end phase 11 (PUSH); final aperture median 0.000; closest |gripper_to_object| median 0.016; closest |object_to_goal| median 0.271; min goal_error median 0.282 (success < 0.070)
- **terminated:ee_ground_collision** — 4/32 failing envs (envs [5, 17, 18, 31]); typical end phase 11 (PUSH); final aperture median 0.001; closest |gripper_to_object| median 0.057; closest |object_to_goal| median 0.252; min goal_error median 0.264 (success < 0.070)

## Failing envs

### env 0 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 21, 10 (PUSH_DESCEND): 29, 11 (PUSH): 550
- aperture min -0.004 / max 0.076 / final 0.013
- closest |gripper_to_object| 0.018 at step 408 (final 0.025)
- closest |object_to_goal| 0.179 at step 513 (final 0.193); goal_error min 0.191 at step 560 / final 0.191 (success < 0.070)
- non-grasp task: object moved 0.033 m toward the goal (|object_to_goal| 0.193 at the end); max rise 0.013 m

### env 1 — object_moved_short
- ended in phase 10 (PUSH_DESCEND) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 159, 10 (PUSH_DESCEND): 139, 11 (PUSH): 302
- aperture min -0.003 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 481 (final 0.112)
- closest |object_to_goal| 0.233 at step 468 (final 0.240); goal_error min 0.243 at step 580 / final 0.243 (success < 0.070)
- non-grasp task: object moved 0.025 m toward the goal (|object_to_goal| 0.240 at the end); max rise 0.000 m

### env 2 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 163, 10 (PUSH_DESCEND): 150, 11 (PUSH): 287
- aperture min -0.006 / max 0.076 / final 0.016
- closest |gripper_to_object| 0.015 at step 93 (final 0.023)
- closest |object_to_goal| 0.225 at step 578 (final 0.238); goal_error min 0.234 at step 594 / final 0.234 (success < 0.070)
- non-grasp task: object moved 0.032 m toward the goal (|object_to_goal| 0.238 at the end); max rise 0.002 m

### env 3 — object_not_moved
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 238, 10 (PUSH_DESCEND): 173, 11 (PUSH): 189
- aperture min -0.002 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.016 at step 268 (final 0.051)
- closest |object_to_goal| 0.276 at step 541 (final 0.277); goal_error min 0.287 at step 567 / final 0.287 (success < 0.070)
- non-grasp task: object moved 0.017 m toward the goal (|object_to_goal| 0.277 at the end); max rise 0.000 m

### env 4 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 18, 10 (PUSH_DESCEND): 27, 11 (PUSH): 555
- aperture min -0.005 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.017 at step 295 (final 0.040)
- closest |object_to_goal| 0.147 at step 548 (final 0.161); goal_error min 0.156 at step 597 / final 0.156 (success < 0.070)
- non-grasp task: object moved 0.083 m toward the goal (|object_to_goal| 0.161 at the end); max rise 0.000 m

### env 5 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 21 steps; TERMINATED by `ee_ground_collision` at step 20
- most steps in phase 10 (PUSH_DESCEND); steps per phase 9 (PUSH_HOVER): 8, 10 (PUSH_DESCEND): 10, 11 (PUSH): 3
- aperture min 0.001 / max 0.076 / final 0.003
- closest |gripper_to_object| 0.054 at step 17 (final 0.063)
- closest |object_to_goal| 0.259 at step 6 (final 0.267); goal_error min 0.271 at step 1 / final 0.271 (success < 0.070)
- non-grasp task: object moved 0.018 m toward the goal (|object_to_goal| 0.267 at the end); max rise 0.000 m

### env 6 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 34, 10 (PUSH_DESCEND): 38, 11 (PUSH): 528
- aperture min -0.005 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.017 at step 522 (final 0.037)
- closest |object_to_goal| 0.228 at step 578 (final 0.244); goal_error min 0.239 at step 589 / final 0.239 (success < 0.070)
- non-grasp task: object moved 0.023 m toward the goal (|object_to_goal| 0.244 at the end); max rise 0.000 m

### env 7 — object_not_moved
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 202, 10 (PUSH_DESCEND): 176, 11 (PUSH): 222
- aperture min -0.004 / max 0.076 / final -0.004
- closest |gripper_to_object| 0.018 at step 428 (final 0.030)
- closest |object_to_goal| 0.252 at step 537 (final 0.265); goal_error min 0.261 at step 489 / final 0.262 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.265 at the end); max rise 0.000 m

### env 8 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 40, 10 (PUSH_DESCEND): 24, 11 (PUSH): 536
- aperture min -0.002 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 141 (final 0.030)
- closest |object_to_goal| 0.201 at step 440 (final 0.211); goal_error min 0.213 at step 568 / final 0.213 (success < 0.070)
- non-grasp task: object moved 0.091 m toward the goal (|object_to_goal| 0.211 at the end); max rise 0.000 m

### env 9 — object_not_moved
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 65, 10 (PUSH_DESCEND): 82, 11 (PUSH): 453
- aperture min -0.002 / max 0.076 / final 0.007
- closest |gripper_to_object| 0.015 at step 468 (final 0.035)
- closest |object_to_goal| 0.215 at step 593 (final 0.230); goal_error min 0.228 at step 596 / final 0.228 (success < 0.070)
- non-grasp task: object moved 0.018 m toward the goal (|object_to_goal| 0.230 at the end); max rise 0.000 m

### env 10 — object_moved_short
- ended in phase 9 (PUSH_HOVER) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 175, 10 (PUSH_DESCEND): 148, 11 (PUSH): 277
- aperture min -0.006 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.017 at step 143 (final 0.034)
- closest |object_to_goal| 0.263 at step 591 (final 0.273); goal_error min 0.273 at step 596 / final 0.273 (success < 0.070)
- non-grasp task: object moved 0.038 m toward the goal (|object_to_goal| 0.273 at the end); max rise 0.000 m

### env 11 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 141, 10 (PUSH_DESCEND): 145, 11 (PUSH): 314
- aperture min -0.005 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.016 at step 217 (final 0.034)
- closest |object_to_goal| 0.218 at step 588 (final 0.236); goal_error min 0.229 at step 592 / final 0.229 (success < 0.070)
- non-grasp task: object moved 0.028 m toward the goal (|object_to_goal| 0.236 at the end); max rise 0.000 m

### env 12 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 35, 10 (PUSH_DESCEND): 39, 11 (PUSH): 526
- aperture min -0.003 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.019 at step 479 (final 0.042)
- closest |object_to_goal| 0.198 at step 568 (final 0.214); goal_error min 0.208 at step 577 / final 0.209 (success < 0.070)
- non-grasp task: object moved 0.087 m toward the goal (|object_to_goal| 0.214 at the end); max rise 0.000 m

### env 13 — object_not_moved
- ended in phase 9 (PUSH_HOVER) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 179, 10 (PUSH_DESCEND): 174, 11 (PUSH): 247
- aperture min -0.006 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.017 at step 239 (final 0.020)
- closest |object_to_goal| 0.232 at step 584 (final 0.235); goal_error min 0.241 at step 599 / final 0.241 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.235 at the end); max rise 0.001 m

### env 14 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 46, 10 (PUSH_DESCEND): 57, 11 (PUSH): 497
- aperture min -0.005 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.016 at step 183 (final 0.035)
- closest |object_to_goal| 0.202 at step 457 (final 0.214); goal_error min 0.208 at step 599 / final 0.208 (success < 0.070)
- non-grasp task: object moved 0.025 m toward the goal (|object_to_goal| 0.214 at the end); max rise 0.000 m

### env 15 — object_not_moved
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 199, 10 (PUSH_DESCEND): 165, 11 (PUSH): 236
- aperture min -0.004 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.014 at step 522 (final 0.037)
- closest |object_to_goal| 0.271 at step 587 (final 0.288); goal_error min 0.282 at step 571 / final 0.283 (success < 0.070)
- non-grasp task: object moved 0.020 m toward the goal (|object_to_goal| 0.288 at the end); max rise 0.001 m

### env 16 — object_moved_short
- ended in phase 10 (PUSH_DESCEND) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 240, 10 (PUSH_DESCEND): 184, 11 (PUSH): 176
- aperture min -0.005 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 414 (final 0.096)
- closest |object_to_goal| 0.269 at step 590 (final 0.278); goal_error min 0.282 at step 580 / final 0.282 (success < 0.070)
- non-grasp task: object moved 0.023 m toward the goal (|object_to_goal| 0.278 at the end); max rise 0.001 m

### env 17 — terminated:ee_ground_collision
- ended in phase 9 (PUSH_HOVER) after 22 steps; TERMINATED by `ee_ground_collision` at step 21
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 15, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.059 at step 17 (final 0.066)
- closest |object_to_goal| 0.246 at step 4 (final 0.259); goal_error min 0.257 at step 5 / final 0.257 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.259 at the end); max rise 0.000 m

### env 18 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 19 steps; TERMINATED by `ee_ground_collision` at step 18
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 12, 10 (PUSH_DESCEND): 6, 11 (PUSH): 1
- aperture min 0.001 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.059 at step 16 (final 0.066)
- closest |object_to_goal| 0.231 at step 18 (final 0.231); goal_error min 0.241 at step 1 / final 0.241 (success < 0.070)
- non-grasp task: object moved 0.002 m toward the goal (|object_to_goal| 0.231 at the end); max rise 0.000 m

### env 19 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 40, 10 (PUSH_DESCEND): 36, 11 (PUSH): 524
- aperture min -0.002 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.017 at step 270 (final 0.045)
- closest |object_to_goal| 0.159 at step 578 (final 0.177); goal_error min 0.169 at step 589 / final 0.169 (success < 0.070)
- non-grasp task: object moved 0.113 m toward the goal (|object_to_goal| 0.177 at the end); max rise 0.000 m

### env 20 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 77, 10 (PUSH_DESCEND): 66, 11 (PUSH): 457
- aperture min -0.004 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.016 at step 282 (final 0.032)
- closest |object_to_goal| 0.216 at step 581 (final 0.223); goal_error min 0.228 at step 591 / final 0.228 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.223 at the end); max rise 0.000 m

### env 21 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 165, 10 (PUSH_DESCEND): 181, 11 (PUSH): 254
- aperture min -0.004 / max 0.076 / final 0.001
- closest |gripper_to_object| 0.018 at step 84 (final 0.035)
- closest |object_to_goal| 0.214 at step 584 (final 0.227); goal_error min 0.224 at step 588 / final 0.225 (success < 0.070)
- non-grasp task: object moved 0.036 m toward the goal (|object_to_goal| 0.227 at the end); max rise 0.000 m

### env 22 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 133, 10 (PUSH_DESCEND): 117, 11 (PUSH): 350
- aperture min -0.006 / max 0.076 / final -0.001
- closest |gripper_to_object| 0.017 at step 353 (final 0.038)
- closest |object_to_goal| 0.256 at step 483 (final 0.260); goal_error min 0.267 at step 599 / final 0.267 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.260 at the end); max rise 0.000 m

### env 23 — object_moved_short
- ended in phase 9 (PUSH_HOVER) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 223, 10 (PUSH_DESCEND): 165, 11 (PUSH): 212
- aperture min -0.006 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.016 at step 276 (final 0.093)
- closest |object_to_goal| 0.272 at step 466 (final 0.291); goal_error min 0.283 at step 589 / final 0.284 (success < 0.070)
- non-grasp task: object moved 0.020 m toward the goal (|object_to_goal| 0.291 at the end); max rise 0.000 m

### env 24 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 53, 10 (PUSH_DESCEND): 42, 11 (PUSH): 505
- aperture min -0.005 / max 0.076 / final -0.000
- closest |gripper_to_object| 0.018 at step 275 (final 0.035)
- closest |object_to_goal| 0.189 at step 543 (final 0.193); goal_error min 0.200 at step 582 / final 0.201 (success < 0.070)
- non-grasp task: object moved 0.047 m toward the goal (|object_to_goal| 0.193 at the end); max rise 0.000 m

### env 25 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 84, 10 (PUSH_DESCEND): 97, 11 (PUSH): 419
- aperture min -0.006 / max 0.076 / final -0.002
- closest |gripper_to_object| 0.016 at step 296 (final 0.034)
- closest |object_to_goal| 0.207 at step 595 (final 0.226); goal_error min 0.219 at step 595 / final 0.220 (success < 0.070)
- non-grasp task: object moved 0.056 m toward the goal (|object_to_goal| 0.226 at the end); max rise 0.000 m

### env 26 — object_not_moved
- ended in phase 9 (PUSH_HOVER) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 234, 10 (PUSH_DESCEND): 183, 11 (PUSH): 183
- aperture min -0.004 / max 0.076 / final 0.004
- closest |gripper_to_object| 0.018 at step 482 (final 0.020)
- closest |object_to_goal| 0.278 at step 479 (final 0.292); goal_error min 0.289 at step 599 / final 0.289 (success < 0.070)
- non-grasp task: object moved 0.013 m toward the goal (|object_to_goal| 0.292 at the end); max rise 0.000 m

### env 27 — object_not_moved
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 239, 10 (PUSH_DESCEND): 174, 11 (PUSH): 187
- aperture min -0.005 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.016 at step 142 (final 0.033)
- closest |object_to_goal| 0.280 at step 596 (final 0.291); goal_error min 0.293 at step 533 / final 0.294 (success < 0.070)
- non-grasp task: object moved 0.013 m toward the goal (|object_to_goal| 0.291 at the end); max rise 0.000 m

### env 28 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 230, 10 (PUSH_DESCEND): 168, 11 (PUSH): 202
- aperture min -0.006 / max 0.076 / final 0.010
- closest |gripper_to_object| 0.016 at step 472 (final 0.033)
- closest |object_to_goal| 0.276 at step 580 (final 0.297); goal_error min 0.289 at step 567 / final 0.290 (success < 0.070)
- non-grasp task: object moved 0.022 m toward the goal (|object_to_goal| 0.297 at the end); max rise 0.000 m

### env 29 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 42, 10 (PUSH_DESCEND): 26, 11 (PUSH): 532
- aperture min -0.002 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 532 (final 0.022)
- closest |object_to_goal| 0.186 at step 507 (final 0.199); goal_error min 0.196 at step 566 / final 0.196 (success < 0.070)
- non-grasp task: object moved 0.084 m toward the goal (|object_to_goal| 0.199 at the end); max rise 0.000 m

### env 30 — object_moved_short
- ended in phase 11 (PUSH) after 600 steps; ran to time-out
- most steps in phase 11 (PUSH); steps per phase 9 (PUSH_HOVER): 54, 10 (PUSH_DESCEND): 80, 11 (PUSH): 466
- aperture min -0.004 / max 0.076 / final 0.002
- closest |gripper_to_object| 0.017 at step 50 (final 0.035)
- closest |object_to_goal| 0.195 at step 594 (final 0.209); goal_error min 0.207 at step 578 / final 0.207 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.209 at the end); max rise 0.000 m

### env 31 — terminated:ee_ground_collision
- ended in phase 11 (PUSH) after 21 steps; TERMINATED by `ee_ground_collision` at step 20
- most steps in phase 9 (PUSH_HOVER); steps per phase 9 (PUSH_HOVER): 12, 10 (PUSH_DESCEND): 8, 11 (PUSH): 1
- aperture min 0.000 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.053 at step 19 (final 0.064)
- closest |object_to_goal| 0.275 at step 17 (final 0.276); goal_error min 0.286 at step 4 / final 0.286 (success < 0.070)
- non-grasp task: object moved 0.006 m toward the goal (|object_to_goal| 0.276 at the end); max rise 0.000 m

## Successful envs (one line each)

