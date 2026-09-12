# diagnose — Mjlab-Tool-Pull-Franka

2026-09-09T21:07:42 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 600 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `ToolPullClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = CHECK, 4 = CLEAR, 5 = ADVANCE, 6 = SWEEP, 7 = PULL, 8 = RELEASE, 9 = PUSH_HOVER, 10 = PUSH_DESCEND, 11 = PUSH, 12 = DONE

## Histograms

- end phase, FAILING envs (32): 1 (DESCEND): 32
- end phase, successful envs (0): —
- most-steps phase, failing envs: 0 (HOVER): 32
- failure class: terminated:ee_ground_collision: 32
- termination among failing envs: ee_ground_collision: 32

## Reading

- **terminated:ee_ground_collision** — 32/32 failing envs (envs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]); typical end phase 1 (DESCEND); final aperture median 0.080; closest |gripper_to_object| median 0.245; closest |object_to_goal| median 0.250; min goal_error median 0.261 (success < 0.070)

## Failing envs

### env 0 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 62 steps; TERMINATED by `ee_ground_collision` at step 61
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 46, 1 (DESCEND): 16
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.270 at step 30 (final 0.336)
- closest |object_to_goal| 0.236 at step 50 (final 0.242); goal_error min 0.246 at step 4 / final 0.246 (success < 0.070)
- non-grasp task: object moved 0.004 m toward the goal (|object_to_goal| 0.242 at the end); max rise 0.000 m

### env 1 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 47, 1 (DESCEND): 16
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.321 at step 60 (final 0.326)
- closest |object_to_goal| 0.278 at step 40 (final 0.286); goal_error min 0.291 at step 3 / final 0.291 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.286 at the end); max rise 0.000 m

### env 2 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 20
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.306 at step 15 (final 0.383)
- closest |object_to_goal| 0.286 at step 17 (final 0.293); goal_error min 0.295 at step 3 / final 0.295 (success < 0.070)
- non-grasp task: object moved 0.009 m toward the goal (|object_to_goal| 0.293 at the end); max rise 0.000 m

### env 3 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 53 steps; TERMINATED by `ee_ground_collision` at step 52
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 35, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.200 at step 45 (final 0.234)
- closest |object_to_goal| 0.248 at step 47 (final 0.265); goal_error min 0.259 at step 4 / final 0.259 (success < 0.070)
- non-grasp task: object moved 0.018 m toward the goal (|object_to_goal| 0.265 at the end); max rise 0.000 m

### env 4 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 32, 1 (DESCEND): 24
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.213 at step 9 (final 0.310)
- closest |object_to_goal| 0.223 at step 42 (final 0.241); goal_error min 0.233 at step 1 / final 0.233 (success < 0.070)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.241 at the end); max rise 0.000 m

### env 5 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 62 steps; TERMINATED by `ee_ground_collision` at step 61
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 41, 1 (DESCEND): 21
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.223 at step 18 (final 0.324)
- closest |object_to_goal| 0.201 at step 43 (final 0.216); goal_error min 0.212 at step 4 / final 0.212 (success < 0.070)
- non-grasp task: object moved 0.007 m toward the goal (|object_to_goal| 0.216 at the end); max rise 0.000 m

### env 6 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 64 steps; TERMINATED by `ee_ground_collision` at step 63
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 38, 1 (DESCEND): 26
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.280 at step 20 (final 0.360)
- closest |object_to_goal| 0.245 at step 22 (final 0.268); goal_error min 0.257 at step 1 / final 0.257 (success < 0.070)
- non-grasp task: object moved 0.012 m toward the goal (|object_to_goal| 0.268 at the end); max rise 0.000 m

### env 7 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 32, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.220 at step 14 (final 0.241)
- closest |object_to_goal| 0.229 at step 24 (final 0.235); goal_error min 0.240 at step 5 / final 0.240 (success < 0.070)
- non-grasp task: object moved 0.000 m toward the goal (|object_to_goal| 0.235 at the end); max rise 0.000 m

### env 8 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 40, 1 (DESCEND): 20
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.226 at step 15 (final 0.405)
- closest |object_to_goal| 0.253 at step 48 (final 0.260); goal_error min 0.264 at step 3 / final 0.264 (success < 0.070)
- non-grasp task: object moved 0.006 m toward the goal (|object_to_goal| 0.260 at the end); max rise 0.000 m

### env 9 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.170 at step 13 (final 0.370)
- closest |object_to_goal| 0.259 at step 37 (final 0.264); goal_error min 0.270 at step 4 / final 0.270 (success < 0.070)
- non-grasp task: object moved 0.010 m toward the goal (|object_to_goal| 0.264 at the end); max rise 0.000 m

### env 10 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 54 steps; TERMINATED by `ee_ground_collision` at step 53
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 36, 1 (DESCEND): 18
- aperture min 0.076 / max 0.083 / final 0.082
- closest |gripper_to_object| 0.215 at step 15 (final 0.346)
- closest |object_to_goal| 0.233 at step 39 (final 0.239); goal_error min 0.243 at step 5 / final 0.243 (success < 0.070)
- non-grasp task: object moved 0.013 m toward the goal (|object_to_goal| 0.239 at the end); max rise 0.000 m

### env 11 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 17
- aperture min 0.076 / max 0.083 / final 0.083
- closest |gripper_to_object| 0.202 at step 16 (final 0.356)
- closest |object_to_goal| 0.272 at step 20 (final 0.283); goal_error min 0.283 at step 4 / final 0.283 (success < 0.070)
- non-grasp task: object moved 0.006 m toward the goal (|object_to_goal| 0.283 at the end); max rise 0.000 m

### env 12 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 21
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.234 at step 25 (final 0.263)
- closest |object_to_goal| 0.248 at step 6 (final 0.263); goal_error min 0.257 at step 4 / final 0.257 (success < 0.070)
- non-grasp task: object moved 0.015 m toward the goal (|object_to_goal| 0.263 at the end); max rise 0.000 m

### env 13 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 71 steps; TERMINATED by `ee_ground_collision` at step 70
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 53, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.392 at step 28 (final 0.407)
- closest |object_to_goal| 0.293 at step 35 (final 0.312); goal_error min 0.305 at step 1 / final 0.305 (success < 0.070)
- non-grasp task: object moved 0.022 m toward the goal (|object_to_goal| 0.312 at the end); max rise 0.000 m

### env 14 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 41 steps; TERMINATED by `ee_ground_collision` at step 40
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 26, 1 (DESCEND): 15
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.270 at step 37 (final 0.281)
- closest |object_to_goal| 0.266 at step 7 (final 0.280); goal_error min 0.276 at step 4 / final 0.276 (success < 0.070)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.280 at the end); max rise 0.000 m

### env 15 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 65 steps; TERMINATED by `ee_ground_collision` at step 64
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 47, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.350 at step 64 (final 0.350)
- closest |object_to_goal| 0.262 at step 58 (final 0.266); goal_error min 0.274 at step 5 / final 0.274 (success < 0.070)
- non-grasp task: object moved 0.009 m toward the goal (|object_to_goal| 0.266 at the end); max rise 0.000 m

### env 16 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.300 at step 43 (final 0.303)
- closest |object_to_goal| 0.235 at step 0 (final 0.239); goal_error min 0.246 at step 1 / final 0.246 (success < 0.070)
- non-grasp task: object moved 0.000 m toward the goal (|object_to_goal| 0.239 at the end); max rise 0.000 m

### env 17 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 33, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.411 at step 38 (final 0.443)
- closest |object_to_goal| 0.281 at step 17 (final 0.289); goal_error min 0.293 at step 4 / final 0.293 (success < 0.070)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.289 at the end); max rise 0.000 m

### env 18 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 75 steps; TERMINATED by `ee_ground_collision` at step 74
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 56, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.257 at step 29 (final 0.308)
- closest |object_to_goal| 0.223 at step 25 (final 0.233); goal_error min 0.236 at step 4 / final 0.236 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.233 at the end); max rise 0.000 m

### env 19 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 42, 1 (DESCEND): 16
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.217 at step 56 (final 0.227)
- closest |object_to_goal| 0.221 at step 49 (final 0.242); goal_error min 0.233 at step 5 / final 0.233 (success < 0.070)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.242 at the end); max rise 0.000 m

### env 20 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 21
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.224 at step 16 (final 0.327)
- closest |object_to_goal| 0.210 at step 55 (final 0.225); goal_error min 0.221 at step 5 / final 0.221 (success < 0.070)
- non-grasp task: object moved 0.021 m toward the goal (|object_to_goal| 0.225 at the end); max rise 0.000 m

### env 21 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 58 steps; TERMINATED by `ee_ground_collision` at step 57
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.205 at step 55 (final 0.211)
- closest |object_to_goal| 0.211 at step 3 (final 0.212); goal_error min 0.222 at step 4 / final 0.222 (success < 0.070)
- non-grasp task: object moved 0.002 m toward the goal (|object_to_goal| 0.212 at the end); max rise 0.000 m

### env 22 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 23
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.315 at step 13 (final 0.442)
- closest |object_to_goal| 0.269 at step 33 (final 0.287); goal_error min 0.280 at step 1 / final 0.280 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.287 at the end); max rise 0.000 m

### env 23 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 55 steps; TERMINATED by `ee_ground_collision` at step 54
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 16
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.207 at step 19 (final 0.319)
- closest |object_to_goal| 0.252 at step 49 (final 0.264); goal_error min 0.262 at step 3 / final 0.262 (success < 0.070)
- non-grasp task: object moved 0.002 m toward the goal (|object_to_goal| 0.264 at the end); max rise 0.000 m

### env 24 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 60 steps; TERMINATED by `ee_ground_collision` at step 59
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 41, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.323 at step 56 (final 0.329)
- closest |object_to_goal| 0.276 at step 20 (final 0.287); goal_error min 0.288 at step 4 / final 0.288 (success < 0.070)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.287 at the end); max rise 0.000 m

### env 25 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 55 steps; TERMINATED by `ee_ground_collision` at step 54
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 16
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.304 at step 19 (final 0.385)
- closest |object_to_goal| 0.243 at step 51 (final 0.255); goal_error min 0.255 at step 4 / final 0.255 (success < 0.070)
- non-grasp task: object moved 0.023 m toward the goal (|object_to_goal| 0.255 at the end); max rise 0.000 m

### env 26 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 51 steps; TERMINATED by `ee_ground_collision` at step 50
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 33, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.227 at step 14 (final 0.275)
- closest |object_to_goal| 0.232 at step 20 (final 0.243); goal_error min 0.244 at step 5 / final 0.244 (success < 0.070)
- non-grasp task: object moved 0.010 m toward the goal (|object_to_goal| 0.243 at the end); max rise 0.000 m

### env 27 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 56 steps; TERMINATED by `ee_ground_collision` at step 55
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39, 1 (DESCEND): 17
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.259 at step 51 (final 0.278)
- closest |object_to_goal| 0.292 at step 43 (final 0.303); goal_error min 0.303 at step 3 / final 0.303 (success < 0.070)
- non-grasp task: object moved 0.011 m toward the goal (|object_to_goal| 0.303 at the end); max rise 0.000 m

### env 28 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 63 steps; TERMINATED by `ee_ground_collision` at step 62
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 26
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.284 at step 18 (final 0.362)
- closest |object_to_goal| 0.276 at step 33 (final 0.286); goal_error min 0.286 at step 3 / final 0.286 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.286 at the end); max rise 0.000 m

### env 29 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 54 steps; TERMINATED by `ee_ground_collision` at step 53
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37, 1 (DESCEND): 17
- aperture min 0.076 / max 0.084 / final 0.083
- closest |gripper_to_object| 0.259 at step 10 (final 0.401)
- closest |object_to_goal| 0.283 at step 13 (final 0.301); goal_error min 0.293 at step 4 / final 0.293 (success < 0.070)
- non-grasp task: object moved 0.016 m toward the goal (|object_to_goal| 0.301 at the end); max rise 0.000 m

### env 30 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 62 steps; TERMINATED by `ee_ground_collision` at step 61
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 44, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.173 at step 59 (final 0.182)
- closest |object_to_goal| 0.209 at step 47 (final 0.219); goal_error min 0.221 at step 5 / final 0.221 (success < 0.070)
- non-grasp task: object moved 0.008 m toward the goal (|object_to_goal| 0.219 at the end); max rise 0.000 m

### env 31 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 57 steps; TERMINATED by `ee_ground_collision` at step 56
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 42, 1 (DESCEND): 15
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.201 at step 13 (final 0.330)
- closest |object_to_goal| 0.262 at step 32 (final 0.270); goal_error min 0.274 at step 1 / final 0.274 (success < 0.070)
- non-grasp task: object moved 0.014 m toward the goal (|object_to_goal| 0.270 at the end); max rise 0.000 m

## Successful envs (one line each)

