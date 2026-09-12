# diagnose — Mjlab-Lift-Cube-Franka

2026-09-09T15:04:53 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 1000 · device cuda:0  
**97/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `LiftCubeClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = LIFT

## Histograms

- end phase, FAILING envs (31): 2 (CLOSE): 14, 1 (DESCEND): 8, 0 (HOVER): 7, 3 (LIFT): 2
- end phase, successful envs (97): 3 (LIFT): 81, 1 (DESCEND): 9, 2 (CLOSE): 7
- most-steps phase, failing envs: 0 (HOVER): 26, 1 (DESCEND): 5
- failure class: terminated:ee_ground_collision: 26, never_lifted: 5
- termination among failing envs: ee_ground_collision: 26, none: 5
- failing envs that lifted the object: 0/31; lifted then lost: 0
- success step (successful envs): median 196, max 369

## Reading

- **terminated:ee_ground_collision** — 26/31 failing envs (envs [4, 9, 11, 15, 25, 33, 40, 46, 48, 51, 55, 67, 68, 71, 73, 78, 82, 84, 89, 91, 100, 102, 105, 106, 108, 120]); typical end phase 2 (CLOSE); final aperture median 0.080; closest |gripper_to_object| median 0.010; closest |object_to_goal| median 0.286; min goal_error median 0.291 (success < 0.050)
- **never_lifted** — 5/31 failing envs (envs [1, 30, 74, 77, 110]); typical end phase 3 (LIFT); final aperture median 0.080; closest |gripper_to_object| median 0.018; closest |object_to_goal| median 0.271; min goal_error median 0.283 (success < 0.050)

## Failing envs

### env 1 — never_lifted
- ended in phase 0 (HOVER) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 271, 1 (DESCEND): 497, 2 (CLOSE): 75, 3 (LIFT): 157
- aperture min 0.000 / max 0.085 / final 0.080
- closest |gripper_to_object| 0.017 at step 99 (final 0.233)
- closest |object_to_goal| 0.218 at step 69 (final 0.288); goal_error min 0.227 at step 1 / final 0.283 (success < 0.050)
- object never lifted (max rise 0.008 m)

### env 4 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 93 steps; TERMINATED by `ee_ground_collision` at step 92
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 22
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.008 at step 91 (final 0.015)
- closest |object_to_goal| 0.375 at step 1 (final 0.389); goal_error min 0.383 at step 1 / final 0.393 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 9 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 90 steps; TERMINATED by `ee_ground_collision` at step 89
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 18, 2 (CLOSE): 1
- aperture min 0.076 / max 0.083 / final 0.081
- closest |gripper_to_object| 0.012 at step 89 (final 0.012)
- closest |object_to_goal| 0.112 at step 42 (final 0.127); goal_error min 0.116 at step 1 / final 0.125 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 11 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 82 steps; TERMINATED by `ee_ground_collision` at step 81
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 58, 1 (DESCEND): 22, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.007 at step 81 (final 0.007)
- closest |object_to_goal| 0.183 at step 18 (final 0.187); goal_error min 0.182 at step 1 / final 0.193 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 15 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 91 steps; TERMINATED by `ee_ground_collision` at step 90
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 20
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.013 at step 89 (final 0.018)
- closest |object_to_goal| 0.424 at step 1 (final 0.433); goal_error min 0.431 at step 1 / final 0.436 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 25 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 91 steps; TERMINATED by `ee_ground_collision` at step 90
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 18, 2 (CLOSE): 2
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.003 at step 88 (final 0.014)
- closest |object_to_goal| 0.351 at step 0 (final 0.368); goal_error min 0.353 at step 1 / final 0.369 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 30 — never_lifted
- ended in phase 3 (LIFT) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 186, 1 (DESCEND): 605, 2 (CLOSE): 75, 3 (LIFT): 134
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.025 at step 213 (final 0.116)
- closest |object_to_goal| 0.351 at step 0 (final 0.412); goal_error min 0.354 at step 1 / final 0.408 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 33 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 50 steps; TERMINATED by `ee_ground_collision` at step 49
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 50
- aperture min 0.063 / max 0.080 / final 0.074
- closest |gripper_to_object| 0.059 at step 48 (final 0.066)
- closest |object_to_goal| 0.074 at step 0 (final 0.119); goal_error min 0.072 at step 1 / final 0.116 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 40 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 94 steps; TERMINATED by `ee_ground_collision` at step 93
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 21, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 92 (final 0.018)
- closest |object_to_goal| 0.158 at step 1 (final 0.195); goal_error min 0.168 at step 1 / final 0.185 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 46 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 89 steps; TERMINATED by `ee_ground_collision` at step 88
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 15, 2 (CLOSE): 3
- aperture min 0.068 / max 0.080 / final 0.068
- closest |gripper_to_object| 0.007 at step 88 (final 0.007)
- closest |object_to_goal| 0.290 at step 0 (final 0.305); goal_error min 0.294 at step 1 / final 0.307 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 48 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 89 steps; TERMINATED by `ee_ground_collision` at step 88
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 18
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.012 at step 88 (final 0.012)
- closest |object_to_goal| 0.325 at step 6 (final 0.330); goal_error min 0.329 at step 1 / final 0.337 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 51 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 92 steps; TERMINATED by `ee_ground_collision` at step 91
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 68, 1 (DESCEND): 22, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.006 at step 89 (final 0.022)
- closest |object_to_goal| 0.084 at step 0 (final 0.109); goal_error min 0.089 at step 1 / final 0.105 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 55 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 92 steps; TERMINATED by `ee_ground_collision` at step 91
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19, 2 (CLOSE): 2
- aperture min 0.076 / max 0.083 / final 0.083
- closest |gripper_to_object| 0.009 at step 88 (final 0.015)
- closest |object_to_goal| 0.541 at step 87 (final 0.556); goal_error min 0.550 at step 1 / final 0.554 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 67 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 50 steps; TERMINATED by `ee_ground_collision` at step 49
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 50
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.096 at step 49 (final 0.096)
- closest |object_to_goal| 0.256 at step 8 (final 0.271); goal_error min 0.269 at step 1 / final 0.269 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 68 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 39 steps; TERMINATED by `ee_ground_collision` at step 38
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 39
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.167 at step 36 (final 0.169)
- closest |object_to_goal| 0.325 at step 1 (final 0.342); goal_error min 0.335 at step 1 / final 0.341 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 71 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 93 steps; TERMINATED by `ee_ground_collision` at step 92
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 20, 2 (CLOSE): 2
- aperture min 0.076 / max 0.083 / final 0.080
- closest |gripper_to_object| 0.002 at step 92 (final 0.002)
- closest |object_to_goal| 0.347 at step 48 (final 0.371); goal_error min 0.359 at step 1 / final 0.359 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 73 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 87 steps; TERMINATED by `ee_ground_collision` at step 86
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 14, 2 (CLOSE): 2
- aperture min 0.076 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.011 at step 85 (final 0.013)
- closest |object_to_goal| 0.508 at step 35 (final 0.532); goal_error min 0.518 at step 1 / final 0.545 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 74 — never_lifted
- ended in phase 1 (DESCEND) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 265, 1 (DESCEND): 503, 2 (CLOSE): 75, 3 (LIFT): 157
- aperture min 0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.018 at step 112 (final 0.086)
- closest |object_to_goal| 0.237 at step 1 (final 0.293); goal_error min 0.235 at step 1 / final 0.296 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 77 — never_lifted
- ended in phase 1 (DESCEND) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 157, 1 (DESCEND): 611, 2 (CLOSE): 75, 3 (LIFT): 157
- aperture min -0.000 / max 0.084 / final 0.080
- closest |gripper_to_object| 0.021 at step 114 (final 0.166)
- closest |object_to_goal| 0.448 at step 201 (final 0.475); goal_error min 0.459 at step 146 / final 0.464 (success < 0.050)
- object never lifted (max rise 0.013 m)

### env 78 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 73 steps; TERMINATED by `ee_ground_collision` at step 72
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 51, 1 (DESCEND): 22
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.010 at step 72 (final 0.010)
- closest |object_to_goal| 0.276 at step 19 (final 0.293); goal_error min 0.283 at step 1 / final 0.288 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 82 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 70 steps; TERMINATED by `ee_ground_collision` at step 69
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 48, 1 (DESCEND): 21, 2 (CLOSE): 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.004 at step 68 (final 0.012)
- closest |object_to_goal| 0.096 at step 0 (final 0.109); goal_error min 0.102 at step 1 / final 0.106 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 84 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 37 steps; TERMINATED by `ee_ground_collision` at step 36
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37
- aperture min 0.075 / max 0.080 / final 0.076
- closest |gripper_to_object| 0.079 at step 36 (final 0.079)
- closest |object_to_goal| 0.190 at step 0 (final 0.206); goal_error min 0.201 at step 36 / final 0.201 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 89 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 90 steps; TERMINATED by `ee_ground_collision` at step 89
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 19
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.010 at step 88 (final 0.014)
- closest |object_to_goal| 0.561 at step 1 (final 0.566); goal_error min 0.562 at step 1 / final 0.573 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 91 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 85 steps; TERMINATED by `ee_ground_collision` at step 84
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 61, 1 (DESCEND): 22, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.004 at step 83 (final 0.007)
- closest |object_to_goal| 0.281 at step 26 (final 0.291); goal_error min 0.288 at step 1 / final 0.294 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 100 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 37 steps; TERMINATED by `ee_ground_collision` at step 36
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 37
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.160 at step 36 (final 0.160)
- closest |object_to_goal| 0.195 at step 0 (final 0.209); goal_error min 0.202 at step 1 / final 0.212 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 102 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 94 steps; TERMINATED by `ee_ground_collision` at step 93
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 22, 2 (CLOSE): 1
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.010 at step 93 (final 0.010)
- closest |object_to_goal| 0.343 at step 0 (final 0.349); goal_error min 0.343 at step 1 / final 0.357 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 105 — terminated:ee_ground_collision
- ended in phase 0 (HOVER) after 42 steps; TERMINATED by `ee_ground_collision` at step 41
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 42
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.091 at step 41 (final 0.091)
- closest |object_to_goal| 0.248 at step 7 (final 0.267); goal_error min 0.260 at step 15 / final 0.260 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 106 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 89 steps; TERMINATED by `ee_ground_collision` at step 88
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 65, 1 (DESCEND): 22, 2 (CLOSE): 2
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.011 at step 88 (final 0.011)
- closest |object_to_goal| 0.295 at step 1 (final 0.320); goal_error min 0.299 at step 1 / final 0.315 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 108 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 96 steps; TERMINATED by `ee_ground_collision` at step 95
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 71, 1 (DESCEND): 25
- aperture min 0.076 / max 0.081 / final 0.081
- closest |gripper_to_object| 0.007 at step 92 (final 0.020)
- closest |object_to_goal| 0.407 at step 2 (final 0.419); goal_error min 0.412 at step 1 / final 0.419 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 110 — never_lifted
- ended in phase 3 (LIFT) after 1000 steps; ran to time-out
- most steps in phase 1 (DESCEND); steps per phase 0 (HOVER): 196, 1 (DESCEND): 534, 2 (CLOSE): 90, 3 (LIFT): 180
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.012 at step 106 (final 0.123)
- closest |object_to_goal| 0.271 at step 0 (final 0.350); goal_error min 0.283 at step 1 / final 0.342 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 120 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 94 steps; TERMINATED by `ee_ground_collision` at step 93
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 69, 1 (DESCEND): 23, 2 (CLOSE): 2
- aperture min 0.076 / max 0.080 / final 0.080
- closest |gripper_to_object| 0.003 at step 92 (final 0.014)
- closest |object_to_goal| 0.086 at step 0 (final 0.099); goal_error min 0.094 at step 1 / final 0.103 (success < 0.050)
- object never lifted (max rise 0.000 m)

## Successful envs (one line each)

- env 0: success at step 259, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 129
- env 2: success at step 156, end phase 3 (LIFT), aperture min 0.008, min |go| 0.002, lifted at 131
- env 3: success at step 181, end phase 1 (DESCEND), aperture min 0.010, min |go| 0.006, lifted at 128
- env 5: success at step 168, end phase 3 (LIFT), aperture min 0.012, min |go| 0.001, lifted at 149
- env 6: success at step 181, end phase 3 (LIFT), aperture min 0.035, min |go| 0.004, lifted at 124
- env 7: success at step 185, end phase 1 (DESCEND), aperture min 0.000, min |go| 0.004, lifted at 135
- env 8: success at step 151, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 132
- env 10: success at step 213, end phase 1 (DESCEND), aperture min 0.000, min |go| 0.003, lifted at 155
- env 12: success at step 186, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 111
- env 13: success at step 198, end phase 3 (LIFT), aperture min 0.043, min |go| 0.006, lifted at 129
- env 14: success at step 205, end phase 3 (LIFT), aperture min 0.015, min |go| 0.002, lifted at 139
- env 16: success at step 191, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 144
- env 17: success at step 246, end phase 2 (CLOSE), aperture min 0.013, min |go| 0.007, lifted at 119
- env 18: success at step 218, end phase 3 (LIFT), aperture min 0.043, min |go| 0.009, lifted at 133
- env 19: success at step 213, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 129
- env 20: success at step 204, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 163
- env 21: success at step 171, end phase 3 (LIFT), aperture min 0.043, min |go| 0.005, lifted at 135
- env 22: success at step 198, end phase 2 (CLOSE), aperture min 0.011, min |go| 0.008, lifted at 130
- env 23: success at step 179, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 152
- env 24: success at step 178, end phase 3 (LIFT), aperture min 0.014, min |go| 0.001, lifted at 145
- env 26: success at step 228, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 142
- env 27: success at step 330, end phase 3 (LIFT), aperture min 0.005, min |go| 0.002, lifted at 272
- env 28: success at step 173, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 139
- env 29: success at step 177, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 127
- env 31: success at step 205, end phase 3 (LIFT), aperture min 0.043, min |go| 0.000, lifted at 117
- env 32: success at step 192, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 135
- env 34: success at step 242, end phase 1 (DESCEND), aperture min 0.008, min |go| 0.004, lifted at 185
- env 35: success at step 177, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 115
- env 36: success at step 169, end phase 2 (CLOSE), aperture min 0.016, min |go| 0.007, lifted at 134
- env 37: success at step 320, end phase 3 (LIFT), aperture min 0.001, min |go| 0.005, lifted at 263
- env 38: success at step 184, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 120
- env 39: success at step 196, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 148
- env 41: success at step 201, end phase 1 (DESCEND), aperture min 0.009, min |go| 0.006, lifted at 135
- env 42: success at step 189, end phase 3 (LIFT), aperture min 0.054, min |go| 0.001, lifted at 131
- env 43: success at step 171, end phase 3 (LIFT), aperture min 0.043, min |go| 0.005, lifted at 137
- env 44: success at step 185, end phase 2 (CLOSE), aperture min 0.014, min |go| 0.005, lifted at 137
- env 45: success at step 224, end phase 3 (LIFT), aperture min 0.044, min |go| 0.003, lifted at 137
- env 47: success at step 169, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 131
- env 49: success at step 142, end phase 3 (LIFT), aperture min 0.006, min |go| 0.005, lifted at 130
- env 50: success at step 158, end phase 3 (LIFT), aperture min 0.011, min |go| 0.003, lifted at 137
- env 52: success at step 161, end phase 3 (LIFT), aperture min 0.007, min |go| 0.003, lifted at 134
- env 53: success at step 213, end phase 1 (DESCEND), aperture min 0.010, min |go| 0.005, lifted at 165
- env 54: success at step 227, end phase 3 (LIFT), aperture min 0.044, min |go| 0.006, lifted at 133
- env 56: success at step 214, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 130
- env 57: success at step 151, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 116
- env 58: success at step 145, end phase 3 (LIFT), aperture min 0.005, min |go| 0.004, lifted at 122
- env 59: success at step 158, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 124
- env 60: success at step 151, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 135
- env 61: success at step 194, end phase 3 (LIFT), aperture min 0.011, min |go| 0.001, lifted at 142
- env 62: success at step 185, end phase 3 (LIFT), aperture min 0.045, min |go| 0.002, lifted at 144
- env 63: success at step 369, end phase 3 (LIFT), aperture min 0.002, min |go| 0.010, lifted at 252
- env 64: success at step 185, end phase 2 (CLOSE), aperture min 0.011, min |go| 0.003, lifted at 132
- env 65: success at step 197, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 132
- env 66: success at step 181, end phase 1 (DESCEND), aperture min 0.017, min |go| 0.002, lifted at 136
- env 69: success at step 204, end phase 3 (LIFT), aperture min 0.007, min |go| 0.002, lifted at 147
- env 70: success at step 214, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 139
- env 72: success at step 189, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 131
- env 75: success at step 181, end phase 3 (LIFT), aperture min 0.011, min |go| 0.001, lifted at 127
- env 76: success at step 230, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 141
- env 79: success at step 209, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 119
- env 80: success at step 205, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 145
- env 81: success at step 247, end phase 3 (LIFT), aperture min 0.041, min |go| 0.008, lifted at 120
- env 83: success at step 191, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 153
- env 85: success at step 153, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 135
- env 86: success at step 199, end phase 3 (LIFT), aperture min 0.017, min |go| 0.002, lifted at 145
- env 87: success at step 224, end phase 3 (LIFT), aperture min 0.042, min |go| 0.013, lifted at 143
- env 88: success at step 321, end phase 3 (LIFT), aperture min 0.005, min |go| 0.008, lifted at 266
- env 90: success at step 215, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 138
- env 92: success at step 160, end phase 3 (LIFT), aperture min 0.026, min |go| 0.002, lifted at 117
- env 93: success at step 214, end phase 3 (LIFT), aperture min 0.044, min |go| 0.002, lifted at 129
- env 94: success at step 245, end phase 1 (DESCEND), aperture min 0.003, min |go| 0.007, lifted at 132
- env 95: success at step 199, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 144
- env 96: success at step 299, end phase 2 (CLOSE), aperture min 0.018, min |go| 0.004, lifted at 125
- env 97: success at step 243, end phase 3 (LIFT), aperture min 0.042, min |go| 0.006, lifted at 137
- env 98: success at step 196, end phase 3 (LIFT), aperture min 0.014, min |go| 0.002, lifted at 141
- env 99: success at step 170, end phase 3 (LIFT), aperture min 0.044, min |go| 0.005, lifted at 138
- env 101: success at step 157, end phase 3 (LIFT), aperture min 0.043, min |go| 0.000, lifted at 143
- env 103: success at step 160, end phase 3 (LIFT), aperture min 0.043, min |go| 0.002, lifted at 120
- env 104: success at step 167, end phase 3 (LIFT), aperture min 0.000, min |go| 0.002, lifted at 141
- env 107: success at step 215, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 141
- env 109: success at step 176, end phase 3 (LIFT), aperture min 0.043, min |go| 0.000, lifted at 136
- env 111: success at step 184, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 111
- env 112: success at step 202, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 131
- env 113: success at step 216, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 143
- env 114: success at step 156, end phase 3 (LIFT), aperture min 0.044, min |go| 0.001, lifted at 130
- env 115: success at step 217, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 134
- env 116: success at step 164, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 127
- env 117: success at step 212, end phase 2 (CLOSE), aperture min 0.012, min |go| 0.003, lifted at 151
- env 118: success at step 214, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 112
- env 119: success at step 170, end phase 3 (LIFT), aperture min 0.006, min |go| 0.001, lifted at 126
- env 121: success at step 174, end phase 3 (LIFT), aperture min 0.043, min |go| 0.003, lifted at 143
- env 122: success at step 348, end phase 3 (LIFT), aperture min 0.001, min |go| 0.006, lifted at 260
- env 123: success at step 204, end phase 3 (LIFT), aperture min 0.044, min |go| 0.004, lifted at 131
- env 124: success at step 249, end phase 3 (LIFT), aperture min 0.043, min |go| 0.004, lifted at 141
- env 125: success at step 207, end phase 3 (LIFT), aperture min 0.043, min |go| 0.001, lifted at 106
- env 126: success at step 231, end phase 3 (LIFT), aperture min 0.043, min |go| 0.007, lifted at 161
- env 127: success at step 222, end phase 1 (DESCEND), aperture min 0.004, min |go| 0.007, lifted at 130
