# diagnose — Mjlab-Stack-Cube-Franka

2026-09-09T13:15:37 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 1000 · device cuda:0  
**15/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 28:31, gripper_to_object 37:40, object_to_goal 40:43  
Teacher `StackObjectClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = CLOSE, 3 = LIFT, 4 = CARRY, 5 = PLACE, 6 = RELEASE, 7 = RETREAT, 8 = DONE

## Histograms

- end phase, FAILING envs (17): 4 (CARRY): 12, 1 (DESCEND): 2, 2 (CLOSE): 1, 8 (DONE): 1, 6 (RELEASE): 1
- end phase, successful envs (15): 8 (DONE): 14, 5 (PLACE): 1
- most-steps phase, failing envs: 4 (CARRY): 12, 0 (HOVER): 3, 8 (DONE): 1, 5 (PLACE): 1
- failure class: never_lifted: 9, grasp_lost: 4, terminated:ee_ground_collision: 4
- termination among failing envs: none: 13, ee_ground_collision: 4
- failing envs that lifted the object: 5/17; lifted then lost: 5
- success step (successful envs): median 116, max 176

## Reading

- **never_lifted** — 9/17 failing envs (envs [1, 3, 8, 9, 20, 21, 24, 25, 26]); typical end phase 4 (CARRY); final aperture median 0.000; closest |gripper_to_object| median 0.012; closest |object_to_goal| median 0.226; min goal_error median 0.240 (success < 0.030)
- **grasp_lost** — 4/17 failing envs (envs [2, 12, 18, 30]); typical end phase 4 (CARRY); final aperture median 0.000; closest |gripper_to_object| median 0.008; closest |object_to_goal| median 0.149; min goal_error median 0.153 (success < 0.030); lost at step median 104
- **terminated:ee_ground_collision** — 4/17 failing envs (envs [7, 10, 13, 14]); typical end phase 1 (DESCEND); final aperture median 0.078; closest |gripper_to_object| median 0.010; closest |object_to_goal| median 0.279; min goal_error median 0.289 (success < 0.030)

## Failing envs

### env 1 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 36, 1 (DESCEND): 61, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 869
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.011 at step 49 (final 0.648)
- closest |object_to_goal| 0.226 at step 339 (final 0.238); goal_error min 0.240 at step 122 / final 0.240 (success < 0.030)
- object never lifted (max rise 0.017 m)

### env 2 — grasp_lost
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 32, 1 (DESCEND): 19, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 915
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.010 at step 49 (final 0.849)
- closest |object_to_goal| 0.217 at step 71 (final 0.312); goal_error min 0.221 at step 71 / final 0.306 (success < 0.030)
- object LIFTED at step 70 (max rise 0.067 m); LOST at step 88 (aperture collapsed while the object moved away)

### env 3 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 36, 1 (DESCEND): 15, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 915
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.007 at step 51 (final 0.766)
- closest |object_to_goal| 0.155 at step 42 (final 0.201); goal_error min 0.164 at step 48 / final 0.196 (success < 0.030)
- object never lifted (max rise 0.016 m)

### env 7 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 36 steps; TERMINATED by `ee_ground_collision` at step 35
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 24, 1 (DESCEND): 12
- aperture min 0.076 / max 0.081 / final 0.080
- closest |gripper_to_object| 0.016 at step 34 (final 0.018)
- closest |object_to_goal| 0.284 at step 35 (final 0.284); goal_error min 0.294 at step 34 / final 0.294 (success < 0.030)
- object never lifted (max rise 0.001 m)

### env 8 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 33, 1 (DESCEND): 14, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 919
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.015 at step 46 (final 0.723)
- closest |object_to_goal| 0.185 at step 732 (final 0.202); goal_error min 0.192 at step 65 / final 0.197 (success < 0.030)
- object never lifted (max rise 0.014 m)

### env 9 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 34, 1 (DESCEND): 27, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 905
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.005 at step 60 (final 0.806)
- closest |object_to_goal| 0.289 at step 53 (final 0.330); goal_error min 0.298 at step 53 / final 0.326 (success < 0.030)
- object never lifted (max rise 0.013 m)

### env 10 — terminated:ee_ground_collision
- ended in phase 2 (CLOSE) after 33 steps; TERMINATED by `ee_ground_collision` at step 32
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 21, 1 (DESCEND): 9, 2 (CLOSE): 3
- aperture min 0.076 / max 0.084 / final 0.076
- closest |gripper_to_object| 0.006 at step 30 (final 0.018)
- closest |object_to_goal| 0.275 at step 32 (final 0.275); goal_error min 0.284 at step 14 / final 0.285 (success < 0.030)
- object never lifted (max rise 0.003 m)

### env 12 — grasp_lost
- ended in phase 8 (DONE) after 1000 steps; ran to time-out
- most steps in phase 8 (DONE); steps per phase 0 (HOVER): 32, 1 (DESCEND): 20, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 30, 5 (PLACE): 71, 6 (RELEASE): 12, 7 (RETREAT): 30, 8 (DONE): 771
- aperture min -0.000 / max 0.085 / final 0.080
- closest |gripper_to_object| 0.010 at step 51 (final 0.774)
- closest |object_to_goal| 0.055 at step 127 (final 0.083); goal_error min 0.058 at step 126 / final 0.088 (success < 0.030)
- object LIFTED at step 75 (max rise 0.116 m); LOST at step 146 (aperture collapsed while the object moved away)

### env 13 — terminated:ee_ground_collision
- ended in phase 1 (DESCEND) after 38 steps; TERMINATED by `ee_ground_collision` at step 37
- most steps in phase 0 (HOVER); steps per phase 0 (HOVER): 27, 1 (DESCEND): 11
- aperture min 0.076 / max 0.087 / final 0.083
- closest |gripper_to_object| 0.013 at step 35 (final 0.018)
- closest |object_to_goal| 0.329 at step 34 (final 0.340); goal_error min 0.337 at step 34 / final 0.337 (success < 0.030)
- object never lifted (max rise 0.000 m)

### env 14 — terminated:ee_ground_collision
- ended in phase 6 (RELEASE) after 184 steps; TERMINATED by `ee_ground_collision` at step 183
- most steps in phase 5 (PLACE); steps per phase 0 (HOVER): 26, 1 (DESCEND): 15, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 31, 5 (PLACE): 71, 6 (RELEASE): 7
- aperture min -0.000 / max 0.084 / final 0.056
- closest |gripper_to_object| 0.004 at step 52 (final 0.850)
- closest |object_to_goal| 0.066 at step 110 (final 0.120); goal_error min 0.068 at step 110 / final 0.115 (success < 0.030)
- object LIFTED at step 63 (max rise 0.147 m); LOST at step 111 (aperture collapsed while the object moved away)

### env 18 — grasp_lost
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 36, 1 (DESCEND): 34, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 896
- aperture min -0.000 / max 0.087 / final 0.000
- closest |gripper_to_object| 0.006 at step 69 (final 0.726)
- closest |object_to_goal| 0.304 at step 61 (final 0.383); goal_error min 0.309 at step 60 / final 0.373 (success < 0.030)
- object LIFTED at step 93 (max rise 0.020 m); LOST at step 100 (aperture collapsed while the object moved away)

### env 20 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 26, 1 (DESCEND): 18, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 922
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.008 at step 43 (final 1.157)
- closest |object_to_goal| 0.196 at step 47 (final 0.203); goal_error min 0.200 at step 50 / final 0.210 (success < 0.030)
- object never lifted (max rise 0.010 m)

### env 21 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 42, 1 (DESCEND): 61, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 863
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.014 at step 57 (final 0.793)
- closest |object_to_goal| 0.278 at step 26 (final 0.332); goal_error min 0.292 at step 1 / final 0.340 (success < 0.030)
- object never lifted (max rise 0.012 m)

### env 24 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 30, 1 (DESCEND): 29, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 907
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.014 at step 59 (final 1.081)
- closest |object_to_goal| 0.281 at step 52 (final 0.288); goal_error min 0.287 at step 53 / final 0.292 (success < 0.030)
- object never lifted (max rise 0.015 m)

### env 25 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 27, 1 (DESCEND): 22, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 917
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.012 at step 51 (final 1.116)
- closest |object_to_goal| 0.193 at step 562 (final 0.198); goal_error min 0.203 at step 51 / final 0.204 (success < 0.030)
- object never lifted (max rise 0.013 m)

### env 26 — never_lifted
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 36, 1 (DESCEND): 34, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 896
- aperture min -0.000 / max 0.091 / final 0.000
- closest |gripper_to_object| 0.012 at step 70 (final 0.674)
- closest |object_to_goal| 0.274 at step 254 (final 0.280); goal_error min 0.285 at step 77 / final 0.285 (success < 0.030)
- object never lifted (max rise 0.012 m)

### env 30 — grasp_lost
- ended in phase 4 (CARRY) after 1000 steps; ran to time-out
- most steps in phase 4 (CARRY); steps per phase 0 (HOVER): 34, 1 (DESCEND): 28, 2 (CLOSE): 12, 3 (LIFT): 22, 4 (CARRY): 904
- aperture min -0.000 / max 0.085 / final 0.000
- closest |gripper_to_object| 0.005 at step 60 (final 1.180)
- closest |object_to_goal| 0.081 at step 957 (final 0.094); goal_error min 0.086 at step 118 / final 0.094 (success < 0.030)
- object LIFTED at step 88 (max rise 0.145 m); LOST at step 109 (aperture collapsed while the object moved away)

## Successful envs (one line each)

- env 0: success at step 139, end phase 8 (DONE), aperture min 0.045, min |go| 0.004, lifted at 73
- env 4: success at step 114, end phase 8 (DONE), aperture min 0.046, min |go| 0.000, lifted at 54
- env 5: success at step 116, end phase 8 (DONE), aperture min 0.004, min |go| 0.003, lifted at 79
- env 6: success at step 176, end phase 8 (DONE), aperture min 0.002, min |go| 0.007, lifted at 93
- env 11: success at step 132, end phase 8 (DONE), aperture min 0.052, min |go| 0.002, lifted at 76
- env 15: success at step 119, end phase 8 (DONE), aperture min 0.039, min |go| 0.006, lifted at 67
- env 16: success at step 163, end phase 8 (DONE), aperture min 0.044, min |go| 0.003, lifted at 79
- env 17: success at step 116, end phase 8 (DONE), aperture min -0.013, min |go| 0.009, lifted at 75
- env 19: success at step 116, end phase 8 (DONE), aperture min 0.044, min |go| 0.006, lifted at 71
- env 22: success at step 98, end phase 8 (DONE), aperture min 0.039, min |go| 0.001, lifted at 52
- env 23: success at step 113, end phase 8 (DONE), aperture min 0.045, min |go| 0.002, lifted at 67
- env 27: success at step 122, end phase 8 (DONE), aperture min -0.001, min |go| 0.005, lifted at 46
- env 28: success at step 101, end phase 5 (PLACE), aperture min -0.000, min |go| 0.009, lifted at 62
- env 29: success at step 129, end phase 8 (DONE), aperture min 0.045, min |go| 0.003, lifted at 74
- env 31: success at step 102, end phase 8 (DONE), aperture min 0.061, min |go| 0.003, lifted at 53
