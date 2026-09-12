# diagnose — Mjlab-Open-Door-Franka

2026-09-09T13:25:33 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 150 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `OpenDoorClassicalPolicy`; primary object `door`; mechanism joint `door_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_door_collision'].  
Phases: unnamed — numbers are the teacher's `_phase` values.

## Histograms

- end phase, FAILING envs (32): 3: 25, 1: 7
- end phase, successful envs (0): —
- most-steps phase, failing envs: 3: 24, 1: 8
- failure class: mechanism_not_moved: 21, never_reached: 8, mechanism_short: 3
- termination among failing envs: none: 32

## Reading

- **mechanism_not_moved** — 21/32 failing envs (envs [1, 2, 3, 6, 7, 8, 10, 13, 14, 15, 16, 17, 18, 19, 20, 22, 24, 25, 28, 29, 30]); typical end phase 3; final aperture median 0.067; closest |gripper_to_object| median 0.034; closest |object_to_goal| median 0.346; min goal_error median 1.565 (success < 0.100); mechanism reached median 0.005 of target 1.571
- **never_reached** — 8/32 failing envs (envs [0, 4, 5, 9, 12, 21, 23, 27]); typical end phase 1; final aperture median 0.050; closest |gripper_to_object| median 0.070; closest |object_to_goal| median 0.346; min goal_error median 1.571 (success < 0.100); mechanism reached median 0.000 of target 1.571
- **mechanism_short** — 3/32 failing envs (envs [11, 26, 31]); typical end phase 3; final aperture median -0.014; closest |gripper_to_object| median 0.024; closest |object_to_goal| median 0.224; min goal_error median 0.943 (success < 0.100); mechanism reached median 0.628 of target 1.571

## Failing envs

### env 0 — never_reached
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 35, 1: 31, 2: 5, 3: 79
- aperture min 0.037 / max 0.092 / final 0.037
- closest |gripper_to_object| 0.066 at step 63 (final 0.076)
- closest |object_to_goal| 0.345 at step 70 (final 0.364); goal_error min 1.567 at step 146 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.003 (final -0.003) of target 1.571

### env 1 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.053 / max 0.093 / final 0.083
- closest |gripper_to_object| 0.031 at step 95 (final 0.046)
- closest |object_to_goal| 0.346 at step 148 (final 0.363); goal_error min 1.571 at step 149 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 2 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.041 / max 0.092 / final 0.054
- closest |gripper_to_object| 0.034 at step 106 (final 0.040)
- closest |object_to_goal| 0.346 at step 16 (final 0.351); goal_error min 1.566 at step 90 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.005 (final -0.000) of target 1.571

### env 3 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.056 / max 0.091 / final 0.083
- closest |gripper_to_object| 0.056 at step 97 (final 0.067)
- closest |object_to_goal| 0.346 at step 70 (final 0.360); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 4 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 118
- aperture min 0.052 / max 0.090 / final 0.077
- closest |gripper_to_object| 0.074 at step 64 (final 0.085)
- closest |object_to_goal| 0.346 at step 87 (final 0.358); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 5 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 117
- aperture min 0.049 / max 0.088 / final 0.050
- closest |gripper_to_object| 0.074 at step 119 (final 0.093)
- closest |object_to_goal| 0.347 at step 37 (final 0.352); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 6 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 35, 1: 31, 2: 5, 3: 79
- aperture min 0.061 / max 0.089 / final 0.083
- closest |gripper_to_object| 0.029 at step 115 (final 0.037)
- closest |object_to_goal| 0.346 at step 24 (final 0.356); goal_error min 1.568 at step 143 / final 1.572 (success < 0.100)
- mechanism joint: reached 0.003 (final -0.001) of target 1.571

### env 7 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 31, 2: 5, 3: 81
- aperture min 0.044 / max 0.088 / final 0.046
- closest |gripper_to_object| 0.053 at step 117 (final 0.070)
- closest |object_to_goal| 0.347 at step 54 (final 0.362); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 8 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.028 / max 0.090 / final 0.066
- closest |gripper_to_object| 0.030 at step 110 (final 0.046)
- closest |object_to_goal| 0.346 at step 61 (final 0.352); goal_error min 1.566 at step 63 / final 1.572 (success < 0.100)
- mechanism joint: reached 0.004 (final -0.001) of target 1.571

### env 9 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 36, 1: 114
- aperture min 0.030 / max 0.086 / final 0.050
- closest |gripper_to_object| 0.069 at step 58 (final 0.098)
- closest |object_to_goal| 0.346 at step 8 (final 0.366); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 10 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 34, 1: 31, 2: 5, 3: 80
- aperture min 0.008 / max 0.088 / final 0.045
- closest |gripper_to_object| 0.036 at step 107 (final 0.041)
- closest |object_to_goal| 0.348 at step 15 (final 0.362); goal_error min 1.571 at step 1 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.000 (final -0.004) of target 1.571

### env 11 — mechanism_short
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 34, 1: 39, 2: 5, 3: 72
- aperture min -0.014 / max 0.093 / final -0.014
- closest |gripper_to_object| 0.027 at step 113 (final 0.042)
- closest |object_to_goal| 0.246 at step 124 (final 0.360); goal_error min 1.028 at step 124 / final 1.582 (success < 0.100)
- mechanism joint: reached 0.542 (final -0.012) of target 1.571

### env 12 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 117
- aperture min 0.046 / max 0.091 / final 0.075
- closest |gripper_to_object| 0.073 at step 59 (final 0.084)
- closest |object_to_goal| 0.346 at step 86 (final 0.362); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 13 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 34, 1: 31, 2: 5, 3: 80
- aperture min 0.032 / max 0.091 / final 0.067
- closest |gripper_to_object| 0.030 at step 138 (final 0.048)
- closest |object_to_goal| 0.332 at step 145 (final 0.337); goal_error min 1.445 at step 149 / final 1.445 (success < 0.100)
- mechanism joint: reached 0.126 (final 0.126) of target 1.571

### env 14 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.050 / max 0.089 / final 0.076
- closest |gripper_to_object| 0.057 at step 87 (final 0.059)
- closest |object_to_goal| 0.344 at step 140 (final 0.363); goal_error min 1.564 at step 149 / final 1.564 (success < 0.100)
- mechanism joint: reached 0.007 (final 0.007) of target 1.571

### env 15 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.009 / max 0.089 / final 0.039
- closest |gripper_to_object| 0.040 at step 128 (final 0.047)
- closest |object_to_goal| 0.346 at step 56 (final 0.362); goal_error min 1.571 at step 1 / final 1.575 (success < 0.100)
- mechanism joint: reached 0.000 (final -0.004) of target 1.571

### env 16 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 31, 2: 5, 3: 81
- aperture min 0.039 / max 0.092 / final 0.072
- closest |gripper_to_object| 0.030 at step 145 (final 0.037)
- closest |object_to_goal| 0.346 at step 48 (final 0.360); goal_error min 1.557 at step 137 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.014 (final -0.003) of target 1.571

### env 17 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 32, 1: 31, 2: 5, 3: 82
- aperture min 0.015 / max 0.092 / final 0.047
- closest |gripper_to_object| 0.035 at step 81 (final 0.056)
- closest |object_to_goal| 0.345 at step 144 (final 0.367); goal_error min 1.565 at step 111 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.005 (final -0.003) of target 1.571

### env 18 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.035 / max 0.091 / final 0.072
- closest |gripper_to_object| 0.030 at step 128 (final 0.048)
- closest |object_to_goal| 0.338 at step 63 (final 0.349); goal_error min 1.526 at step 63 / final 1.562 (success < 0.100)
- mechanism joint: reached 0.044 (final 0.008) of target 1.571

### env 19 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.042 / max 0.090 / final 0.061
- closest |gripper_to_object| 0.031 at step 61 (final 0.034)
- closest |object_to_goal| 0.345 at step 110 (final 0.367); goal_error min 1.537 at step 113 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.033 (final -0.003) of target 1.571

### env 20 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 32, 1: 31, 2: 5, 3: 82
- aperture min 0.043 / max 0.090 / final 0.068
- closest |gripper_to_object| 0.050 at step 110 (final 0.054)
- closest |object_to_goal| 0.347 at step 99 (final 0.362); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 21 — never_reached
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 31, 1: 33, 2: 5, 3: 81
- aperture min 0.035 / max 0.090 / final 0.064
- closest |gripper_to_object| 0.064 at step 149 (final 0.064)
- closest |object_to_goal| 0.345 at step 17 (final 0.352); goal_error min 1.563 at step 137 / final 1.568 (success < 0.100)
- mechanism joint: reached 0.007 (final 0.002) of target 1.571

### env 22 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.050 / max 0.090 / final 0.076
- closest |gripper_to_object| 0.058 at step 88 (final 0.062)
- closest |object_to_goal| 0.347 at step 96 (final 0.361); goal_error min 1.555 at step 149 / final 1.555 (success < 0.100)
- mechanism joint: reached 0.016 (final 0.016) of target 1.571

### env 23 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 118
- aperture min 0.032 / max 0.089 / final 0.038
- closest |gripper_to_object| 0.071 at step 58 (final 0.090)
- closest |object_to_goal| 0.347 at step 64 (final 0.357); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 24 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 31, 2: 5, 3: 81
- aperture min 0.027 / max 0.092 / final 0.059
- closest |gripper_to_object| 0.036 at step 141 (final 0.045)
- closest |object_to_goal| 0.343 at step 144 (final 0.354); goal_error min 1.497 at step 149 / final 1.497 (success < 0.100)
- mechanism joint: reached 0.074 (final 0.074) of target 1.571

### env 25 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 31, 2: 5, 3: 78
- aperture min 0.053 / max 0.093 / final 0.083
- closest |gripper_to_object| 0.029 at step 132 (final 0.040)
- closest |object_to_goal| 0.343 at step 144 (final 0.360); goal_error min 1.545 at step 94 / final 1.555 (success < 0.100)
- mechanism joint: reached 0.025 (final 0.016) of target 1.571

### env 26 — mechanism_short
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 101, 2: 5, 3: 10
- aperture min -0.014 / max 0.093 / final -0.014
- closest |gripper_to_object| 0.024 at step 67 (final 0.095)
- closest |object_to_goal| 0.208 at step 81 (final 0.364); goal_error min 0.880 at step 82 / final 1.579 (success < 0.100)
- mechanism joint: reached 0.691 (final -0.009) of target 1.571

### env 27 — never_reached
- ended in phase 1 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 36, 1: 114
- aperture min 0.031 / max 0.092 / final 0.038
- closest |gripper_to_object| 0.066 at step 61 (final 0.076)
- closest |object_to_goal| 0.347 at step 9 (final 0.365); goal_error min 1.571 at step 1 / final 1.571 (success < 0.100)
- mechanism joint: reached 0.000 (final 0.000) of target 1.571

### env 28 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 31, 2: 5, 3: 81
- aperture min 0.058 / max 0.090 / final 0.084
- closest |gripper_to_object| 0.024 at step 147 (final 0.029)
- closest |object_to_goal| 0.326 at step 113 (final 0.330); goal_error min 1.444 at step 119 / final 1.475 (success < 0.100)
- mechanism joint: reached 0.127 (final 0.096) of target 1.571

### env 29 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 35, 1: 31, 2: 5, 3: 79
- aperture min 0.042 / max 0.092 / final 0.052
- closest |gripper_to_object| 0.032 at step 122 (final 0.046)
- closest |object_to_goal| 0.344 at step 73 (final 0.357); goal_error min 1.547 at step 77 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.023 (final -0.003) of target 1.571

### env 30 — mechanism_not_moved
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 32, 1: 31, 2: 5, 3: 82
- aperture min 0.034 / max 0.092 / final 0.066
- closest |gripper_to_object| 0.035 at step 133 (final 0.041)
- closest |object_to_goal| 0.345 at step 21 (final 0.362); goal_error min 1.571 at step 1 / final 1.574 (success < 0.100)
- mechanism joint: reached 0.000 (final -0.003) of target 1.571

### env 31 — mechanism_short
- ended in phase 3 after 150 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 56, 2: 5, 3: 56
- aperture min -0.017 / max 0.091 / final -0.014
- closest |gripper_to_object| 0.010 at step 69 (final 0.036)
- closest |object_to_goal| 0.224 at step 85 (final 0.352); goal_error min 0.943 at step 90 / final 1.580 (success < 0.100)
- mechanism joint: reached 0.628 (final -0.009) of target 1.571

## Successful envs (one line each)

