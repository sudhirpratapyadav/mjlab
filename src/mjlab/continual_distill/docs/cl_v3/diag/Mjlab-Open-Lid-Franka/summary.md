# diagnose — Mjlab-Open-Lid-Franka

2026-09-09T13:26:27 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 250 · device cuda:0  
**2/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `OpenLidClassicalPolicy`; primary object `asset`; mechanism joint `lid_hinge`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_lid_collision'].  
Phases: unnamed — numbers are the teacher's `_phase` values.

## Histograms

- end phase, FAILING envs (30): 3: 25, 1: 3, 2: 2
- end phase, successful envs (2): 3: 2
- most-steps phase, failing envs: 3: 19, 1: 11
- failure class: mechanism_not_moved: 16, mechanism_short: 14
- termination among failing envs: none: 30
- success step (successful envs): median 123, max 125

## Reading

- **mechanism_not_moved** — 16/30 failing envs (envs [0, 2, 7, 8, 9, 11, 12, 13, 15, 16, 18, 22, 24, 25, 28, 29]); typical end phase 3; final aperture median 0.000; closest |gripper_to_object| median 0.009; closest |object_to_goal| median 0.143; min goal_error median 1.310 (success < 0.200); mechanism reached median 0.000 of target -1.309
- **mechanism_short** — 14/30 failing envs (envs [1, 3, 4, 5, 6, 10, 14, 17, 19, 20, 21, 27, 30, 31]); typical end phase 3; final aperture median 0.012; closest |gripper_to_object| median 0.008; closest |object_to_goal| median 0.079; min goal_error median 0.821 (success < 0.200); mechanism reached median -0.488 of target -1.309

## Failing envs

### env 0 — mechanism_not_moved
- ended in phase 1 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 34, 1: 147, 2: 24, 3: 45
- aperture min 0.000 / max 0.081 / final 0.079
- closest |gripper_to_object| 0.010 at step 107 (final 0.056)
- closest |object_to_goal| 0.140 at step 90 (final 0.156); goal_error min 1.235 at step 89 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.074 (final 0.001) of target -1.309

### env 1 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 31, 1: 69, 2: 24, 3: 126
- aperture min 0.000 / max 0.081 / final 0.022
- closest |gripper_to_object| 0.008 at step 164 (final 0.049)
- closest |object_to_goal| 0.093 at step 239 (final 0.104); goal_error min 0.933 at step 244 / final 0.955 (success < 0.200)
- mechanism joint: reached -0.376 (final -0.354) of target -1.309

### env 2 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 51, 2: 12, 3: 154
- aperture min -0.000 / max 0.083 / final 0.000
- closest |gripper_to_object| 0.013 at step 103 (final 0.091)
- closest |object_to_goal| 0.143 at step 187 (final 0.156); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 3 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 30, 1: 80, 2: 30, 3: 110
- aperture min -0.000 / max 0.081 / final -0.000
- closest |gripper_to_object| 0.007 at step 164 (final 0.104)
- closest |object_to_goal| 0.080 at step 111 (final 0.143); goal_error min 0.815 at step 109 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.494 (final 0.001) of target -1.309

### env 4 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 26, 1: 116, 2: 30, 3: 78
- aperture min 0.000 / max 0.084 / final 0.025
- closest |gripper_to_object| 0.008 at step 242 (final 0.036)
- closest |object_to_goal| 0.067 at step 149 (final 0.157); goal_error min 0.758 at step 150 / final 1.320 (success < 0.200)
- mechanism joint: reached -0.551 (final 0.011) of target -1.309

### env 5 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 28, 1: 113, 2: 30, 3: 79
- aperture min 0.000 / max 0.082 / final 0.003
- closest |gripper_to_object| 0.008 at step 229 (final 0.094)
- closest |object_to_goal| 0.077 at step 96 (final 0.151); goal_error min 0.827 at step 98 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.482 (final 0.001) of target -1.309

### env 6 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 35, 1: 82, 2: 24, 3: 109
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.007 at step 150 (final 0.087)
- closest |object_to_goal| 0.028 at step 88 (final 0.147); goal_error min 0.456 at step 88 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.853 (final 0.001) of target -1.309

### env 7 — mechanism_not_moved
- ended in phase 1 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 95, 2: 36, 3: 86
- aperture min 0.000 / max 0.080 / final 0.027
- closest |gripper_to_object| 0.006 at step 195 (final 0.080)
- closest |object_to_goal| 0.144 at step 138 (final 0.155); goal_error min 1.299 at step 203 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.010 (final 0.001) of target -1.309

### env 8 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 71, 2: 18, 3: 128
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.007 at step 99 (final 0.093)
- closest |object_to_goal| 0.143 at step 3 (final 0.163); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 9 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 27, 1: 56, 2: 18, 3: 149
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.008 at step 122 (final 0.099)
- closest |object_to_goal| 0.143 at step 70 (final 0.157); goal_error min 1.301 at step 70 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.008 (final 0.001) of target -1.309

### env 10 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 30, 1: 112, 2: 30, 3: 78
- aperture min 0.000 / max 0.082 / final 0.014
- closest |gripper_to_object| 0.010 at step 144 (final 0.108)
- closest |object_to_goal| 0.103 at step 246 (final 0.113); goal_error min 0.981 at step 245 / final 1.106 (success < 0.200)
- mechanism joint: reached -0.328 (final -0.203) of target -1.309

### env 11 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 36, 1: 48, 2: 12, 3: 154
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.012 at step 107 (final 0.092)
- closest |object_to_goal| 0.143 at step 34 (final 0.154); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 12 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 28, 1: 61, 2: 18, 3: 143
- aperture min -0.000 / max 0.082 / final 0.000
- closest |gripper_to_object| 0.006 at step 133 (final 0.100)
- closest |object_to_goal| 0.138 at step 78 (final 0.155); goal_error min 1.183 at step 77 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.126 (final 0.001) of target -1.309

### env 13 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 27, 1: 85, 2: 24, 3: 114
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.015 at step 133 (final 0.096)
- closest |object_to_goal| 0.134 at step 77 (final 0.151); goal_error min 1.189 at step 76 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.120 (final 0.001) of target -1.309

### env 14 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 28, 1: 114, 2: 30, 3: 78
- aperture min 0.001 / max 0.082 / final 0.001
- closest |gripper_to_object| 0.005 at step 228 (final 0.118)
- closest |object_to_goal| 0.043 at step 163 (final 0.160); goal_error min 0.621 at step 166 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.688 (final 0.001) of target -1.309

### env 15 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 29, 1: 57, 2: 18, 3: 146
- aperture min -0.000 / max 0.080 / final -0.000
- closest |gripper_to_object| 0.010 at step 128 (final 0.089)
- closest |object_to_goal| 0.144 at step 171 (final 0.149); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 16 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 33, 1: 86, 2: 24, 3: 107
- aperture min -0.000 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.006 at step 185 (final 0.085)
- closest |object_to_goal| 0.143 at step 63 (final 0.160); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 17 — mechanism_short
- ended in phase 1 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 102, 2: 30, 3: 87
- aperture min 0.000 / max 0.081 / final 0.056
- closest |gripper_to_object| 0.003 at step 111 (final 0.185)
- closest |object_to_goal| 0.093 at step 190 (final 0.153); goal_error min 0.966 at step 189 / final 1.312 (success < 0.200)
- mechanism joint: reached -0.343 (final 0.003) of target -1.309

### env 18 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 32, 1: 91, 2: 18, 3: 109
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.021 at step 98 (final 0.090)
- closest |object_to_goal| 0.139 at step 80 (final 0.153); goal_error min 1.240 at step 81 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.069 (final 0.001) of target -1.309

### env 19 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 34, 1: 70, 2: 24, 3: 122
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.006 at step 174 (final 0.083)
- closest |object_to_goal| 0.133 at step 83 (final 0.148); goal_error min 1.172 at step 84 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.137 (final 0.001) of target -1.309

### env 20 — mechanism_short
- ended in phase 2 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 121, 2: 30, 3: 68
- aperture min 0.000 / max 0.093 / final 0.034
- closest |gripper_to_object| 0.009 at step 200 (final 0.035)
- closest |object_to_goal| 0.071 at step 105 (final 0.170); goal_error min 0.783 at step 104 / final 1.331 (success < 0.200)
- mechanism joint: reached -0.526 (final 0.022) of target -1.309

### env 21 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 28, 1: 73, 2: 18, 3: 131
- aperture min -0.000 / max 0.086 / final -0.000
- closest |gripper_to_object| 0.008 at step 105 (final 0.093)
- closest |object_to_goal| 0.113 at step 76 (final 0.163); goal_error min 1.040 at step 74 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.269 (final 0.001) of target -1.309

### env 22 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 25, 1: 55, 2: 18, 3: 152
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.015 at step 85 (final 0.077)
- closest |object_to_goal| 0.144 at step 63 (final 0.147); goal_error min 1.296 at step 68 / final 1.310 (success < 0.200)
- mechanism joint: reached -0.013 (final 0.001) of target -1.309

### env 24 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 31, 1: 79, 2: 24, 3: 116
- aperture min -0.000 / max 0.082 / final -0.000
- closest |gripper_to_object| 0.005 at step 140 (final 0.079)
- closest |object_to_goal| 0.144 at step 68 (final 0.150); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 25 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 32, 1: 47, 2: 12, 3: 159
- aperture min -0.000 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.010 at step 98 (final 0.080)
- closest |object_to_goal| 0.142 at step 219 (final 0.159); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 27 — mechanism_short
- ended in phase 2 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 31, 1: 102, 2: 33, 3: 84
- aperture min 0.000 / max 0.084 / final 0.069
- closest |gripper_to_object| 0.010 at step 111 (final 0.016)
- closest |object_to_goal| 0.076 at step 192 (final 0.154); goal_error min 0.783 at step 191 / final 1.344 (success < 0.200)
- mechanism joint: reached -0.526 (final 0.035) of target -1.309

### env 28 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 28, 1: 54, 2: 18, 3: 150
- aperture min -0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.005 at step 91 (final 0.090)
- closest |object_to_goal| 0.144 at step 69 (final 0.164); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 29 — mechanism_not_moved
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 32, 1: 102, 2: 30, 3: 86
- aperture min 0.000 / max 0.081 / final 0.000
- closest |gripper_to_object| 0.003 at step 96 (final 0.079)
- closest |object_to_goal| 0.142 at step 10 (final 0.160); goal_error min 1.310 at step 17 / final 1.310 (success < 0.200)
- mechanism joint: reached 0.000 (final 0.001) of target -1.309

### env 30 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 1; steps per phase 0: 33, 1: 95, 2: 30, 3: 92
- aperture min 0.001 / max 0.083 / final 0.010
- closest |gripper_to_object| 0.010 at step 171 (final 0.063)
- closest |object_to_goal| 0.114 at step 230 (final 0.129); goal_error min 1.117 at step 248 / final 1.120 (success < 0.200)
- mechanism joint: reached -0.192 (final -0.189) of target -1.309

### env 31 — mechanism_short
- ended in phase 3 after 250 steps; ran to time-out
- most steps in phase 3; steps per phase 0: 28, 1: 6, 2: 6, 3: 210
- aperture min 0.025 / max 0.080 / final 0.029
- closest |gripper_to_object| 0.016 at step 34 (final 0.035)
- closest |object_to_goal| 0.020 at step 170 (final 0.028); goal_error min 0.354 at step 163 / final 0.380 (success < 0.200)
- mechanism joint: reached -0.955 (final -0.929) of target -1.309

## Successful envs (one line each)

- env 23: success at step 122, end phase 3, aperture min 0.022, min |go| 0.016
- env 26: success at step 125, end phase 3, aperture min 0.016, min |go| 0.015
