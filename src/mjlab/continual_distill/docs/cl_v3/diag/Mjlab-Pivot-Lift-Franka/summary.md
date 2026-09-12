# diagnose — Mjlab-Pivot-Lift-Franka

2026-09-09T14:15:15 · HEAD `b9b0563` · n = 32 (1 × 32 envs) · episode_length 300 · device cuda:0  
**0/32 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `PivotLiftClassicalPolicy`; primary object `object`; mechanism joint `None`; termination terms ['time_out', 'ee_ground_collision', 'object_out_of_bounds']; contact sensors ['ee_ground_collision'].  
Phases: 0 = HOVER, 1 = SEAT, 2 = PIVOT, 3 = CLEAR, 4 = RE_HOVER, 5 = DESCEND, 6 = CLOSE, 7 = LIFT

## Histograms

- end phase, FAILING envs (32): 7 (LIFT): 18, 5 (DESCEND): 9, 6 (CLOSE): 5
- end phase, successful envs (0): —
- most-steps phase, failing envs: 5 (DESCEND): 18, 2 (PIVOT): 14
- failure class: never_lifted: 24, never_reached: 8
- termination among failing envs: none: 32
- failing envs that lifted the object: 0/32; lifted then lost: 0

## Reading

- **never_lifted** — 24/32 failing envs (envs [0, 1, 2, 3, 5, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 26, 28, 29, 30, 31]); typical end phase 7 (LIFT); final aperture median 0.002; closest |gripper_to_object| median 0.039; closest |object_to_goal| median 0.241; min goal_error median 0.253 (success < 0.050)
- **never_reached** — 8/32 failing envs (envs [4, 6, 10, 11, 20, 22, 25, 27]); typical end phase 5 (DESCEND); final aperture median -0.011; closest |gripper_to_object| median 0.248; closest |object_to_goal| median 0.253; min goal_error median 0.265 (success < 0.050)

## Failing envs

### env 0 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 30, 1 (SEAT): 25, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 1
- aperture min -0.010 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.040 at step 142 (final 0.092)
- closest |object_to_goal| 0.241 at step 173 (final 0.255); goal_error min 0.251 at step 163 / final 0.251 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 1 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 33, 1 (SEAT): 16, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 8
- aperture min -0.009 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.045 at step 118 (final 0.081)
- closest |object_to_goal| 0.187 at step 41 (final 0.195); goal_error min 0.198 at step 14 / final 0.199 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 2 — never_lifted
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 38, 1 (SEAT): 17, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 25, 5 (DESCEND): 70, 6 (CLOSE): 15, 7 (LIFT): 25
- aperture min -0.009 / max 0.080 / final 0.078
- closest |gripper_to_object| 0.013 at step 250 (final 0.196)
- closest |object_to_goal| 0.247 at step 250 (final 0.261); goal_error min 0.257 at step 13 / final 0.258 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 3 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 34, 1 (SEAT): 18, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 5
- aperture min -0.009 / max 0.082 / final 0.001
- closest |gripper_to_object| 0.039 at step 120 (final 0.079)
- closest |object_to_goal| 0.215 at step 216 (final 0.223); goal_error min 0.226 at step 159 / final 0.226 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 4 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.012
- closest |gripper_to_object| 0.247 at step 78 (final 0.262)
- closest |object_to_goal| 0.263 at step 72 (final 0.277); goal_error min 0.274 at step 15 / final 0.274 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 5 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 33, 1 (SEAT): 20, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 4
- aperture min -0.009 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.039 at step 122 (final 0.090)
- closest |object_to_goal| 0.262 at step 284 (final 0.266); goal_error min 0.272 at step 15 / final 0.273 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 6 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.017 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.256 at step 179 (final 0.272)
- closest |object_to_goal| 0.205 at step 156 (final 0.227); goal_error min 0.216 at step 14 / final 0.216 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 7 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 36, 1 (SEAT): 18, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 25, 5 (DESCEND): 57, 6 (CLOSE): 29, 7 (LIFT): 25
- aperture min -0.009 / max 0.081 / final 0.062
- closest |gripper_to_object| 0.013 at step 210 (final 0.025)
- closest |object_to_goal| 0.242 at step 91 (final 0.265); goal_error min 0.255 at step 55 / final 0.255 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 8 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 38, 1 (SEAT): 16, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 3
- aperture min -0.009 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.039 at step 145 (final 0.075)
- closest |object_to_goal| 0.274 at step 225 (final 0.288); goal_error min 0.286 at step 162 / final 0.286 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 9 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 31, 1 (SEAT): 22, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 4
- aperture min -0.010 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.039 at step 129 (final 0.071)
- closest |object_to_goal| 0.203 at step 173 (final 0.223); goal_error min 0.215 at step 160 / final 0.215 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 10 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.242 at step 91 (final 0.259)
- closest |object_to_goal| 0.312 at step 203 (final 0.326); goal_error min 0.325 at step 12 / final 0.325 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 11 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.012
- closest |gripper_to_object| 0.249 at step 185 (final 0.275)
- closest |object_to_goal| 0.329 at step 100 (final 0.338); goal_error min 0.342 at step 14 / final 0.342 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 12 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 35, 1 (SEAT): 19, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 2
- aperture min -0.010 / max 0.088 / final 0.002
- closest |gripper_to_object| 0.041 at step 146 (final 0.095)
- closest |object_to_goal| 0.282 at step 28 (final 0.286); goal_error min 0.293 at step 15 / final 0.294 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 13 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 31, 1 (SEAT): 16, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 10
- aperture min -0.009 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.044 at step 127 (final 0.069)
- closest |object_to_goal| 0.321 at step 26 (final 0.334); goal_error min 0.332 at step 12 / final 0.333 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 14 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 32, 1 (SEAT): 18, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 6
- aperture min -0.009 / max 0.080 / final 0.001
- closest |gripper_to_object| 0.046 at step 124 (final 0.089)
- closest |object_to_goal| 0.233 at step 168 (final 0.254); goal_error min 0.245 at step 14 / final 0.246 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 15 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 34, 1 (SEAT): 17, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 24, 5 (DESCEND): 59, 6 (CLOSE): 30, 7 (LIFT): 26
- aperture min -0.009 / max 0.082 / final 0.075
- closest |gripper_to_object| 0.017 at step 220 (final 0.033)
- closest |object_to_goal| 0.171 at step 10 (final 0.180); goal_error min 0.183 at step 14 / final 0.185 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 16 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 31, 1 (SEAT): 16, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 9
- aperture min -0.009 / max 0.084 / final 0.000
- closest |gripper_to_object| 0.042 at step 99 (final 0.120)
- closest |object_to_goal| 0.259 at step 238 (final 0.274); goal_error min 0.271 at step 16 / final 0.271 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 17 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 36, 1 (SEAT): 17, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 3
- aperture min -0.009 / max 0.091 / final 0.001
- closest |gripper_to_object| 0.042 at step 133 (final 0.085)
- closest |object_to_goal| 0.181 at step 164 (final 0.195); goal_error min 0.194 at step 14 / final 0.194 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 18 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 33, 1 (SEAT): 15, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 8
- aperture min -0.011 / max 0.080 / final 0.000
- closest |gripper_to_object| 0.038 at step 116 (final 0.112)
- closest |object_to_goal| 0.244 at step 8 (final 0.266); goal_error min 0.257 at step 155 / final 0.257 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 19 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 30, 1 (SEAT): 19, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 22, 5 (DESCEND): 57, 6 (CLOSE): 30, 7 (LIFT): 32
- aperture min -0.009 / max 0.081 / final 0.027
- closest |gripper_to_object| 0.017 at step 277 (final 0.054)
- closest |object_to_goal| 0.176 at step 182 (final 0.190); goal_error min 0.189 at step 299 / final 0.189 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 20 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.252 at step 197 (final 0.262)
- closest |object_to_goal| 0.243 at step 295 (final 0.252); goal_error min 0.256 at step 12 / final 0.256 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 21 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 35, 1 (SEAT): 19, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 3
- aperture min -0.009 / max 0.085 / final 0.001
- closest |gripper_to_object| 0.042 at step 135 (final 0.077)
- closest |object_to_goal| 0.266 at step 220 (final 0.275); goal_error min 0.278 at step 162 / final 0.278 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 22 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.242 at step 199 (final 0.247)
- closest |object_to_goal| 0.236 at step 272 (final 0.256); goal_error min 0.246 at step 14 / final 0.246 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 23 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 40, 1 (SEAT): 16, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 24, 5 (DESCEND): 66, 6 (CLOSE): 19, 7 (LIFT): 25
- aperture min -0.010 / max 0.080 / final 0.073
- closest |gripper_to_object| 0.016 at step 296 (final 0.027)
- closest |object_to_goal| 0.203 at step 193 (final 0.221); goal_error min 0.215 at step 13 / final 0.217 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 24 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 33, 1 (SEAT): 23, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 1
- aperture min -0.010 / max 0.088 / final 0.002
- closest |gripper_to_object| 0.043 at step 127 (final 0.070)
- closest |object_to_goal| 0.277 at step 228 (final 0.297); goal_error min 0.291 at step 166 / final 0.291 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 25 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.014 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.238 at step 176 (final 0.246)
- closest |object_to_goal| 0.236 at step 258 (final 0.239); goal_error min 0.249 at step 14 / final 0.249 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 26 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 37, 1 (SEAT): 18, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 2
- aperture min -0.011 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.039 at step 132 (final 0.070)
- closest |object_to_goal| 0.253 at step 166 (final 0.256); goal_error min 0.263 at step 15 / final 0.264 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 27 — never_reached
- ended in phase 5 (DESCEND) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 41, 1 (SEAT): 31, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 61, 5 (DESCEND): 57
- aperture min -0.015 / max 0.076 / final -0.011
- closest |gripper_to_object| 0.260 at step 177 (final 0.270)
- closest |object_to_goal| 0.277 at step 78 (final 0.284); goal_error min 0.288 at step 15 / final 0.288 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 28 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 40, 1 (SEAT): 19, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 13
- aperture min -0.009 / max 0.084 / final 0.005
- closest |gripper_to_object| 0.043 at step 141 (final 0.075)
- closest |object_to_goal| 0.203 at step 170 (final 0.216); goal_error min 0.213 at step 167 / final 0.213 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 29 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 37, 1 (SEAT): 20, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 17, 5 (DESCEND): 101, 6 (CLOSE): 15
- aperture min -0.009 / max 0.090 / final 0.003
- closest |gripper_to_object| 0.040 at step 141 (final 0.080)
- closest |object_to_goal| 0.251 at step 15 (final 0.259); goal_error min 0.264 at step 13 / final 0.264 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 30 — never_lifted
- ended in phase 7 (LIFT) after 300 steps; ran to time-out
- most steps in phase 5 (DESCEND); steps per phase 0 (HOVER): 34, 1 (SEAT): 21, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 18, 5 (DESCEND): 101, 6 (CLOSE): 15, 7 (LIFT): 1
- aperture min -0.010 / max 0.080 / final 0.002
- closest |gripper_to_object| 0.039 at step 144 (final 0.100)
- closest |object_to_goal| 0.182 at step 243 (final 0.205); goal_error min 0.193 at step 16 / final 0.195 (success < 0.050)
- object never lifted (max rise 0.000 m)

### env 31 — never_lifted
- ended in phase 6 (CLOSE) after 300 steps; ran to time-out
- most steps in phase 2 (PIVOT); steps per phase 0 (HOVER): 39, 1 (SEAT): 17, 2 (PIVOT): 90, 3 (CLEAR): 20, 4 (RE_HOVER): 24, 5 (DESCEND): 62, 6 (CLOSE): 23, 7 (LIFT): 25
- aperture min -0.009 / max 0.088 / final 0.064
- closest |gripper_to_object| 0.015 at step 215 (final 0.024)
- closest |object_to_goal| 0.164 at step 276 (final 0.169); goal_error min 0.174 at step 163 / final 0.176 (success < 0.050)
- object never lifted (max rise 0.000 m)

## Successful envs (one line each)

