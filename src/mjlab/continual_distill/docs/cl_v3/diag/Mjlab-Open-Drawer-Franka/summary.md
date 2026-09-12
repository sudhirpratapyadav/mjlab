# diagnose — Mjlab-Open-Drawer-Franka

2026-09-09T13:35:33 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 150 · device cuda:0  
**126/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `OpenDrawerClassicalPolicy`; primary object `drawer`; mechanism joint `drawer_slide`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_drawer_collision'].  
Phases: 0 = HOVER, 1 = DESCEND, 2 = PULL

## Histograms

- end phase, FAILING envs (2): 2 (PULL): 2
- end phase, successful envs (126): 2 (PULL): 125, 1 (DESCEND): 1
- most-steps phase, failing envs: 2 (PULL): 2
- failure class: mechanism_short: 2
- termination among failing envs: none: 2
- success step (successful envs): median 92, max 148

## Reading

- **mechanism_short** — 2/2 failing envs (envs [29, 116]); typical end phase 2 (PULL); final aperture median 0.000; closest |gripper_to_object| median 0.023; closest |object_to_goal| median 0.159; min goal_error median 0.166 (success < 0.020); mechanism reached median -0.084 of target -0.250

## Failing envs

### env 29 — mechanism_short
- ended in phase 2 (PULL) after 150 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (HOVER): 42, 1 (DESCEND): 17, 2 (PULL): 91
- aperture min -0.004 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.029 at step 60 (final 0.153)
- closest |object_to_goal| 0.173 at step 145 (final 0.177); goal_error min 0.179 at step 149 / final 0.179 (success < 0.020)
- mechanism joint: reached -0.071 (final -0.071) of target -0.250

### env 116 — mechanism_short
- ended in phase 2 (PULL) after 150 steps; ran to time-out
- most steps in phase 2 (PULL); steps per phase 0 (HOVER): 42, 1 (DESCEND): 17, 2 (PULL): 91
- aperture min -0.002 / max 0.076 / final 0.000
- closest |gripper_to_object| 0.017 at step 61 (final 0.158)
- closest |object_to_goal| 0.146 at step 149 (final 0.146); goal_error min 0.154 at step 149 / final 0.154 (success < 0.020)
- mechanism joint: reached -0.096 (final -0.096) of target -0.250

## Successful envs (one line each)

- env 0: success at step 92, end phase 2 (PULL), aperture min -0.001, min |go| 0.018
- env 1: success at step 82, end phase 2 (PULL), aperture min -0.007, min |go| 0.014
- env 2: success at step 81, end phase 2 (PULL), aperture min -0.004, min |go| 0.013
- env 3: success at step 94, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 4: success at step 100, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 5: success at step 96, end phase 2 (PULL), aperture min -0.001, min |go| 0.015
- env 6: success at step 92, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 7: success at step 110, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 8: success at step 90, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 9: success at step 89, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 10: success at step 98, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 11: success at step 103, end phase 2 (PULL), aperture min -0.003, min |go| 0.015
- env 12: success at step 102, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 13: success at step 86, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 14: success at step 81, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 15: success at step 103, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 16: success at step 79, end phase 2 (PULL), aperture min -0.007, min |go| 0.017
- env 17: success at step 113, end phase 2 (PULL), aperture min -0.008, min |go| 0.015
- env 18: success at step 88, end phase 2 (PULL), aperture min -0.005, min |go| 0.017
- env 19: success at step 91, end phase 2 (PULL), aperture min -0.006, min |go| 0.014
- env 20: success at step 95, end phase 2 (PULL), aperture min -0.016, min |go| 0.016
- env 21: success at step 86, end phase 2 (PULL), aperture min -0.004, min |go| 0.013
- env 22: success at step 93, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 23: success at step 86, end phase 2 (PULL), aperture min -0.004, min |go| 0.013
- env 24: success at step 99, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 25: success at step 97, end phase 2 (PULL), aperture min -0.008, min |go| 0.013
- env 26: success at step 103, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 27: success at step 127, end phase 2 (PULL), aperture min -0.004, min |go| 0.020
- env 28: success at step 80, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 30: success at step 102, end phase 2 (PULL), aperture min -0.010, min |go| 0.019
- env 31: success at step 88, end phase 2 (PULL), aperture min -0.003, min |go| 0.013
- env 32: success at step 84, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 33: success at step 100, end phase 2 (PULL), aperture min -0.002, min |go| 0.016
- env 34: success at step 97, end phase 2 (PULL), aperture min -0.006, min |go| 0.015
- env 35: success at step 91, end phase 2 (PULL), aperture min -0.006, min |go| 0.014
- env 36: success at step 91, end phase 2 (PULL), aperture min -0.002, min |go| 0.013
- env 37: success at step 76, end phase 2 (PULL), aperture min -0.012, min |go| 0.014
- env 38: success at step 90, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 39: success at step 86, end phase 2 (PULL), aperture min -0.008, min |go| 0.016
- env 40: success at step 90, end phase 2 (PULL), aperture min -0.005, min |go| 0.015
- env 41: success at step 99, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 42: success at step 90, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 43: success at step 95, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 44: success at step 80, end phase 2 (PULL), aperture min -0.006, min |go| 0.017
- env 45: success at step 105, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 46: success at step 83, end phase 2 (PULL), aperture min -0.005, min |go| 0.016
- env 47: success at step 81, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 48: success at step 100, end phase 2 (PULL), aperture min -0.003, min |go| 0.015
- env 49: success at step 85, end phase 2 (PULL), aperture min -0.003, min |go| 0.015
- env 50: success at step 148, end phase 2 (PULL), aperture min -0.000, min |go| 0.017
- env 51: success at step 79, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 52: success at step 101, end phase 2 (PULL), aperture min -0.004, min |go| 0.020
- env 53: success at step 91, end phase 2 (PULL), aperture min -0.005, min |go| 0.017
- env 54: success at step 95, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 55: success at step 104, end phase 2 (PULL), aperture min -0.006, min |go| 0.014
- env 56: success at step 92, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 57: success at step 81, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 58: success at step 137, end phase 2 (PULL), aperture min -0.000, min |go| 0.022
- env 59: success at step 148, end phase 2 (PULL), aperture min -0.003, min |go| 0.013
- env 60: success at step 80, end phase 2 (PULL), aperture min -0.003, min |go| 0.012
- env 61: success at step 105, end phase 2 (PULL), aperture min -0.000, min |go| 0.015
- env 62: success at step 78, end phase 2 (PULL), aperture min -0.004, min |go| 0.015
- env 63: success at step 90, end phase 1 (DESCEND), aperture min -0.003, min |go| 0.016
- env 64: success at step 108, end phase 2 (PULL), aperture min -0.002, min |go| 0.016
- env 65: success at step 100, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 66: success at step 96, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 67: success at step 94, end phase 2 (PULL), aperture min -0.006, min |go| 0.015
- env 68: success at step 93, end phase 2 (PULL), aperture min -0.007, min |go| 0.015
- env 69: success at step 94, end phase 2 (PULL), aperture min -0.012, min |go| 0.016
- env 70: success at step 83, end phase 2 (PULL), aperture min -0.005, min |go| 0.015
- env 71: success at step 78, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 72: success at step 96, end phase 2 (PULL), aperture min -0.006, min |go| 0.018
- env 73: success at step 145, end phase 2 (PULL), aperture min -0.012, min |go| 0.017
- env 74: success at step 90, end phase 2 (PULL), aperture min -0.003, min |go| 0.013
- env 75: success at step 87, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 76: success at step 83, end phase 2 (PULL), aperture min -0.007, min |go| 0.014
- env 77: success at step 89, end phase 2 (PULL), aperture min -0.006, min |go| 0.015
- env 78: success at step 86, end phase 2 (PULL), aperture min -0.006, min |go| 0.014
- env 79: success at step 98, end phase 2 (PULL), aperture min -0.002, min |go| 0.015
- env 80: success at step 96, end phase 2 (PULL), aperture min -0.006, min |go| 0.015
- env 81: success at step 128, end phase 2 (PULL), aperture min -0.000, min |go| 0.015
- env 82: success at step 83, end phase 2 (PULL), aperture min -0.007, min |go| 0.017
- env 83: success at step 86, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 84: success at step 102, end phase 2 (PULL), aperture min -0.005, min |go| 0.025
- env 85: success at step 87, end phase 2 (PULL), aperture min -0.010, min |go| 0.017
- env 86: success at step 87, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 87: success at step 80, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 88: success at step 105, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 89: success at step 86, end phase 2 (PULL), aperture min -0.012, min |go| 0.019
- env 90: success at step 91, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 91: success at step 145, end phase 2 (PULL), aperture min -0.001, min |go| 0.016
- env 92: success at step 102, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 93: success at step 82, end phase 2 (PULL), aperture min -0.005, min |go| 0.015
- env 94: success at step 72, end phase 2 (PULL), aperture min -0.012, min |go| 0.017
- env 95: success at step 87, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 96: success at step 84, end phase 2 (PULL), aperture min -0.002, min |go| 0.012
- env 97: success at step 98, end phase 2 (PULL), aperture min -0.003, min |go| 0.013
- env 98: success at step 98, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 99: success at step 79, end phase 2 (PULL), aperture min -0.003, min |go| 0.013
- env 100: success at step 83, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 101: success at step 91, end phase 2 (PULL), aperture min -0.006, min |go| 0.016
- env 102: success at step 104, end phase 2 (PULL), aperture min -0.004, min |go| 0.015
- env 103: success at step 75, end phase 2 (PULL), aperture min -0.004, min |go| 0.013
- env 104: success at step 79, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 105: success at step 77, end phase 2 (PULL), aperture min -0.002, min |go| 0.016
- env 106: success at step 94, end phase 2 (PULL), aperture min -0.001, min |go| 0.013
- env 107: success at step 83, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 108: success at step 108, end phase 2 (PULL), aperture min -0.003, min |go| 0.015
- env 109: success at step 77, end phase 2 (PULL), aperture min -0.004, min |go| 0.015
- env 110: success at step 121, end phase 2 (PULL), aperture min -0.004, min |go| 0.015
- env 111: success at step 96, end phase 2 (PULL), aperture min -0.005, min |go| 0.013
- env 112: success at step 100, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 113: success at step 88, end phase 2 (PULL), aperture min -0.001, min |go| 0.017
- env 114: success at step 144, end phase 2 (PULL), aperture min -0.001, min |go| 0.015
- env 115: success at step 93, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 117: success at step 87, end phase 2 (PULL), aperture min -0.002, min |go| 0.018
- env 118: success at step 83, end phase 2 (PULL), aperture min -0.010, min |go| 0.014
- env 119: success at step 94, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 120: success at step 113, end phase 2 (PULL), aperture min -0.004, min |go| 0.014
- env 121: success at step 106, end phase 2 (PULL), aperture min -0.005, min |go| 0.014
- env 122: success at step 106, end phase 2 (PULL), aperture min -0.005, min |go| 0.015
- env 123: success at step 138, end phase 2 (PULL), aperture min -0.003, min |go| 0.014
- env 124: success at step 102, end phase 2 (PULL), aperture min -0.002, min |go| 0.014
- env 125: success at step 141, end phase 2 (PULL), aperture min -0.006, min |go| 0.018
- env 126: success at step 93, end phase 2 (PULL), aperture min -0.006, min |go| 0.013
- env 127: success at step 109, end phase 2 (PULL), aperture min -0.005, min |go| 0.015
