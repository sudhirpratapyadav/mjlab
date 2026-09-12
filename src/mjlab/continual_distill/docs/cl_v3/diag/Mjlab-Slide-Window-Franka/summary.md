# diagnose — Mjlab-Slide-Window-Franka

2026-09-09T13:58:19 · HEAD `b9b0563` · n = 128 (4 × 32 envs) · episode_length 150 · device cuda:0  
**128/128 succeeded** (orientation only — quote SR from `test_classical` at n = 128 on the frozen spec).

Obs slices used: robot_joint_pos 0:9, gripper_pos 25:28, gripper_to_object 40:43, object_to_goal 43:46  
Teacher `SlideWindowClassicalPolicy`; primary object `asset`; mechanism joint `window_slide`; termination terms ['time_out', 'ee_ground_collision']; contact sensors ['ee_ground_collision', 'ee_window_collision'].  
Phases: 0 = ALIGN, 1 = SEAT, 2 = PUSH

## Histograms

- end phase, FAILING envs (0): —
- end phase, successful envs (128): 2 (PUSH): 109, 1 (SEAT): 19
- most-steps phase, failing envs: —
- failure class: —
- termination among failing envs: —
- success step (successful envs): median 58, max 74

## Failing envs

(none)

## Successful envs (one line each)

- env 0: success at step 55, end phase 2 (PUSH), aperture min -0.016, min |go| 0.034
- env 1: success at step 60, end phase 1 (SEAT), aperture min -0.024, min |go| 0.013
- env 2: success at step 58, end phase 2 (PUSH), aperture min -0.016, min |go| 0.027
- env 3: success at step 52, end phase 2 (PUSH), aperture min -0.011, min |go| 0.030
- env 4: success at step 58, end phase 2 (PUSH), aperture min -0.010, min |go| 0.044
- env 5: success at step 59, end phase 1 (SEAT), aperture min -0.014, min |go| 0.033
- env 6: success at step 56, end phase 2 (PUSH), aperture min -0.018, min |go| 0.031
- env 7: success at step 55, end phase 2 (PUSH), aperture min -0.008, min |go| 0.021
- env 8: success at step 55, end phase 2 (PUSH), aperture min -0.017, min |go| 0.017
- env 9: success at step 58, end phase 2 (PUSH), aperture min -0.015, min |go| 0.030
- env 10: success at step 58, end phase 2 (PUSH), aperture min -0.015, min |go| 0.034
- env 11: success at step 57, end phase 2 (PUSH), aperture min -0.016, min |go| 0.034
- env 12: success at step 58, end phase 2 (PUSH), aperture min -0.010, min |go| 0.041
- env 13: success at step 52, end phase 2 (PUSH), aperture min -0.019, min |go| 0.016
- env 14: success at step 61, end phase 2 (PUSH), aperture min -0.013, min |go| 0.018
- env 15: success at step 58, end phase 2 (PUSH), aperture min -0.011, min |go| 0.017
- env 16: success at step 57, end phase 2 (PUSH), aperture min -0.012, min |go| 0.020
- env 17: success at step 59, end phase 1 (SEAT), aperture min -0.007, min |go| 0.031
- env 18: success at step 56, end phase 2 (PUSH), aperture min -0.016, min |go| 0.018
- env 19: success at step 55, end phase 1 (SEAT), aperture min -0.017, min |go| 0.050
- env 20: success at step 55, end phase 2 (PUSH), aperture min -0.008, min |go| 0.035
- env 21: success at step 60, end phase 2 (PUSH), aperture min -0.015, min |go| 0.025
- env 22: success at step 58, end phase 2 (PUSH), aperture min -0.017, min |go| 0.020
- env 23: success at step 61, end phase 2 (PUSH), aperture min -0.007, min |go| 0.023
- env 24: success at step 59, end phase 1 (SEAT), aperture min -0.020, min |go| 0.017
- env 25: success at step 53, end phase 2 (PUSH), aperture min -0.008, min |go| 0.036
- env 26: success at step 59, end phase 2 (PUSH), aperture min -0.014, min |go| 0.034
- env 27: success at step 58, end phase 2 (PUSH), aperture min -0.017, min |go| 0.016
- env 28: success at step 53, end phase 2 (PUSH), aperture min -0.015, min |go| 0.023
- env 29: success at step 62, end phase 2 (PUSH), aperture min -0.012, min |go| 0.022
- env 30: success at step 59, end phase 2 (PUSH), aperture min -0.014, min |go| 0.029
- env 31: success at step 48, end phase 2 (PUSH), aperture min -0.018, min |go| 0.027
- env 32: success at step 56, end phase 2 (PUSH), aperture min -0.011, min |go| 0.029
- env 33: success at step 54, end phase 1 (SEAT), aperture min -0.019, min |go| 0.023
- env 34: success at step 51, end phase 1 (SEAT), aperture min -0.013, min |go| 0.049
- env 35: success at step 55, end phase 2 (PUSH), aperture min -0.016, min |go| 0.022
- env 36: success at step 62, end phase 2 (PUSH), aperture min -0.016, min |go| 0.019
- env 37: success at step 54, end phase 1 (SEAT), aperture min -0.015, min |go| 0.038
- env 38: success at step 57, end phase 2 (PUSH), aperture min -0.011, min |go| 0.041
- env 39: success at step 54, end phase 2 (PUSH), aperture min -0.019, min |go| 0.017
- env 40: success at step 58, end phase 2 (PUSH), aperture min -0.016, min |go| 0.040
- env 41: success at step 56, end phase 2 (PUSH), aperture min -0.019, min |go| 0.028
- env 42: success at step 58, end phase 2 (PUSH), aperture min -0.020, min |go| 0.024
- env 43: success at step 61, end phase 2 (PUSH), aperture min -0.017, min |go| 0.018
- env 44: success at step 54, end phase 1 (SEAT), aperture min -0.005, min |go| 0.060
- env 45: success at step 55, end phase 2 (PUSH), aperture min -0.016, min |go| 0.026
- env 46: success at step 58, end phase 2 (PUSH), aperture min -0.013, min |go| 0.021
- env 47: success at step 62, end phase 2 (PUSH), aperture min -0.012, min |go| 0.019
- env 48: success at step 54, end phase 2 (PUSH), aperture min -0.017, min |go| 0.022
- env 49: success at step 58, end phase 1 (SEAT), aperture min -0.017, min |go| 0.028
- env 50: success at step 59, end phase 2 (PUSH), aperture min -0.017, min |go| 0.020
- env 51: success at step 62, end phase 2 (PUSH), aperture min -0.016, min |go| 0.027
- env 52: success at step 65, end phase 2 (PUSH), aperture min -0.015, min |go| 0.022
- env 53: success at step 60, end phase 2 (PUSH), aperture min -0.016, min |go| 0.018
- env 54: success at step 53, end phase 1 (SEAT), aperture min -0.018, min |go| 0.022
- env 55: success at step 63, end phase 1 (SEAT), aperture min -0.024, min |go| 0.033
- env 56: success at step 60, end phase 2 (PUSH), aperture min -0.017, min |go| 0.026
- env 57: success at step 53, end phase 1 (SEAT), aperture min -0.015, min |go| 0.014
- env 58: success at step 55, end phase 2 (PUSH), aperture min -0.015, min |go| 0.026
- env 59: success at step 56, end phase 2 (PUSH), aperture min -0.016, min |go| 0.016
- env 60: success at step 56, end phase 2 (PUSH), aperture min -0.010, min |go| 0.022
- env 61: success at step 60, end phase 2 (PUSH), aperture min -0.007, min |go| 0.040
- env 62: success at step 61, end phase 2 (PUSH), aperture min -0.014, min |go| 0.022
- env 63: success at step 68, end phase 2 (PUSH), aperture min -0.012, min |go| 0.040
- env 64: success at step 61, end phase 2 (PUSH), aperture min -0.014, min |go| 0.035
- env 65: success at step 62, end phase 2 (PUSH), aperture min -0.015, min |go| 0.029
- env 66: success at step 61, end phase 2 (PUSH), aperture min -0.019, min |go| 0.021
- env 67: success at step 57, end phase 2 (PUSH), aperture min -0.015, min |go| 0.039
- env 68: success at step 64, end phase 2 (PUSH), aperture min -0.015, min |go| 0.019
- env 69: success at step 63, end phase 2 (PUSH), aperture min -0.018, min |go| 0.022
- env 70: success at step 55, end phase 2 (PUSH), aperture min -0.010, min |go| 0.038
- env 71: success at step 52, end phase 2 (PUSH), aperture min -0.009, min |go| 0.036
- env 72: success at step 57, end phase 2 (PUSH), aperture min -0.009, min |go| 0.024
- env 73: success at step 53, end phase 2 (PUSH), aperture min -0.023, min |go| 0.027
- env 74: success at step 55, end phase 2 (PUSH), aperture min -0.014, min |go| 0.029
- env 75: success at step 58, end phase 2 (PUSH), aperture min -0.015, min |go| 0.035
- env 76: success at step 64, end phase 2 (PUSH), aperture min -0.016, min |go| 0.038
- env 77: success at step 52, end phase 2 (PUSH), aperture min -0.013, min |go| 0.018
- env 78: success at step 59, end phase 2 (PUSH), aperture min -0.013, min |go| 0.032
- env 79: success at step 60, end phase 2 (PUSH), aperture min -0.015, min |go| 0.029
- env 80: success at step 57, end phase 2 (PUSH), aperture min -0.011, min |go| 0.041
- env 81: success at step 55, end phase 1 (SEAT), aperture min -0.018, min |go| 0.030
- env 82: success at step 52, end phase 2 (PUSH), aperture min -0.016, min |go| 0.030
- env 83: success at step 51, end phase 2 (PUSH), aperture min -0.013, min |go| 0.029
- env 84: success at step 54, end phase 2 (PUSH), aperture min -0.011, min |go| 0.022
- env 85: success at step 56, end phase 2 (PUSH), aperture min -0.014, min |go| 0.020
- env 86: success at step 63, end phase 2 (PUSH), aperture min -0.008, min |go| 0.022
- env 87: success at step 61, end phase 2 (PUSH), aperture min -0.014, min |go| 0.036
- env 88: success at step 58, end phase 2 (PUSH), aperture min -0.015, min |go| 0.036
- env 89: success at step 74, end phase 2 (PUSH), aperture min -0.008, min |go| 0.013
- env 90: success at step 54, end phase 2 (PUSH), aperture min -0.010, min |go| 0.025
- env 91: success at step 55, end phase 2 (PUSH), aperture min -0.014, min |go| 0.018
- env 92: success at step 55, end phase 2 (PUSH), aperture min -0.009, min |go| 0.034
- env 93: success at step 61, end phase 1 (SEAT), aperture min -0.016, min |go| 0.037
- env 94: success at step 63, end phase 2 (PUSH), aperture min -0.015, min |go| 0.022
- env 95: success at step 55, end phase 1 (SEAT), aperture min -0.021, min |go| 0.044
- env 96: success at step 63, end phase 1 (SEAT), aperture min -0.015, min |go| 0.025
- env 97: success at step 53, end phase 2 (PUSH), aperture min -0.015, min |go| 0.043
- env 98: success at step 55, end phase 1 (SEAT), aperture min -0.016, min |go| 0.034
- env 99: success at step 59, end phase 2 (PUSH), aperture min -0.014, min |go| 0.031
- env 100: success at step 60, end phase 2 (PUSH), aperture min -0.018, min |go| 0.023
- env 101: success at step 62, end phase 1 (SEAT), aperture min -0.015, min |go| 0.023
- env 102: success at step 60, end phase 2 (PUSH), aperture min -0.016, min |go| 0.027
- env 103: success at step 58, end phase 2 (PUSH), aperture min -0.016, min |go| 0.039
- env 104: success at step 61, end phase 2 (PUSH), aperture min -0.015, min |go| 0.019
- env 105: success at step 62, end phase 2 (PUSH), aperture min -0.018, min |go| 0.017
- env 106: success at step 59, end phase 2 (PUSH), aperture min -0.017, min |go| 0.022
- env 107: success at step 54, end phase 2 (PUSH), aperture min -0.006, min |go| 0.046
- env 108: success at step 48, end phase 2 (PUSH), aperture min -0.021, min |go| 0.022
- env 109: success at step 63, end phase 2 (PUSH), aperture min -0.006, min |go| 0.015
- env 110: success at step 55, end phase 2 (PUSH), aperture min -0.019, min |go| 0.019
- env 111: success at step 59, end phase 2 (PUSH), aperture min -0.020, min |go| 0.018
- env 112: success at step 52, end phase 2 (PUSH), aperture min -0.018, min |go| 0.019
- env 113: success at step 61, end phase 2 (PUSH), aperture min -0.017, min |go| 0.031
- env 114: success at step 57, end phase 2 (PUSH), aperture min -0.013, min |go| 0.018
- env 115: success at step 60, end phase 2 (PUSH), aperture min -0.011, min |go| 0.020
- env 116: success at step 62, end phase 2 (PUSH), aperture min -0.020, min |go| 0.056
- env 117: success at step 58, end phase 2 (PUSH), aperture min -0.013, min |go| 0.020
- env 118: success at step 59, end phase 2 (PUSH), aperture min -0.017, min |go| 0.054
- env 119: success at step 59, end phase 2 (PUSH), aperture min -0.014, min |go| 0.046
- env 120: success at step 58, end phase 2 (PUSH), aperture min -0.010, min |go| 0.043
- env 121: success at step 62, end phase 2 (PUSH), aperture min -0.011, min |go| 0.038
- env 122: success at step 59, end phase 2 (PUSH), aperture min -0.016, min |go| 0.021
- env 123: success at step 54, end phase 2 (PUSH), aperture min -0.014, min |go| 0.024
- env 124: success at step 59, end phase 2 (PUSH), aperture min -0.006, min |go| 0.032
- env 125: success at step 59, end phase 2 (PUSH), aperture min -0.017, min |go| 0.018
- env 126: success at step 58, end phase 2 (PUSH), aperture min -0.019, min |go| 0.017
- env 127: success at step 57, end phase 2 (PUSH), aperture min -0.015, min |go| 0.025
