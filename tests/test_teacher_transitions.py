"""Gripper continuity at acquisition state-machine boundaries."""

import numpy as np
import pytest

from mjlab.continual_distill.classical.reorient_object import (
  CLOSE_SQUEEZE,
  GRIPPER_CLOSED,
  GRIPPER_OPEN,
  P_CLOSE,
  P_HOVER,
  P_LIFT,
  ReorientObjectClassicalPolicy,
)


@pytest.mark.parametrize(
  "aperture, expected_phase, expected_grip",
  [(0.052, P_LIFT, GRIPPER_CLOSED), (0.0, P_HOVER, GRIPPER_OPEN)],
)
def test_reorient_close_transition(aperture, expected_phase, expected_grip):
  policy = ReorientObjectClassicalPolicy(1)
  policy._phase[0] = P_CLOSE
  policy._closing[0] = True
  policy._close_k[0] = CLOSE_SQUEEZE - 1
  policy._ap_ema[0] = aperture
  policy._hold[0] = policy._fk(policy.default_qpos)[0]
  _, _, grip = policy._target_error(0, np.zeros(60))
  assert policy._phase[0] == expected_phase
  assert grip == expected_grip
