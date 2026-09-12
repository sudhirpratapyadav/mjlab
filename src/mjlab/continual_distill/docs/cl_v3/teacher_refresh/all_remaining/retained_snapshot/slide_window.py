"""Slide the sash using a front pinch on its exposed vertical handle.

A side approach lets the collidable finger body hit the sash stile. Approach from
mount -x with a wrist tilted 70 degrees downward, close across the bar's width,
then retain the measured grasp offset while translating toward the goal. The
mount yaw is estimated from policy observations; no simulator state is read.
"""
from mjlab.continual_distill.classical.front_pinch_slider import FrontPinchSliderPolicy

PHASE_NAMES = {0: 'APPROACH', 1: 'SEAT', 2: 'CLOSE', 3: 'TRANSPORT'}


class SlideWindowClassicalPolicy(FrontPinchSliderPolicy):
  VERTICAL_BAR = True
  TILT = 70.0
