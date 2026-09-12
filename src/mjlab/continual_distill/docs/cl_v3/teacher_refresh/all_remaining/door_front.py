"""Use the measured front-pinch slider approach on the vertical door pull."""
from mjlab.continual_distill.classical.front_pinch_slider import FrontPinchSliderPolicy
class OpenDoorClassicalPolicy(FrontPinchSliderPolicy):
  VERTICAL_BAR=True
  TILT=70.
  SEAT_X=-.015
