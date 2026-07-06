"""Hand-coded (classical) teacher policies for continual distillation.

Each policy is a drop-in replacement for an NN teacher: it receives only the
policy observation vector and returns joint-position actions. All logic
(state machine, differential IK on a private MuJoCo model) lives inside.
"""

from mjlab.continual_distill.classical.open_door import OpenDoorClassicalPolicy
from mjlab.continual_distill.classical.open_drawer import OpenDrawerClassicalPolicy
from mjlab.continual_distill.classical.push_button import PushButtonClassicalPolicy
from mjlab.continual_distill.classical.push_cuboid import PushCuboidClassicalPolicy

__all__ = [
  "OpenDoorClassicalPolicy",
  "OpenDrawerClassicalPolicy",
  "PushButtonClassicalPolicy",
  "PushCuboidClassicalPolicy",
]
