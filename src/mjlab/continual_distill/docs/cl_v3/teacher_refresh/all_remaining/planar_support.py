"""Candidate: analytical gravity support replaces the learned height bias."""
from mjlab.continual_distill.classical.push_cuboid import PushCuboidClassicalPolicy as Push
from mjlab.continual_distill.classical.drag_pull import DragPullClassicalPolicy as Drag
class Support:
  gravity_compensation=True
  Z_KI=0.
  Z_BIAS_MAX=0.
  Z_FF_PER_LEAD=0.
class PushCuboidClassicalPolicy(Support,Push):
  pass
class DragPullClassicalPolicy(Support,Drag):
  pass
