"""Hand-coded (classical) teacher policies for continual distillation.

Each policy is a drop-in replacement for an NN teacher: it receives only the
policy observation vector and returns joint-position actions. All logic
(state machine, differential IK on a private MuJoCo model) lives inside.

Measured success rates for every task live in
``continual_distill/docs/benchmark/CLASSICAL_TEACHERS.md``. Several teachers here are
weak or failing and are documented as such — check the table before assuming a task has
a usable teacher.
"""

from mjlab.continual_distill.classical.axial_extract import (
  AxialExtractClassicalPolicy,
)
from mjlab.continual_distill.classical.cage_drag import CageDragClassicalPolicy
from mjlab.continual_distill.classical.drag_pull import DragPullClassicalPolicy
from mjlab.continual_distill.classical.edge_grasp import EdgeGraspClassicalPolicy
from mjlab.continual_distill.classical.flip_switch import FlipSwitchClassicalPolicy
from mjlab.continual_distill.classical.lift_object import (
  LiftCubeClassicalPolicy,
  LiftCylinderClassicalPolicy,
  LiftEllipsoidClassicalPolicy,
  LiftObjectClassicalPolicy,
  LiftSphereClassicalPolicy,
)
from mjlab.continual_distill.classical.open_door import OpenDoorClassicalPolicy
from mjlab.continual_distill.classical.open_drawer import OpenDrawerClassicalPolicy
from mjlab.continual_distill.classical.open_lid import OpenLidClassicalPolicy
from mjlab.continual_distill.classical.peg_insertion import PegInsertionClassicalPolicy
from mjlab.continual_distill.classical.place_in_container import (
  PlaceInContainerClassicalPolicy,
)
from mjlab.continual_distill.classical.push_button import PushButtonClassicalPolicy
from mjlab.continual_distill.classical.push_cuboid import PushCuboidClassicalPolicy
from mjlab.continual_distill.classical.push_disc import PushDiscClassicalPolicy
from mjlab.continual_distill.classical.push_flap import PushFlapClassicalPolicy
from mjlab.continual_distill.classical.reach_target import ReachTargetClassicalPolicy
from mjlab.continual_distill.classical.reorient_object import (
  ReorientObjectClassicalPolicy,
)
from mjlab.continual_distill.classical.rotate_valve import RotateValveClassicalPolicy
from mjlab.continual_distill.classical.slide_window import SlideWindowClassicalPolicy
from mjlab.continual_distill.classical.stack_object import StackObjectClassicalPolicy
from mjlab.continual_distill.classical.strike_slide import StrikeSlideClassicalPolicy
from mjlab.continual_distill.classical.pivot_lift import PivotLiftClassicalPolicy
from mjlab.continual_distill.classical.throw_to_bin import ThrowToBinClassicalPolicy
from mjlab.continual_distill.classical.tool_pull import ToolPullClassicalPolicy
from mjlab.continual_distill.classical.topple_block import ToppleBlockClassicalPolicy
from mjlab.continual_distill.classical.turn_lever import TurnLeverClassicalPolicy

#: Task ID -> classical teacher. The single source of truth for which tasks have a
#: scripted teacher; ``test_classical`` and dataset collection both read this.
CLASSICAL_POLICIES = {
  "Mjlab-Drag-Pull-Franka": DragPullClassicalPolicy,
  "Mjlab-Cage-Drag-Franka": CageDragClassicalPolicy,
  "Mjlab-Reach-Target-Franka": ReachTargetClassicalPolicy,
  "Mjlab-Lift-Cube-Franka": LiftCubeClassicalPolicy,
  "Mjlab-Lift-Cylinder-Franka": LiftCylinderClassicalPolicy,
  "Mjlab-Lift-Sphere-Franka": LiftSphereClassicalPolicy,
  "Mjlab-Lift-Ellipsoid-Franka": LiftEllipsoidClassicalPolicy,
  "Mjlab-Push-Cuboid-Franka": PushCuboidClassicalPolicy,
  "Mjlab-Push-Disc-Franka": PushDiscClassicalPolicy,
  "Mjlab-Push-Button-Franka": PushButtonClassicalPolicy,
  "Mjlab-Open-Door-Franka": OpenDoorClassicalPolicy,
  "Mjlab-Open-Drawer-Franka": OpenDrawerClassicalPolicy,
  "Mjlab-Turn-Lever-Franka": TurnLeverClassicalPolicy,
  "Mjlab-Rotate-Valve-Franka": RotateValveClassicalPolicy,
  "Mjlab-Flip-Switch-Franka": FlipSwitchClassicalPolicy,
  "Mjlab-Slide-Window-Franka": SlideWindowClassicalPolicy,
  "Mjlab-Open-Lid-Franka": OpenLidClassicalPolicy,
  "Mjlab-Stack-Cube-Franka": StackObjectClassicalPolicy,
  "Mjlab-Peg-Insertion-Franka": PegInsertionClassicalPolicy,
  "Mjlab-Place-In-Container-Franka": PlaceInContainerClassicalPolicy,
  "Mjlab-Reorient-Object-Franka": ReorientObjectClassicalPolicy,
  "Mjlab-Tool-Pull-Franka": ToolPullClassicalPolicy,
  "Mjlab-Topple-Block-Franka": ToppleBlockClassicalPolicy,
  "Mjlab-Push-Flap-Franka": PushFlapClassicalPolicy,
  "Mjlab-Axial-Extract-Franka": AxialExtractClassicalPolicy,
  "Mjlab-Edge-Grasp-Franka": EdgeGraspClassicalPolicy,
  "Mjlab-Strike-Slide-Franka": StrikeSlideClassicalPolicy,
  "Mjlab-Pivot-Lift-Franka": PivotLiftClassicalPolicy,
  "Mjlab-Throw-To-Bin-Franka": ThrowToBinClassicalPolicy,
}

__all__ = [
  "CLASSICAL_POLICIES",
  "AxialExtractClassicalPolicy",
  "CageDragClassicalPolicy",
  "DragPullClassicalPolicy",
  "EdgeGraspClassicalPolicy",
  "FlipSwitchClassicalPolicy",
  "LiftCubeClassicalPolicy",
  "LiftCylinderClassicalPolicy",
  "LiftEllipsoidClassicalPolicy",
  "LiftObjectClassicalPolicy",
  "LiftSphereClassicalPolicy",
  "OpenDoorClassicalPolicy",
  "OpenDrawerClassicalPolicy",
  "OpenLidClassicalPolicy",
  "PegInsertionClassicalPolicy",
  "PivotLiftClassicalPolicy",
  "PlaceInContainerClassicalPolicy",
  "PushButtonClassicalPolicy",
  "PushCuboidClassicalPolicy",
  "PushDiscClassicalPolicy",
  "PushFlapClassicalPolicy",
  "ReachTargetClassicalPolicy",
  "ReorientObjectClassicalPolicy",
  "RotateValveClassicalPolicy",
  "SlideWindowClassicalPolicy",
  "StackObjectClassicalPolicy",
  "StrikeSlideClassicalPolicy",
  "ThrowToBinClassicalPolicy",
  "ToolPullClassicalPolicy",
  "ToppleBlockClassicalPolicy",
  "TurnLeverClassicalPolicy",
]
