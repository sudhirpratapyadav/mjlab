import mujoco
import numpy as np

from mjlab.sim import SimulationCfg
from mjlab.sim.free_body_implicitfast import standalone_free_dofs


def test_compatibility_is_opt_in():
  assert SimulationCfg().free_body_implicitfast_compat is False


def test_only_standalone_free_bodies_are_selected():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body name="leaf"><freejoint/><geom type="sphere" size=".1"/></body>
    <body name="arm"><joint type="hinge"/><geom type="sphere" size=".1"/></body>
    <body name="coupled"><freejoint/><geom type="sphere" size=".1"/>
      <body name="child" pos="0 0 .3"><joint/><geom type="sphere" size=".1"/></body>
    </body></worldbody></mujoco>''')
  selected = standalone_free_dofs(model)
  np.testing.assert_array_equal(selected[:6], np.ones(6, dtype=np.int32))
  assert not selected[6:].any()


def test_no_free_body_needs_no_correction():
  model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
    <body><joint/><geom type="sphere" size=".1"/></body>
    </worldbody></mujoco>''')
  assert not standalone_free_dofs(model).any()
