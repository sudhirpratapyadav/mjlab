"""Teachers must decode the actual target pose, including nonzero hinge offsets."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mjlab.continual_distill.classical.mechanism_geometry import joint_goal_displacement


@pytest.mark.parametrize('asset,joint,axis,radius,target', [
  ('switch', 'switch_hinge', [0, 1, 0], [0, 0, .062], .5235988),
  ('valve', 'valve_hinge', [1, 0, 0], [-.04, .09, 0], 4.712389),
  ('lid', 'lid_hinge', [0, -1, 0], [-.19, 0, .01], -1.308997),
  ('flap', 'flap_hinge', [0, 0, 1], [-.012, .18, 0], -1.2217305),
])
def test_goal_chord_matches_analytic_hinge_rotation(asset, joint, axis, radius, target):
  radius = np.array(radius)
  expected = Rotation.from_rotvec(np.array(axis) * target).apply(radius) - radius
  chord = joint_goal_displacement(asset, joint, target)
  np.testing.assert_allclose(chord, expected, atol=1e-8)
  assert not chord.flags.writeable
