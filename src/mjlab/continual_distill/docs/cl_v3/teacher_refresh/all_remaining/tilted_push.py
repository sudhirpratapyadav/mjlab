"""Candidate: incline the pushing face to unload the floor contact."""
import numpy as np
from mjlab.continual_distill.classical.push_cuboid import PushCuboidClassicalPolicy as Original
class PushCuboidClassicalPolicy(Original):
  def _target_error(self,i,obs):
    err,rot,grip=super()._target_error(i,obs)
    if rot is not None and rot.ndim==2:
      a=np.deg2rad(15);c,s=np.cos(a),np.sin(a)
      rot=rot@np.array([[c,0,s],[0,1,0],[-s,0,c]])
    return err,rot,grip
