import numpy as np
from mjlab.continual_distill.classical.lift_object import LiftCubeClassicalPolicy as Original
from mjlab.continual_distill.classical.base import lowest_hand_z
class LiftCubeClassicalPolicy(Original):
  def _target_error(self,i,obs):
    ph=int(self._phase[i])
    err,rot,grip=super()._target_error(i,obs)
    q=self.default_qpos+obs[:9]
    ee,r=self._fk(q)
    if ph in (1,2):
      err[2]=max(self.GRASP_SITE_Z-ee[2],-.006 if ee[2]<.12 else -.02)
    low=lowest_hand_z(ee,r,float(max(q[7],q[8])))
    err[2]=max(err[2],.012-low)
    return err,rot,grip
