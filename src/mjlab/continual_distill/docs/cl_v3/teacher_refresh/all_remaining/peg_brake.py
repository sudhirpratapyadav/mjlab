"""Apply verified acquisition braking to the existing peg transport controller."""
from mjlab.continual_distill.classical.peg_insertion import PegInsertionClassicalPolicy as Original
from mjlab.continual_distill.classical.base import lowest_hand_z
class PegInsertionClassicalPolicy(Original):
  def _target_error(self,i,obs):
    ph=int(self._phase[i])
    err,rot,grip=super()._target_error(i,obs)
    q=self.default_qpos+obs[:9]
    p,r=self._fk(q)
    if ph in (1,2):err[2]=max(self.lift_grasp_site_z-p[2],-.006 if p[2]<.15 else -.02)
    err[2]=max(err[2],.012-lowest_hand_z(p,r,max(q[7],q[8])))
    return err,rot,grip
