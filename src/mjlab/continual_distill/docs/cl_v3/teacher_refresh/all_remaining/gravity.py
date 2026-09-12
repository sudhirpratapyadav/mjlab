import mujoco
import numpy as np
from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.base import FRANKA_ACTION_SCALE
class GravitySupport:
  def _act_single(self,i,obs):
    action=super()._act_single(i,obs)
    q=self.default_qpos+obs[:9]
    self._fk(q)
    self.data.qvel[:]=0
    mujoco.mj_forward(self.model,self.data)
    gains=np.array([self.model.actuator("actuator"+str(k)).gainprm[0] for k in range(1,8)])
    offset=self.data.qfrc_bias[self._arm_dofadr]/gains
    action[:7]+=offset/FRANKA_ACTION_SCALE
    return action
for original in set(CLASSICAL_POLICIES.values()):
  globals()[original.__name__]=type(original.__name__,(GravitySupport,original),{})
