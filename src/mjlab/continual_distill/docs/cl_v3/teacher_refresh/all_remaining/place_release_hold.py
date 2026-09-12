"""Hold the actual release command while opening inside the basket."""
import numpy as np
from mjlab.continual_distill.classical.place_in_container import PlaceInContainerClassicalPolicy as Original
from mjlab.continual_distill.classical.base import FRANKA_ACTION_SCALE
class PlaceInContainerClassicalPolicy(Original):
  def reset(self,env_ids=None):
    super().reset(env_ids)
    if env_ids is None:self.release_q=np.full((self.num_envs,7),np.nan)
    else:self.release_q[env_ids]=np.nan
  def _act_single(self,i,obs):
    if self._phase[i]==6 and self._phase_steps[i]<60:
      if np.isnan(self.release_q[i]).any():self.release_q[i]=self._q_cmd[i]
      a=np.zeros(8,np.float32)
      a[:7]=(self.release_q[i]-self.default_qpos[:7])/FRANKA_ACTION_SCALE
      a[7]=1.
      return a
    self.release_q[i]=np.nan
    return super()._act_single(i,obs)
