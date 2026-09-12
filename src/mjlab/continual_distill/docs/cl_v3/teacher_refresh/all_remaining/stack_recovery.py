"""Retry failed releases within the original episode budget."""
import numpy as np
from mjlab.continual_distill.classical.staged_stack import StagedStackPolicy
class StackObjectClassicalPolicy(StagedStackPolicy):
  def reset(self,env_ids=None):
    super().reset(env_ids)
    if env_ids is None:self.placed_error=np.zeros((self.num_envs,3));self.placed_n=np.zeros(self.num_envs,int)
    else:self.placed_error[env_ids]=0;self.placed_n[env_ids]=0
  def _act_single(self,i,obs):
    if self._phase[i]==6 and self._phase_steps[i]>60:
      self.placed_error[i]+=.2*(obs[40:43]-self.placed_error[i]);self.placed_n[i]+=1
      d=self.placed_error[i]
      if self.placed_n[i]>=20 and (np.linalg.norm(d[:2])>.04 or abs(d[2])>.02):
        self._rewind(i)
        self.goal_n[i]=0;self.hold_n[i]=0;self.integ[i]=0;self.stable[i]=0
        self.release_q[i]=np.nan;self.placed_error[i]=0;self.placed_n[i]=0
    return super()._act_single(i,obs)
