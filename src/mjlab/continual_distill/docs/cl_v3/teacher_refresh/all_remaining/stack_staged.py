"""Candidate: safe cube acquisition, raised transport, slow placement, joint hold on release."""
import numpy as np
from mjlab.continual_distill.classical.lift_object import LiftCubeClassicalPolicy
from mjlab.continual_distill.classical.base import FRANKA_ACTION_SCALE
DOWN=np.array([0.,0.,-1.])
class StackObjectClassicalPolicy(LiftCubeClassicalPolicy):
  def reset(self,env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      n=self.num_envs
      self.goal=np.zeros((n,3));self.goal_n=np.zeros(n,int)
      self.hold=np.zeros((n,3));self.hold_n=np.zeros(n,int)
      self.integ=np.zeros((n,3));self.stable=np.zeros(n,int)
      self.release_q=np.full((n,7),np.nan)
    else:
      self.goal_n[env_ids]=0;self.hold_n[env_ids]=0;self.integ[env_ids]=0;self.stable[env_ids]=0;self.release_q[env_ids]=np.nan
  def _act_single(self,i,obs):
    if self._phase[i]==6:
      if np.isnan(self.release_q[i]).any():self.release_q[i]=self._q_cmd[i]
      if self._phase_steps[i]<16:
        a=np.zeros(8,np.float32);a[:7]=(self.release_q[i]-self.default_qpos[:7])/FRANKA_ACTION_SCALE;a[7]=1.;return a
    return super()._act_single(i,obs)
  def _target_error(self,i,obs):
    q=self.default_qpos+obs[:9];ee,rot=self._fk(q);gto=obs[37:40]
    goal=ee+gto+obs[40:43]
    if self.goal_n[i]==0:self.goal[i]=goal
    elif self._phase[i]<3:self.goal[i]+=(goal-self.goal[i])/min(100,self.goal_n[i]+1)
    self.goal_n[i]+=1
    ph=int(self._phase[i])
    if ph<=3:
      mapped=np.zeros(60);mapped[:18]=obs[:18];mapped[40:43]=gto;mapped[43:46]=obs[40:43]
      err,axis,grip=super()._target_error(i,mapped)
      if ph==3:
        sample=rot.T@gto
        if self.hold_n[i]==0:self.hold[i]=sample
        else:self.hold[i]+=.15*(sample-self.hold[i])
        self.hold_n[i]+=1
        if self._phase_steps[i]>=self.CLIMB_STEPS and self._phase[i]==3:
          self._phase[i]=4;self._phase_steps[i]=0;self.integ[i]=0
      else:self.hold_n[i]=0
      return err,axis,grip
    self.max_dq=.045
    if ph in (4,5):
      obj=ee+rot@self.hold[i];d=self.goal[i]-obj
      self.integ[i,:2]=np.clip(self.integ[i,:2]+.04*d[:2],-.04,.04)
      err=d+self.integ[i]
      if ph==4:
        err[2]+=.10
        if np.linalg.norm(d[:2])<.014 and abs(err[2])<.02:
          self.stable[i]+=1
          if self.stable[i]>=5:self._phase[i]=5;self._phase_steps[i]=0;self.stable[i]=0
        else:self.stable[i]=0
      else:
        err[2]=max(d[2]+.006,-.004)
        if np.linalg.norm(d[:2])>.025:err[2]=max(err[2],0.)
        if np.linalg.norm(d[:2])<.018 and abs(d[2]+.006)<.015:
          self.stable[i]+=1
          if self.stable[i]>=5:self._phase[i]=6;self._phase_steps[i]=0
        else:self.stable[i]=0
      err[2]=max(err[2],.050-ee[2])
      return err,DOWN,-1.
    return np.array([0.,0.,max(0.,.22-ee[2])]),DOWN,1.
