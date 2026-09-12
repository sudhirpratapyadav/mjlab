import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import mujoco
from staged_grasp import StagedGrasp
from mjlab.continual_distill.classical.stack_object import closing_frame,quat_yaw_wxyz
class PegInsertionClassicalPolicy(StagedGrasp):
  GRASP_SITE_Z=.078
  OBJ_CENTER_Z=0.
  HELD_TOL=.14
  def reset(self,env_ids=None):
    super().reset(env_ids)
    if env_ids is None:
      self.relative_rot=np.zeros((self.num_envs,3,3));self.rot_n=np.zeros(self.num_envs,int)
    else:self.rot_n[env_ids]=0;self.relative_rot[env_ids]=0
  def _act_single(self,i,obs):
    mapped=np.zeros(51);mapped[:25]=obs[:25];mapped[37:40]=obs[40:43];mapped[40:43]=obs[43:46]
    return super()._act_single(i,mapped)
  def _target_error(self,i,obs):
    ph=int(self._phase[i]);q=self.default_qpos+obs[:9];_,hand=self._fk(q)
    quat=obs[21:25].copy();quat/=np.linalg.norm(quat)
    obj=np.empty(9);mujoco.mju_quat2Mat(obj,quat);obj=obj.reshape(3,3)
    if ph==3:
      self.relative_rot[i]+=hand.T@obj;self.rot_n[i]+=1
    elif ph<3:self.relative_rot[i]=0;self.rot_n[i]=0
    err,axis,grip=super()._target_error(i,obs)
    if ph<4 or self.rot_n[i]==0:
      axis=closing_frame(quat_yaw_wxyz(quat))
      self.orientation_weight=.3;self.posture_weight=.005
    else:
      u,_,v=np.linalg.svd(self.relative_rot[i]);rel=u@v
      if np.linalg.det(rel)<0:u[:,-1]*=-1;rel=u@v
      axis=rel.T
      self.orientation_weight=.6;self.posture_weight=.0005
    return err,axis,grip
