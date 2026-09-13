"""Native CPU parity for the production primitive box contact option."""
import copy

import mujoco
import numpy as np
import pytest
import torch
from conftest import get_test_device

from mjlab.sim import MujocoCfg, Simulation, SimulationCfg


@pytest.mark.parametrize('sideways', [False, True])
def test_resting_box_manifold_settles_without_changing_native_model(sideways):
  model = mujoco.MjModel.from_xml_string('''<mujoco>
    <default><geom condim="3" friction="1 .03 .003" solref=".01 1"/></default>
    <worldbody>
      <geom name="floor" type="box" pos="0 0 .002" size=".0835 .0835 .002"/>
      <body pos="0 0 .027"><freejoint/>
        <inertial pos=".00005 -.00004 -.00004" quat=".996575 -.063327 -.039454 .035648"
          mass=".05" diaginertia="1.650e-5 1.663e-5 1.664e-5"/>
        <geom name="cube" type="box" size=".023 .0226 .0226"/>
      </body>
    </worldbody></mujoco>''')
  assert not SimulationCfg().primitive_box_box_compat
  original = copy.copy(model)
  cfg = SimulationCfg(primitive_box_box_compat=True, elliptic_hessian_compat=True,
                      nconmax=32, njmax=64,
                      mujoco=MujocoCfg(timestep=.005, cone='elliptic', jacobian='dense'))
  device = get_test_device()
  sim = Simulation(16,cfg,model,device)
  assert sim.mj_model.opt.disableflags == original.opt.disableflags
  assert sim.wp_model.opt.disableflags == (original.opt.disableflags | int(mujoco.mjtDisableBit.mjDSBL_NATIVECCD))
  for field in ['body_mass','body_inertia','geom_size','geom_friction','geom_solref','geom_solimp']:
    np.testing.assert_array_equal(getattr(model,field),getattr(original,field))
  cpus=[]
  poses=[]
  for lane in range(16):
    d=mujoco.MjData(model)
    # Different yaw, small initial roll/pitch and four side orientations.
    pitch=lane%4*np.pi/2+.006 if sideways else .006
    q=np.empty(4);mujoco.mju_euler2Quat(q,np.array([.008*(-1)**lane,pitch,lane*.37]),'xyz')
    d.qpos[:]=[0,0,.028,*q]
    mujoco.mj_forward(model,d)
    poses.append(d.qpos.copy());cpus.append(d)
  sim.data.qpos[:]=torch.as_tensor(np.stack(poses),device=device,dtype=torch.float32)
  sim.data.qvel.zero_();sim.data.qacc_warmstart.zero_();sim.forward()
  for _ in range(400):
    sim.step()
    for d in cpus:mujoco.mj_step(model,d)
  sim.forward()
  qvel=sim.data.qvel.cpu().numpy();qpos=sim.data.qpos.cpu().numpy()
  assert np.isfinite(qpos).all() and np.isfinite(qvel).all()
  assert np.all(np.linalg.norm(qvel[:,:3],axis=1)<.03)
  assert np.all(np.linalg.norm(qvel[:,3:],axis=1)<.3)
  # Keep every side orientation in the physical support check. Native CPU 3.11.1
  # drops all contacts for one tilted side case (recorded separately), so use
  # the independent collider support geometry for this broader set.
  def bottom_height(pose):
    rotation=np.empty(9);mujoco.mju_quat2Mat(rotation,pose[3:].astype(np.float64))
    return pose[2]-np.abs(rotation.reshape(3,3)[2])@model.geom_size[1]
  for pose in qpos:
    bottom=bottom_height(pose)
    assert .004-.0001 < bottom < .004+.0001
  if not sideways:
    # Different trajectories can settle on different faces of this slightly
    # rectangular cube. Compare the physical support clearance, not root Z.
    np.testing.assert_allclose([bottom_height(p) for p in qpos],
                               [bottom_height(d.qpos) for d in cpus],atol=2e-5,rtol=0)
  n=int(sim.data.nacon[0]);worlds=sim.data.contact.worldid[:n].cpu().numpy()
  assert np.all(np.bincount(worlds,minlength=16)==4)
  if not sideways:
    for d in cpus:
      mujoco.mj_forward(model,d)
      assert d.ncon==4
      assert np.linalg.norm(d.qvel[3:])<.3
