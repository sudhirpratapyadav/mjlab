"""Compare the production cone kernel to independent energy derivatives."""
import numpy as np
import pytest
import torch
import warp as wp
from mujoco_warp._src import solver, types

from mjlab.sim import SimulationCfg
from mjlab.sim.elliptic_hessian import dense_cone_hessian, elliptic_hessian_scope


@pytest.mark.parametrize('dim', [3, 4, 6])
@pytest.mark.parametrize('scale', [1e-10, 4e-6, 1., 100.])
def test_kernel_matches_cone_energy_hessian(dim, scale):
  rng = np.random.default_rng(7)
  worlds = 3
  nv = 4
  mu = np.float32(1/np.sqrt(10))
  friction = np.array([1., .7, .03, .003, .002], np.float32)
  residual = np.zeros((worlds, 6), np.float32)
  jac = rng.normal(size=(worlds, 6, nv)).astype(np.float32)
  expected = []
  for i, ratio in enumerate([-3., .2, .9]):
    unit = rng.normal(size=dim-1)
    unit /= np.linalg.norm(unit)
    residual[i, 0] = ratio*scale
    residual[i, 1:dim] = unit*scale/friction[:dim-1]
    r = torch.tensor(residual[i, :dim], dtype=torch.float64)
    weights = torch.tensor(np.r_[mu, friction[:dim-1]], dtype=torch.float64)
    def energy(x):
      u = x*weights
      return .5/(float(mu)**2*(1+float(mu)**2))*(u[0]-float(mu)*u[1:].norm()).square()
    h = torch.autograd.functional.hessian(energy, r).numpy()
    expected.append(jac[i, :dim].T @ h @ jac[i, :dim])
  rows, cols = np.triu_indices(nv)
  def a(x, dtype=wp.float32):
    return wp.array(x, dtype=dtype, device='cpu')
  out = wp.zeros((worlds, nv, nv), device='cpu')
  states = np.full((worlds, 6), int(types.ConstraintState.CONE), np.int32)
  addresses = np.tile(np.arange(6), (worlds, 1)).astype(np.int32)
  wp.launch(dense_cone_hessian, dim=(worlds, len(rows)), inputs=[
    a([mu]), a(rows, wp.int32), a(cols, wp.int32), a(np.full(worlds, -.001)),
    a(np.zeros(worlds)), a(np.tile(friction, (worlds, 1)), types.vec5),
    a(np.full(worlds, dim), wp.int32), a(addresses, wp.int32),
    a(np.arange(worlds), wp.int32), a(jac), a(np.ones((worlds, 6))),
    a(states, wp.int32), worlds, a([worlds], wp.int32), a(residual),
    a(np.zeros(worlds, dtype=bool), wp.bool), 1, worlds], outputs=[out], device='cpu')
  upper = out.numpy()
  for i in range(worlds):
    actual = upper[i]+np.triu(upper[i], 1).T
    np.testing.assert_allclose(actual, expected[i], atol=2e-5, rtol=3e-5)
    assert np.linalg.eigvalsh(actual).min() > -2e-5


def test_captured_contact_old_denominator_is_indefinite():
  mu = .3162277638912201
  residual = np.array([3.814697265625e-6, -3.306313374196179e-6, -2.510565991542535e-6])
  n, u = mu*residual[0], residual[1:]
  t = np.linalg.norm(u)
  old = np.empty((3, 3))
  old[0, 0] = 1
  old[0, 1:] = old[1:, 0] = -mu*u/t
  old[1:, 1:] = mu*n/max(t**3, 1e-15)*np.outer(u, u)+(mu*mu-mu*n/t)*np.eye(2)
  assert t**3 < 1e-15
  assert np.linalg.eigvalsh(old).min() < 0


def test_dispatch_is_scoped_and_default_is_unchanged():
  original = solver._update_gradient_JTCJ_dense
  assert not SimulationCfg().elliptic_hessian_compat
  with pytest.raises(RuntimeError):
    with elliptic_hessian_scope(True):
      assert solver._update_gradient_JTCJ_dense is dense_cone_hessian
      with elliptic_hessian_scope(True):
        assert solver._update_gradient_JTCJ_dense is dense_cone_hessian
      raise RuntimeError('test restoration')
  assert solver._update_gradient_JTCJ_dense is original
  with elliptic_hessian_scope(False):
    assert solver._update_gradient_JTCJ_dense is original
