"""Opt-in dense elliptic Hessian compatibility with native MuJoCo.

The installed Warp solver floors T**3 at mjMINVAL independently of T. For
0 < T < 1e-5 this alters the cone Hessian and can destroy positive
semidefiniteness. Normalize tangential residuals first and evaluate the same
Hessian as a sum of outer products. No force, constraint state, model, solver
tolerance, or integration setting changes.

Reference: MuJoCo engine_core_constraint.c, mj_constraintUpdate_impl, middle
cone zone. Kernel interface/dispatch adapted from MuJoCo Warp solver.py
(Copyright 2025 The Newton Developers, Apache-2.0).
"""
from contextlib import contextmanager

import warp as wp
from mujoco_warp._src import solver, types


@wp.kernel(enable_backward=False)
def dense_cone_hessian(
  opt_impratio_invsqrt: wp.array[float],
  dof_tri_row: wp.array[int],
  dof_tri_col: wp.array[int],
  contact_dist_in: wp.array[float],
  contact_includemargin_in: wp.array[float],
  contact_friction_in: wp.array[types.vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array2d[int],
  contact_worldid_in: wp.array[int],
  efc_J_in: wp.array3d[float],
  efc_D_in: wp.array2d[float],
  efc_state_in: wp.array2d[int],
  naconmax_in: int,
  nacon_in: wp.array[int],
  ctx_Jaref_in: wp.array2d[float],
  ctx_done_in: wp.array[bool],
  nblocks_perblock: int,
  dim_block: int,
  ctx_h_out: wp.array3d[float],
):
  start, element = wp.tid()
  a = dof_tri_row[element]
  b = dof_tri_col[element]
  for block in range(nblocks_perblock):
    con = start + block*dim_block
    if con >= wp.min(nacon_in[0], naconmax_in):
      return
    world = contact_worldid_in[con]
    if ctx_done_in[world]:
      continue
    dim = contact_dim_in[con]
    if dim == 1 or contact_dist_in[con] >= contact_includemargin_in[con]:
      continue
    row = contact_efc_address_in[con, 0]
    if row < 0:
      continue
    if efc_state_in[world, row] != types.ConstraintState.CONE:
      continue
    friction = contact_friction_in[con]
    mu = friction[0]*opt_impratio_invsqrt[world % opt_impratio_invsqrt.shape[0]]
    denominator = mu*mu*(1.0+mu*mu)
    if denominator == 0.0:
      continue
    dm = efc_D_in[world, row]/denominator
    n = mu*ctx_Jaref_in[world, row]
    u = types.vec5(0.0)
    za = types.vec5(0.0)
    zb = types.vec5(0.0)
    norm2 = float(0.0)
    for j in range(1, dim):
      k = contact_efc_address_in[con, j]
      if k >= 0:
        u[j-1] = friction[j-1]*ctx_Jaref_in[world, k]
        za[j-1] = friction[j-1]*efc_J_in[world, k, a]
        zb[j-1] = friction[j-1]*efc_J_in[world, k, b]
        norm2 += u[j-1]*u[j-1]
    t = wp.max(wp.sqrt(norm2), types.MJ_MINVAL)
    unit = u/t
    pa = wp.dot(unit, za)
    pb = wp.dot(unit, zb)
    va = mu*efc_J_in[world, row, a]-mu*pa
    vb = mu*efc_J_in[world, row, b]-mu*pb
    perpendicular = wp.dot(za-unit*pa, zb-unit*pb)
    h = dm*(va*vb+(mu*mu-mu*(n/t))*perpendicular)
    wp.atomic_add(ctx_h_out[world, a], b, h)


@contextmanager
def elliptic_hessian_scope(enabled: bool):
  """Scope dispatch to this simulation's graph capture/eager call only."""
  if not enabled:
    yield
    return
  original = solver._update_gradient_JTCJ_dense
  solver._update_gradient_JTCJ_dense = dense_cone_hessian
  try:
    yield
  finally:
    solver._update_gradient_JTCJ_dense = original
