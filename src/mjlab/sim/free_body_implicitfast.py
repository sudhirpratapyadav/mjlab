"""Optional compatibility with MuJoCo's implicitfast free-body gyro solve.

MuJoCo's native implicitfast re-solves standalone free bodies with gyroscopic
derivatives. Some mujoco-warp versions omit that correction and gain energy for
rapid asymmetric spins. This stepper composes existing Warp derivative/solve
routines and substitutes only the decoupled free-body accelerations. Articulated
body accelerations retain Warp's original implicitfast result.

Uses private mujoco-warp APIs; validate against the installed CPU engine before
enabling. No model parameters, actions, reset rules or state clipping are changed.
"""
import mujoco
import numpy as np
import warp as wp
from mujoco_warp._src import derivative, forward, smooth
from mujoco_warp._src.types import DisableBit, IntegratorType


def standalone_free_dofs(model: mujoco.MjModel) -> np.ndarray:
  """Only free joints on leaf bodies, matching native MuJoCo's eligible case."""
  selected = np.zeros(model.nv, dtype=np.int32)
  for joint in range(model.njnt):
    body = int(model.jnt_bodyid[joint])
    if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
      continue
    if np.any(model.body_parentid[1:] == body):
      continue
    address = int(model.jnt_dofadr[joint])
    selected[address:address+6] = 1
  return selected


@wp.kernel
def _replace_free_acceleration(mask: wp.array[int], full: wp.array2d[float], fast: wp.array2d[float]):
  world, dof = wp.tid()
  if mask[dof] != 0:
    fast[world, dof] = full[world, dof]


class FreeBodyImplicitFastStepper:
  def __init__(self, cpu_model, device):
    selected = standalone_free_dofs(cpu_model)
    self.has_free = bool(selected.any())
    self.mask = wp.array(selected, dtype=int, device=device)

  def __call__(self, model, data):
    if model.opt.integrator != IntegratorType.IMPLICITFAST or not self.has_free:
      forward.step(model, data)
      return
    forward.forward(model, data)
    if ~(model.opt.disableflags | ~(DisableBit.ACTUATION | DisableBit.SPRING | DisableBit.DAMPER)):
      fast_matrix = wp.empty((data.nworld, model.nC), dtype=float)
      factors = wp.empty_like(data.qLD)
      diagonal = wp.empty((data.nworld, model.nv), dtype=float)
      derivative.deriv_smooth_vel(model, data, fast_matrix)
      acceleration = wp.empty((data.nworld, model.nv), dtype=float)
      smooth.factor_solve_i(model, data, fast_matrix, factors, diagonal, acceleration, data.efc.Ma)
    else:
      acceleration = wp.clone(data.qacc)

    full_matrix = wp.empty(data.M.shape, dtype=float)
    derivative.deriv_smooth_vel(model, data, full_matrix)
    wp.launch(forward._map_m2d, dim=(data.nworld, model.nD),
              inputs=[model.mapM2D, full_matrix], outputs=[data.qLU])
    # RNE returns the derivative of the inverse-dynamics bias, which enters
    # forward force with a minus sign: M - h*d(force)/dv = M + h*d(bias)/dv.
    derivative.deriv_rne_vel(model, data, data.qLU, flg_subtract=False)
    full_acceleration = wp.empty((data.nworld, model.nv), dtype=float)
    smooth.factor_solve_lu(model, data, data.qLU, full_acceleration, data.efc.Ma)
    wp.launch(_replace_free_acceleration, dim=(data.nworld, model.nv),
              inputs=[self.mask, full_acceleration], outputs=[acceleration])
    forward._advance(model, data, acceleration)
