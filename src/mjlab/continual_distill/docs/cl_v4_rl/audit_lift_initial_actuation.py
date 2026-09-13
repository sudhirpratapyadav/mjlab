"""Native CPU torque/sensitivity diagnostic of one recorded initial state."""
import json
from pathlib import Path
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.manipulation.franka_interface import CENTER, HALF_RANGE

HERE = Path(__file__).resolve().parent


def main():
  output = HERE / 'evidence/lift_s1_initial_actuation_audit.json'
  assert not output.exists()
  ev = json.loads((HERE / 'evidence/RL-002-S1-m23489-val-20260914.json').read_text())
  with np.load(Path(ev['trace_dir']) / 'trace.npz') as z:
    trace = {k: z[k] for k in z.files}
  saved = torch.load(ev['checkpoint'], map_location='cpu', weights_only=False)
  std = saved['model_state_dict']['log_std'].exp().numpy()[:7]
  cfg = load_env_cfg(ev['task']); cfg.scene.num_envs = 1
  m = Scene(cfg.scene, 'cpu').compile(); cfg.sim.mujoco.apply(m)
  assert m.opt.timestep == .005 and ev['records'][0]['steps'] >= 50
  joints = [m.joint(f'robot/joint{i}').id for i in range(1, 8)]
  dofs = m.jnt_dofadr[joints]
  acts = [m.actuator(f'robot/actuator{i}').id for i in range(1, 8)]
  m.geom_friction[:] = trace['initial_model_geom_friction'][0]
  center, half = np.asarray(CENTER[:7]), np.asarray(HALF_RANGE[:7])
  first = trace['ctrl'][1, 0].copy()
  normalized = (first[acts] - center) / half
  def initial():
    d = mujoco.MjData(m)
    for field in ['qpos', 'qvel', 'mocap_pos', 'mocap_quat', 'ctrl']:
      getattr(d, field)[:] = trace[field][0, 0]
    for adr, kind in zip(m.jnt_qposadr, m.jnt_type):
      if kind == mujoco.mjtJoint.mjJNT_FREE:
        d.qpos[adr:adr+3] -= trace['origins'][0]
    d.mocap_pos[:] -= trace['origins'][0]
    return d
  d = initial(); velocities, forces = [], []
  for tick in range(200):
    d.ctrl[:] = trace['ctrl'][1+tick//4, 0]
    mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
    forces.append(d.actuator_force[acts].copy())
    if tick % 4 == 3:
      velocities.append(d.qvel[dofs].copy())
  forces, velocities = np.asarray(forces), np.asarray(velocities)
  limits = m.actuator_forcerange[acts]
  at_limit = (np.isclose(forces, limits[:, 0], atol=1e-4) |
              np.isclose(forces, limits[:, 1], atol=1e-4)) & m.actuator_forcelimited[acts].astype(bool)
  def endpoint(control):
    d = initial(); d.ctrl[:] = control
    for _ in range(4):
      mujoco.mj_step(m, d)
    assert np.isfinite(d.qvel).all() and not d.warning.number.any()
    return d.qvel[dofs].copy()
  baseline = endpoint(first)
  probes = []
  for j in range(7):
    for sign in [-1, 1]:
      control = first.copy()
      action = np.clip(normalized[j] + sign*std[j], -1, 1)
      control[acts[j]] = center[j] + half[j]*action
      v = endpoint(control)
      probes.append(dict(joint=j+1, sign=sign, saved_normalized_std=float(std[j]),
        applied_target_change_rad=float(control[acts[j]]-first[acts[j]]),
        max_arm_velocity_change_rad_s=float(np.abs(v-baseline).max())))
  report = dict(checkpoint_sha256=ev['checkpoint_sha256'], env_id=0,
    first_second_cpu_gpu_velocity_rmse=float(np.sqrt(np.mean((velocities-trace['qvel'][1:51,0][:,dofs])**2))),
    force_limits=limits.tolist(), torque_limit_fraction_first_second=at_limit.mean(axis=0).tolist(),
    baseline_first_step_velocity=baseline.tolist(), probes=probes,
    method='Preregistered single-state nativeCPU cold replay and14 one-step sensitivity probes. Original state/friction and approved clipped affineaction mapping. No policy episodes/training data or changes to the running S3 trial.')
  output.write_text(json.dumps(report, indent=2) + '\n')
  print(json.dumps(report, indent=2))


if __name__ == '__main__':
  main()
