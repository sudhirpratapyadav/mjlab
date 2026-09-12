"""Inspect the first nonfinite Newton search on a captured state; diagnostic only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import warp as wp
from mujoco_warp._src import solver

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scripts.train import TrainConfig
from rl_recipes import apply_recipe


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--failure', type=Path, required=True)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--lane', type=int, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error('Output already exists')
  saved = torch.load(args.failure, map_location='cpu', weights_only=False)
  state = saved['previous_state']
  manifest = json.loads((args.failure.parent/'manifest.json').read_text())
  cfg = TrainConfig.from_task(manifest['task'])
  apply_recipe(cfg, manifest['recipe'])
  cfg.env.scene.num_envs = len(state['qpos'])
  cfg.env.seed = manifest['seed']
  cfg.env.sim.free_body_implicitfast_compat = manifest.get('free_body_implicitfast_compat', False)
  env = ManagerBasedRlEnv(cfg.env, device='cuda:0')
  samples = []
  original = solver._update_gradient

  class FoundNonfinite(Exception):
    pass

  def inspect(m, d, ctx, *pos, **kw):
    original(m, d, ctx, *pos, **kw)
    h = wp.to_torch(ctx.h)[args.lane].cpu().numpy().copy()
    upper = np.triu(h).astype(np.float64)
    symmetric = upper + np.triu(upper, 1).T
    search = wp.to_torch(ctx.search)[args.lane].cpu().numpy().copy()
    grad = wp.to_torch(ctx.grad)[args.lane].cpu().numpy().copy()
    row = dict(index=len(samples), hessian_finite=bool(np.isfinite(h).all()),
               gradient_finite=bool(np.isfinite(grad).all()), search_finite=bool(np.isfinite(search).all()),
               hessian=symmetric.tolist(), gradient=grad.tolist(),
               eigenvalues=np.linalg.eigvalsh(symmetric).tolist() if np.isfinite(h).all() else None,
               constraints=int(wp.to_torch(d.nefc)[args.lane]))
    samples.append(row)
    if not row['search_finite']:
      contacts = []
      ncon = int(wp.to_torch(d.nacon)[0])
      world = wp.to_torch(d.contact.worldid)[:ncon].cpu().numpy()
      addresses = wp.to_torch(d.contact.efc_address).cpu().numpy()
      friction = wp.to_torch(d.contact.friction).cpu().numpy()
      dims = wp.to_torch(d.contact.dim).cpu().numpy()
      states = wp.to_torch(d.efc.state)[args.lane].cpu().numpy()
      residual = wp.to_torch(ctx.Jaref)[args.lane].cpu().numpy()
      diagonal = wp.to_torch(d.efc.D)[args.lane].cpu().numpy()
      impratio = wp.to_torch(m.opt.impratio_invsqrt).cpu().numpy()
      for i in np.flatnonzero(world == args.lane):
        ids = addresses[i, :dims[i]]
        if (ids < 0).any():
          continue
        mu = float(friction[i, 0]*impratio[args.lane % len(impratio)])
        tangential = residual[ids[1:]]*friction[i, :dims[i]-1]
        contacts.append(dict(state=int(states[ids[0]]), residual=residual[ids].tolist(),
                             friction=friction[i].tolist(), mu=mu, D=float(diagonal[ids[0]]),
                             tangent_norm=float(np.linalg.norm(tangential))))
      row['contacts'] = contacts
      raise FoundNonfinite()

  try:
    env.reset()
    for name in ('qpos', 'qvel', 'ctrl', 'qacc_warmstart', 'mocap_pos', 'mocap_quat'):
      getattr(env.sim.data, name)[:] = state[name].to(env.device)
    env.sim.forward()
    env.sim.data.qacc_warmstart[:] = state['qacc_warmstart'].to(env.device)
    env.action_manager.process_action(state['actions'].to(env.device))
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.use_cuda_graph = False
    env.sim.wp_model.opt.graph_conditional = False
    solver._update_gradient = inspect
    stopped = False
    try:
      env.sim.step()
    except FoundNonfinite:
      stopped = True
    report = dict(task=manifest['task'], source_lane=args.lane, worlds=env.num_envs,
                  solver=str(env.sim.mj_model.opt.solver), cone=str(env.sim.mj_model.opt.cone),
                  sparse=env.sim.wp_model.is_sparse,
                  stopped_at_nonfinite_search=stopped, samples=samples,
                  method='Captured full batch; eager first physics substep, inspect every full Newton Hessian/search. Stop before integrating a nonfinite search.',
                  limitations='Original randomized model friction and solver caches unavailable. No training/configuration changes; diagnostic eager launch differs from production CUDA graph.')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print({k:v for k,v in report.items() if k!='samples'})
    for row in samples:
      print({k:v for k,v in row.items() if k not in ('hessian', 'gradient')})
  finally:
    solver._update_gradient = original
    env.close()


if __name__ == '__main__':
  main()
