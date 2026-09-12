"""Check whether solver-cache clearing changes a fresh first-episode reset."""
import json
from pathlib import Path

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.benchmark import active_cl_tasks
from mjlab.tasks.registry import load_env_cfg


def main():
  output = Path(__file__).resolve().parent/'evidence/cold_solver_reset.json'
  if output.exists():
    raise FileExistsError(output)
  rows = []
  for task in active_cl_tasks():
    cfg = load_env_cfg(task)
    cfg.scene.num_envs = 1
    cfg.seed = 20260914
    env = ManagerBasedRlEnv(cfg,device='cuda:0')
    try:
      before = env.sim.data.qacc_warmstart.clone()
      already_zero = bool(torch.equal(before,torch.zeros_like(before)))
      env.sim.reset_solver_state()
      unchanged = bool(torch.equal(before,env.sim.data.qacc_warmstart))
      rows.append({'task':task,'cold_warmstart_max_abs':float(before.abs().max()),
                   'cold_cache_already_zero':already_zero,'new_clear_is_noop':unchanged})
      print('COLD_RESET',task,already_zero,unchanged,flush=True)
    finally:
      env.close()
  output.write_text(json.dumps({'rows':rows,'all_cold_resets_unchanged':all(r['new_clear_is_noop'] for r in rows),
                               'scope':'Fresh registered environments before first reset, one world per task. Does not replace teacher success evaluation.'},indent=2)+'\n')


if __name__ == '__main__':
  main()
