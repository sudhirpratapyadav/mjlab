"""Real-physics readiness for the four remaining task-specific reward recipes."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from bounded_policy import BoundedActorCritic
from rl_recipes import apply_recipe
from wandb_config import configure, ENTITY, PROJECT

HERE = Path(__file__).resolve().parent


def main():
  output = HERE/'evidence/remaining_preflight_v1.json'
  if output.exists():
    raise FileExistsError(output)
  rows = []
  for task, recipe in [('Edge-Grasp','edge_v1'),('Pivot-Lift','pivot_v1'),
                       ('Strike-Slide','strike_v1'),('Throw-To-Bin','throw_v1')]:
    cfg = TrainConfig.from_task(f'Mjlab-{task}-Franka')
    fields = ('observations','actions','commands','terminations','events','scene','sim','episode_length_s')
    before = {name:repr(getattr(cfg.env,name)) for name in fields}
    apply_recipe(cfg,recipe)
    assert before == {name:repr(getattr(cfg.env,name)) for name in fields}
    cfg.env.scene.num_envs = 32
    cfg.env.seed = 20260912
    env = ManagerBasedRlEnv(cfg.env,device='cuda:0')
    wrapped = RslRlVecEnvWrapper(env,clip_actions=1.)
    try:
      obs = wrapped.get_observations()
      values = asdict(cfg.agent.policy)
      values.pop('class_name')
      policy = BoundedActorCritic(obs,cfg.agent.obs_groups,8,**values).to(env.device)
      total = 0.
      for _ in range(100):
        with torch.no_grad():
          policy.update_normalization(obs)
          actions = policy.act(obs)
          obs, reward, *_ = wrapped.step(actions)
        assert obs['policy'].shape == (32,60) and actions.shape == (32,8)
        assert all(torch.isfinite(v).all() for v in (obs['policy'],reward,env.sim.data.qpos,env.sim.data.qvel))
        total += float(reward.mean())
      rows.append({'task':task,'recipe':recipe,'finite':True,'environments':32,'steps':100,'mean_return':total,'benchmark_invariants':list(fields)})
      print('PASS',rows[-1],flush=True)
    finally:
      wrapped.close()
  report = {'cases':rows,'source_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'source_patch_sha256':hashlib.sha256(subprocess.check_output(['git','diff','HEAD'])).hexdigest(),
            'interface':'franka_shared_60_v2','seed':20260912}
  output.write_text(json.dumps(report,indent=2)+'\n')
  configure()
  import wandb
  with wandb.init(entity=ENTITY,project=PROJECT,name='remaining-four-reward-preflight',job_type='readiness') as run:
    run.log({'finite_cases':len(rows)})
    artifact = wandb.Artifact('remaining-four-reward-preflight',type='readiness')
    artifact.add_file(str(output))
    run.log_artifact(artifact)


if __name__ == '__main__':
  main()
