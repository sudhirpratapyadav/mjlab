"""Two-update PPO plumbing check, not a teacher-training or success experiment."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

here = Path(__file__).resolve().parent
run = here / "runs/PREFLIGHT-PPO-001"
run.mkdir(parents=True, exist_ok=True)
task = "Mjlab-Reach-Target-Franka"
cfg = load_env_cfg(task)
cfg.scene.num_envs = 64
cfg.seed = 20260912
rl = load_rl_cfg(task)
rl.seed = 20260912
rl.logger = "tensorboard"
torch.manual_seed(rl.seed)
env = ManagerBasedRlEnv(cfg, device="cuda:0")
wrapped = RslRlVecEnvWrapper(env, clip_actions=rl.clip_actions)
try:
  runner = OnPolicyRunner(wrapped, asdict(rl), str(run), device="cuda:0")
  before = {k: v.detach().clone() for k, v in runner.alg.policy.named_parameters()}
  losses = []
  original_update = runner.alg.update

  def capture_update():
    result = original_update()
    losses.append({k: float(v) for k, v in result.items()})
    return result

  runner.alg.update = capture_update
  start = time.monotonic()
  runner.learn(num_learning_iterations=2)
  elapsed = time.monotonic() - start
  assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters())
  changed = any(
    not torch.equal(before[k], v) for k, v in runner.alg.policy.named_parameters()
  )
  assert changed
  assert all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in losses)
  checkpoint = run / "smoke.pt"
  runner.save(str(checkpoint), infos={"purpose": "plumbing-only", "task": task})
  obs = wrapped.get_observations()
  with torch.no_grad():
    expected = runner.get_inference_policy()(obs).clone()
  restored = OnPolicyRunner(wrapped, asdict(rl), None, device="cuda:0")
  restored.load(str(checkpoint), map_location="cuda:0")
  with torch.no_grad():
    actual = restored.get_inference_policy()(obs)
  torch.testing.assert_close(actual, expected)
  assert restored.alg.optimizer.state_dict()["state"]
  result = dict(
    task=task,
    status="pass",
    purpose="plumbing-only; no success evaluation",
    num_envs=64,
    updates=2,
    transitions=64 * 24 * 2,
    training_seconds=elapsed,
    parameters_changed=changed,
    losses=losses,
    checkpoint_reload=True,
    optimizer_restored=True,
    peak_gpu_memory_mib=torch.cuda.max_memory_allocated() / 1024**2,
  )
  (here / "evidence/ppo_smoke.json").write_text(json.dumps(result, indent=2) + "\n")
  print("SMOKE_RESULT", result, flush=True)
finally:
  env.close()
