"""Strict first-episode teacher evaluation with phase and MuJoCo state traces."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.continual_distill.classical.episode_evaluation import TerminalSuccessCapture
from mjlab.continual_distill.classical.render_rollout import (
  _build_env,
  _success_term,
  source_fingerprint,
)
from mjlab.envs import ManagerBasedRlEnv


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--tasks", nargs="+", required=True)
  parser.add_argument("--n", type=int, default=32)
  parser.add_argument("--seed", type=int, default=20260911)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--teacher-file", type=Path)
  parser.add_argument("--device", default="cpu")
  args = parser.parse_args()
  fingerprint = source_fingerprint()
  for name in args.tasks:
    task = f"Mjlab-{name}-Franka"
    folder = args.out / task
    folder.mkdir(parents=True, exist_ok=True)
    ManagerBasedRlEnv.seed(args.seed)
    env = _build_env(task, args.n, args.device, False, 0, 0)
    try:
      policy_class = CLASSICAL_POLICIES[task]
      if args.teacher_file:
        spec = importlib.util.spec_from_file_location(
          "measured_teacher", args.teacher_file
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy_class = getattr(module, policy_class.__name__)
      policy = policy_class(num_envs=args.n)
      obs, _ = env.reset()
      policy.reset()
      cmd = _success_term(env)
      alive = np.ones(args.n, dtype=bool)
      success = np.zeros(args.n, dtype=bool)
      outcomes = [None] * args.n
      trace = []

      def snapshot(env=env, cmd=cmd):
        state = {
          k: getattr(env.sim.data, k).cpu().numpy().copy()
          for k in ("qpos", "qvel", "mocap_pos", "mocap_quat")
        }
        state.update(
          {f"metric_{k}": v.cpu().numpy().copy() for k, v in cmd.metrics.items()}
        )
        return state

      terminal = {}
      final_states = {
        k: np.zeros_like(getattr(env.sim.data, k).cpu().numpy())
        for k in ("qpos", "qvel", "mocap_pos", "mocap_quat")
      }

      def before_reset(ids, cmd=cmd, snapshot=snapshot, terminal=terminal):
        cmd._update_command()
        state = snapshot()
        for i in ids.tolist():
          terminal[i] = {k: v[i] for k, v in state.items()}

      with TerminalSuccessCapture(env, cmd, before_reset) as capture:
        for step in range(int(env.max_episode_length)):
          state = snapshot()
          state["phase"] = policy._phase.copy()
          state["alive"] = alive.copy()
          state["obs"] = obs["policy"].cpu().numpy().copy()
          trace.append(state)
          action = policy(state["obs"], active_env_ids=np.flatnonzero(alive))
          obs, _, terminated, truncated, _ = capture.step(
            torch.from_numpy(action).to(args.device)
          )
          done = (terminated | truncated).cpu().numpy()
          good = (
            torch.where(
              terminated | truncated, capture.terminal_success, cmd.episode_success > 0
            )
            .cpu()
            .numpy()
          )
          ending = alive & (done | good)
          current = snapshot()
          for i in np.flatnonzero(ending):
            success[i] = good[i]
            final = terminal.get(i, {k: v[i] for k, v in current.items()})
            for key in final_states:
              final_states[key][i] = final[key]
            outcomes[i] = dict(
              env=int(i),
              success=bool(good[i]),
              steps=step + 1,
              phase=int(policy._phase[i]),
              terminated=bool(terminated[i]),
              timeout=bool(truncated[i]),
              metrics={
                k.removeprefix("metric_"): float(v)
                for k, v in final.items()
                if k.startswith("metric_")
              },
            )
          alive &= ~ending
          terminal.clear()
          if step % 100 == 0:
            print(
              name,
              step,
              "alive",
              int(alive.sum()),
              "success",
              int(success.sum()),
              flush=True,
            )
          if not alive.any():
            break
      np.savez_compressed(
        folder / "trace.npz", **{k: np.stack([s[k] for s in trace]) for k in trace[0]}
      )
      np.savez_compressed(
        folder / "final_states.npz",
        **final_states,
        origins=env.scene.env_origins.cpu().numpy(),
      )
      result = dict(
        task_id=task,
        n=args.n,
        num_success=int(success.sum()),
        sr=float(success.mean()),
        stats_seed=args.seed,
        stats_batch_size=args.n,
        source_sha256=fingerprint,
        evaluation_protocol="single_episode_no_reset_retries",
        task_revision="2026-09-11-corrected-semantics",
        device=args.device,
        outcomes=outcomes,
      )
      if args.teacher_file:
        result["teacher_override"] = str(args.teacher_file)
        result["teacher_override_sha256"] = hashlib.sha256(
          args.teacher_file.read_bytes()
        ).hexdigest()
      (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
      print("RESULT", name, result["num_success"], args.n, flush=True)
    finally:
      env.close()


if __name__ == "__main__":
  main()
