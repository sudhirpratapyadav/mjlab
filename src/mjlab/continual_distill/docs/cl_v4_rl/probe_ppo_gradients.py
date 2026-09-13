"""Observe PPO gradient clipping in a separate diagnostic run, preserving updates.

Accepts the same CLI as train_teacher.py. Records every attempted minibatch,
including KL-rejected attempts. These runs are diagnostics, not teacher candidates.
"""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import torch

from teacher_runner import TeacherRunner


def observed_clip(parameters, max_norm, *args, policy, original, emit, **kwargs):
  parameters = list(parameters)
  names = {id(p): name for name, p in policy.named_parameters()}
  groups = {"actor": [], "critic": [], "exploration": [], "other": []}
  for parameter in parameters:
    if parameter.grad is None:
      continue
    name = names[id(parameter)]
    group = ("actor" if name.startswith("actor.") else
             "critic" if name.startswith("critic.") else
             "exploration" if name in ("std", "log_std") else "other")
    groups[group].append(parameter.grad.detach().norm(2))
  norms = {name: float(torch.stack(values).norm(2)) if values else 0.
           for name, values in groups.items()}
  # Execute the original clipping operation exactly once, with identical arguments.
  result = original(parameters, max_norm, *args, **kwargs)
  total = float(result)
  emit(dict(pre_clip_l2=norms, total_norm_returned=total, max_norm=float(max_norm),
            common_clip_multiplier=min(1., float(max_norm)/(total+1e-6))))
  return result


class GradientProbeRunner(TeacherRunner):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    original_update = self.alg.update
    original_clip = torch.nn.utils.clip_grad_norm_
    output = Path(self.logger.log_dir) / "gradient-clipping.jsonl"
    assert not output.exists()
    self._gradient_probe_update = 0

    def update():
      self._gradient_probe_update += 1
      rows = []
      if self._gradient_probe_update == 1:
        storage = self.alg.storage
        def digest(value):
          return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        fingerprints = {f"obs/{name}": digest(value)
                        for name, value in storage.observations.items()}
        fingerprints["actions"] = digest(storage.actions)
        fingerprints["old_mean"] = digest(storage.mu)
        fingerprints["old_std"] = digest(storage.sigma)
        (output.parent / "rollout-fingerprint.json").write_text(
          json.dumps(fingerprints, indent=2)+"\n")

      def clip(parameters, max_norm, *arguments, **options):
        return observed_clip(parameters, max_norm, *arguments,
                             policy=self.alg.policy, original=original_clip,
                             emit=rows.append, **options)

      try:
        with patch("torch.nn.utils.clip_grad_norm_", clip):
          return original_update()
      finally:
        with output.open("a") as stream:
          for index, row in enumerate(rows):
            stream.write(json.dumps(dict(update=self._gradient_probe_update,
              attempted_minibatch=index, **row), allow_nan=False)+"\n")

    self.alg.update = update


if __name__ == "__main__":
  import train_teacher
  train_teacher.TeacherRunner = GradientProbeRunner
  train_teacher.main()
