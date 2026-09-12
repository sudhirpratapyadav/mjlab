"""Publish a completed strict evaluation without rerunning its episodes."""

import argparse
import json
from pathlib import Path

from wandb_config import ENTITY, PROJECT, configure


def publish(path):
  path = Path(path)
  data = json.loads(path.read_text())
  configure()
  import wandb
  with wandb.init(entity=ENTITY, project=PROJECT, name=path.stem, job_type="strict-evaluation", config={k:v for k,v in data.items() if k != "records"}) as run:
    run.log({"strict/successes": data["successes"], "strict/episodes": data["episodes"], "strict/success_rate": data["success_rate"]})
    artifact = wandb.Artifact(path.stem, type="strict-evaluation", metadata={"checkpoint_sha256":data["checkpoint_sha256"], "seed":data["seed"]})
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    print("EVALUATION_URL", run.url, flush=True)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("result", type=Path)
  publish(parser.parse_args().result)
