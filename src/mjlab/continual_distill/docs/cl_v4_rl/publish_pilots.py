"""Copy the original six pilot histories and retained evidence to Sudhir's project."""

import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

from wandb_config import ENTITY, PROJECT, configure

HERE = Path(__file__).resolve().parent
PILOTS = (
  ("RL-001-reach-s20260912", "Mjlab-Reach-Target-Franka", "completed"),
  ("RL-002-lift-s20260912", "Mjlab-Lift-Cube-Franka", "interrupted_for_wandb"),
  ("RL-003-push-cuboid-s20260912", "Mjlab-Push-Cuboid-Franka", "invalid_action_std"),
  ("RL-004-drag-pull-s20260912", "Mjlab-Drag-Pull-Franka", "interrupted_for_wandb"),
  ("RL-005-reorient-s20260912", "Mjlab-Reorient-Object-Franka", "interrupted_for_wandb"),
  ("RL-006-topple-s20260912", "Mjlab-Topple-Block-Franka", "interrupted_for_wandb"),
)


def main():
  configure()
  import wandb

  records = []
  for name, task, status in PILOTS:
    directory = HERE / "runs" / name
    agent = yaml.load((directory / "params/agent.yaml").read_text(), Loader=yaml.FullLoader)
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}).Reload()
    history = {}
    for tag in accumulator.Tags()["scalars"]:
      for event in accumulator.Scalars(tag):
        history.setdefault(event.step, {})[tag] = event.value
    checkpoint = max(directory.glob("model_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    run = wandb.init(
      entity=ENTITY, project=PROJECT, id="import-" + name[:6].lower(),
      name=name, job_type="imported-pilot", resume="allow",
      config={"task": task, "original_agent_config": agent, "interface": "franka_shared_60_v2",
              "source_run_dir": str(directory), "imported_from": "domimagi-iitj"},
    )
    for step, values in sorted(history.items()):
      run.log(values, step=step)
    run.summary.update({"source_status": status, "last_checkpoint": checkpoint.name,
                        "checkpoint_sha256": digest, "source_history_steps": len(history)})
    artifact = wandb.Artifact(name + "-evidence", type="pilot-evidence", metadata={"task": task, "source_status": status})
    artifact.add_file(str(checkpoint))
    for path in sorted((directory / "params").glob("*.yaml")):
      artifact.add_file(str(path), name="params/" + path.name)
    for path in directory.glob("events.out.tfevents.*"):
      artifact.add_file(str(path), name="tensorboard/" + path.name)
    for path in (directory / "git").glob("*.diff"):
      artifact.add_file(str(path), name="source/" + path.name)
    for path in sorted((HERE / "evidence").glob(name[:6] + "-*.json")):
      result = json.loads(path.read_text())
      artifact.add_file(str(path), name="evaluations/" + path.name)
      phase = "confirmation" if "confirm" in path.name else "validation"
      run.summary.update({f"strict/{phase}/successes": result["successes"],
                          f"strict/{phase}/episodes": result["episodes"],
                          f"strict/{phase}/success_rate": result["success_rate"],
                          f"strict/{phase}/seed": result["seed"]})
    run.log_artifact(artifact)
    record = {"source_run": name, "wandb_run_id": run.id, "url": run.url,
              "source_status": status, "history_steps": len(history), "checkpoint_sha256": digest}
    run.finish()
    remote = wandb.Api(timeout=30).run(f"{ENTITY}/{PROJECT}/{record['wandb_run_id']}")
    assert remote.summary["source_history_steps"] == len(history)
    assert remote.summary["checkpoint_sha256"] == digest
    record["remote_verified"] = True
    records.append(record)
    (HERE / "evidence/wandb_pilot_imports.json").write_text(json.dumps(records, indent=2) + "\n")
    print("VERIFIED_IMPORT", name, record["url"], flush=True)


if __name__ == "__main__":
  main()
