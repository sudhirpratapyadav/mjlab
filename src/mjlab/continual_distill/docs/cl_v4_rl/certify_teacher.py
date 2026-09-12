"""Retain a reviewed teacher only after both strict held-out batches pass."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import imageio_ffmpeg
import torch
import yaml

HERE=Path(__file__).resolve().parent


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--validation",type=Path,required=True)
  parser.add_argument("--confirmation",type=Path,required=True)
  parser.add_argument("--video-result",type=Path,required=True)
  parser.add_argument("--review",type=Path,required=True)
  args=parser.parse_args()
  validation=json.loads(args.validation.read_text())
  confirmation=json.loads(args.confirmation.read_text())
  video_result=json.loads(args.video_result.read_text())
  review=json.loads(args.review.read_text())
  checkpoint=Path(validation["checkpoint"])
  checkpoint_sha=digest(checkpoint)
  for result in (validation,confirmation):
    assert result["episodes"]>=128 and result["successes"]/result["episodes"]>0.90
    assert result["successes"]==sum(row["success"] for row in result["records"])
    assert result["deterministic"] and result["first_episode_only"]
    assert result["checkpoint_sha256"]==checkpoint_sha
    assert result["task"]==validation["task"]
  assert validation["seed"]!=confirmation["seed"]
  assert review["verdict"]=="pass" and review["checkpoint_sha256"]==checkpoint_sha
  assert video_result["checkpoint_sha256"]==checkpoint_sha
  assert video_result["interface"]=="franka_shared_60_v2"
  video_dir=Path(video_result["trace_dir"])
  videos=json.loads((video_dir/"videos.json").read_text())
  assert "success" in videos["clips"]
  if any(r["successes"]<r["episodes"] for r in (validation,confirmation)):
    assert "failure" in videos["clips"]
  for clip in videos["clips"].values():
    frames,seconds=imageio_ffmpeg.count_frames_and_secs(clip["path"])
    assert frames==clip["frames"] and abs(seconds-clip["encoded_seconds"])<0.05
  saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
  state=saved["model_state_dict"]
  assert state["actor.0.weight"].shape[1]==60 and state["actor.6.weight"].shape[0]==8
  assert saved["optimizer_state_dict"]["state"]
  assert all(torch.isfinite(value).all() for value in state.values())
  normalizers={key:value for key,value in state.items() if "obs_normalizer" in key}
  assert len(normalizers)==8 and normalizers["actor_obs_normalizer.count"]>0
  agent=yaml.load((checkpoint.parent/"params/agent.yaml").read_text(),Loader=yaml.FullLoader)
  assert agent["clip_actions"]==1.0
  assert "franka_shared_60_v2" in (checkpoint.parent/"params/env.yaml").read_text()
  task=validation["task"]
  destination=HERE/"runs/teachers"/task
  destination.mkdir(parents=True,exist_ok=False)
  shutil.copy2(checkpoint,destination/"model.pt")
  torch.save(normalizers,destination/"normalizers.pt")
  shutil.copytree(checkpoint.parent/"params",destination/"params")
  for source in (args.validation,args.confirmation,args.video_result,args.review,video_dir/"videos.json"):
    shutil.copy2(source,destination/source.name)
  for name in ("manifest.json","source.patch"):
    if (checkpoint.parent/name).exists():
      shutil.copy2(checkpoint.parent/name,destination/name)
  certificate={"task":task,"certified_utc":datetime.now(timezone.utc).isoformat(),
               "checkpoint":str(destination/"model.pt"),"checkpoint_sha256":checkpoint_sha,
               "normalizers":str(destination/"normalizers.pt"),"normalizers_sha256":digest(destination/"normalizers.pt"),
               "interface":"franka_shared_60_v2","validation":{"successes":validation["successes"],"episodes":validation["episodes"],"seed":validation["seed"],"path":str(args.validation.resolve())},
               "confirmation":{"successes":confirmation["successes"],"episodes":confirmation["episodes"],"seed":confirmation["seed"],"path":str(args.confirmation.resolve())},
               "review":str(args.review.resolve()),"videos":videos["clips"],
               "criterion":"Both independent measured rates strictly >90%; not a statistical lower-confidence-bound guarantee"}
  (destination/"certificate.json").write_text(json.dumps(certificate,indent=2)+'\n')
  from wandb_config import configure,ENTITY,PROJECT
  configure()
  import wandb
  with wandb.init(entity=ENTITY,project=PROJECT,name=task+"-certified",job_type="certification",config={"task":task,"checkpoint_sha256":checkpoint_sha,"interface":"franka_shared_60_v2"}) as run:
    run.log({"certified":1,"strict/validation_success_rate":validation["success_rate"],"strict/confirmation_success_rate":confirmation["success_rate"]})
    artifact=wandb.Artifact(task+"-teacher",type="certified-teacher",metadata={"task":task,"checkpoint_sha256":checkpoint_sha})
    artifact.add_dir(str(destination))
    for label,clip in videos["clips"].items():
      artifact.add_file(clip["path"],name=f"videos/{label}.mp4")
    logged=run.log_artifact(artifact,aliases=["verified"])
    logged.wait()
    certificate["wandb_run_url"]=run.url
    certificate["wandb_artifact"]=logged.qualified_name
  (destination/"certificate.json").write_text(json.dumps(certificate,indent=2)+'\n')
  (HERE/"evidence"/f"{task}-certificate.json").write_text(json.dumps(certificate,indent=2)+'\n')
  print("CERTIFIED",task,certificate["wandb_run_url"])


if __name__=="__main__":
  main()
