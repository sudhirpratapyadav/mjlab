"""Render success and highest-return failure from recorded strict first episodes."""

import argparse
import json
import os
from pathlib import Path

from allocation import GPUS

# EGL enumerates physical devices separately from CUDA's UUID remapping.
# Set the physical index before importing MuJoCo's rendering backend.
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if visible not in GPUS.values():
  raise RuntimeError("Render inside the assigned GPU1–7 allocation")
os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = str(next(i for i,uuid in GPUS.items() if uuid == visible))

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg


def main():
  parser=argparse.ArgumentParser(description=__doc__)
  parser.add_argument("result",type=Path)
  args=parser.parse_args()
  result=json.loads(args.result.read_text())
  directory=Path(result["trace_dir"])
  traces=np.load(directory/"trace.npz")
  cfg=load_env_cfg(result["task"])
  cfg.scene.num_envs=1
  cfg.seed=result["seed"]
  env=ManagerBasedRlEnv(cfg,device="cuda:0")
  clips={}
  try:
    env.reset()
    model=env.sim.mj_model
    model.vis.global_.offwidth=960
    model.vis.global_.offheight=540
    data=mujoco.MjData(model)
    renderer=mujoco.Renderer(model,540,960)
    camera=mujoco.MjvCamera()
    camera.lookat[:]=[0.43,0,0.21]
    camera.distance=1.30
    camera.azimuth=60
    camera.elevation=-28
    options=mujoco.MjvOption()
    options.geomgroup[3]=0
    options.sitegroup[:]=0
    for success,label in ((True,"success"),(False,"failure")):
      candidates=[row for row in result["records"] if row["success"]==success]
      if not candidates:
        continue
      outcome=max(candidates,key=lambda row:row["return"])
      index,steps=outcome["env_id"],outcome["steps"]
      states={name:traces[name][:steps+1,index].copy() for name in ("qpos","qvel","mocap_pos","mocap_quat")}
      for address,kind in zip(model.jnt_qposadr,model.jnt_type):
        if kind==mujoco.mjtJoint.mjJNT_FREE:
          states["qpos"][:,address:address+3]-=traces["origins"][index]
      states["mocap_pos"]-=traces["origins"][index]
      frames=list(range(2,steps+1,2))
      if not frames or frames[-1]!=steps:
        frames.append(steps)
      path=directory/f"{label}.mp4"
      writer=imageio_ffmpeg.write_frames(str(path),(960,540),fps=25,codec="libx264",pix_fmt_out="yuv420p",macro_block_size=1,output_params=["-crf","20","-preset","fast","-threads","2"],ffmpeg_log_level="error")
      writer.send(None)
      try:
        for frame_index,state_index in enumerate(frames):
          for name in states:
            getattr(data,name)[:]=states[name][state_index]
          mujoco.mj_forward(model,data)
          renderer.update_scene(data,camera=camera,scene_option=options)
          image=renderer.render().copy()
          writer.send(np.ascontiguousarray(image))
          if frame_index in (0,len(frames)//2,len(frames)-1):
            Image.fromarray(image).save(directory/f"{label}-{frame_index:04d}.png")
      finally:
        writer.close()
      clips[label]={"path":str(path),"outcome":outcome,"seed":result["seed"],"frames":len(frames),"fps":25,"simulation_seconds":steps*0.02,"encoded_seconds":len(frames)/25,"source":"exact first-episode states; rendering only, no resimulation"}
      print("RENDERED",label,index,steps,flush=True)
    renderer.close()
  finally:
    env.close()
  metadata={"task":result["task"],"checkpoint_sha256":result["checkpoint_sha256"],"clips":clips,"all_measured_succeeded":result["successes"]==result["episodes"]}
  (directory/"videos.json").write_text(json.dumps(metadata,indent=2)+'\n')
  from wandb_config import configure,ENTITY,PROJECT
  configure()
  import wandb
  with wandb.init(entity=ENTITY,project=PROJECT,name=args.result.stem+"-videos",job_type="video-review",config={"task":result["task"],"checkpoint_sha256":result["checkpoint_sha256"],"seed":result["seed"]}) as run:
    run.log({label:wandb.Video(clip["path"],format="mp4") for label,clip in clips.items()})
    artifact=wandb.Artifact(args.result.stem+"-videos",type="evaluation-videos")
    for path in directory.glob("*.png"):
      artifact.add_file(str(path))
    artifact.add_file(str(directory/"videos.json"))
    run.log_artifact(artifact)


if __name__=="__main__":
  main()
