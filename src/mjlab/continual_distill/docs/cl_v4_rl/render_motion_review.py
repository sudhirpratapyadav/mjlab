"""Render fixed paired episodes at50fps/real time, directly from stored states."""
import argparse
import json
import os
from pathlib import Path
from allocation import GPUS

visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
if visible not in GPUS.values():raise RuntimeError('Use an allocated GPU1–7 UUID')
os.environ['MUJOCO_GL']='egl'
os.environ['MUJOCO_EGL_DEVICE_ID']=str(next(i for i,u in GPUS.items() if u==visible))
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image
from mjlab.scene import Scene
from mjlab.tasks.registry import load_env_cfg


def main():
  p=argparse.ArgumentParser(description=__doc__)
  p.add_argument('--baseline',type=Path,required=True)
  p.add_argument('--candidate',type=Path,required=True)
  p.add_argument('--output',type=Path,required=True)
  a=p.parse_args()
  if a.output.exists():p.error('Output exists')
  a.output.mkdir(parents=True)
  baseline=json.loads(a.baseline.read_text());candidate=json.loads(a.candidate.read_text())
  assert baseline['task']==candidate['task']=='Mjlab-Lift-Cube-Franka' and baseline['seed']==candidate['seed']
  cfg=load_env_cfg(baseline['task']);cfg.scene.num_envs=1;m=Scene(cfg.scene,'cpu').compile();cfg.sim.mujoco.apply(m)
  assert abs(cfg.decimation*m.opt.timestep-.02)<1e-12
  m.vis.global_.offwidth=960;m.vis.global_.offheight=540;d=mujoco.MjData(m)
  renderer=mujoco.Renderer(m,540,960);camera=mujoco.MjvCamera();camera.lookat[:]=[.43,0,.21];camera.distance=1.3;camera.azimuth=60;camera.elevation=-28
  options=mujoco.MjvOption();options.geomgroup[3]=0;options.sitegroup[:]=0
  rows=[];initial={};friction={}
  try:
    for label,ev in [('before',baseline),('after',candidate)]:
      with np.load(Path(ev['trace_dir'])/'trace.npz') as z:t={k:z[k] for k in ['qpos','qvel','mocap_pos','mocap_quat','origins','initial_model_geom_friction']}
      for lane in [0,1,2]:
        r=next(x for x in ev['records'] if x['env_id']==lane);end=r['steps'];states={k:t[k][:end+1,lane].copy() for k in ['qpos','qvel','mocap_pos','mocap_quat']}
        for adr,kind in zip(m.jnt_qposadr,m.jnt_type):
          if kind==mujoco.mjtJoint.mjJNT_FREE:states['qpos'][:,adr:adr+3]-=t['origins'][lane]
        states['mocap_pos']-=t['origins'][lane]
        if label=='before':
          initial[lane]={k:v[0].copy() for k,v in states.items()}
          friction[lane]=t['initial_model_geom_friction'][lane].copy()
        else:
          for k in states:np.testing.assert_allclose(states[k][0],initial[lane][k],rtol=0,atol=1e-6)
          np.testing.assert_array_equal(t['initial_model_geom_friction'][lane],friction[lane])
        path=a.output/f'{label}-{lane}.mp4';writer=imageio_ffmpeg.write_frames(str(path),(960,540),fps=50,codec='libx264',pix_fmt_out='yuv420p',macro_block_size=1,output_params=['-crf','20','-preset','fast','-threads','2'],ffmpeg_log_level='error');writer.send(None)
        try:
          for step in range(1,end+1):
            for k in states:getattr(d,k)[:]=states[k][step]
            mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera,scene_option=options);frame=renderer.render().copy();writer.send(np.ascontiguousarray(frame))
            if step in [1,end//2,end]:Image.fromarray(frame).save(a.output/f'{label}-{lane}-{step:04d}.png')
        finally:writer.close()
        rows.append(dict(label=label,env_id=lane,success=r['success'],frames=end,fps=50,simulation_seconds=end*.02,encoded_seconds=end/50,video=path.name,poster=f'{label}-{lane}-0001.png',checkpoint_sha256=ev['checkpoint_sha256']))
        print(label,lane,r['success'],'rendered',flush=True)
  finally:renderer.close()
  (a.output/'videos.json').write_text(json.dumps(dict(task=baseline['task'],seed=baseline['seed'],paired_initial_states_verified=True,paired_friction_verified=True,method='Fixed preregistered envs0/1/2, identical initial states and randomized friction; each recorded20ms state rendered once at50fps. No resimulation/interpolation/time scaling or video smoothing.',clips=rows),indent=2)+'\n')


if __name__=='__main__':main()
