"""Compare constant-control holding/release from recorded grasps on GPU and CPU."""
import argparse
import copy
import json
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.manipulation.mdp.task_geometry import grasped, touching
from rl_recipes import apply_recipe


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--evaluation',type=Path,required=True)
  parser.add_argument('--output',type=Path,required=True)
  parser.add_argument('--gyro',action='store_true')
  parser.add_argument('--elliptic-hessian',action='store_true',help='Diagnostic native-equivalent dense elliptic Hessian')
  parser.add_argument('--use-trace-friction',action='store_true')
  parser.add_argument('--fixed-grip-target',type=float)
  parser.add_argument('--failures-only',action='store_true')
  parser.add_argument('--stride',type=int,default=4)
  parser.add_argument('--fixed-only',action='store_true',help='Run only the fixed-grip current-arm intervention')
  parser.add_argument('--save-state',type=Path,help='Save matched CPU/GPU trajectory and final contact buffers for diagnosis')
  parser.add_argument('--primitive-box-box',action='store_true',help='Diagnostic GPU primitive box-box dispatch; CPU benchmark model remains unchanged')
  parser.add_argument('--use-trace-controls',action='store_true',help='Freeze actual recorded terminal actuator targets instead of re-querying the policy')
  parser.add_argument('--recorded-only',action='store_true',help='Run only the actual recorded terminal-control hold')
  args = parser.parse_args()
  evaluation = json.loads(args.evaluation.read_text())
  assert evaluation['task'] in ('Mjlab-Lift-Cube-Franka', 'Mjlab-Throw-To-Bin-Franka', 'Mjlab-Place-In-Container-Franka', 'Mjlab-Reorient-Object-Franka', 'Mjlab-Peg-Insertion-Franka')
  assert args.stride>=1
  assert not args.fixed_only or args.fixed_grip_target is not None
  assert not args.recorded_only or args.use_trace_controls
  assert not args.save_state or args.fixed_only or args.recorded_only, 'State recording requires one control case'
  checkpoint = Path(evaluation['checkpoint'])
  manifest = json.loads((checkpoint.parent/'manifest.json').read_text())
  cfg = TrainConfig.from_task(evaluation['task'])
  apply_recipe(cfg,manifest['recipe'])
  cfg.env.sim.free_body_implicitfast_compat = args.gyro or manifest.get('free_body_implicitfast_compat', False)
  cfg.env.sim.elliptic_hessian_compat = args.elliptic_hessian or manifest.get('elliptic_hessian_compat', False)
  cfg.env.sim.primitive_box_box_compat = args.primitive_box_box or manifest.get('primitive_box_box_compat', False)
  records = [r for r in evaluation['records'] if r['reason']=='timeout' and (not args.failures_only or not r['success'])][::args.stride]
  assert records, 'No selected episodes'
  cfg.env.scene.num_envs = len(records)
  cfg.env.seed = 20260914
  with np.load(Path(evaluation['trace_dir'])/'trace.npz') as archive:
    trace = {name:archive[name] for name in archive.files}
  fields = ['qpos','qvel','mocap_pos','mocap_quat']
  initial = {name:np.stack([trace[name][r['steps'],r['env_id']] for r in records]) for name in fields}
  source_origins = np.stack([trace['origins'][r['env_id']] for r in records])
  env = ManagerBasedRlEnv(cfg.env,device='cuda:0')
  wrapped = RslRlVecEnvWrapper(env,clip_actions=1.)
  try:
    env.reset()
    model = env.sim.mj_model
    if args.use_trace_friction:
      values = trace['initial_model_geom_friction'][[r['env_id'] for r in records]]
      env.sim.model.geom_friction[:] = torch.as_tensor(values,device=env.device)
    command = env.command_manager.get_term(next(iter(cfg.env.commands)))
    object_prefix=command.cfg.asset_name+'/'
    joint = next(i for i in range(model.njnt) if model.joint(i).name.startswith(object_prefix) and model.jnt_type[i]==mujoco.mjtJoint.mjJNT_FREE)
    qadr,vadr = int(model.jnt_qposadr[joint]),int(model.jnt_dofadr[joint])
    local = {name:values.copy() for name,values in initial.items()}
    for address,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:
        local['qpos'][:,address:address+3] -= source_origins
    local['mocap_pos'] -= source_origins[:,None]
    placed = {name:values.copy() for name,values in local.items()}
    for address,kind in zip(model.jnt_qposadr,model.jnt_type):
      if kind==mujoco.mjtJoint.mjJNT_FREE:
        placed['qpos'][:,address:address+3] += env.scene.env_origins.cpu().numpy()
    placed['mocap_pos'] += env.scene.env_origins.cpu().numpy()[:,None]
    def restore():
      for name,values in placed.items():
        getattr(env.sim.data,name)[:]=torch.as_tensor(values,device=env.device)
      env.sim.data.qacc_warmstart.zero_()
      if evaluation['task']=='Mjlab-Reorient-Object-Franka':
        anchor=np.stack([trace['qpos'][0,r['env_id'],qadr:qadr+3] for r in records])-source_origins
        command.target_pos[:]=torch.as_tensor(anchor,device=env.device)+env.scene.env_origins
      else:
        command.target_pos[:]=torch.as_tensor(placed['mocap_pos'][:,1],device=env.device)
      env.sim.forward()
      env.scene.update(dt=env.physics_dt)
    restore()
    if args.use_trace_controls:
      assert 'ctrl' in trace, 'This trace predates actual control recording'
      policy_ctrl=np.stack([trace['ctrl'][r['steps'],r['env_id']] for r in records])
    else:
      agent = yaml.load((checkpoint.parent/'params/agent.yaml').read_text(),Loader=yaml.FullLoader)
      agent['policy'].setdefault('class_name',cfg.agent.policy.class_name)
      agent['algorithm'].setdefault('class_name',cfg.agent.algorithm.class_name)
      runner = OnPolicyRunner(wrapped,agent,None,device=env.device)
      runner.load(str(checkpoint),load_optimizer=False,map_location=env.device)
      with torch.inference_mode():
        action = runner.get_inference_policy()(wrapped.get_observations())
      env.action_manager.process_action(action)
      env.action_manager.apply_action()
      env.scene.write_data_to_sim()
      policy_ctrl = env.sim.data.ctrl.cpu().numpy().copy()
    randomized_fields = {event.params.get('field') for event in cfg.env.events.values()
                         if event.domain_randomization}
    assert randomized_fields<={'geom_friction'},randomized_fields
    friction = env.sim.model.geom_friction.cpu().numpy().copy()
    cpu_models = [copy.copy(model) for _ in records]
    for lane,cpu_model in enumerate(cpu_models):
      cpu_model.geom_friction[:] = friction[lane]
    assert model.nu==8 and all(model.actuator_trnid[i,0]==model.joint(f'robot/joint{i+1}').id for i in range(7))
    cases = []
    frozen_mode='frozen_recorded_targets' if args.use_trace_controls else 'frozen_policy_targets'
    modes = [frozen_mode,'hold_current_arm','hold_current_arm_gentle_grip']
    if args.fixed_grip_target is not None:
      modes.append('hold_current_arm_fixed_grip')
    if args.fixed_only:
      modes=['hold_current_arm_fixed_grip']
    if args.recorded_only:
      modes=[frozen_mode]
    for mode in modes:
      ctrl = policy_ctrl.copy()
      if mode!=frozen_mode:
        ctrl[:,:7]=local['qpos'][:,:7]
      if mode=='hold_current_arm_gentle_grip':
        ctrl[:,7]=np.maximum(local['qpos'][:,7]-.002,0)
      if mode=='hold_current_arm_fixed_grip':
        ctrl[:,7]=args.fixed_grip_target
      restore()
      env.sim.data.ctrl[:]=torch.as_tensor(ctrl,device=env.device)
      cpus = [mujoco.MjData(cpu_model) for cpu_model in cpu_models]
      for lane,cpu in enumerate(cpus):
        for name in fields:
          getattr(cpu,name)[:]=local[name][lane]
        cpu.ctrl[:]=ctrl[lane]
        mujoco.mj_forward(cpu_models[lane],cpu)
        cpu.qacc_warmstart[:]=0
      history={name:[] for name in ['gpu_qpos','gpu_qvel','cpu_qpos','cpu_qvel']}
      def record_state():
        if not args.save_state:
          return
        for name in ['qpos','qvel']:
          history['gpu_'+name].append(getattr(env.sim.data,name).cpu().numpy().copy())
          history['cpu_'+name].append(np.stack([getattr(cpu,name).copy() for cpu in cpus]))
      record_state()
      for step in range(400):
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
        for cpu_model,cpu in zip(cpu_models,cpus):
          mujoco.mj_step(cpu_model,cpu)
        record_state()
      env.sim.forward()
      env.scene.update(dt=env.physics_dt)
      for cpu_model,cpu in zip(cpu_models,cpus):
        mujoco.mj_forward(cpu_model,cpu)
      gpu_qvel=env.sim.data.qvel.cpu().numpy().copy()
      gpu_qpos=env.sim.data.qpos.cpu().numpy().copy()
      command._update_metrics()
      native_grasp=grasped(command).cpu().numpy()
      cpu_v=np.stack([cpu.qvel.copy() for cpu in cpus])
      summary={}
      for name,vel in [('gpu',gpu_qvel),('cpu',cpu_v)]:
        linear=np.linalg.norm(vel[:,vadr:vadr+3],axis=1)
        angular=np.linalg.norm(vel[:,vadr+3:vadr+6],axis=1)
        native_linear=.1 if evaluation['task']=='Mjlab-Lift-Cube-Franka' else getattr(command.cfg,'settle_speed',.03)
        native_angular=.5 if evaluation['task']=='Mjlab-Lift-Cube-Franka' else .3
        summary[name]=dict(settled=int(((linear<.1)&(angular<.5)).sum()),
                           native_settled=int(((linear<native_linear)&(angular<native_angular)).sum()),
                           median_linear_speed_m_s=float(np.median(linear)),
                           median_angular_speed_rad_s=float(np.median(angular)),
                           linear_speeds=linear.tolist(),angular_speeds=angular.tolist())
      summary['gpu'].update(grasped=int(native_grasp.sum()),native_success=int(command.compute_success().sum()),
                            robot_contact_by_lane=touching(command,command.robot,command.object).cpu().tolist(),
                            finite=bool(np.isfinite(gpu_qvel).all() and np.isfinite(gpu_qpos).all()))
      gpu_pos = gpu_qpos[:,qadr:qadr+3]-env.scene.env_origins.cpu().numpy()
      cpu_pos = np.stack([cpu.qpos[qadr:qadr+3].copy() for cpu in cpus])
      for name,positions in [('gpu',gpu_pos),('cpu',cpu_pos)]:
        summary[name].update(median_height_m=float(np.median(positions[:,2])),
                             positions_local_m=positions.tolist(),
                             goal_errors_m=np.linalg.norm(positions-local['mocap_pos'][:,1],axis=1).tolist())
      pads = [model.geom('robot/'+name).id for name in ['left_finger_pad','right_finger_pad']]
      object_ids = {i for i in range(model.ngeom) if model.geom(i).name.startswith(object_prefix)}
      cpu_grasp = []
      for cpu in cpus:
        cpu_grasp.append(all(any(contact.dist<=.001 and ((contact.geom[0]==pad and contact.geom[1] in object_ids) or (contact.geom[1]==pad and contact.geom[0] in object_ids)) for contact in cpu.contact) for pad in pads))
      summary['cpu']['grasped'] = sum(cpu_grasp)
      summary['cpu']['warnings']=[cpu.warning.number.tolist() for cpu in cpus]
      if args.save_state:
        nacon=int(env.sim.data.nacon[0])
        contact=env.sim.data.contact
        saved={name:np.stack(values) for name,values in history.items()}
        saved.update(ctrl=ctrl,friction=friction,mocap_pos=local['mocap_pos'],mocap_quat=local['mocap_quat'],
                     gpu_origins=env.scene.env_origins.cpu().numpy(),lanes=np.array([r['env_id'] for r in records]),
                     gpu_qacc=env.sim.data.qacc.cpu().numpy(),cpu_qacc=np.stack([cpu.qacc.copy() for cpu in cpus]))
        for name in ['geom','worldid','dist','pos','frame']:
          saved['gpu_contact_'+name]=getattr(contact,name)[:nacon].cpu().numpy()
        args.save_state.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.save_state,**saved)
        summary['cpu']['object_contacts']=[[
          dict(geoms=[model.geom(int(i)).name for i in c.geom],dist=float(c.dist),pos=c.pos.tolist(),frame=c.frame.tolist())
          for c in cpu.contact if any(int(g) in object_ids for g in c.geom)
        ] for cpu in cpus]
        geoms=saved['gpu_contact_geom'];worlds=saved['gpu_contact_worldid']
        summary['gpu']['object_contacts']=[[
          dict(geoms=[model.geom(int(i)).name for i in geoms[j]],dist=float(saved['gpu_contact_dist'][j]),pos=saved['gpu_contact_pos'][j].tolist(),frame=saved['gpu_contact_frame'][j].tolist())
          for j in np.flatnonzero(worlds==lane) if any(int(g) in object_ids for g in geoms[j])
        ] for lane in range(len(records))]
      cases.append(dict(mode=mode,**summary))
      print(mode,{name:{k:v for k,v in row.items() if k not in ['warnings','linear_speeds','angular_speeds','positions_local_m','goal_errors_m','object_contacts','robot_contact_by_lane']} for name,row in summary.items()},flush=True)
    report=dict(task=evaluation['task'],checkpoint_sha256=evaluation['checkpoint_sha256'],
                gyro_correction=cfg.env.sim.free_body_implicitfast_compat,
                elliptic_hessian_compat=cfg.env.sim.elliptic_hessian_compat,
                diagnostic_settle_limits=dict(linear_m_s=.1,angular_rad_s=.5),
                native_settle_limits=dict(linear_m_s=native_linear,angular_rad_s=native_angular),
                episodes=len(records),lanes=[r['env_id'] for r in records],
                model_parameters_matched_between_cpu_gpu=['geom_friction'],
                source_episode_randomization_reconstructed=args.use_trace_friction,physics_dt=env.physics_dt,
                fixed_grip_target=args.fixed_grip_target,failures_only=args.failures_only,stride=args.stride,
                state_archive=str(args.save_state) if args.save_state else None,
                gpu_primitive_box_box=cfg.env.sim.primitive_box_box_compat,
                actual_recorded_controls=args.use_trace_controls,
                method=f'Every {args.stride}th timeout terminal state, failed-only={args.failures_only}, cold solver cache,400 native steps with constant XML position-actuator controls. '+('Actual applied terminal controls restored from trace; ' if args.use_trace_controls else 'Policy queried once from restored observations; ')+'alternatives hold current arm joints, optionally reduce gripper closure to2mm or use the reported fixed finger target. CPU and GPU receive identical local state/controls and matched per-world friction. '+('Original evaluation friction is restored from the trace. ' if args.use_trace_friction else 'This probe resamples the registered friction distribution. ')+'No policy training, no first-episode success evaluation; steady-control intervention only.',cases=cases)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
  finally:
    wrapped.close()


if __name__=='__main__':
  main()
