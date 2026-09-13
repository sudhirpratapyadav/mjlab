import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.scripts.train import TrainConfig
from mjlab.tasks.manipulation.mdp.rewards import articulation_task_reward


@pytest.mark.parametrize("task,recipe", [
  ("Mjlab-Drag-Pull-Franka","baseline_long"),
  ("Mjlab-Push-Cuboid-Franka","stable_v1"),
  ("Mjlab-Push-Button-Franka","mechanism_v1"),
  ("Mjlab-Open-Drawer-Franka","mechanism_v1"),
  ("Mjlab-Lift-Cube-Franka","lift_v1"),
  ("Mjlab-Reorient-Object-Franka","reorient_v1"),
  ("Mjlab-Lift-Cube-Franka","lift_v2"),
  ("Mjlab-Lift-Cube-Franka","lift_v3"),
  ("Mjlab-Lift-Cube-Franka","lift_v4"),
  ("Mjlab-Lift-Cube-Franka","lift_v5"),
  ("Mjlab-Lift-Cube-Franka","lift_v6"),
  ("Mjlab-Lift-Cube-Franka","lift_smooth_v1"),
  ("Mjlab-Lift-Cube-Franka","lift_smooth_v2"),
  ("Mjlab-Reorient-Object-Franka","reorient_v2"),
  ("Mjlab-Reorient-Object-Franka","reorient_v3"),
  ("Mjlab-Reorient-Object-Franka","reorient_v4"),
  ("Mjlab-Reorient-Object-Franka","reorient_v5"),
  ("Mjlab-Reorient-Object-Franka","reorient_v6"),
  ("Mjlab-Reorient-Object-Franka","reorient_v7"),
  ("Mjlab-Cage-Drag-Franka","cage_v1"),
  ("Mjlab-Cage-Drag-Franka","cage_v2"),
  ("Mjlab-Cage-Drag-Franka","cage_v3"),
  ("Mjlab-Cage-Drag-Franka","cage_v4"),
  ("Mjlab-Open-Lid-Franka","lid_v1"),
  ("Mjlab-Open-Lid-Franka","lid_v2"),
  ("Mjlab-Edge-Grasp-Franka","edge_v1"),
  ("Mjlab-Pivot-Lift-Franka","pivot_v1"),
  ("Mjlab-Strike-Slide-Franka","strike_v1"),
  ("Mjlab-Throw-To-Bin-Franka","throw_v1"),
  ("Mjlab-Edge-Grasp-Franka","edge_v2"),
  ("Mjlab-Edge-Grasp-Franka","edge_v6"),
  ("Mjlab-Pivot-Lift-Franka","pivot_v2"),
  ("Mjlab-Throw-To-Bin-Franka","throw_v2"),
  ("Mjlab-Throw-To-Bin-Franka","throw_v3"),
  ("Mjlab-Throw-To-Bin-Franka","throw_v4"),
  ("Mjlab-Stack-Cube-Franka","completion_v1"),
  ("Mjlab-Place-In-Container-Franka","completion_v1"),
  ("Mjlab-Peg-Insertion-Franka","completion_v1"),
  ("Mjlab-Stack-Cube-Franka","completion_v2"),
  ("Mjlab-Place-In-Container-Franka","completion_v2"),
  ("Mjlab-Peg-Insertion-Franka","completion_v2"),
  ("Mjlab-Stack-Cube-Franka","completion_v3"),
  ("Mjlab-Place-In-Container-Franka","completion_v3"),
  ("Mjlab-Peg-Insertion-Franka","completion_v3"),
  ("Mjlab-Stack-Cube-Franka","completion_v4"),
  ("Mjlab-Place-In-Container-Franka","completion_v4"),
  ("Mjlab-Peg-Insertion-Franka","completion_v4"),
  ("Mjlab-Stack-Cube-Franka","completion_v5"),
  ("Mjlab-Place-In-Container-Franka","completion_v5"),
  ("Mjlab-Peg-Insertion-Franka","completion_v5"),
  ("Mjlab-Stack-Cube-Franka","completion_v6"),
  ("Mjlab-Place-In-Container-Franka","completion_v6"),
  ("Mjlab-Peg-Insertion-Franka","completion_v6"),
  ("Mjlab-Peg-Insertion-Franka","peg_v3"),
  ("Mjlab-Peg-Insertion-Franka","peg_v4"),
  ("Mjlab-Reorient-Object-Franka","reorient_v9"),
  ("Mjlab-Reorient-Object-Franka","reorient_v11"),
  ("Mjlab-Strike-Slide-Franka","strike_v5"),
])
def test_recipes_preserve_benchmark(task,recipe,monkeypatch):
  stage=Path(__file__).resolve().parents[1]/"src/mjlab/continual_distill/docs/cl_v4_rl"
  monkeypatch.syspath_prepend(str(stage))
  recipes=importlib.import_module("rl_recipes")
  cfg=TrainConfig.from_task(task)
  fields=("observations","actions","commands","terminations","events","scene","sim","episode_length_s")
  before={field:repr(getattr(cfg.env,field)) for field in fields}
  recipes.apply_recipe(cfg,recipe)
  assert before=={field:repr(getattr(cfg.env,field)) for field in fields}
  assert cfg.agent.clip_actions==1.0


@pytest.mark.parametrize('recipe,velocity_weight,acceleration_weight,action_weight',[
  ('lift_smooth_v1',-2.,-.0002,-5.),('lift_smooth_v2',-10.,-.001,-50.)])
def test_lift_smoothing_preserves_task_and_penalizes_arm_motion(monkeypatch,recipe,velocity_weight,acceleration_weight,action_weight):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Lift-Cube-Franka') for _ in range(2))
  recipes.apply_recipe(old,'lift_v9');recipes.apply_recipe(new,recipe)
  assert asdict(old.agent)==asdict(new.agent)
  vel=new.env.rewards['joint_vel_penalty'];acc=new.env.rewards['smooth_arm_acceleration']
  arm=tuple(f'joint{i}' for i in range(1,8))
  assert tuple(vel.params['robot_asset_cfg'].joint_names)==arm
  assert tuple(acc.params['asset_cfg'].joint_names)==arm
  assert vel.weight==velocity_weight and vel.params['max_vel']==1 and acc.weight==acceleration_weight
  # Large finger velocities/accelerations must not enter an arm-only metric.
  robot=SimpleNamespace(data=SimpleNamespace(joint_vel=torch.tensor([[.5]*7+[100.,100.],[2.]*7+[0.,0.]]),joint_acc=torch.tensor([[0.]*7+[10000.,10000.],[10.]*7+[0.,0.]])))
  env=SimpleNamespace(scene={'robot':robot})
  select=SimpleNamespace(name='robot',joint_ids=list(range(7)))
  vp=vel.func(env,max_vel=1.,robot_asset_cfg=select)
  ap=acc.func(env,asset_cfg=select)
  torch.testing.assert_close(vp,torch.tensor([0.,7.]))
  torch.testing.assert_close(ap,torch.tensor([0.,700.]))
  assert (vel.weight*vp)[1]<0 and (acc.weight*ap)[1]<0
  change=new.env.rewards['action_rate_l2']
  actions=torch.tensor([[0.]*8,[.5]*8])
  env.action_manager=SimpleNamespace(action=actions,prev_action=torch.zeros_like(actions))
  assert change.weight==action_weight
  torch.testing.assert_close(change.weight*change.func(env),torch.tensor([0.,2*action_weight]))
  for name in ['joint_vel_penalty','action_rate_l2']:new.env.rewards[name]=old.env.rewards[name]
  del new.env.rewards['smooth_arm_acceleration']
  assert repr(old.env)==repr(new.env)


def test_peg_lift_credit_changes_only_two_reward_weights(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage)); recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Peg-Insertion-Franka') for _ in range(2))
  recipes.apply_recipe(old,'peg_v3'); recipes.apply_recipe(new,'peg_v4')
  assert asdict(old.agent)==asdict(new.agent)
  params=new.env.rewards['stack'].params
  assert params['lift_weight']==12. and params.pop('native_completion_weight')==30.
  params['lift_weight']=4.
  assert repr(old.env)==repr(new.env)


def test_edge_clearance_preserves_benchmark_and_bounded_waypoint(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  reward=importlib.import_module('remaining_reward')
  old,new=(TrainConfig.from_task('Mjlab-Edge-Grasp-Franka') for _ in range(2))
  recipes.apply_recipe(old,'edge_v5');recipes.apply_recipe(new,'edge_v6')
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('arrival_clearance')==.03
  assert repr(old.env)==repr(new.env)
  heights=torch.tensor([-.05,0.,.020,.053,.10])
  target=torch.zeros(5,3);hand=torch.stack([torch.zeros(5),torch.zeros(5),heights],dim=1)
  axis=torch.tensor([1.,0.,0.]).expand(5,3)
  out=reward.edge_clearance_waypoint(target,hand,axis,.03)
  assert torch.isfinite(out).all() and (out[:,0]>=-.03).all() and (out[:,0]<=0).all()
  assert (out[1:,0]<out[:-1,0]).all() and abs(out[1,0])<.00054
  assert abs(out[3,0]+.03)<.00005 and not out[:,1:].any()
  assert not target.any()  # Inputs must remain unchanged.
  shift=torch.tensor([3.,-4.,.7])
  torch.testing.assert_close(reward.edge_clearance_waypoint(target+shift,hand+shift,axis,.03),out+shift)


def test_reorient_quiet_score_requires_axis_progress_and_both_speeds(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage)); recipes=importlib.import_module('rl_recipes')
  orientation=torch.tensor([1.,1.,1.,1.,0.])
  linear=torch.tensor([0.,.123,0.,.123,0.])
  angular=torch.tensor([0.,0.,.999,.999,0.])
  saved=[v.clone() for v in (orientation,linear,angular)]
  score=recipes.reorient_quiet_score(orientation,linear,angular)
  assert score[0]>score[1]>score[3]>0
  assert score[0]>score[2]>score[3] and score[4]==0
  assert (score>=0).all() and (score<=1).all()
  for value,before in zip((orientation,linear,angular),saved):
    torch.testing.assert_close(value,before,rtol=0,atol=0)


def test_mechanism_approach_recovers_signal_without_changing_joint_goal():
  distances=torch.tensor([0.689,0.8,0.939])
  obj=SimpleNamespace(site_names=[],data=SimpleNamespace(root_link_pos_w=torch.stack((distances,torch.zeros(3),torch.zeros(3)),dim=1)))
  robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.zeros(3,1,3)))
  command=SimpleNamespace(target_value=torch.ones(3),_joint_value=lambda:torch.zeros(3),cfg=SimpleNamespace(success_threshold=0.1))
  env=SimpleNamespace(scene={"object":obj,"robot":robot},command_manager=SimpleNamespace(get_term=lambda _:command))
  params={"command_name":"task","robot_asset_cfg":SimpleNamespace(name="robot",site_ids=[0])}
  old=articulation_task_reward(env,**params)
  new=articulation_task_reward(env,approach_scale=0.3,**params)
  assert old.max()<1e-5
  assert new.min()>0.04
  assert torch.all(new[:-1]>new[1:])
  torch.testing.assert_close(articulation_task_reward(env,"task"),articulation_task_reward(env,"task",approach_scale=0.3))


def test_cage_guidance_rewards_starting_transport_but_never_invalid_pinch(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes=importlib.import_module('rl_recipes')
  geometry=importlib.import_module('mjlab.tasks.manipulation.mdp.task_geometry')
  monkeypatch.setattr(geometry,'between_fingers',lambda _:torch.ones(2,dtype=torch.bool))
  robot=SimpleNamespace(data=SimpleNamespace(site_pos_w=torch.tensor([[[.4,0.,.057]],[[.402,0.,.057]]]),site_quat_w=torch.tensor([[[0.,1.,0.,0.]],[[0.,1.,0.,0.]]])))
  obj=SimpleNamespace(site_names=[],data=SimpleNamespace(root_link_pos_w=torch.tensor([[.4,0.,.02],[.4,0.,.02]])))
  command=SimpleNamespace(robot=robot,robot_cfg=SimpleNamespace(site_ids=[0]),object=obj,target_pos=torch.tensor([[.6,0.,.02],[.6,0.,.02]]),min_aperture=torch.full((2,),.08),_aperture=lambda:torch.full((2,),.08),cfg=SimpleNamespace(aperture_min=.06949),caged_progress=torch.zeros(2),required_progress=torch.full((2,),.005),compute_success=lambda:torch.zeros(2,dtype=torch.bool))
  env=SimpleNamespace(device='cpu',num_envs=2,command_manager=SimpleNamespace(get_term=lambda _:command))
  old=recipes.cage_approach_reward(env,'cage_drag')
  guided=recipes.cage_approach_reward(env,'cage_drag',transport_guidance=True)
  assert old[1]<old[0]  # Moving inside the gap initially hurts the old approach.
  assert guided[1]>guided[0]
  command.min_aperture[:]=.06
  assert torch.equal(recipes.cage_approach_reward(env,'cage_drag',transport_guidance=True),torch.zeros(2))


def test_cage_target_places_trailing_pad_on_rear_face_for_both_directions(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes=importlib.import_module('rl_recipes')
  positions=torch.tensor([[.4,0.,.023],[.4,0.,.023]])
  corners=positions[:,None]+torch.cartesian_prod(*[torch.tensor([-.023,.023])]*3)[None]
  grip=torch.tensor([[.4,0.,.04],[.4,0.,.04]])
  offsets=torch.tensor([[[-.055,0.,-.01],[.055,0.,-.01]]]).expand(2,-1,-1)
  pads=grip[:,None]+offsets
  direction=torch.tensor([[1.,0.,0.],[-1.,0.,0.]])
  target=recipes.cage_contact_target(positions,corners,pads,grip,direction)
  resulting_pads=target[:,None]+offsets
  # Positive direction uses the negative-side pad's inner face, and vice versa.
  torch.testing.assert_close(resulting_pads[0,0,0]+.0076,positions[0,0]-.023+.001)
  torch.testing.assert_close(resulting_pads[1,1,0]-.0076,positions[1,0]+.023-.001)
  torch.testing.assert_close(resulting_pads.mean(1)[:,2],positions[:,2])
  # The transport target must encourage forward motion in either direction,
  # without shifting vertical/lateral placement or mutating physical state.
  before = [t.clone() for t in (positions,corners,pads,grip,direction)]
  driven = recipes.cage_contact_target(positions,corners,pads,grip,direction,contact_drive=.012)
  torch.testing.assert_close(driven-target,.011*direction)
  for source, saved in zip((positions,corners,pads,grip,direction),before):
    torch.testing.assert_close(source,saved)
def test_lift_settling_bonus_distinguishes_tight_grip_translation_and_spin(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes=importlib.import_module('rl_recipes')
  # Same captured near-goal pose; independently vary closure load and motion.
  distance=torch.full((5,),.025)
  linear=torch.tensor([0.,0.,.27,0.,0.])
  angular=torch.tensor([0.,0.,0.,1.87,0.])
  actual=torch.full((5,),.023)
  target=torch.tensor([.021,.001,.021,.021,.021])
  held=torch.tensor([1.,1.,1.,1.,0.])
  score=recipes.lift_settling_bonus(distance,linear,angular,actual,target,held)
  assert (score[0]>score[1:4]).all() and score[4]==0
  # Settling far from the goal must not receive the near-goal quiet bonus.
  distant=recipes.lift_settling_bonus(torch.full((5,),.3),linear,angular,actual,target,held)
  assert score[0]>distant[0]+2


def test_lift_settling_recipe_preserves_native_environment_and_agent(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Lift-Cube-Franka') for _ in range(2))
  recipes.apply_recipe(old,'lift_v6');recipes.apply_recipe(new,'lift_v7')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('settle_grip')
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_lift_arm_hold_score_has_signal_at_recorded_error_and_preserves_inputs(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  distance=torch.tensor([.013,.013,.013,.3,.013])
  error=torch.tensor([0.,.05,1.046,0.,0.])
  held=torch.tensor([1.,1.,1.,1.,0.]);before=[v.clone() for v in (distance,error,held)]
  score=recipes.lift_arm_hold_bonus(distance,error,held)
  assert score[0]>score[1]>score[2]>.1
  assert score[3]<.003 and score[4]==0
  for value,saved in zip((distance,error,held),before):torch.testing.assert_close(value,saved)


def test_lift_arm_recipe_only_changes_opt_in_reward(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Lift-Cube-Franka') for _ in range(2))
  recipes.apply_recipe(old,'lift_v8');recipes.apply_recipe(new,'lift_v9')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('arm_hold_weight')==15.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_lift_completion_recipe_preserves_native_benchmark(monkeypatch):
  import importlib
  from dataclasses import asdict
  from pathlib import Path
  from mjlab.scripts.train import TrainConfig
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Lift-Cube-Franka') for _ in range(2))
  recipes.apply_recipe(old,'lift_v7');recipes.apply_recipe(new,'lift_v8')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  params=new.env.rewards['reach_object'].params
  assert params.pop('quiet_weight')==15.
  assert params.pop('native_completion_weight')==25.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_reorient_axis_score_has_progress_beyond_ninety_degrees(monkeypatch):
  import importlib
  from pathlib import Path
  import torch
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  recipes=importlib.import_module('rl_recipes')
  angles=torch.deg2rad(torch.tensor([170.,135.,95.,85.,45.,10.],requires_grad=True))
  angles.retain_grad()
  score=recipes.reorient_axis_score(angles.cos())
  assert (score[1:]>score[:-1]).all()
  score.sum().backward()
  assert (angles.grad<0).all()
  assert recipes.reorient_axis_score(torch.tensor([-1.,1.]),True).tolist()==[1.,1.]
  assert recipes.reorient_axis_score(torch.tensor([-1.,1.])).tolist()==[0.,1.]


def test_reorient_axis_recipe_preserves_native_benchmark(monkeypatch):
  import importlib
  from dataclasses import asdict
  from pathlib import Path
  from mjlab.scripts.train import TrainConfig
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Reorient-Object-Franka') for _ in range(2))
  recipes.apply_recipe(old,'reorient_v7');recipes.apply_recipe(new,'reorient_v8')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  params=new.env.rewards['reach_object'].params
  assert params.pop('continuous_orientation')
  assert params.pop('native_completion_weight')==25.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_peg_lift_recipe_preserves_interface_physics_and_native_gate(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Peg-Insertion-Franka') for _ in range(2))
  recipes.apply_recipe(old,'peg_v1');recipes.apply_recipe(new,'peg_v2')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['stack'].params.pop('lift_weight')==4.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_reorient_steady_actions_preserve_benchmark_and_gate_on_orientation(monkeypatch):
  import importlib
  from dataclasses import asdict
  from pathlib import Path
  import torch
  from mjlab.scripts.train import TrainConfig
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Reorient-Object-Franka') for _ in range(2))
  recipes.apply_recipe(old,'reorient_v9');recipes.apply_recipe(new,'reorient_v10')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('steady_action_weight')==3.
  assert repr(old.env.rewards)==repr(new.env.rewards)
  action=torch.tensor([0.,.01,.1,1.]).unsqueeze(-1).expand(4,8);previous=torch.zeros_like(action)
  score=recipes.reorient_steady_action_score(torch.ones(4),action,previous)
  assert score[0]==1 and torch.all(score[1:]<score[:-1])
  assert not recipes.reorient_steady_action_score(torch.zeros(4),action,previous).any()


def test_strike_returns_changes_only_rollout_and_lambda(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage)); recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Strike-Slide-Franka') for _ in range(2))
  recipes.apply_recipe(old,'strike_v4');recipes.apply_recipe(new,'strike_v5')
  assert repr(old.env)==repr(new.env)
  assert new.agent.num_steps_per_env==96 and new.agent.algorithm.lam==.99
  assert new.agent.algorithm.gamma==old.agent.algorithm.gamma==.995
  assert 500*new.agent.num_steps_per_env==2000*old.agent.num_steps_per_env
  new.agent.num_steps_per_env=old.agent.num_steps_per_env
  new.agent.algorithm.lam=old.agent.algorithm.lam
  assert asdict(old.agent)==asdict(new.agent)


def test_reorient_stronger_settling_keeps_native_bonus_dominant(monkeypatch):
  from dataclasses import asdict
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage)); recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Reorient-Object-Franka') for _ in range(2))
  recipes.apply_recipe(old,'reorient_v10');recipes.apply_recipe(new,'reorient_v11')
  assert asdict(old.agent)==asdict(new.agent)
  p=new.env.rewards['reach_object'].params
  assert p['quiet_weight']==12 and p['steady_action_weight']==12
  assert 6+3+6+p['quiet_weight']+p['steady_action_weight']<p['native_completion_weight']==50
  for key in ('quiet_weight','steady_action_weight','native_completion_weight'):
    p[key]=old.env.rewards['reach_object'].params[key]
  assert repr(old.env)==repr(new.env)
