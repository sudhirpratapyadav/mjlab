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
