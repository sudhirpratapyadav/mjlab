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
  ("Mjlab-Reorient-Object-Franka","reorient_v2"),
  ("Mjlab-Reorient-Object-Franka","reorient_v3"),
  ("Mjlab-Reorient-Object-Franka","reorient_v4"),
  ("Mjlab-Cage-Drag-Franka","cage_v1"),
  ("Mjlab-Cage-Drag-Franka","cage_v2"),
  ("Mjlab-Open-Lid-Franka","lid_v1"),
  ("Mjlab-Open-Lid-Franka","lid_v2"),
  ("Mjlab-Edge-Grasp-Franka","edge_v1"),
  ("Mjlab-Pivot-Lift-Franka","pivot_v1"),
  ("Mjlab-Strike-Slide-Franka","strike_v1"),
  ("Mjlab-Throw-To-Bin-Franka","throw_v1"),
  ("Mjlab-Stack-Cube-Franka","completion_v1"),
  ("Mjlab-Place-In-Container-Franka","completion_v1"),
  ("Mjlab-Peg-Insertion-Franka","completion_v1"),
  ("Mjlab-Stack-Cube-Franka","completion_v2"),
  ("Mjlab-Place-In-Container-Franka","completion_v2"),
  ("Mjlab-Peg-Insertion-Franka","completion_v2"),
  ("Mjlab-Stack-Cube-Franka","completion_v3"),
  ("Mjlab-Place-In-Container-Franka","completion_v3"),
  ("Mjlab-Peg-Insertion-Franka","completion_v3"),
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
