"""Shaping predictions must distinguish correct flights from plausible misses."""
import importlib
from pathlib import Path
import torch


def module(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  return importlib.import_module('remaining_reward')


def test_slide_prediction_distinguishes_undershoot_overshoot_and_sideways(monkeypatch):
  reward = module(monkeypatch)
  start = torch.tensor([[.4,0.]]).repeat(4,1)
  mu = .04
  speed = (2*mu*9.81*.5)**.5
  velocity = torch.tensor([[speed,0.],[speed*.5,0.],[speed*1.5,0.],[0.,speed]])
  landing = reward.sliding_endpoint(start,velocity,mu)
  goal = torch.tensor([.9,0.])
  error = torch.linalg.vector_norm(landing-goal,dim=-1)
  assert error[0]<1e-6 and (error[1:]>.3).all()
  torch.testing.assert_close(reward.sliding_endpoint(start,torch.zeros_like(velocity),mu),start)


def test_strike_target_crosses_wedge_contact_without_mutating_state(monkeypatch):
  reward = module(monkeypatch)
  position = torch.tensor([[.4,.1,.0127],[.4,-.1,1.0127]])
  original = position.clone()
  direction = torch.tensor([[1.,0.],[-1.,0.]])
  floor = torch.tensor([0.,1.])
  old = reward.strike_approach_target(position,direction,floor)
  driven = reward.strike_approach_target(position,direction,floor,.012)
  # Independently derive the first pad/cylinder contact in the horizontal
  # plane from registered geometry:38.1mm puck radius,30mm half-gap,
  # 8.75mm pad half-length. The old seating waypoint leaves a real gap.
  contact_standoff = (.0381**2-.030**2)**.5+.00875
  old_standoff = ((position[:,:2]-old[:,:2])*direction).sum(-1)
  new_standoff = ((position[:,:2]-driven[:,:2])*direction).sum(-1)
  assert (old_standoff>contact_standoff+.002).all()
  assert (new_standoff<contact_standoff-.008).all()
  torch.testing.assert_close(position,original)
  torch.testing.assert_close(driven[:,2],floor+.028)
  torch.testing.assert_close(old[:,1],driven[:,1])


def test_strike_drive_recipe_preserves_benchmark_and_policy(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes = importlib.import_module('rl_recipes')
  old,new = (TrainConfig.from_task('Mjlab-Strike-Slide-Franka') for _ in range(2))
  recipes.apply_recipe(old,'strike_v1')
  recipes.apply_recipe(new,'strike_v2')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('contact_drive')==.012
  assert new.env.rewards['reach_object'].params.pop('precision_weight')==2.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_edge_side_alignment_rejects_wrong_heading_and_vertical_closing_sign(monkeypatch):
  import math
  reward = module(monkeypatch)
  pitch = math.radians(20)
  # Site x=-world y, closing axis mostly downward, approach mostly +world x.
  rotation = torch.tensor([[0.,-math.sin(pitch),math.cos(pitch)],
                           [-1.,0.,0.],[0.,-math.cos(pitch),-math.sin(pitch)]])
  wrong_heading = torch.diag(torch.tensor([-1.,-1.,1.]))@rotation
  wrong_roll = rotation@torch.diag(torch.tensor([-1.,-1.,1.]))
  matrices = torch.stack([rotation,wrong_heading,wrong_roll])
  axes = [matrices[:,:,i] for i in range(3)]
  far = torch.tensor([[1.,0.,0.]]).expand(3,-1)
  score = reward.edge_side_alignment(axes,far)
  torch.testing.assert_close(score[0],torch.tensor(1.))
  assert (score[1:]<.5).all()
  # Heading distinction was missing from the old abs(vertical-closing) factor.
  torch.testing.assert_close(matrices[0,2,1].abs(),matrices[1,2,1].abs())


def test_edge_side_recipe_preserves_benchmark_and_agent(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes = importlib.import_module('rl_recipes')
  old,new = (TrainConfig.from_task('Mjlab-Edge-Grasp-Franka') for _ in range(2))
  recipes.apply_recipe(old,'edge_v2')
  recipes.apply_recipe(new,'edge_v3')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  params = new.env.rewards['reach_object'].params
  assert params.pop('side_wrist') and params.pop('contact_geometry')
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_ballistic_prediction_checks_rim_crossing_and_downward_branch(monkeypatch):
  reward = module(monkeypatch)
  position = torch.tensor([[.5,0.,.4],[.5,0.,.05],[.5,0.,.4]])
  velocity = torch.tensor([[1.,0.,0.],[1.,0.,0.],[1.,0.,-1.]])
  height = torch.full((3,),.14)
  landing,valid = reward.ballistic_endpoint(position,velocity,height)
  assert valid.tolist()==[True,False,True]
  expected = .5+(2*(.4-.14)/9.81)**.5
  torch.testing.assert_close(landing[0,0],torch.tensor(expected))
  assert landing[2,0]<landing[0,0]
  assert torch.isfinite(landing).all()


def test_pivot_ramp_clearance_and_opening_discriminate_flat_top_press(monkeypatch):
  import math
  reward=module(monkeypatch)
  angle=math.radians(70)
  closing=torch.tensor([[math.sin(angle),0.,math.cos(angle)]]).expand(3,-1)
  approach=torch.tensor([[math.cos(angle),0.,-math.sin(angle)]]).expand(3,-1)
  axes=[torch.cross(closing,approach,dim=-1),closing,approach]
  gap=torch.tensor([.08,.03,.08]);floor=torch.tensor([0.,0.,2.])
  corner=torch.tensor([[.35,0.,.005],[.35,0.,.005],[.35,0.,2.005]])
  target=reward.pivot_ramp_target(corner,gap,closing,approach,floor)
  # Independently enumerate both pad boxes using their XML offsets/half-sizes.
  signs=torch.cartesian_prod(*[torch.tensor([-1.,1.])]*3)
  for sign in [-1,1]:
    center=target+sign*(gap/2+.0076)[:,None]*closing+.0037*approach
    corners=center[:,None]+signs[None,:,0,None]*.00875*axes[0][:,None]+signs[None,:,1,None]*.0076*closing[:,None]+signs[None,:,2,None]*.0082*approach[:,None]
    assert (corners[:,:,2]>=floor[:,None]+.001-1e-6).all()
  score=reward.pivot_approach_score(torch.zeros(3),axes,closing,approach,gap,torch.zeros(3))
  assert score[0]>score[1] and score.max()<=1.5
  torch.testing.assert_close(score[0],score[2])
  reversed_axes=[-axes[0],axes[1],-axes[2]]
  wrong=reward.pivot_approach_score(torch.zeros(3),reversed_axes,closing,approach,gap,torch.zeros(3))
  assert (wrong<score).all()


def test_pivot_ramp_recipe_preserves_native_benchmark_and_agent(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Pivot-Lift-Franka') for _ in range(2))
  recipes.apply_recipe(old,'pivot_v2');recipes.apply_recipe(new,'pivot_v3')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  params=new.env.rewards['reach_object'].params
  assert params.pop('ramp_geometry') and params.pop('contact_geometry')
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_strike_precision_favors_correct_speed_and_settled_endpoint(monkeypatch):
  reward=module(monkeypatch)
  # At a half-metre target distance, the gradient must increase an undershoot
  # speed and decrease an overshoot speed under the physical sliding model.
  speeds=torch.tensor([.5,.9],requires_grad=True)
  velocities=torch.stack([speeds,torch.zeros_like(speeds)],dim=-1)
  predicted=reward.sliding_endpoint(torch.zeros(2,2),velocities,.04)
  error=torch.linalg.vector_norm(predicted-torch.tensor([.5,0.]),dim=-1)
  score=reward.strike_endpoint_precision(error,torch.full((2,),.5),speeds)
  score.sum().backward()
  assert speeds.grad[0]>0 and speeds.grad[1]<0
  settled=reward.strike_endpoint_precision(torch.zeros(2),torch.zeros(2),torch.tensor([0.,1.]))
  assert settled[0]>settled[1] and torch.isfinite(settled).all()


def test_strike_precision_recipe_preserves_native_benchmark(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Strike-Slide-Franka') for _ in range(2))
  recipes.apply_recipe(old,'strike_v2');recipes.apply_recipe(new,'strike_v3')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  assert new.env.rewards['reach_object'].params.pop('endpoint_precision_weight')==4.
  assert repr(old.env.rewards)==repr(new.env.rewards)


def test_strike_endpoint_recipe_keeps_success_above_all_shaping(monkeypatch):
  from dataclasses import asdict
  from mjlab.scripts.train import TrainConfig
  module(monkeypatch)
  recipes=importlib.import_module('rl_recipes')
  old,new=(TrainConfig.from_task('Mjlab-Strike-Slide-Franka') for _ in range(2))
  recipes.apply_recipe(old,'strike_v3');recipes.apply_recipe(new,'strike_v4')
  for field in ('observations','actions','commands','terminations','events','scene','sim','episode_length_s'):
    assert repr(getattr(old.env,field))==repr(getattr(new.env,field))
  assert asdict(old.agent)==asdict(new.agent)
  params=new.env.rewards['reach_object'].params
  # Both endpoint exponentials are <=1, orientation factors <=1, wedge <=1.5.
  max_uncompleted=(1+params['precision_weight'])*1.5+4+3+2*params['endpoint_precision_weight']
  assert max_uncompleted < params['native_weight']
  assert params.pop('native_weight')==40.
  assert params['endpoint_precision_weight']==8.
  params['endpoint_precision_weight']=4.
  assert repr(old.env.rewards)==repr(new.env.rewards)
