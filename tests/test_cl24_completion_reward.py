import importlib
from pathlib import Path
from types import SimpleNamespace
import torch


def load_reward(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'))
  return importlib.import_module('completion_reward')


def test_native_completion_beats_lingering_carry_or_nearby_failure(monkeypatch):
  module = load_reward(monkeypatch)
  rng = torch.Generator().manual_seed(42)
  n = 4096
  zeros,ones = torch.zeros(n),torch.ones(n)
  random = lambda: torch.rand(n,generator=rng)
  carrying = module.completion_score(random(),random(),ones,random(),random(),zeros,zeros)
  released_failure = module.completion_score(random(),random(),zeros,random(),random(),random(),zeros)
  # Even the weakest completed state beats any noncompleted stage. The strict
  # bit comes from the native predicate, not a position-only proxy.
  completed = module.completion_score(zeros,zeros,zeros,zeros,zeros,zeros,ones)
  assert completed.min() > carrying.max()
  assert completed.min() > released_failure.max()
  # Opening at an unrelated location has no stand-alone completion reward.
  unrelated = module.completion_score(zeros,ones,zeros,zeros,zeros,zeros,zeros)
  assert torch.equal(unrelated,zeros)


def test_peg_grasp_uses_upper_body_in_each_environment(monkeypatch):
  module = load_reward(monkeypatch)
  tip = torch.tensor([[.5,.1,.01],[3.5,-1.9,.01]])
  corners = torch.stack([tip,tip+torch.tensor([0.,0.,.12])],dim=1)
  monkeypatch.setattr(module,'tracking_position',lambda _:tip)
  monkeypatch.setattr(module,'object_corners',lambda _:corners)
  command = SimpleNamespace(object=object(),cfg=SimpleNamespace(insertion=True))
  target = module.safe_grasp_target(command)
  torch.testing.assert_close(target[:,:2],tip[:,:2])
  assert ((target[:,2]>.08)&(target[:,2]<.12)).all()
  torch.testing.assert_close(tip[:,2],torch.tensor([.01,.01]))


def test_closure_path_has_no_approach_barrier(monkeypatch):
  module = load_reward(monkeypatch)
  distance = torch.linspace(.10,0.,1001)
  for aperture in [.075,.060,.040]:
    score = torch.exp(-distance/.12)*(1+module.smooth_closure_bonus(distance,torch.full_like(distance,aperture)))
    assert torch.all(score[1:]>=score[:-1])
  open_score = module.smooth_closure_bonus(torch.tensor(.01),torch.tensor(.075))
  closed_score = module.smooth_closure_bonus(torch.tensor(.01),torch.tensor(.04))
  assert closed_score > open_score


def test_enclosed_grasp_rejects_closed_pads_pressing_cube_top(monkeypatch):
  module = load_reward(monkeypatch)
  geometry = importlib.import_module('mjlab.tasks.manipulation.mdp.task_geometry')
  corners = torch.cartesian_prod(*[torch.tensor([-.023,.023])]*3).unsqueeze(0).expand(3,-1,-1)
  monkeypatch.setattr(geometry,'object_corners',lambda _:corners)
  # First pair presses the cube with a gap too narrow to enclose it. Second
  # encloses it with real bilateral contact. Third encloses it without contact.
  pads = torch.tensor([[[0.,-.0168,.025],[0.,.0168,.025]],
                       [[0.,-.031,0.],[0.,.031,0.]],
                       [[0.,-.031,0.],[0.,.031,0.]]])
  robot = SimpleNamespace(geom_names=['left_finger_pad','right_finger_pad'],
                         data=SimpleNamespace(site_pos_w=torch.zeros(3,1,3),
                           site_quat_w=torch.tensor([[[1.,0.,0.,0.]]]).expand(3,-1,-1),geom_pos_w=pads))
  command = SimpleNamespace(robot=robot,object=object(),robot_cfg=SimpleNamespace(site_ids=[0]),num_envs=3,device='cpu')
  monkeypatch.setattr(module,'grasped',lambda _:torch.tensor([True,True,False]))
  assert module.enclosed_grasp(command).tolist()==[False,True,False]


def test_capture_aperture_prefers_object_width_over_empty_closure(monkeypatch):
  module = load_reward(monkeypatch)
  width=torch.tensor([.046,.070])
  distance=torch.full((2,),.03)
  closed=module.aperture_fit_score(distance,torch.full((2,),.015),width)
  fitting=module.aperture_fit_score(distance,width+.012,width)
  assert torch.all(fitting>closed+.5)
  # At contact, zero clearance still receives nearly maximum fit credit.
  contact=module.aperture_fit_score(torch.zeros(2),width,width)
  assert torch.all(contact>.98)
  # No hard open/close switch at the old35mm boundary.
  path=torch.linspace(.10,0.,1001)
  score=module.aperture_fit_score(path,torch.full_like(path,.058),torch.full_like(path,.046))
  assert (score[1:]-score[:-1]).abs().max()<.003


def test_contact_capture_prefers_light_squeeze_over_stationary_open_cage(monkeypatch):
  module = load_reward(monkeypatch)
  width = torch.tensor([.046,.065])
  distance = torch.full((2,),.006)
  score = lambda gap: module.aperture_fit_score(distance,gap,width,squeeze=True)
  # At a correctly approached cube, the old positive clearance pays more for
  # hovering open. The new variant rewards closure into contact while retaining
  # object width, so closing empty fingers remains strongly disfavored.
  assert torch.all(score(width-.002)>score(width+.008))
  assert torch.all(score(width-.002)>score(torch.zeros(2))+.8)
  far = torch.full((2,),.20)
  assert torch.all(module.aperture_fit_score(far,width+.016,width,squeeze=True)>
                   module.aperture_fit_score(far,width-.004,width,squeeze=True))


def test_increased_peg_lift_credit_still_requires_grasp_and_prefers_completion(monkeypatch):
  module=load_reward(monkeypatch)
  one,zero=torch.tensor(1.),torch.tensor(0.)
  low=module.completion_score(one,one,one,zero,zero,zero,zero,lift_weight=4.)
  lifted=module.completion_score(one,one,one,one,zero,zero,zero,lift_weight=4.)
  assert lifted-low==4.
  assert module.completion_score(zero,zero,zero,one,zero,zero,zero,lift_weight=4.)==0
  best_carry=module.completion_score(one,one,one,one,one,zero,zero,lift_weight=4.)
  weakest_complete=module.completion_score(zero,zero,zero,zero,zero,zero,one,lift_weight=4.)
  assert best_carry<weakest_complete
