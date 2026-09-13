"""Bounded means must survive optimization and checkpoint reload unchanged."""
import importlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import torch
import copy
import pytest
from tensordict import TensorDict
from mjlab.scripts.train import TrainConfig


def test_separate_clipping_protects_actor_from_critic_scale_and_restores_patch(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  module=importlib.import_module('teacher_runner')
  class Policy(torch.nn.Module):
    def __init__(self):
      super().__init__()
      self.actor=torch.nn.Linear(1,1,bias=False)
      self.critic=torch.nn.Linear(1,1,bias=False)
      self.log_std=torch.nn.Parameter(torch.zeros(1))
  policy=Policy()
  original=torch.nn.utils.clip_grad_norm_
  def update(critic_gradient):
    policy.actor.weight.grad=torch.full_like(policy.actor.weight,3.)
    policy.log_std.grad=torch.full_like(policy.log_std,4.)
    policy.critic.weight.grad=torch.full_like(policy.critic.weight,critic_gradient)
    return torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
  for magnitude in (10.,1e6):
    combined=module.separately_clipped_update(policy,lambda:update(magnitude))
    assert float(combined)==pytest.approx((25.+magnitude**2)**.5)
    torch.testing.assert_close(policy.actor.weight.grad,torch.tensor([[.3]]))
    torch.testing.assert_close(policy.log_std.grad,torch.tensor([.4]))
    torch.testing.assert_close(policy.critic.weight.grad,torch.tensor([[.5]]))
    assert torch.nn.utils.clip_grad_norm_ is original
  # Default clipping still applies the library's shared bound after the context.
  update(1e6)
  assert float(policy.actor.weight.grad)<2e-6
  def fail():
    raise RuntimeError('synthetic update failure')
  with pytest.raises(RuntimeError,match='synthetic'):
    module.separately_clipped_update(policy,fail)
  assert torch.nn.utils.clip_grad_norm_ is original


@pytest.mark.parametrize('override,expected', [(None,1e-4),(5e-5,5e-5)])
def test_resume_learning_rate_survives_adam_restore(monkeypatch,override,expected):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  module=importlib.import_module('teacher_runner')
  model=torch.nn.Linear(2,1)
  original=torch.optim.Adam(model.parameters(),lr=1e-4)
  model(torch.ones(1,2)).sum().backward();original.step()
  saved=copy.deepcopy(original.state_dict())
  restored=torch.optim.Adam(model.parameters(),lr=3e-4)
  def parent_load(self,*args,**kwargs):
    self.alg.optimizer.load_state_dict(saved)
    return {'restored':True}
  monkeypatch.setattr(module.OnPolicyRunner,'load',parent_load)
  runner=module.TeacherRunner.__new__(module.TeacherRunner)
  runner.alg=SimpleNamespace(optimizer=restored,learning_rate=3e-4)
  runner.resume_gripper_std=runner.resume_gripper_mean=None
  runner.resume_noise_scale=None
  runner.resume_arm_std=None
  runner.learning_rate_override=override
  assert runner.load('checkpoint')['restored']
  assert restored.param_groups[0]['lr']==runner.alg.learning_rate==expected
  # Changing step size must preserve accumulated Adam moments and step counts.
  for key,state in saved['state'].items():
    for name,value in state.items():
      torch.testing.assert_close(restored.state_dict()['state'][key][name],value)


def test_bounded_policy_initialization_and_roundtrip(monkeypatch):
  stage = Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  recipes = importlib.import_module('rl_recipes')
  module = importlib.import_module('bounded_policy')
  cfg = TrainConfig.from_task('Mjlab-Cage-Drag-Franka')
  recipes.apply_recipe(cfg,'cage_v1')
  obs = TensorDict({'policy':torch.randn(64,60),'critic':torch.randn(64,60)},batch_size=[64])
  values = asdict(cfg.agent.policy)
  values.pop('class_name')
  policy = module.BoundedActorCritic(obs, cfg.agent.obs_groups, 8, **values)
  mean = policy.act_inference(obs)
  assert mean.shape == (64,8) and mean.abs().max() <= 1
  assert mean[:,-1].min() > .98
  action = cfg.env.actions['robot_joint_pos']
  init = cfg.env.scene.entities['robot'].init_state.joint_pos
  for i in range(7):
    name = f'joint{i+1}'
    mapped = cfg.agent.policy.initial_mean[i]*action.scale[name]+action.offset[name]
    assert abs(mapped-init[name]) < 1e-6
  sampled = policy.act(obs)
  loss = -policy.get_actions_log_prob(sampled.detach()).mean() + policy.evaluate(obs).square().mean()
  optimizer = torch.optim.Adam(policy.parameters(),lr=1e-4)
  optimizer.zero_grad();loss.backward();optimizer.step()
  policy.eval()
  clone = module.BoundedActorCritic(obs,cfg.agent.obs_groups,8,**values)
  clone.load_state_dict(policy.state_dict());clone.eval()
  torch.testing.assert_close(policy.act_inference(obs),clone.act_inference(obs))
  assert torch.isfinite(policy.log_std.exp()).all() and (policy.log_std.exp()>0).all()
  # Even extreme normalized observations cannot create an unbounded mean.
  huge = TensorDict({'policy':torch.randn(64,60)*1e6,'critic':torch.randn(64,60)},batch_size=[64])
  assert policy.act_inference(huge).abs().max() <= 1

  # A resume intervention must preserve deterministic behavior, normalization,
  # and all learned parameters and optimizer state outside gripper exploration.
  runner_module = importlib.import_module('teacher_runner')
  before = copy.deepcopy(policy.state_dict())
  moments = {p:copy.deepcopy(v) for p,v in optimizer.state.items()}
  output_before = policy.act_inference(obs).detach().clone()
  runner_module.reset_gripper_exploration(policy, optimizer, .2)
  torch.testing.assert_close(policy.act_inference(obs), output_before, rtol=0, atol=0)
  for name, value in policy.state_dict().items():
    if name == 'log_std':
      torch.testing.assert_close(value[:-1], before[name][:-1], rtol=0, atol=0)
      assert value[-1].exp().item() == pytest.approx(.2)
    else:
      torch.testing.assert_close(value, before[name], rtol=0, atol=0)
  for parameter, old_state in moments.items():
    for key, old in old_state.items():
      new = optimizer.state[parameter][key]
      if parameter is policy.log_std and key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        assert new[-1] == 0
        torch.testing.assert_close(new[:-1], old[:-1], rtol=0, atol=0)
      else:
        torch.testing.assert_close(new, old, rtol=0, atol=0)
  after = copy.deepcopy(policy.state_dict())
  for invalid in (0., -1., float('nan'), float('inf')):
    with pytest.raises(ValueError):
      runner_module.reset_gripper_exploration(policy, optimizer, invalid)
  for name, value in policy.state_dict().items():
    torch.testing.assert_close(value, after[name], rtol=0, atol=0)

  before = copy.deepcopy(policy.state_dict())
  moments = {p:copy.deepcopy(v) for p,v in optimizer.state.items()}
  output_before = policy.act_inference(obs).detach().clone()
  runner_module.reset_gripper_output(policy,optimizer,.5)
  output_after = policy.act_inference(obs)
  torch.testing.assert_close(output_after[:,:7],output_before[:,:7],rtol=0,atol=0)
  torch.testing.assert_close(output_after[:,7],torch.full((64,),.5))
  for name,value in policy.state_dict().items():
    if name in ('actor.6.weight','actor.6.bias'):
      torch.testing.assert_close(value[:-1],before[name][:-1],rtol=0,atol=0)
    else:
      torch.testing.assert_close(value,before[name],rtol=0,atol=0)
  head = policy.actor[-2]
  for parameter,old_state in moments.items():
    for key,old in old_state.items():
      new = optimizer.state[parameter][key]
      if (parameter is head.weight or parameter is head.bias) and key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        assert (new[-1]==0).all()
        torch.testing.assert_close(new[:-1],old[:-1],rtol=0,atol=0)
      else:
        torch.testing.assert_close(new,old,rtol=0,atol=0)
  for invalid in (-1.,1.,float('nan'),float('inf')):
    with pytest.raises(ValueError):
      runner_module.reset_gripper_output(policy,optimizer,invalid)

  before = copy.deepcopy(policy.state_dict())
  moments = {p:copy.deepcopy(v) for p,v in optimizer.state.items()}
  output_before = policy.act_inference(obs).detach().clone()
  runner_module.scale_policy_exploration(policy,optimizer,.5)
  torch.testing.assert_close(policy.act_inference(obs),output_before,rtol=0,atol=0)
  for name,value in policy.state_dict().items():
    if name == 'log_std':
      torch.testing.assert_close(value.exp(),.5*before[name].exp())
    else:
      torch.testing.assert_close(value,before[name],rtol=0,atol=0)
  for parameter,old_state in moments.items():
    for key,old in old_state.items():
      new=optimizer.state[parameter][key]
      if parameter is policy.log_std and key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        assert (new==0).all()
      else:
        torch.testing.assert_close(new,old,rtol=0,atol=0)
  after=copy.deepcopy(policy.state_dict())
  for invalid in (0.,-1.,float('nan'),float('inf')):
    with pytest.raises(ValueError):
      runner_module.scale_policy_exploration(policy,optimizer,invalid)
  for name,value in policy.state_dict().items():
    torch.testing.assert_close(value,after[name],rtol=0,atol=0)


def test_arm_exploration_reset_preserves_means_gripper_and_other_adam_state(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  runner=importlib.import_module('teacher_runner')
  bounded=importlib.import_module('bounded_policy')
  obs=TensorDict({'policy':torch.randn(32,60),'critic':torch.randn(32,60)},batch_size=[32])
  policy=bounded.BoundedActorCritic(obs,{'policy':['policy'],'critic':['critic']},8,
      actor_obs_normalization=True,critic_obs_normalization=True,noise_std_type='log')
  optimizer=torch.optim.Adam(policy.parameters(),lr=1e-4,amsgrad=True)
  loss=policy.act_inference(obs).square().mean()+policy.evaluate(obs).square().mean()+policy.log_std.square().sum()+policy.log_std.sum()
  loss.backward();optimizer.step()
  state=copy.deepcopy(policy.state_dict())
  moments={p:copy.deepcopy(v) for p,v in optimizer.state.items()}
  mean=policy.act_inference(obs).detach().clone()
  runner.reset_arm_exploration(policy,optimizer,.03)
  torch.testing.assert_close(policy.act_inference(obs),mean,rtol=0,atol=0)
  for name,value in policy.state_dict().items():
    if name=='log_std':
      torch.testing.assert_close(value[:7].exp(),torch.full((7,),.03))
      torch.testing.assert_close(value[7:],state[name][7:],rtol=0,atol=0)
    else:torch.testing.assert_close(value,state[name],rtol=0,atol=0)
  for parameter,previous in moments.items():
    for name,value in previous.items():
      current=optimizer.state[parameter][name]
      if parameter is policy.log_std and name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        assert (current[:7]==0).all()
        torch.testing.assert_close(current[7:],value[7:],rtol=0,atol=0)
      else:torch.testing.assert_close(current,value,rtol=0,atol=0)
  state=copy.deepcopy(policy.state_dict())
  for invalid in (0.,-1.,float('nan'),float('inf')):
    with pytest.raises(ValueError):runner.reset_arm_exploration(policy,optimizer,invalid)
  for name,value in policy.state_dict().items():torch.testing.assert_close(value,state[name],rtol=0,atol=0)


def test_distribution_drift_is_read_only_and_matches_gaussian_kl(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage))
  module=importlib.import_module('teacher_runner')
  bounded=importlib.import_module('bounded_policy')
  obs=TensorDict({'policy':torch.randn(32,60),'critic':torch.randn(32,60)},batch_size=[32])
  policy=bounded.BoundedActorCritic(obs,{'policy':['policy'],'critic':['critic']},8,
      actor_obs_normalization=True,critic_obs_normalization=True,noise_std_type='log')
  old_mean=policy.act_inference(obs).detach().clone();old_std=policy.log_std.exp().detach().expand_as(old_mean).clone()
  assert module.distribution_drift(policy,obs,old_mean,old_std)['kl_mean']==pytest.approx(0,abs=1e-6)
  with torch.no_grad():policy.actor[-2].bias[0]+=.1
  state=copy.deepcopy(policy.state_dict());rng=torch.random.get_rng_state().clone()
  actual=module.distribution_drift(policy,obs,old_mean,old_std)
  expected=torch.distributions.kl_divergence(torch.distributions.Normal(old_mean,old_std),
      torch.distributions.Normal(policy.act_inference(obs),policy.log_std.exp())).sum(-1)
  assert actual['kl_mean']==pytest.approx(float(expected.detach().mean()),abs=1e-6)
  assert actual['kl_max']==pytest.approx(float(expected.detach().max()),abs=1e-6)
  assert actual['kl_mean']>0
  assert torch.equal(rng,torch.random.get_rng_state())
  for name,value in policy.state_dict().items():torch.testing.assert_close(value,state[name],rtol=0,atol=0)


def test_kl_guard_rolls_back_rejected_adam_updates_and_rng(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));module=importlib.import_module('teacher_runner')
  policy=torch.nn.Linear(1,1,bias=False)
  with torch.inference_mode():
    policy.register_buffer('rollout_std',torch.ones(1))
  with torch.no_grad():policy.weight.fill_(0.)
  optimizer=torch.optim.Adam(policy.parameters(),lr=.2)
  algorithm=SimpleNamespace(policy=policy,optimizer=optimizer,storage=SimpleNamespace(step=24),learning_rate=.2,schedule='fixed')
  draws=[]
  def update():
    draws.append(torch.rand(1).item())
    optimizer.zero_grad();(policy(torch.ones(1,1))-1).square().mean().backward();optimizer.step()
    algorithm.storage.step=0
    return {'value':1.}
  measure=lambda:{'kl_mean':float(policy.weight.detach().square().sum())}
  rng=torch.random.get_rng_state().clone()
  loss,diagnostics=module.guarded_update(algorithm,update,measure,.003)
  assert diagnostics['Diagnostics/kl_guard_backtracks']==2
  assert len(set(draws))==1
  assert optimizer.state[policy.weight]['step']==1
  assert algorithm.storage.step==0 and algorithm.learning_rate==pytest.approx(.05)
  torch.testing.assert_close(policy.weight,torch.full_like(policy.weight,.05))
  accepted_rng=torch.random.get_rng_state().clone()
  torch.random.set_rng_state(rng);torch.rand(1)
  assert torch.equal(accepted_rng,torch.random.get_rng_state())
  # Exhaustion must restore accumulated Adam state and the unconsumed rollout.
  algorithm.storage.step=24;before=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(optimizer.state_dict());rng=torch.random.get_rng_state().clone()
  with pytest.raises(RuntimeError,match='exhausted'):
    module.guarded_update(algorithm,update,lambda:{'kl_mean':1.},.03,max_attempts=2)
  assert algorithm.storage.step==24 and algorithm.learning_rate==pytest.approx(.05)
  assert torch.equal(rng,torch.random.get_rng_state())
  for k,v in before.items():torch.testing.assert_close(policy.state_dict()[k],v,rtol=0,atol=0)
  for k,v in adam['state'].items():
    for n,t in v.items():torch.testing.assert_close(optimizer.state_dict()['state'][k][n],t,rtol=0,atol=0)


def test_kl_guard_ceiling_retries_larger_step_without_losing_rollback(monkeypatch):
  stage=Path(__file__).resolve().parents[1]/'src/mjlab/continual_distill/docs/cl_v4_rl'
  monkeypatch.syspath_prepend(str(stage));module=importlib.import_module('teacher_runner')
  policy=torch.nn.Linear(1,1,bias=False)
  with torch.no_grad():policy.weight.zero_()
  optimizer=torch.optim.Adam(policy.parameters(),lr=.2)
  algorithm=SimpleNamespace(policy=policy,optimizer=optimizer,storage=SimpleNamespace(step=24),learning_rate=.2,schedule='fixed')
  def update():
    optimizer.zero_grad();(policy(torch.ones(1,1))-1).square().mean().backward();optimizer.step()
    algorithm.storage.step=0
    return {'value':1.}
  anchor=policy.weight.detach().clone()
  measure=lambda:{'kl_mean':float((policy.weight.detach()-anchor).square().sum())}
  _,first=module.guarded_update(algorithm,update,measure,.003,learning_rate_ceiling=.2)
  assert first['Diagnostics/kl_guard_backtracks']==2
  assert algorithm.learning_rate==pytest.approx(.05)
  # A fresh rollout that admits a larger step starts at the ceiling again.
  anchor=policy.weight.detach().clone();algorithm.storage.step=24
  _,second=module.guarded_update(algorithm,update,measure,.05,learning_rate_ceiling=.2)
  assert second['Diagnostics/kl_guard_backtracks']==0
  assert second['Diagnostics/kl_guard_accepted_kl']<=.05
  assert algorithm.learning_rate==pytest.approx(.2)
  assert optimizer.state[policy.weight]['step']==2
  # A failed attempt with a different ceiling restores the previous accepted LR,
  # parameters, optimizer moments and rollout, rather than retaining the ceiling.
  algorithm.storage.step=24;before=copy.deepcopy(policy.state_dict());adam=copy.deepcopy(optimizer.state_dict())
  with pytest.raises(RuntimeError,match='exhausted'):
    module.guarded_update(algorithm,update,lambda:{'kl_mean':1.},.03,max_attempts=2,learning_rate_ceiling=.4)
  assert algorithm.learning_rate==pytest.approx(.2) and algorithm.storage.step==24
  assert optimizer.param_groups[0]['lr']==pytest.approx(.2)
  for k,v in before.items():torch.testing.assert_close(policy.state_dict()[k],v,rtol=0,atol=0)
  for k,v in adam['state'].items():
    for n,t in v.items():torch.testing.assert_close(optimizer.state_dict()['state'][k][n],t,rtol=0,atol=0)
  for invalid in [0.,-1.,float('nan'),float('inf')]:
    with pytest.raises(ValueError,match='ceiling'):
      module.guarded_update(algorithm,update,measure,.03,learning_rate_ceiling=invalid)
