import pytest
import torch
from scipy.spatial.transform import Rotation

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.mdp.goal_orientation import nearest_axis_goal
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_apply


@pytest.mark.parametrize(
  "symmetric,axis", [(False, [0.0, 0.0, 1.0]), (True, [1.0, 0.0, 0.0])]
)
def test_axis_goal_preserves_valid_orientations_and_aligns_tilted_ones(symmetric, axis):
  q = torch.tensor(
    Rotation.random(30, random_state=12).as_quat(scalar_first=True), dtype=torch.float32
  )
  body = torch.tensor(axis)
  target = torch.tensor([[0.0, 0.0, 1.0]]).expand(30, -1)
  goal = nearest_axis_goal(q, body, target, symmetric)
  alignment = (quat_apply(goal, body.expand(30, -1)) * target).sum(-1)
  if symmetric:
    alignment = alignment.abs()
  torch.testing.assert_close(alignment, torch.ones(30), atol=1e-5, rtol=0)
  unchanged = nearest_axis_goal(goal, body, target, symmetric)
  assert torch.all((unchanged * goal).sum(-1).abs() > 1 - 1e-5)


@pytest.mark.parametrize(
  "task", ["Mjlab-Reorient-Object-Franka", "Mjlab-Topple-Block-Franka"]
)
def test_ghost_follows_xy_but_does_not_move_success_anchor(task):
  cfg = load_env_cfg(task, test=True)
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg, device="cpu")
  try:
    env.reset()
    env.sim.forward()
    cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
    anchor = cmd.target_pos.clone()
    pose = torch.cat(
      [
        cmd.object.data.root_link_pos_w.clone(),
        cmd.object.data.root_link_quat_w.clone(),
      ],
      -1,
    )
    pose[:, 0] += 0.04
    cmd.object.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    cmd._update_command()
    env.sim.forward()
    ghost = env.scene["mocap_goal"].data.root_link_pos_w
    torch.testing.assert_close(
      ghost[:, :2], cmd._object_pos()[:, :2], atol=1e-5, rtol=0
    )
    torch.testing.assert_close(cmd.target_pos, anchor)
    pose[:, 0] += 1
    cmd.object.write_root_link_pose_to_sim(pose)
    env.sim.forward()
    cmd._update_command()
    env.sim.forward()
    ghost = env.scene["mocap_goal"].data.root_link_pos_w
    assert torch.linalg.norm(ghost[:, :2] - anchor[:, :2]) < cmd.cfg.max_drift
    torch.testing.assert_close(cmd.target_pos, anchor)
  finally:
    env.close()
