"""Construct explicit MuJoCo states for task-contract tests and review renders.

These are geometric oracles, not policy rollouts or success-rate measurements.
"""

import mujoco
import numpy as np
import torch
from scipy.optimize import least_squares


def native_state(env, index=0):
  model = env.sim.mj_model
  data = mujoco.MjData(model)
  for field in ("qpos", "qvel", "mocap_pos", "mocap_quat"):
    getattr(data, field)[:] = getattr(env.sim.data, field)[index].cpu().numpy()
  mujoco.mj_forward(model, data)
  return model, data


def place(env, name, positions, quaternion=(1.0, 0.0, 0.0, 0.0)):
  obj = env.scene[name]
  positions = torch.as_tensor(positions, dtype=torch.float32, device=env.device)
  if positions.ndim == 1:
    positions = positions.expand(env.num_envs, 3)
  quat = torch.as_tensor(quaternion, dtype=torch.float32, device=env.device)
  if quat.ndim == 1:
    quat = quat.expand(env.num_envs, 4)
  obj.write_root_link_pose_to_sim(torch.cat([positions, quat], -1))
  obj.write_root_link_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device))
  env.sim.forward()


def position_hand(env, positions, opening=0.04, rotation=None):
  """Solve full wrist-pose IK in the actual compiled scene, then write both fingers."""
  rotation = np.diag([1.0, -1.0, -1.0]) if rotation is None else np.asarray(rotation)
  robot = env.scene["robot"]
  poses = robot.data.joint_pos.clone()
  site = int(robot.data.indexing.site_ids[robot.site_names.index("gripper")])
  ids = np.array(
    [
      int(robot.data.indexing.joint_ids[robot.joint_names.index(f"joint{i}")])
      for i in range(1, 8)
    ]
  )
  finger_ids = [robot.joint_names.index(n) for n in ("finger_joint1", "finger_joint2")]
  for index in range(env.num_envs):
    model, data = native_state(env, index)
    qids = model.jnt_qposadr[ids]
    lower, upper = model.jnt_range[ids].T
    target = np.asarray(positions[index].cpu())

    def residual(q, data=data, qids=qids, model=model, target=target):
      data.qpos[qids] = q
      mujoco.mj_forward(model, data)
      return np.r_[
        5 * (data.site_xpos[site] - target),
        0.3 * (data.site_xmat[site].reshape(3, 3) - rotation).ravel(),
      ]

    guesses = [
      data.qpos[qids].copy(),
      np.array([0, 0.3, 0, -1.57, 0, 2, -0.785]),
      np.array([0, -0.3, 0, -2.0, 0, 2.0, 0.785]),
    ]
    best = None
    for guess in guesses:
      result = least_squares(
        residual,
        np.clip(guess, lower + 1e-5, upper - 1e-5),
        bounds=(lower, upper),
        max_nfev=250,
      )
      if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
        best = result
      if (
        np.linalg.norm(result.fun[:3]) < 0.0025
        and np.linalg.norm(result.fun[3:]) < 0.005
      ):
        break
    residual(best.x)
    error = np.linalg.norm(data.site_xpos[site] - target)
    if error > 0.003:
      raise RuntimeError(f"Wrist oracle cannot reach {target}: residual {error:.4f} m")
    for i in range(7):
      poses[index, robot.joint_names.index(f"joint{i + 1}")] = float(best.x[i])
  poses[:, finger_ids] = opening
  robot.write_joint_position_to_sim(poses)
  robot.write_joint_velocity_to_sim(torch.zeros_like(poses))
  env.sim.forward()


def cage_at_goal(env, command):
  """Two enclosed states demonstrate 10 mm of progress with open fingers."""
  goal = command.target_pos.clone()
  for offset in (-0.01, 0.0):
    pos = goal.clone()
    pos[:, 0] += offset
    hand = pos.clone()
    hand[:, 2] += 0.009
    position_hand(env, hand)
    place(env, command.cfg.asset_name, pos)
    command._update_metrics()


def held_at_goal(env, command):
  """Bilateral pad contact at the task's own goal; plate/board stand on edge."""
  name = command.cfg.asset_name
  pos = command.target_pos.clone()
  rotation = None
  if name == "board":
    # Present the thin side toward the wall. A board normal along world Y can
    # intersect the wall at an otherwise valid center goal near its face.
    quat = (2**-0.5, 0.0, 2**-0.5, 0.0)
    opening, offset = 0.0095, 0.035
    rotation = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
  elif name == "plate":
    # Thin direction along the horizontal jaw axis; grasp the upper edge.
    quat = (2**-0.5, 2**-0.5, 0.0, 0.0)
    opening = 0.0091
    offset = 0.052
  else:
    quat = (1.0, 0.0, 0.0, 0.0)
    opening, offset = 0.0225, 0.004
  hand = pos.clone()
  hand[:, 2] += offset
  position_hand(env, hand, opening=opening, rotation=rotation)
  place(env, name, pos, quat)
  command._update_metrics()


def tick_metrics(env, command):
  env.sim.forward()
  command._update_metrics()
  command._update_command()
  env.sim.forward()


def goal_oracle(env, command):
  """Construct a goal-state witness, including contact histories where required."""
  name = type(command).__name__
  if name == "CageDragCommand":
    cage_at_goal(env, command)
  elif name == "PivotLiftCommand":
    wall = command.wall.data.root_link_pos_w.clone()
    p = wall.clone()
    # Positive-y rotation: far lower corner touches the wall and the ground.
    p[:, 0] -= 0.0525 + (0.06 + 0.01) * 2**-0.5 - 0.0005
    p[:, 2] = env.scene.env_origins[:, 2] + (0.06 + 0.01) * 2**-0.5
    place(env, command.cfg.asset_name, p, (np.cos(np.pi / 8), 0, np.sin(np.pi / 8), 0))
    command._update_metrics()
    if not command.pivoted.all():
      raise RuntimeError("wall-pivot witness failed to produce a board/wall contact")
    held_at_goal(env, command)
  elif name in ("LiftingCommand", "EdgeGraspCommand"):
    held_at_goal(env, command)
  elif name == "StackingCommand":
    p = command.target_pos.clone()
    if not command.cfg.insertion:
      p[:, 2] -= 0.0005
    place(env, command.cfg.asset_name, p, command.base.data.root_link_quat_w)
  elif name == "PlaceInContainerCommand":
    p = command.target_pos.clone()
    p[:, 2] -= 0.0005
    place(env, command.cfg.asset_name, p)
  elif name == "ReorientObjectCommand":
    from mjlab.tasks.manipulation.mdp.task_geometry import object_corners

    p = command.target_pos.clone()
    if command.cfg.symmetric_axis:
      quat = (2**-0.5, 0, 2**-0.5, 0)
    else:
      quat = (1.0, 0.0, 0.0, 0.0)
    place(env, command.cfg.asset_name, p, quat)
    # Use the asset's actual support height rather than an assumed bottle/block
    # size, so the representative goal witness also rests on the ground.
    p[:, 2] += env.scene.env_origins[:, 2] - object_corners(command.object)[
      :, :, 2
    ].amin(1)
    place(env, command.cfg.asset_name, p, quat)
  elif name == "ToolPullCommand":
    # Isolate the contact/history contract at a clear reachable height. These are
    # constructed states, not a claim that a teacher executes this trajectory.
    hand = env.scene.env_origins + torch.tensor([0.4, 0, 0.24], device=env.device)
    position_hand(env, hand, opening=0.0105)
    tool = hand + torch.tensor([0.09, 0, -0.004], device=env.device)
    place(env, command.cfg.tool_asset_name, tool)
    puck = tool + torch.tensor([0.1685, 0.07, 0.0], device=env.device)
    place(env, command.cfg.asset_name, puck)
    command._update_metrics()
    if not command.tool_used.all():
      raise RuntimeError(
        "tool-contact witness failed to touch both pads/tool and tool/puck"
      )
    place(env, command.cfg.asset_name, command.target_pos.clone())
  elif name == "ReachingCommand":
    position_hand(env, command.target_pos)
  elif hasattr(command, "target_value"):
    command.asset.write_joint_position_to_sim(command.target_value[:, None])
  elif hasattr(command, "target_angle"):
    command.door.write_joint_position_to_sim(command.target_angle[:, None])
  elif hasattr(command, "target_distance"):
    asset = command.drawer if hasattr(command, "drawer") else command.button
    asset.write_joint_position_to_sim(command.target_distance[:, None])
  else:
    place(env, command.cfg.asset_name, command.target_pos.clone())
  tick_metrics(env, command)
  for index in range(env.num_envs):
    model, data = native_state(env, index)
    for contact in data.contact:
      if contact.dist < -0.003:
        names = [model.geom(g).name for g in contact.geom]
        raise RuntimeError(
          f"Goal witness interpenetrates {names}: {contact.dist:.4f} m"
        )
  return command.compute_success()
