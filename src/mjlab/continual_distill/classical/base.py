"""Base class for classical (hand-coded) teacher policies.

The policy is a drop-in replacement for an NN teacher: input is the raw policy
observation batch (num_envs, obs_dim), output is joint-position actions
(num_envs, 8) in the same convention as the RL policies
(JointPositionAction with use_default_offset=True, scale=FRANKA_ACTION_SCALE).

Internally it keeps a private CPU MuJoCo model of the Franka arm for forward
kinematics and Jacobians, and runs a damped-least-squares differential IK step
per environment (same formulation as mjlab's DifferentialIKAction).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Franka action convention (see franka_constants.py / JointPositionActionCfg).
FRANKA_ACTION_SCALE = 0.04

# Default joint positions (arm 7 + fingers 2). The env's joint_pos_rel
# observation and the JointPositionAction offset are both relative to the
# robot cfg's init pose — WHICH DIFFERS PER TASK:
#   door/drawer/button -> get_franka_robot_cfg_neutral (NEUTRAL_QPOS)
#   lift/push_cube/cuboid/disc -> get_franka_robot_cfg (HOME_QPOS)
# Using the wrong one puts FK/Jacobian at a fantasy configuration.
NEUTRAL_QPOS = np.array(
  [0.0, -1.0, 0.0, -1.57079, 0.0, 2.0, 0.785, 0.04, 0.04], dtype=np.float64
)
HOME_QPOS = np.array(
  [0.0, 0.3, 0.0, -1.57079, 0.0, 2.0, -0.7853, 0.04, 0.04], dtype=np.float64
)

ARM_JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]
EE_SITE_NAME = "gripper"


def _panda_xml_path() -> str:
  here = Path(__file__).resolve()
  src_mjlab = here.parents[2]  # .../src/mjlab
  return str(
    src_mjlab / "asset_zoo" / "robots" / "franka_emika_panda" / "xmls" / "panda.xml"
  )


def rot_error_vec(r_cur: np.ndarray, r_des: np.ndarray) -> np.ndarray:
  """Orientation error as a rotation vector (world frame), from column cross
  products: e = 0.5 * sum_i cur_col_i x des_col_i."""
  return 0.5 * (
    np.cross(r_cur[:, 0], r_des[:, 0])
    + np.cross(r_cur[:, 1], r_des[:, 1])
    + np.cross(r_cur[:, 2], r_des[:, 2])
  )


def axis_error_vec(r_cur: np.ndarray, axis_des: np.ndarray) -> np.ndarray:
  """Align only the EE z-axis (approach axis) with ``axis_des``; yaw is free.

  Constraining the full orientation makes top-down poses at the edge of the
  Franka workspace infeasible; axis-only alignment keeps a DOF free.
  """
  return np.cross(r_cur[:, 2], axis_des)


class ClassicalPolicyBase:
  """Differential-IK helper base for scripted Franka policies.

  Subclasses implement `_target_pose(env_idx, obs_i)` returning
  (target_pos (3,), target_rot (3,3) or None, gripper_action (float)).
  """

  # DLS parameters. Tuned for stable convergence when iterated through the
  # position servos: kp>2 causes limit-cycle oscillation; posture_weight must
  # stay well below the position term or the solve stalls short of the target.
  # Default joint pose of the task's robot cfg (override per policy).
  DEFAULT_QPOS = NEUTRAL_QPOS

  kp_task = 1.0
  damping = 0.2
  max_dq = 0.05
  position_weight = 1.0
  orientation_weight = 0.3
  posture_weight = 0.005
  # Inner IK iterations per control step (internal model only, ~µs each).
  ik_iters = 30
  # Cap on the position error fed to the solver per step. Large raw errors
  # (e.g. 0.7m descents from the start pose) make the arm cut curved paths
  # that can swing the wrist into the ground; capping tracks a straight line.
  max_pos_err = 0.08

  def __init__(self, num_envs: int):
    self.num_envs = num_envs
    self.model = mujoco.MjModel.from_xml_path(_panda_xml_path())
    self.data = mujoco.MjData(self.model)

    self._arm_qadr = np.array(
      [self.model.joint(n).qposadr[0] for n in ARM_JOINT_NAMES]
    )
    self._arm_dofadr = np.array(
      [self.model.joint(n).dofadr[0] for n in ARM_JOINT_NAMES]
    )
    self._site_id = self.model.site(EE_SITE_NAME).id
    self.default_qpos = self.DEFAULT_QPOS.copy()
    self._posture_target = self.default_qpos[:7].copy()

    self.reset()

  # -- interface ------------------------------------------------------------

  # Command-lead: when > 0, the joint command integrates from the previous
  # COMMAND (not the lagging actual position), up to this many rad ahead of
  # the actual joints. Sustained lead = sustained servo force — needed to
  # pull articulated objects (door/drawer) at speed. 0 disables (default).
  cmd_lead_max = 0.0
  step_clip_mode = "uniform"  # or "per_joint" (legacy behavior)

  def reset(self, env_ids=None) -> None:
    """Reset per-env state-machine state. Subclasses may extend."""
    if env_ids is None:
      self._phase = np.zeros(self.num_envs, dtype=np.int64)
      self._phase_steps = np.zeros(self.num_envs, dtype=np.int64)
      self._q_cmd = np.full((self.num_envs, 7), np.nan)
    else:
      self._phase[env_ids] = 0
      self._phase_steps[env_ids] = 0
      self._q_cmd[env_ids] = np.nan

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    """obs (num_envs, 60) -> actions (num_envs, 8)."""
    obs = np.asarray(obs, dtype=np.float64)
    actions = np.zeros((obs.shape[0], 8), dtype=np.float32)
    for i in range(obs.shape[0]):
      actions[i] = self._act_single(i, obs[i])
      self._phase_steps[i] += 1
    return actions

  # -- per-env --------------------------------------------------------------

  def _act_single(self, i: int, obs_i: np.ndarray) -> np.ndarray:
    q_abs = self.default_qpos + obs_i[0:9]  # joint_pos_rel -> absolute
    pos_err, target_rot, gripper_a = self._target_error(i, obs_i)

    # Per-axis cap (NOT norm scaling): scaling the whole vector starves small
    # components — e.g. a +6cm "hold altitude" correction vanishes next to a
    # 30cm lateral error, and the arm sags into the ground while reaching.
    pos_err = np.clip(pos_err, -self.max_pos_err, self.max_pos_err)

    # Solve IK to convergence on the internal model (decoupled from motion):
    # commanding one DLS step per control step couples solver dynamics with
    # servo dynamics and either oscillates or crawls. Here we find the joint
    # configuration for the current waypoint, then step toward it at a
    # bounded rate — a clean joint-space trajectory for the servos to track.
    ee_pos0, _ = self._fk(q_abs)
    target_pos = ee_pos0 + pos_err
    q_goal = q_abs.copy()
    for _ in range(self.ik_iters):
      ee_pos, _ = self._fk(q_goal)
      dq = self._ik_step(q_goal, target_pos - ee_pos, target_rot)
      q_goal[:7] += dq
      if np.linalg.norm(dq) < 1e-4:
        break

    # Uniform scaling by default, NOT per-joint clipping: clipping distorts
    # the joint displacement direction on large strides (the EE veers off the
    # commanded path — e.g. sagging below a drag waypoint until a grasp
    # disengages). Policies tuned under the old clipping behavior (cuboid
    # shepherding) can opt back in via step_clip_mode = "per_joint".
    if self.cmd_lead_max > 0.0 and not np.any(np.isnan(self._q_cmd[i])):
      q_start = self._q_cmd[i]
    else:
      q_start = q_abs[:7]
    step = q_goal[:7] - q_start
    if self.step_clip_mode == "per_joint":
      step = np.clip(step, -self.max_dq, self.max_dq)
    else:
      peak = float(np.max(np.abs(step)))
      if peak > self.max_dq:
        step = step * (self.max_dq / peak)
    q_des = q_start + step
    if self.cmd_lead_max > 0.0:
      q_des = np.clip(
        q_des, q_abs[:7] - self.cmd_lead_max, q_abs[:7] + self.cmd_lead_max
      )
    self._q_cmd[i] = q_des

    action = np.zeros(8, dtype=np.float32)
    action[:7] = (q_des - self.default_qpos[:7]) / FRANKA_ACTION_SCALE
    action[7] = gripper_a
    return action

  def _target_error(self, i: int, obs_i: np.ndarray):
    """Return (pos_err (3,), target_rot (3,3) or None, gripper_action).

    ``pos_err`` is desired_pos - current_gripper_pos expressed in world axes.
    Observations are world-frame but include per-env scene origin offsets, so
    subclasses must build the error from *relative* observation terms (e.g.
    ``gripper_to_object``) where the offsets cancel.
    """
    raise NotImplementedError

  # -- kinematics -----------------------------------------------------------

  def _fk(self, q_abs: np.ndarray):
    """Set qpos, run FK, return (ee_pos (3,), ee_rot (3,3))."""
    self.data.qpos[self._arm_qadr] = q_abs[:7]
    # Fingers: use observed values so FK matches env geometry closely.
    fq = self.model.joint("finger_joint1").qposadr[0]
    self.data.qpos[fq] = q_abs[7]
    fq2 = self.model.joint("finger_joint2").qposadr[0]
    self.data.qpos[fq2] = q_abs[8]
    mujoco.mj_kinematics(self.model, self.data)
    mujoco.mj_comPos(self.model, self.data)
    ee_pos = self.data.site_xpos[self._site_id].copy()
    ee_rot = self.data.site_xmat[self._site_id].reshape(3, 3).copy()
    return ee_pos, ee_rot

  def _ik_step(
    self, q_abs: np.ndarray, pos_err: np.ndarray, target_rot: np.ndarray | None
  ) -> np.ndarray:
    """One DLS IK step: returns arm joint displacement (7,)."""
    _ee_pos, ee_rot = self._fk(q_abs)

    jacp = np.zeros((3, self.model.nv))
    jacr = np.zeros((3, self.model.nv))
    mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self._site_id)
    jp = jacp[:, self._arm_dofadr]  # (3,7)
    jr = jacr[:, self._arm_dofadr]

    pos_dx = self.kp_task * pos_err
    w_pos, w_ori, w_post = (
      self.position_weight,
      self.orientation_weight,
      self.posture_weight,
    )

    jtj = w_pos * jp.T @ jp
    jtdx = w_pos * jp.T @ pos_dx
    if target_rot is not None and w_ori > 0:
      if target_rot.ndim == 1:  # (3,) axis -> align EE z-axis only, yaw free
        ori_dx = self.kp_task * axis_error_vec(ee_rot, target_rot)
      else:  # (3,3) full rotation target
        ori_dx = self.kp_task * rot_error_vec(ee_rot, target_rot)
      jtj += w_ori * jr.T @ jr
      jtdx += w_ori * jr.T @ ori_dx

    # Posture regularization + damping on the diagonal.
    jtj[np.diag_indices(7)] += w_post + self.damping**2
    jtdx += w_post * (self._posture_target - q_abs[:7])

    dq = np.linalg.solve(jtj, jtdx)
    # Scale uniformly instead of clipping per joint: per-joint clipping
    # distorts the solution direction (a clipped shoulder turns a net-up
    # move into a net-down one).
    peak = float(np.max(np.abs(dq)))
    if peak > self.max_dq:
      dq = dq * (self.max_dq / peak)
    return dq
