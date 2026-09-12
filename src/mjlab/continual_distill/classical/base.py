"""Base class for classical (hand-coded) teacher policies.

The policy is a drop-in replacement for an NN teacher: input is the raw policy
observation batch (num_envs, obs_dim), output is joint-position actions
(num_envs, 8). Registry teachers produce bounded normalized absolute targets.
Legacy observations retain the old default-offset / 0.04 action convention.

Internally it keeps a private CPU MuJoCo model of the Franka arm for forward
kinematics and Jacobians, and runs a damped-least-squares differential IK step
per environment (same formulation as mjlab's DifferentialIKAction).
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

# Internal legacy teacher action units; converted at the shared-60 boundary.
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


# --- CL-V3 additive helpers (W1-M): hand frame from the observation ----------------
# (mount yaw: see ``mount_yaw_from_obs`` / ``rot_z`` at the bottom of this file.)

def rot_z_mat(psi: float) -> np.ndarray:
  """3x3 rotation about world +z by ``psi`` radians (matrix form of ``rot_z``)."""
  c, s = np.cos(psi), np.sin(psi)
  return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def gripper_rot(obs_i: np.ndarray) -> np.ndarray:
  """Full 3x3 rotation of the gripper site (columns = site axes in world coordinates)
  recovered from ``gripper_orientation`` (obs[28:34] = rows 1 and 2 of site_xmat);
  row 0 = row 1 x row 2 for a right-handed frame."""
  r1 = obs_i[28:31]
  r2 = obs_i[31:34]
  r0 = np.cross(r1, r2)
  return np.stack([r0, r1, r2], axis=0)


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
  legacy_layout = "standard"
  shared_observations = False

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
    jr = np.array([self.model.jnt_range[self.model.joint(n).id] for n in ARM_JOINT_NAMES])
    self._jnt_centre = 0.5 * (jr[:, 0] + jr[:, 1])
    self._jnt_half = 0.5 * (jr[:, 1] - jr[:, 0])
    self._ik_sat = np.zeros(7, dtype=bool)
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
  # CL-V3 (W1-G, additive, default-off): with cmd_ref = "command" the IK target is
  # FK(previous command) + pos_err and the solve starts from the previous command, so
  # the command moves by exactly pos_err per step and the servo's gravity sag stays a
  # CONSTANT offset. With the default "actual" the target is FK(actual) + pos_err: on a
  # descent the command re-anchors to "actual - step" every step and the sag is added
  # to every step (measured: 8 mm/step against a 6 mm/step command, into the floor).
  # Only meaningful together with cmd_lead_max > 0 (the lead clip still bounds the
  # command around the actual joints).
  cmd_ref = "actual"
  cmd_ref_axes = None  # with cmd_ref="command": axes referenced to the command (e.g. (2,)); None = all
  # CL-V3 (additive, default-off): fraction of the remaining JOINT-SPACE delta that is
  # commanded per control step, applied before the max_dq cap. 1.0 = the legacy full
  # stride. The wrist actuators (kp 300, 12 N m) saturate at ~0.04 rad of tracking
  # error, so a full 0.15-0.2 rad stride re-planned every step from the LAGGING actual
  # joints is bang-bang and hunts around the target (measured on Open-Lid: the
  # descent oscillated +-7 cm for 100+ steps). 0.3-0.5 damps that while leaving the
  # far-field speed (max_dq) untouched.
  step_gain = 1.0
  # CL-V3 (additive, default-off): clamp the IK iterate to the arm's joint limits
  # scaled by this factor about each range's centre (the env applies
  # ``soft_joint_pos_limit_factor`` 0.9 to the Franka, i.e. joint 6 <= 3.56 rad). With
  # 0 the solver may converge to a pose the env cannot execute (measured on the valve
  # probe: joint 6 asked for 3.7 rad, clamped at 3.56, the rest of the arm never
  # reconfigured and the hand stalled 5-9 cm high); with the clamp inside the
  # iteration the DLS finds the reachable branch instead.
  ik_joint_limit_factor = 0.0
  # Opt-in support for position servos. Evaluate per task: extra support changes
  # contact loads and is not beneficial for every manipulation strategy.
  gravity_compensation = False

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

  def __call__(self, obs: np.ndarray, *, active_env_ids=None) -> np.ndarray:
    """Registry adapters use shared 60D state and normalized absolute actions."""
    from .shared_interface import legacy_observation, normalized_actions

    obs = np.asarray(obs, dtype=np.float64)
    shared = self.shared_observations
    if shared:
      obs = legacy_observation(obs, self.default_qpos, self.legacy_layout)
    actions = np.zeros((obs.shape[0], 8), dtype=np.float32)
    actions[:, 7] = 1.0
    indices = range(obs.shape[0]) if active_env_ids is None else active_env_ids
    for i in indices:
      actions[i] = self._act_single(i, obs[i])
      self._phase_steps[i] += 1
    return normalized_actions(actions, self.default_qpos) if shared else actions

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
    use_cmd = (
      self.cmd_ref == "command"
      and self.cmd_lead_max > 0.0
      and not np.any(np.isnan(self._q_cmd[i]))
    )
    if use_cmd:
      q_ref = q_abs.copy()
      q_ref[:7] = np.clip(
        self._q_cmd[i], q_abs[:7] - self.cmd_lead_max, q_abs[:7] + self.cmd_lead_max
      )
    else:
      q_ref = q_abs
    ee_pos0, _ = self._fk(q_ref)
    target_pos = ee_pos0 + pos_err
    if use_cmd and self.cmd_ref_axes is not None:
      # Reference the command only on the selected axes (e.g. z: the descent
      # ratchet), keep the others anchored to the actual site (xy orbited otherwise).
      ee_act, _ = self._fk(q_abs)
      for ax in range(3):
        if ax not in self.cmd_ref_axes:
          target_pos[ax] = ee_act[ax] + pos_err[ax]
    q_goal = q_ref.copy()
    for _ in range(self.ik_iters):
      ee_pos, _ = self._fk(q_goal)
      dq = self._ik_step(q_goal, target_pos - ee_pos, target_rot)
      q_goal[:7] += dq
      if self.ik_joint_limit_factor > 0.0:
        h = self._jnt_half * self.ik_joint_limit_factor
        lo, hi = self._jnt_centre - h, self._jnt_centre + h
        # Joints pushed past a limit are frozen for the next iteration (large
        # diagonal weight) so the solve re-routes through the other joints instead
        # of pinning the iterate on the boundary.
        self._ik_sat = ((q_goal[:7] > hi) & (dq > 0)) | ((q_goal[:7] < lo) & (dq < 0))
        q_goal[:7] = np.clip(q_goal[:7], lo, hi)
      if np.linalg.norm(dq) < 1e-4:
        break
    self._ik_sat[:] = False

    # Uniform scaling by default, NOT per-joint clipping: clipping distorts
    # the joint displacement direction on large strides (the EE veers off the
    # commanded path — e.g. sagging below a drag waypoint until a grasp
    # disengages). Policies tuned under the old clipping behavior (cuboid
    # shepherding) can opt back in via step_clip_mode = "per_joint".
    if self.cmd_lead_max > 0.0 and not np.any(np.isnan(self._q_cmd[i])):
      q_start = self._q_cmd[i]
    else:
      q_start = q_abs[:7]
    step = (q_goal[:7] - q_start) * self.step_gain
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
    if self.gravity_compensation:
      self._fk(q_abs)
      self.data.qvel[:] = 0.0
      mujoco.mj_forward(self.model, self.data)
      gains = np.array([
        self.model.actuator(f"actuator{k}").gainprm[0] for k in range(1, 8)
      ])
      action[:7] += (
        self.data.qfrc_bias[self._arm_dofadr] / gains / FRANKA_ACTION_SCALE
      )
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
    jtj[np.diag_indices(7)] += w_post + self.damping**2 + 1e3 * self._ik_sat
    jtdx += w_post * (self._posture_target - q_abs[:7])

    dq = np.linalg.solve(jtj, jtdx)
    # Scale uniformly instead of clipping per joint: per-joint clipping
    # distorts the solution direction (a clipped shoulder turns a net-up
    # move into a net-down one).
    peak = float(np.max(np.abs(dq)))
    if peak > self.max_dq:
      dq = dq * (self.max_dq / peak)
    return dq


# -- CL-V3 mount-frame helpers (additive; nothing in the base class calls them) ----
#
# Under the CL-V3 init spec every mechanism mount carries a yaw band (+-0.26 rad about
# the facing-the-robot pose). The ``object_orientation`` observation term (obs[34:40] in
# the 60-D layout) is rows 1 and 2 of the entity ROOT body's rotation matrix, and for a
# mocap-mounted mechanism the root body IS the mount (``<name>_base``) -- the same
# ``root_link_quat_w`` the command terms rotate their goal markers with. For a yaw-only
# mount R = R_z(psi), so row 1 = (sin psi, cos psi, 0) and psi = atan2(obs[34], obs[35]).
# The mount never moves during an episode, so teachers latch / smooth the value.


# -- CL-V3 (W1-G2): true lowest-hand-point helper (additive; nothing in the base class
# calls it). MEASURED on the private panda model, in the gripper SITE frame (site z = the
# approach axis, +y = the finger-closing axis):
#   finger pad box : centre (0, +-(q_finger + 0.0076), +0.0037), half-extents
#                    (0.0088, 0.0076, 0.0082)  -> lowest point 0.0119 along +z_site
#   hand_capsule   : centre (0, 0, -0.070), axis along site y, half-length 0.060, r 0.040
# Every teacher that guards a descent has been clamping the SITE height, which is only the
# same thing when the hand is exactly vertical: at a 10 deg tilt the OUTER pad corner drops
# ~7 mm below where a site-height guard thinks it is, and with the jaws open (pad centre
# 0.0476 off the axis) that is the whole ground-collision margin. Guard this instead.
def lowest_hand_z(
  ee_pos: np.ndarray, ee_rot: np.ndarray, finger_qpos: float = 0.04
) -> float:
  """World z of the lowest COLLIDABLE point of the hand (pad corners + hand capsule)."""
  rz = ee_rot[2, :]  # world-z row: world_z(v_local) = ee_pos[2] + rz . v_local
  py = float(finger_qpos) + 0.0076
  hx, hy, hz = 0.0088, 0.0076, 0.0082
  # 8 corners of each pad box; the extreme is the sum of the |.| terms about the centre.
  drop = abs(rz[0]) * hx + abs(rz[1]) * hy + abs(rz[2]) * hz
  lo = np.inf
  for sy in (-py, py):
    c = float(rz[0] * 0.0 + rz[1] * sy + rz[2] * 0.0037)
    lo = min(lo, c - drop)
  cap = float(rz[2] * -0.070) - abs(rz[1]) * 0.060 - 0.040
  return float(ee_pos[2] + min(lo, cap))


def rot_z(v: np.ndarray, th: float) -> np.ndarray:
  """Rotate a 3-vector about world +z by ``th`` radians."""
  c, s = np.cos(th), np.sin(th)
  return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def mount_yaw_from_obs(obs_i: np.ndarray, start: int = 34) -> float:
  """Yaw (rad) of a yaw-only-rotated root body from its rot6d observation term.

  ``obs_i[start:start+3]`` is row 1 of the body rotation matrix = (sin psi, cos psi, 0)
  for R_z(psi). Observation noise (+-0.01) gives ~0.01 rad of jitter; callers should
  latch or EMA the value over a few steps.
  """
  r1 = obs_i[start : start + 3]
  return float(np.arctan2(r1[0], r1[1]))


def yaw_down_rot(psi: float) -> np.ndarray:
  """Full 3x3 EE target: approach axis straight DOWN, finger-closing axis (site y)
  along the mount's y axis rotated by ``psi``. Measured on the private panda model:
  at NEUTRAL the site y-axis is world -y with the hand pointing +x, so the top-down
  pose with the same wrist branch is diag(1, -1, -1); yawing the mount rotates it."""
  c, s = np.cos(psi), np.sin(psi)
  rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
  return rz @ np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
