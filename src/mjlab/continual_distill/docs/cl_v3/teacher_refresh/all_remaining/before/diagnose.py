#!/usr/bin/env python3
"""Failure-analysis dump for ANY classical teacher in ``CLASSICAL_POLICIES``.

Runs the same unrendered rollout loop as ``test_classical.py`` and records, per env
and per control step, everything a human needs to say *what physically happened* to
a failing episode: the teacher's phase, the action, the gripper aperture, the
relative object / goal vectors, the mechanism joint (if any), which termination
fired, and the latched success. It then writes a machine-readable trace and a
human-readable per-env summary sorted into failure classes.

Usage (GPU node, through the holder — see docs/cl_v3/AGENT_BRIEF.md)::

    PYTHONPATH=src .venv/bin/python -m mjlab.continual_distill.classical.diagnose \\
        --task Mjlab-Stack-Cube-Franka --num-envs 32 --num-episodes 1 \\
        --out src/mjlab/continual_distill/docs/cl_v3/diag/Mjlab-Stack-Cube-Franka \\
        --device cuda:0 [--dump-every 10] [--print-envs 2] [--seed 0]

    # rebuild summary.md / summary.json from an existing trace (CPU, no sim):
    ... diagnose --resummarize <out-dir> [--lift-thresh 0.02] [--lost-aperture 0.015]

Outputs in ``--out``:

  trace.npz     arrays indexed [env, step, ...] (env = episode * num_envs + env_idx;
                step = control step 0 .. episode_length-1). Steps AFTER an env's first
                done (termination or time-out) are masked ``valid=False`` — the env has
                auto-reset by then and its data belongs to a new episode.
  summary.json  per-env dicts (the same fields as summary.md) + histograms + meta.
  summary.md    the human-readable report. Read this first.

trace.npz columns (all float32 unless stated; NaN / -1 where a field is unavailable):

  valid            [E,T] bool   step belongs to the env's first episode (see above)
  phase            [E,T] int    ``policy._phase[i]`` right AFTER the teacher processed
                                this step's observation = the phase that produced the
                                step's action (teachers transition first, then compute
                                the target); -1 if the teacher has no ``_phase``
  phase_steps      [E,T] int    ``policy._phase_steps[i]`` at the same moment; -1 if absent
  action           [E,T,8]      action sent to the env: [:7] arm joint targets in the
                                JointPositionAction convention ((q_des - default)/0.04),
                                [7] gripper: +1 open, -1 close
  finger_qpos_obs  [E,T,2]      finger joint positions AS THE TEACHER SEES THEM:
                                obs[7:9] + DEFAULT_QPOS[7:9] (obs has ±0.01 noise)
  finger_qpos_sim  [E,T,2]      true finger joint positions from the sim (robot entity)
  aperture         [E,T]        finger_qpos_sim.sum(): 0.08 = fully open, ~0.0 = closed
                                on nothing; a 46 mm cube held between the pads reads
                                ~0.046. Falls back to the obs value if sim read fails.
  gripper_to_object[E,T,3]      the `gripper_to_object` obs term (object - gripper site,
                                world axes, relative); obs[40:43] in the 60-D layout,
                                obs[37:40] in Stack/Peg's 51-D layout — resolved by name
  object_to_goal   [E,T,3]      the `object_to_goal` obs term (goal - object, relative)
  dist_go          [E,T]        |gripper_to_object|
  dist_og          [E,T]        |object_to_goal|
  gripper_pos      [E,T,3]      `gripper_pos` obs term - env origin (env-relative, noisy)
  object_pos       [E,T,3]      primary object root position - env origin (from sim)
  object_quat      [E,T,4]      primary object root quaternion (w,x,y,z) (from sim)
  object_speed     [E,T]        |root linear velocity| of the primary object
  mech_joint       [E,T]        mechanism joint value tracked by the command term
                                (rad or m); NaN for free-object tasks
  goal_error       [E,T]        ``cmd.metrics["goal_error"]`` if the term has it
  success          [E,T] bool   ``cmd.episode_success`` (latched) AFTER the step
  done             [E,T] bool   terminated | truncated AFTER the step
  term_flags       [E,T,K] bool one column per termination term, order in
                                ``term_names`` (typically time_out,
                                ee_ground_collision, object_out_of_bounds)
  contact_found    [E,T,S] bool one column per ContactSensor in the scene (``found``
                                any-matched), order in ``contact_names``
  term_names / contact_names / phase_names   string arrays (labels)
  fingertip_friction [E]        sliding friction of the finger pads for that env
                                (domain-randomized 0.3–1.5 at startup); NaN if unreadable
  mech_target      [E]          command target for the mechanism joint; NaN if none
  spawn_z          [E]          object z at step 0 (env-relative)

Note: the reset joint noise is applied to the finger joints too, so at step 0 the
fingers can sit anywhere in [0, 0.04] each; the aperture is only meaningful once the
teacher has commanded the gripper for a few steps.

Per-env summary fields (summary.md / summary.json):

  outcome            "success" | "fail"
  end_phase          phase at the env's last valid step
  most_phase         phase with the most steps
  steps_in_phase     {phase: count}
  aperture_min/max   over the episode (m)
  min_dist_go        closest |gripper_to_object| reached (m) and the step
  min_dist_og        closest |object_to_goal| reached (m) and the step
  termination        name of the non-time-out termination that ended the episode
                     (null if it ran to time-out) and its step
  lifted             free-object tasks: object z rose > --lift-thresh (2 cm) above
                     spawn_z at some step (first step given); null for mechanisms
  lost_step          first step AFTER the lift at which the aperture had collapsed
                     below --lost-aperture (1.5 cm total) while |gripper_to_object|
                     was growing — i.e. the fingers closed on nothing while the object
                     moved away. null if never lifted / never lost.
  mech_max           mechanisms: max joint value reached vs mech_target
  grasp_task         task-level: some env's teacher opened the gripper and later closed
                     it (a grasp); push / drag teachers hold it closed and are NOT grasp
                     tasks (same value on every env)
  og_progress        non-mechanism: initial |object_to_goal| minus the closest reached
  class              coarse failure class (see _classify), in priority order:
                     terminated:<name>   a non-time-out termination ended the episode
                     near_miss           min goal_error < 1.5 x the command's success_threshold
                     never_reached       closest |gripper_to_object| > 6 cm
                     never_lifted / grasp_lost / lifted_not_at_goal   (grasp tasks)
                     mechanism_not_moved (< 10 % of target) / mechanism_short (< 90 %)
                     object_not_moved (< 2 cm toward goal) / object_moved_short
                     stalled_phase_<n>   fallback

Phase names: ``policy.PHASE_NAMES`` (dict or list) if the teacher defines one, else
module-level ``P_<NAME> = <int>`` constants of the teacher's module, else a built-in
table for teachers whose docstring numbers the phases, else "phase <n>". The numbers
are always the teacher's own ``_phase`` values, so read the teacher when in doubt.

Absolute SR printed here is for orientation only — the measurement protocol is
``test_classical.py`` at n = 128 on the frozen init spec (PLAN.md §5).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mjlab.continual_distill.classical import CLASSICAL_POLICIES
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Default observation slices (60-D manipulation layout; see lift_object_env_cfg.py).
# NOT universal: Stack-Cube / Peg-Insertion use a 51-D two-object layout (base_pos in,
# object_orientation / goal_orientation_diff out), so the slices are resolved by term
# name from the observation manager at runtime (`_obs_slices`) and these are only the
# fallback.
_DEFAULT_OBS_SLICES = {
  "robot_joint_pos": slice(0, 9),
  "gripper_pos": slice(25, 28),
  "gripper_to_object": slice(40, 43),
  "object_to_goal": slice(43, 46),
}


def _obs_slices(env: ManagerBasedRlEnv) -> dict[str, slice]:
  """Term name -> slice of the flat 'policy' observation, from the manager."""
  out = dict(_DEFAULT_OBS_SLICES)
  try:
    om = env.observation_manager
    names = om.active_terms["policy"]
    dims = om.group_obs_term_dim["policy"]
    off = 0
    found = {}
    for name, dim in zip(names, dims):
      n = int(np.prod(dim))
      found[name] = slice(off, off + n)
      off += n
    for k in out:
      if k in found:
        out[k] = found[k]
  except Exception:  # noqa: BLE001
    pass
  return out

# Built-in phase names for teachers that number their phases in the docstring but
# define no P_* constants. Keyed by a substring of the policy class name.
# Entries marked "read off the branch comments" are the tool author's reading of the
# teacher's `if self._phase[i] == n:` blocks — verify against the teacher before quoting.
_PHASE_TABLE: dict[str, dict[int, str]] = {
  "Lift": {0: "HOVER", 1: "DESCEND", 2: "CLOSE", 3: "LIFT"},
  "PushButton": {0: "HOVER", 1: "PRESS"},
  "StrikeSlide": {0: "HOVER", 1: "SEAT", 2: "WINDUP", 3: "STRIKE", 4: "RETREAT"},
  # read off the branch comments:
  "PushCuboid": {0: "ALIGN", 1: "DESCEND", 2: "PUSH", 3: "DONE"},
  "DragPull": {0: "ALIGN", 1: "DESCEND", 2: "DRAG", 3: "DONE"},
  "CageDrag": {0: "ALIGN", 1: "DESCEND", 2: "DRAG", 3: "DONE"},
  "ToolPull": {0: "ALIGN", 1: "DESCEND", 2: "PULL", 3: "DONE"},
  "PushDisc": {0: "ALIGN", 1: "DESCEND", 2: "PUSH", 3: "DONE"},
  "RotateValve": {0: "ALIGN", 1: "SEAT", 2: "ARC_FOLLOW", 3: "PINCH"},
  "ToppleBlock": {0: "HOVER", 1: "SEAT", 2: "PUNCH", 3: "RETREAT"},
  "FlipSwitch": {0: "ALIGN", 1: "SEAT", 2: "STROKE"},
  "SlideWindow": {0: "ALIGN", 1: "SEAT", 2: "PUSH"},
  "OpenDrawer": {0: "HOVER", 1: "DESCEND", 2: "PULL"},
}


def _git_head() -> str:
  try:
    return subprocess.run(
      ["git", "rev-parse", "--short", "HEAD"],
      cwd=_REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
  except Exception:  # noqa: BLE001
    return "unknown"


def _to_np(x) -> np.ndarray | None:
  try:
    if isinstance(x, torch.Tensor):
      return x.detach().cpu().numpy()
    return np.asarray(x)
  except Exception:  # noqa: BLE001
    return None


def _phase_names(policy) -> dict[int, str]:
  """Resolve phase number -> name for this teacher (see module docstring)."""
  names = getattr(policy, "PHASE_NAMES", None)
  if isinstance(names, dict):
    return {int(k): str(v) for k, v in names.items()}
  if isinstance(names, (list, tuple)):
    return {i: str(v) for i, v in enumerate(names)}
  out: dict[int, str] = {}
  for klass in type(policy).__mro__:
    mod = sys.modules.get(klass.__module__)
    if mod is None:
      continue
    for k, v in vars(mod).items():
      if k.startswith("P_") and isinstance(v, (int, np.integer)) and not isinstance(v, bool):
        out.setdefault(int(v), k[2:])
    if out:
      return out
  cname = type(policy).__name__
  for key, table in _PHASE_TABLE.items():
    if key in cname:
      return dict(table)
  return {}


class _SceneProbe:
  """Best-effort accessors for object / mechanism / robot state. Every read is
  wrapped: a teacher or task lacking a field yields NaN, never an exception."""

  def __init__(self, env: ManagerBasedRlEnv, num_envs: int):
    self.env = env
    self.n = num_envs
    self.cmd = env.command_manager.get_term(next(iter(env.command_manager.active_terms)))
    self.obs = _obs_slices(env)
    self.origins = _to_np(env.scene.env_origins)
    if self.origins is None:
      self.origins = np.zeros((num_envs, 3))
    # Primary object entity: the command term's `object`, else its `asset`, else the
    # first non-robot / non-mocap entity in the scene.
    self.obj = None
    self.obj_name = None
    for attr in ("object", "asset", "door", "drawer", "button"):
      ent = getattr(self.cmd, attr, None)
      if ent is not None and hasattr(ent, "data"):
        self.obj, self.obj_name = ent, attr
        break
    if self.obj is None:
      for name, ent in env.scene.entities.items():
        if name not in ("robot",) and not name.startswith("mocap"):
          self.obj, self.obj_name = ent, name
          break
    # Mechanism joint: (entity, joint index) from whichever attribute the term uses.
    self.mech = None
    for ent_attr, idx_attr in (
      ("asset", "joint_idx"), ("door", "hinge_idx"), ("drawer", "slide_idx"),
      ("button", "slide_idx"), ("object", "joint_idx"),
    ):
      ent = getattr(self.cmd, ent_attr, None)
      idx = getattr(self.cmd, idx_attr, None)
      if ent is not None and idx is not None:
        self.mech = (ent, int(idx))
        break
    self.success_threshold = None
    try:
      v = getattr(self.cmd.cfg, "success_threshold", None)
      self.success_threshold = float(v) if v is not None else None
    except Exception:  # noqa: BLE001
      pass
    self.mech_name = None
    if self.mech is not None:
      try:
        self.mech_name = self.mech[0].joint_names[self.mech[1]]
      except Exception:  # noqa: BLE001
        pass
    # Robot fingers.
    self.robot = env.scene["robot"] if "robot" in env.scene.entities else None
    self.finger_idx = None
    if self.robot is not None:
      try:
        jn = list(self.robot.joint_names)
        self.finger_idx = [jn.index("finger_joint1"), jn.index("finger_joint2")]
      except Exception:  # noqa: BLE001
        self.finger_idx = None
    self.contact_names = []
    self.contact_sensors = []
    try:
      for name, sensor in env.scene.sensors.items():
        if type(sensor).__name__ == "ContactSensor":
          self.contact_names.append(name)
          self.contact_sensors.append(sensor)
    except Exception:  # noqa: BLE001
      pass

  def _read_mech_target(self) -> np.ndarray:
    """Per-env target of the mechanism joint (valid only AFTER a reset)."""
    for attr in ("target_value", "target_angle", "target_distance"):
      v = getattr(self.cmd, attr, None)
      if isinstance(v, torch.Tensor) and v.numel() == self.n:
        return _to_np(v).reshape(-1).astype(np.float64)
    return np.full(self.n, np.nan)

  def object_state(self):
    pos = np.full((self.n, 3), np.nan)
    quat = np.full((self.n, 4), np.nan)
    speed = np.full(self.n, np.nan)
    if self.obj is None:
      return pos, quat, speed
    try:
      p = _to_np(self.obj.data.root_link_pos_w)
      pos = p[: self.n] - self.origins[: self.n]
    except Exception:  # noqa: BLE001
      pass
    try:
      quat = _to_np(self.obj.data.root_link_quat_w)[: self.n]
    except Exception:  # noqa: BLE001
      pass
    try:
      v = _to_np(self.obj.data.root_link_lin_vel_w)[: self.n]
      speed = np.linalg.norm(v, axis=-1)
    except Exception:  # noqa: BLE001
      pass
    return pos, quat, speed

  def mech_joint(self) -> np.ndarray:
    if self.mech is None:
      return np.full(self.n, np.nan)
    try:
      return _to_np(self.mech[0].data.joint_pos)[: self.n, self.mech[1]].astype(np.float64)
    except Exception:  # noqa: BLE001
      return np.full(self.n, np.nan)

  def finger_qpos(self) -> np.ndarray:
    if self.robot is None or self.finger_idx is None:
      return np.full((self.n, 2), np.nan)
    try:
      return _to_np(self.robot.data.joint_pos)[: self.n][:, self.finger_idx].astype(np.float64)
    except Exception:  # noqa: BLE001
      return np.full((self.n, 2), np.nan)

  def goal_error(self) -> np.ndarray:
    try:
      v = self.cmd.metrics.get("goal_error")
      if v is not None:
        return _to_np(v).reshape(-1)[: self.n].astype(np.float64)
    except Exception:  # noqa: BLE001
      pass
    return np.full(self.n, np.nan)

  def success(self) -> np.ndarray:
    return _to_np(self.cmd.episode_success).reshape(-1)[: self.n] > 0.5

  def contacts(self) -> np.ndarray:
    out = np.zeros((self.n, len(self.contact_sensors)), dtype=bool)
    for k, s in enumerate(self.contact_sensors):
      try:
        found = s.data.found
        if found is not None:
          out[:, k] = _to_np(found.any(dim=-1))[: self.n]
      except Exception:  # noqa: BLE001
        pass
    return out

  def fingertip_friction(self) -> np.ndarray:
    try:
      ids, _names = self.robot.find_geoms(r".*finger_pad.*")
      gid = _to_np(self.robot.indexing.geom_ids)[np.asarray(ids)]
      fr = _to_np(self.env.sim.model.geom_friction)
      if fr.ndim == 3:  # [world, geom, 3]
        return fr[: self.n, gid[0], 0].astype(np.float64)
      return np.full(self.n, float(fr[gid[0], 0]))
    except Exception:  # noqa: BLE001
      return np.full(self.n, np.nan)


def run_episode(env, policy, probe: _SceneProbe, num_envs: int, device: str,
                dump_every: int, print_envs: int, ep: int) -> dict:
  """One episode for all envs. Returns dict of per-step arrays [env, step, ...]."""
  T = int(env.max_episode_length)
  E = num_envs
  tm = env.termination_manager
  term_names = list(tm.active_terms)
  K = len(term_names)
  S = len(probe.contact_sensors)
  rec = {
    "valid": np.zeros((E, T), bool),
    "phase": np.full((E, T), -1, np.int64),
    "phase_steps": np.full((E, T), -1, np.int64),
    "action": np.full((E, T, 8), np.nan, np.float32),
    "finger_qpos_obs": np.full((E, T, 2), np.nan, np.float32),
    "finger_qpos_sim": np.full((E, T, 2), np.nan, np.float32),
    "aperture": np.full((E, T), np.nan, np.float32),
    "gripper_to_object": np.full((E, T, 3), np.nan, np.float32),
    "object_to_goal": np.full((E, T, 3), np.nan, np.float32),
    "dist_go": np.full((E, T), np.nan, np.float32),
    "dist_og": np.full((E, T), np.nan, np.float32),
    "gripper_pos": np.full((E, T, 3), np.nan, np.float32),
    "object_pos": np.full((E, T, 3), np.nan, np.float32),
    "object_quat": np.full((E, T, 4), np.nan, np.float32),
    "object_speed": np.full((E, T), np.nan, np.float32),
    "mech_joint": np.full((E, T), np.nan, np.float32),
    "goal_error": np.full((E, T), np.nan, np.float32),
    "success": np.zeros((E, T), bool),
    "done": np.zeros((E, T), bool),
    "term_flags": np.zeros((E, T, K), bool),
    "contact_found": np.zeros((E, T, S), bool),
  }
  obs, _ = env.reset()
  policy.reset()
  default_q = np.asarray(getattr(policy, "default_qpos", np.zeros(9)), dtype=np.float64)
  alive = np.ones(E, bool)
  pos0, _q0, _s0 = probe.object_state()
  spawn_z = pos0[:, 2].copy()
  # Targets are (re)sampled by the command term at reset, so read them only now.
  mech_target = probe._read_mech_target()
  friction = probe.fingertip_friction()

  for t in range(T):
    obs_np = obs["policy"].detach().cpu().numpy().astype(np.float64)
    # State BEFORE the action (what the teacher saw).
    jp = probe.obs["robot_joint_pos"]
    rec["finger_qpos_obs"][:, t] = obs_np[:, jp.start + 7:jp.start + 9] + default_q[7:9]
    fq = probe.finger_qpos()
    rec["finger_qpos_sim"][:, t] = fq
    ap = fq.sum(axis=1)
    bad = ~np.isfinite(ap)
    ap[bad] = rec["finger_qpos_obs"][bad, t].sum(axis=1)
    rec["aperture"][:, t] = ap
    gto = obs_np[:, probe.obs["gripper_to_object"]]
    otg = obs_np[:, probe.obs["object_to_goal"]]
    rec["gripper_to_object"][:, t] = gto
    rec["object_to_goal"][:, t] = otg
    rec["dist_go"][:, t] = np.linalg.norm(gto, axis=1)
    rec["dist_og"][:, t] = np.linalg.norm(otg, axis=1)
    rec["gripper_pos"][:, t] = obs_np[:, probe.obs["gripper_pos"]] - probe.origins[:E]
    opos, oquat, ospeed = probe.object_state()
    rec["object_pos"][:, t] = opos
    rec["object_quat"][:, t] = oquat
    rec["object_speed"][:, t] = ospeed
    rec["mech_joint"][:, t] = probe.mech_joint()
    # metrics are zero-initialised at reset and first computed inside step(): the
    # pre-step value at t=0 is meaningless, so leave it NaN.
    rec["goal_error"][:, t] = probe.goal_error() if t > 0 else np.nan
    rec["valid"][:, t] = alive

    actions = policy(obs_np.astype(np.float32))
    rec["action"][:, t] = actions
    # Phase AFTER the call: teachers apply transitions inside the call and then
    # compute the target, so this is the phase that produced this step's action.
    ph = _to_np(getattr(policy, "_phase", None))
    ps = _to_np(getattr(policy, "_phase_steps", None))
    if ph is not None:
      rec["phase"][:, t] = ph[:E]
    if ps is not None:
      rec["phase_steps"][:, t] = ps[:E]
    obs, _r, terminated, truncated, _info = env.step(torch.from_numpy(actions).to(device))

    term = _to_np(terminated).reshape(-1)[:E] > 0
    trunc = _to_np(truncated).reshape(-1)[:E] > 0
    done = term | trunc
    # Success is read BEFORE the auto-reset would clear it? No: env.step already
    # reset done envs, which clears episode_success for them. So for envs that
    # finished this step, success is whatever was latched at the previous step
    # (a success latched on the terminal step itself is then invisible, as it is to
    # test_classical.py).
    succ_now = probe.success()
    prev = rec["success"][:, t - 1] if t > 0 else np.zeros(E, bool)
    rec["success"][:, t] = np.where(done, prev, succ_now | prev) & alive
    rec["done"][:, t] = done & alive
    for k, name in enumerate(term_names):
      try:
        rec["term_flags"][:, t, k] = (_to_np(tm.get_term(name)).reshape(-1)[:E] > 0) & alive
      except Exception:  # noqa: BLE001
        pass
    if S:
      rec["contact_found"][:, t] = probe.contacts()

    if dump_every and (t % dump_every == 0 or done.any()):
      for i in range(min(print_envs, E)):
        if not alive[i]:
          continue
        print(
          f"ep{ep} t={t:3d} env{i:2d} ph={rec['phase'][i, t]:2d}/{rec['phase_steps'][i, t]:3d} "
          f"ap={ap[i]:.3f} |go|={rec['dist_go'][i, t]:.3f} |og|={rec['dist_og'][i, t]:.3f} "
          f"objz={opos[i, 2]:+.3f} mech={rec['mech_joint'][i, t]:+.3f} "
          f"grip={actions[i, 7]:+.1f} succ={int(rec['success'][i, t])} done={int(done[i])}",
          flush=True,
        )
    alive &= ~done
    if not alive.any():
      break

  rec["spawn_z"] = spawn_z
  rec["mech_target"] = mech_target
  rec["fingertip_friction"] = friction
  rec["term_names"] = term_names
  return rec


def _classify(s: dict, is_mech: bool) -> str:
  """Coarse failure class. Order matters: a termination beats everything; then a
  near miss (got within 1.5x the success threshold without latching); then "never
  got to the object"; then task-family specific reasons."""
  if s["termination"]:
    return f"terminated:{s['termination']}"
  thr = s.get("success_threshold")
  if thr and s["min_goal_error"] is not None and s["min_goal_error"] < 1.5 * thr:
    return "near_miss"
  if s["grasp_task"] and s["lifted"]:
    # Reached by definition (it lifted the object); a large |gripper_to_object|
    # afterwards is the object getting away, not the arm never arriving.
    return "grasp_lost" if s["lost_step"] is not None else "lifted_not_at_goal"
  if s["min_dist_go"] is not None and s["min_dist_go"] > 0.06:
    return "never_reached"
  if is_mech:
    if s["mech_max"] is not None and s["mech_target"] is not None:
      prog = abs(s["mech_max"]) / max(abs(s["mech_target"]), 1e-6)
      if prog < 0.1:
        return "mechanism_not_moved"
      if prog < 0.9:
        return "mechanism_short"
    return f"stalled_phase_{s['end_phase']}"
  if s["grasp_task"]:
    return "never_lifted"
  # Non-grasp free-object task (push / drag / strike / topple): did the object move
  # toward the goal at all?
  if s["og_progress"] is not None:
    if s["og_progress"] < 0.02:
      return "object_not_moved"
    return "object_moved_short"
  return f"stalled_phase_{s['end_phase']}"


def _is_grasp_task(rec: dict) -> bool:
  """A grasp teacher OPENS the gripper and CLOSES it later in the same episode;
  push / drag / strike teachers hold it closed (or open) throughout. Decided over
  the whole batch so an env that terminated before its first close is still
  analysed as a grasp episode."""
  g = rec["action"][:, :, 7]
  v = rec["valid"]
  for e in range(g.shape[0]):
    ge = g[e, v[e]]
    opened = np.nonzero(ge > 0)[0]
    closed = np.nonzero(ge < 0)[0]
    if opened.size and closed.size and closed.max() > opened.min():
      return True
  return False


def summarize_env(rec: dict, e: int, args, is_mech: bool, grasp_task: bool,
                  success_threshold: float | None = None) -> dict:
  v = rec["valid"][e]
  n = int(v.sum())
  if n == 0:
    return {"env": e, "outcome": "fail", "steps": 0, "class": "other"}
  last = n - 1
  ph = rec["phase"][e, :n]
  ap = rec["aperture"][e, :n]
  dgo = rec["dist_go"][e, :n]
  dog = rec["dist_og"][e, :n]
  counts = Counter(int(p) for p in ph)
  most = max(counts.items(), key=lambda kv: kv[1])[0] if counts else -1
  success = bool(rec["success"][e, :n].any())
  succ_step = int(np.argmax(rec["success"][e, :n])) if success else None
  term = None
  term_step = None
  for k, name in enumerate(rec["term_names"]):
    if name == "time_out":
      continue
    col = rec["term_flags"][e, :n, k]
    if col.any():
      term, term_step = name, int(np.argmax(col))
      break
  timed_out = bool(rec["done"][e, last]) and term is None

  def fmin(a):
    a = a[np.isfinite(a)]
    return float(a.min()) if a.size else None

  def fmax(a):
    a = a[np.isfinite(a)]
    return float(a.max()) if a.size else None

  def argmin_step(a):
    if not np.isfinite(a).any():
      return None
    return int(np.nanargmin(a))

  # Lift / loss analysis (free objects only).
  lifted = None
  lift_step = None
  max_rise = None
  lost_step = None
  z0 = float(rec["spawn_z"][e]) if np.isfinite(rec["spawn_z"][e]) else None
  og_progress = None
  if not is_mech and z0 is not None:
    if np.isfinite(dog[0]) and np.isfinite(dog).any():
      og_progress = float(dog[0] - np.nanmin(dog))
    rise = rec["object_pos"][e, :n, 2] - z0
    max_rise = fmax(rise)
    up = np.isfinite(rise) & (rise > args.lift_thresh)
    lifted = bool(up.any())
    if lifted:
      lift_step = int(np.argmax(up))
      for t in range(lift_step + 1, n):
        if (
          np.isfinite(ap[t]) and ap[t] < args.lost_aperture
          and np.isfinite(dgo[t]) and np.isfinite(dgo[t - 1])
          and dgo[t] > dgo[t - 1] + 0.003 and dgo[t] > 0.03
        ):
          lost_step = t
          break
  mech_max = None
  mech_final = None
  mech_target = None
  if is_mech:
    mj = rec["mech_joint"][e, :n]
    if np.isfinite(mj).any():
      tgt = rec["mech_target"][e]
      mech_target = float(tgt) if np.isfinite(tgt) else None
      # "max" in the direction of the target (signed progress).
      sgn = np.sign(mech_target) if mech_target else 1.0
      mech_max = float((mj * sgn).max() * sgn)
      mech_final = float(mj[last])
  ge = rec["goal_error"][e, :n]
  s = {
    "env": e,
    "outcome": "success" if success else "fail",
    "success_step": succ_step,
    "steps": n,
    "timed_out": timed_out,
    "end_phase": int(ph[last]),
    "most_phase": int(most),
    "steps_in_phase": {str(k): int(c) for k, c in sorted(counts.items())},
    "aperture_min": fmin(ap),
    "aperture_max": fmax(ap),
    "aperture_final": float(ap[last]) if np.isfinite(ap[last]) else None,
    "min_dist_go": fmin(dgo),
    "min_dist_go_step": argmin_step(dgo),
    "final_dist_go": float(dgo[last]) if np.isfinite(dgo[last]) else None,
    "min_dist_og": fmin(dog),
    "min_dist_og_step": argmin_step(dog),
    "final_dist_og": float(dog[last]) if np.isfinite(dog[last]) else None,
    "min_goal_error": fmin(ge),
    "min_goal_error_step": argmin_step(ge),
    "success_threshold": success_threshold,
    "final_goal_error": float(ge[last]) if np.isfinite(ge[last]) else None,
    "termination": term,
    "termination_step": term_step,
    "grasp_task": bool(grasp_task),
    "og_progress": og_progress,
    "lifted": lifted,
    "lift_step": lift_step,
    "max_rise": max_rise,
    "lost_step": lost_step,
    "mech_max": mech_max,
    "mech_final": mech_final,
    "mech_target": mech_target,
    "fingertip_friction": (
      float(rec["fingertip_friction"][e]) if np.isfinite(rec["fingertip_friction"][e]) else None
    ),
  }
  s["class"] = "success" if success else _classify(s, is_mech)
  return s


def _f(x, nd=3):
  return "null" if x is None else f"{x:.{nd}f}"


def _pn(ph: int, names: dict[int, str]) -> str:
  return f"{ph} ({names[ph]})" if ph in names else str(ph)


def write_summary_md(path: Path, meta: dict, envs: list[dict], names: dict[int, str]) -> None:
  fails = [s for s in envs if s["outcome"] == "fail"]
  succs = [s for s in envs if s["outcome"] == "success"]
  L = []
  L.append(f"# diagnose — {meta['task']}\n")
  L.append(
    f"{meta['date']} · HEAD `{meta['head']}` · n = {meta['n']} "
    f"({meta['num_episodes']} × {meta['num_envs']} envs) · episode_length {meta['episode_length']} "
    f"· device {meta['device']}  "
  )
  L.append(
    f"**{len(succs)}/{len(envs)} succeeded** (orientation only — quote SR from "
    "`test_classical` at n = 128 on the frozen spec).\n"
  )
  if meta.get("obs_slices"):
    L.append("Obs slices used: " + ", ".join(f"{k} {v[0]}:{v[1]}" for k, v in meta["obs_slices"].items()) + "  ")
  L.append(f"Teacher `{meta['policy_class']}`; primary object `{meta['object_entity']}`; "
           f"mechanism joint `{meta['mech_joint']}`; termination terms {meta['term_names']}; "
           f"contact sensors {meta['contact_names']}.  ")
  if names:
    L.append("Phases: " + ", ".join(f"{k} = {v}" for k, v in sorted(names.items())) + "\n")
  else:
    L.append("Phases: unnamed — numbers are the teacher's `_phase` values.\n")

  def hist(rows, key):
    c = Counter(r[key] for r in rows)
    return ", ".join(f"{_pn(k, names) if isinstance(k, int) else k}: {v}" for k, v in
                     sorted(c.items(), key=lambda kv: -kv[1])) or "—"

  L.append("## Histograms\n")
  L.append(f"- end phase, FAILING envs ({len(fails)}): {hist(fails, 'end_phase')}")
  L.append(f"- end phase, successful envs ({len(succs)}): {hist(succs, 'end_phase')}")
  L.append(f"- most-steps phase, failing envs: {hist(fails, 'most_phase')}")
  L.append(f"- failure class: {hist(fails, 'class')}")
  L.append(f"- termination among failing envs: {hist([{'t': s['termination'] or 'none'} for s in fails], 't')}")
  if fails and any(s.get("grasp_task") for s in fails):
    lifted = sum(1 for s in fails if s["lifted"])
    lost = sum(1 for s in fails if s["lost_step"] is not None)
    L.append(f"- failing envs that lifted the object: {lifted}/{len(fails)}; lifted then lost: {lost}")
  fr = [s["fingertip_friction"] for s in envs if s["fingertip_friction"] is not None]
  if fr:
    ffr = [s["fingertip_friction"] for s in fails if s["fingertip_friction"] is not None]
    sfr = [s["fingertip_friction"] for s in succs if s["fingertip_friction"] is not None]
    L.append(
      f"- fingertip friction: failing mean {_f(np.mean(ffr) if ffr else None, 2)} "
      f"(min {_f(min(ffr) if ffr else None, 2)}), successful mean {_f(np.mean(sfr) if sfr else None, 2)} "
      f"(min {_f(min(sfr) if sfr else None, 2)})"
    )
  if succs:
    ss = [s["success_step"] for s in succs if s["success_step"] is not None]
    if ss:
      L.append(f"- success step (successful envs): median {int(np.median(ss))}, max {max(ss)}")
  L.append("")

  # One-line verdict per failure class.
  if fails:
    L.append("## Reading\n")
    by_class: dict[str, list[dict]] = {}
    for s in fails:
      by_class.setdefault(s["class"], []).append(s)
    for cls, rows in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
      ends = Counter(r["end_phase"] for r in rows).most_common(1)[0][0]
      ap = [r["aperture_final"] for r in rows if r["aperture_final"] is not None]
      go = [r["min_dist_go"] for r in rows if r["min_dist_go"] is not None]
      og = [r["min_dist_og"] for r in rows if r["min_dist_og"] is not None]
      line = (
        f"- **{cls}** — {len(rows)}/{len(fails)} failing envs (envs {[r['env'] for r in rows]}); "
        f"typical end phase {_pn(ends, names)}; final aperture median {_f(np.median(ap) if ap else None)}; "
        f"closest |gripper_to_object| median {_f(np.median(go) if go else None)}; "
        f"closest |object_to_goal| median {_f(np.median(og) if og else None)}"
      )
      gem = [r["min_goal_error"] for r in rows if r["min_goal_error"] is not None]
      if gem:
        line += f"; min goal_error median {_f(np.median(gem))}"
        if rows[0].get("success_threshold"):
          line += f" (success < {_f(rows[0]['success_threshold'])})"
      mm = [r["mech_max"] for r in rows if r["mech_max"] is not None]
      if mm:
        line += f"; mechanism reached median {_f(np.median(mm))} of target {_f(rows[0]['mech_target'])}"
      if cls == "grasp_lost":
        ls = [r["lost_step"] for r in rows]
        line += f"; lost at step median {int(np.median(ls))}"
      L.append(line)
    L.append("")

  L.append("## Failing envs\n")
  if not fails:
    L.append("(none)\n")
  for s in fails:
    L.append(f"### env {s['env']} — {s['class']}")
    L.append(
      f"- ended in phase {_pn(s['end_phase'], names)} after {s['steps']} steps"
      + (f"; TERMINATED by `{s['termination']}` at step {s['termination_step']}" if s["termination"] else
         ("; ran to time-out" if s["timed_out"] else "; episode cut short (no done flag)"))
    )
    L.append(f"- most steps in phase {_pn(s['most_phase'], names)}; steps per phase "
             + ", ".join(f"{_pn(int(k), names)}: {v}" for k, v in s["steps_in_phase"].items()))
    L.append(f"- aperture min {_f(s['aperture_min'])} / max {_f(s['aperture_max'])} / final {_f(s['aperture_final'])}")
    L.append(f"- closest |gripper_to_object| {_f(s['min_dist_go'])} at step {s['min_dist_go_step']} (final {_f(s['final_dist_go'])})")
    L.append(f"- closest |object_to_goal| {_f(s['min_dist_og'])} at step {s['min_dist_og_step']} (final {_f(s['final_dist_og'])}); "
             f"goal_error min {_f(s['min_goal_error'])} at step {s['min_goal_error_step']} / final {_f(s['final_goal_error'])}"
             + (f" (success < {_f(s['success_threshold'])})" if s.get("success_threshold") else ""))
    if s["lifted"] is not None:
      if s["lifted"]:
        L.append(f"- object LIFTED at step {s['lift_step']} (max rise {_f(s['max_rise'])} m)"
                 + (f"; LOST at step {s['lost_step']} (aperture collapsed while the object moved away)"
                    if s["lost_step"] is not None else "; never lost"))
      elif s["grasp_task"]:
        L.append(f"- object never lifted (max rise {_f(s['max_rise'])} m)")
      else:
        L.append(f"- non-grasp task: object moved {_f(s['og_progress'])} m toward the goal "
                 f"(|object_to_goal| {_f(s['final_dist_og'])} at the end); max rise {_f(s['max_rise'])} m"
                 + (" — object TIPPED / climbed the pusher" if (s['max_rise'] or 0) > 0.02 else ""))
    if s["mech_target"] is not None:
      L.append(f"- mechanism joint: reached {_f(s['mech_max'])} (final {_f(s['mech_final'])}) of target {_f(s['mech_target'])}")
    if s["fingertip_friction"] is not None:
      L.append(f"- fingertip friction {s['fingertip_friction']:.2f}")
    L.append("")

  L.append("## Successful envs (one line each)\n")
  for s in succs:
    L.append(
      f"- env {s['env']}: success at step {s['success_step']}, end phase {_pn(s['end_phase'], names)}, "
      f"aperture min {_f(s['aperture_min'])}, min |go| {_f(s['min_dist_go'])}"
      + (f", lifted at {s['lift_step']}" if s.get("lifted") else "")
      + (f", friction {s['fingertip_friction']:.2f}" if s["fingertip_friction"] is not None else "")
    )
  path.write_text("\n".join(L) + "\n")


def main() -> None:
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--task", default=None, choices=sorted(CLASSICAL_POLICIES))
  ap.add_argument("--num-envs", type=int, default=32)
  ap.add_argument("--num-episodes", type=int, default=1)
  ap.add_argument("--out", type=Path, default=None)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--seed", type=int, default=None)
  ap.add_argument("--dump-every", type=int, default=0,
                  help="print a per-step line for the first --print-envs envs every N steps (0 = off)")
  ap.add_argument("--print-envs", type=int, default=2)
  ap.add_argument("--lift-thresh", type=float, default=0.02, help="m above spawn z that counts as lifted")
  ap.add_argument("--lost-aperture", type=float, default=0.015,
                  help="total finger aperture (m) below which the fingers are 'closed on nothing'")
  ap.add_argument("--success-threshold", type=float, default=None,
                  help="override the command term's success_threshold used for the "
                       "near_miss class (normally read from the env cfg)")
  ap.add_argument("--resummarize", type=Path, default=None, metavar="DIR",
                  help="skip the rollout: rebuild summary.md/json from DIR/trace.npz "
                       "(+ meta from DIR/summary.json). Use it to retune thresholds.")
  args = ap.parse_args()

  if args.resummarize is not None:
    _resummarize(args.resummarize, args)
    return

  if args.task is None or args.out is None:
    ap.error("--task and --out are required (unless --resummarize DIR)")
  configure_torch_backends()
  device = args.device
  if device.startswith("cuda") and not torch.cuda.is_available():
    print(f"WARNING: cuda not available, falling back to cpu (requested {device})")
    device = "cpu"

  env_cfg = load_env_cfg(args.task, play=False)
  env_cfg.scene.num_envs = args.num_envs
  if args.seed is not None:
    env_cfg.seed = args.seed
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  policy = CLASSICAL_POLICIES[args.task](num_envs=args.num_envs)
  probe = _SceneProbe(env, args.num_envs)
  names = _phase_names(policy)
  is_mech = probe.mech is not None
  print(f"[diagnose] {args.task}: episode_length={int(env.max_episode_length)} "
        f"object={probe.obj_name} mech_joint={probe.mech_name} phases={names or 'unnamed'} "
        f"obs_slices={ {k: (v.start, v.stop) for k, v in probe.obs.items()} } "
        f"terms={list(env.termination_manager.active_terms)}", flush=True)

  recs = []
  try:
    for ep in range(args.num_episodes):
      rec = run_episode(env, policy, probe, args.num_envs, device, args.dump_every, args.print_envs, ep)
      recs.append(rec)
      ns = int(rec["success"].any(axis=1).sum())
      print(f"[diagnose] episode {ep}: {ns}/{args.num_envs} succeeded", flush=True)
  finally:
    env.close()

  # Concatenate episodes along the env axis.
  keys = [k for k in recs[0] if isinstance(recs[0][k], np.ndarray)]
  trace = {k: np.concatenate([r[k] for r in recs], axis=0) for k in keys}
  trace["term_names"] = np.array(recs[0]["term_names"])
  trace["contact_names"] = np.array(probe.contact_names)
  trace["phase_names"] = np.array([f"{k}={v}" for k, v in sorted(names.items())])
  merged = dict(trace)
  merged["term_names"] = recs[0]["term_names"]

  meta = {
    "task": args.task,
    "policy_class": type(policy).__name__,
    "date": datetime.now().isoformat(timespec="seconds"),
    "head": _git_head(),
    "device": device,
    "n": int(trace["valid"].shape[0]),
    "num_envs": args.num_envs,
    "num_episodes": args.num_episodes,
    "episode_length": int(trace["valid"].shape[1]),
    "object_entity": probe.obj_name,
    "mech_joint": probe.mech_name,
    "is_mech": is_mech,
    "success_threshold": probe.success_threshold,
    "obs_slices": {k: [v.start, v.stop] for k, v in probe.obs.items()},
    "term_names": recs[0]["term_names"],
    "contact_names": probe.contact_names,
    "phase_names": {str(k): v for k, v in names.items()},
  }
  args.out.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(args.out / "trace.npz", **trace)
  _write_summaries(args.out, merged, meta, names, args)
  print(f"[diagnose] wrote {args.out}/trace.npz, summary.json, summary.md")


def _write_summaries(out: Path, merged: dict, meta: dict, names: dict[int, str], args) -> None:
  E = merged["valid"].shape[0]
  is_mech = bool(meta.get("is_mech"))
  grasp_task = (not is_mech) and _is_grasp_task(merged)
  thr = args.success_threshold if args.success_threshold is not None else meta.get("success_threshold")
  envs = [summarize_env(merged, e, args, is_mech, grasp_task, thr) for e in range(E)]
  n_succ = sum(s["outcome"] == "success" for s in envs)
  meta = dict(meta)
  meta["success_threshold"] = thr
  meta["num_success"] = n_succ
  meta["grasp_task"] = grasp_task
  meta["lift_thresh"] = args.lift_thresh
  meta["lost_aperture"] = args.lost_aperture
  fails = [s for s in envs if s["outcome"] == "fail"]
  succs = [s for s in envs if s["outcome"] == "success"]
  hists = {
    "end_phase_failing": dict(Counter(str(s["end_phase"]) for s in fails)),
    "end_phase_success": dict(Counter(str(s["end_phase"]) for s in succs)),
    "most_phase_failing": dict(Counter(str(s["most_phase"]) for s in fails)),
    "failure_class": dict(Counter(s["class"] for s in fails)),
    "termination_failing": dict(Counter(str(s["termination"]) for s in fails)),
  }
  (out / "summary.json").write_text(json.dumps({"meta": meta, "histograms": hists, "envs": envs}, indent=1))
  write_summary_md(out / "summary.md", meta, envs, names)
  print(f"[diagnose] {n_succ}/{E} succeeded; failure classes {hists['failure_class']}")


def _resummarize(out: Path, args) -> None:
  """Rebuild summary.md / summary.json from an existing trace.npz."""
  t = np.load(out / "trace.npz")
  merged = {k: t[k] for k in t.files}
  merged["term_names"] = [str(x) for x in t["term_names"]]
  meta = {}
  if (out / "summary.json").exists():
    meta = json.loads((out / "summary.json").read_text()).get("meta", {})
  names = {}
  for item in t["phase_names"]:
    k, _, v = str(item).partition("=")
    names[int(k)] = v
  if not names and meta.get("phase_names"):
    names = {int(k): v for k, v in meta["phase_names"].items()}
  if "is_mech" not in meta:
    meta["is_mech"] = bool(np.isfinite(t["mech_joint"]).any())
  if not names and meta.get("policy_class"):
    # Same lookup as at rollout time, minus the module scan (no policy instance here).
    for key, table in _PHASE_TABLE.items():
      if key in meta["policy_class"]:
        names = dict(table)
        break
  meta["phase_names"] = {str(k): v for k, v in names.items()}
  meta["resummarized"] = datetime.now().isoformat(timespec="seconds")
  _write_summaries(out, merged, meta, names, args)
  print(f"[diagnose] re-summarized {out}")


if __name__ == "__main__":
  main()
